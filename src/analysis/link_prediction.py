"""
Link prediction for airport network with time-split evaluation.

This module implements:
- Time-based train/test splitting (early months vs. later months)
- Negative sampling (non-edges that don't appear in test set)
- Baseline heuristics (common neighbors, Jaccard, Adamic-Adar)
- Embedding-based classification (logistic regression)
- Evaluation metrics (AUC, average precision)

Follows WS4 contract: consumes cleaned flight data and WS1 network outputs
to predict new routes appearing in test period.
"""

import logging
from pathlib import Path
from typing import Optional

import igraph as ig
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger(__name__)


def build_month_split_graphs(
    cleaned_path: str | Path,
    train_months: list[int],
    test_months: list[int],
    filters: dict,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], dict[str, int]]:
    """
    Build train and test edge sets from flight data based on month split.

    Parameters
    ----------
    cleaned_path : str or Path
        Path to cleaned flights parquet
    train_months : list[int]
        Months for training (e.g., [1, 2, 3, 4, 5, 6, 7, 8, 9])
    test_months : list[int]
        Months for testing (e.g., [10, 11, 12])
    filters : dict
        Data filters (year, include_cancelled, carriers)

    Returns
    -------
    train_edges : set[tuple[int, int]]
        Directed edges (origin_id, dest_id) in train period
    test_edges : set[tuple[int, int]]
        New edges appearing in test period (not in train)
    code_to_id : dict[str, int]
        Mapping from airport code to vertex_id

    Notes
    -----
    - Ensures no data leakage: test edges are routes present in test but absent in train
    - Filters cancelled flights if specified
    """
    logger.info(
        f"Building month split: train_months={train_months}, test_months={test_months}"
    )

    lf = pl.scan_parquet(cleaned_path)

    # Apply filters
    year = filters.get("year")
    if year:
        lf = lf.filter(pl.col("YEAR") == year)

    include_cancelled = filters.get("include_cancelled", False)
    if not include_cancelled:
        lf = lf.filter(pl.col("CANCELLED") == 0)

    carriers = filters.get("carriers", "ALL")
    if carriers != "ALL":
        lf = lf.filter(pl.col("OP_UNIQUE_CARRIER").is_in(carriers))

    # Select relevant columns
    lf = lf.select(["MONTH", "ORIGIN", "DEST"])

    # Split by month
    train_lf = lf.filter(pl.col("MONTH").is_in(train_months))
    test_lf = lf.filter(pl.col("MONTH").is_in(test_months))

    train_df = train_lf.collect()
    test_df = test_lf.collect()

    logger.info(f"Train flights: {len(train_df)}, Test flights: {len(test_df)}")

    # Build airport code mapping
    all_airports = set(train_df["ORIGIN"].to_list() + train_df["DEST"].to_list())
    all_airports.update(test_df["ORIGIN"].to_list() + test_df["DEST"].to_list())
    code_to_id = {code: idx for idx, code in enumerate(sorted(all_airports))}

    logger.info(f"Total airports: {len(code_to_id)}")

    # Build edge sets
    train_edges = set()
    for row in train_df.iter_rows(named=True):
        u = code_to_id[row["ORIGIN"]]
        v = code_to_id[row["DEST"]]
        train_edges.add((u, v))

    test_edges_all = set()
    for row in test_df.iter_rows(named=True):
        u = code_to_id[row["ORIGIN"]]
        v = code_to_id[row["DEST"]]
        test_edges_all.add((u, v))

    # Test positives: edges in test but NOT in train
    test_edges = test_edges_all - train_edges

    logger.info(
        f"Train edges: {len(train_edges)}, "
        f"Test edges (all): {len(test_edges_all)}, "
        f"Test edges (new): {len(test_edges)}"
    )

    return train_edges, test_edges, code_to_id


def build_igraph_from_edges(
    edges: set[tuple[int, int]], n_nodes: int, directed: bool = True
) -> ig.Graph:
    """
    Build igraph Graph from edge set.

    Parameters
    ----------
    edges : set[tuple[int, int]]
        Set of directed edges
    n_nodes : int
        Number of nodes
    directed : bool
        Whether graph is directed

    Returns
    -------
    ig.Graph
    """
    edge_list = list(edges)
    g = ig.Graph(n=n_nodes, edges=edge_list, directed=directed)
    logger.info(f"Built graph: N={g.vcount()}, E={g.ecount()}")
    return g


def negative_sample_non_edges(
    n_nodes: int,
    train_edges: set[tuple[int, int]],
    test_positives: set[tuple[int, int]],
    ratio: int,
    seed: int,
) -> set[tuple[int, int]]:
    """
    Sample negative edges (non-edges) that don't appear in train or test.

    Parameters
    ----------
    n_nodes : int
        Number of nodes in graph
    train_edges : set[tuple[int, int]]
        Edges in train graph
    test_positives : set[tuple[int, int]]
        Positive edges in test set
    ratio : int
        Negative-to-positive ratio
    seed : int
        Random seed

    Returns
    -------
    set[tuple[int, int]]
        Sampled negative edges

    Notes
    -----
    Ensures negatives are not in train_edges or test_positives to avoid leakage.
    """
    logger.info(
        f"Negative sampling: n_nodes={n_nodes}, ratio={ratio}, seed={seed}"
    )

    rng = np.random.default_rng(seed)

    n_negatives = len(test_positives) * ratio
    negatives = set()

    # All possible edges (excluding train and test positives)
    forbidden = train_edges | test_positives

    attempts = 0
    max_attempts = n_negatives * 100

    while len(negatives) < n_negatives and attempts < max_attempts:
        u = rng.integers(0, n_nodes)
        v = rng.integers(0, n_nodes)
        if u != v and (u, v) not in forbidden and (u, v) not in negatives:
            negatives.add((u, v))
        attempts += 1

    if len(negatives) < n_negatives:
        logger.warning(
            f"Only sampled {len(negatives)} negatives (target: {n_negatives})"
        )

    logger.info(f"Sampled {len(negatives)} negative edges")
    return negatives


def compute_heuristic_features(
    g: ig.Graph, edge_pairs: list[tuple[int, int]]
) -> np.ndarray:
    """
    Compute baseline heuristic features for edge pairs.

    Features:
        - Common neighbors (CN)
        - Jaccard coefficient
        - Adamic-Adar index
        - Preferential attachment (product of degrees)

    Parameters
    ----------
    g : ig.Graph
        Train graph
    edge_pairs : list[tuple[int, int]]
        List of (u, v) node pairs

    Returns
    -------
    np.ndarray
        Shape (len(edge_pairs), 4) feature matrix
    """
    logger.info(f"Computing heuristic features for {len(edge_pairs)} pairs")

    features = []
    for u, v in edge_pairs:
        u_nbs = set(g.neighbors(u, mode="out"))
        v_nbs = set(g.neighbors(v, mode="out"))

        # Common neighbors
        cn = len(u_nbs & v_nbs)

        # Jaccard
        union_size = len(u_nbs | v_nbs)
        jaccard = cn / union_size if union_size > 0 else 0.0

        # Adamic-Adar
        aa = 0.0
        for w in u_nbs & v_nbs:
            deg_w = g.degree(w, mode="out")
            if deg_w > 1:
                aa += 1.0 / np.log(deg_w)

        # Preferential attachment
        pa = len(u_nbs) * len(v_nbs)

        features.append([cn, jaccard, aa, pa])

    return np.array(features)


def combine_features(
    heuristic_feats: np.ndarray, embedding_feats: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Combine heuristic and embedding features.

    Parameters
    ----------
    heuristic_feats : np.ndarray
        Shape (N, 4) baseline features
    embedding_feats : np.ndarray, optional
        Shape (N, D) embedding-derived features

    Returns
    -------
    np.ndarray
        Concatenated feature matrix
    """
    if embedding_feats is None:
        return heuristic_feats
    return np.hstack([heuristic_feats, embedding_feats])


def evaluate_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> tuple[LogisticRegression, dict]:
    """
    Train logistic regression classifier and evaluate.

    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels (1=positive, 0=negative)
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    seed : int
        Random seed

    Returns
    -------
    model : LogisticRegression
        Fitted classifier
    metrics : dict
        Evaluation metrics (auc, avg_precision)
    """
    logger.info(
        f"Training classifier: train_size={len(X_train)}, test_size={len(X_test)}"
    )

    model = LogisticRegression(
        random_state=seed,
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
    )

    model.fit(X_train, y_train)

    # Predict probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    avg_prec = average_precision_score(y_test, y_pred_proba)

    metrics = {"auc": float(auc), "avg_precision": float(avg_prec)}

    logger.info(f"Metrics: AUC={auc:.4f}, AvgPrec={avg_prec:.4f}")

    return model, metrics


def rank_candidate_edges(
    model: LogisticRegression,
    candidate_edges: list[tuple[int, int]],
    features: np.ndarray,
    id_to_code: dict[int, str],
    top_k: int = 100,
) -> pl.DataFrame:
    """
    Rank candidate edges by predicted probability.

    Parameters
    ----------
    model : LogisticRegression
        Fitted classifier
    candidate_edges : list[tuple[int, int]]
        List of (u, v) pairs to rank
    features : np.ndarray
        Feature matrix for candidates
    id_to_code : dict[int, str]
        Mapping from vertex_id to airport code
    top_k : int
        Number of top predictions to return

    Returns
    -------
    pl.DataFrame
        Columns: origin, dest, score, rank
    """
    logger.info(f"Ranking {len(candidate_edges)} candidate edges")

    scores = model.predict_proba(features)[:, 1]

    # Sort by score descending
    sorted_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(sorted_indices, start=1):
        u, v = candidate_edges[idx]
        origin = id_to_code.get(u, f"ID{u}")
        dest = id_to_code.get(v, f"ID{v}")
        score = float(scores[idx])
        results.append({"origin": origin, "dest": dest, "score": score, "rank": rank})

    return pl.DataFrame(results)


def evaluate_baseline_heuristics(
    g_train: ig.Graph,
    test_positives: list[tuple[int, int]],
    test_negatives: list[tuple[int, int]],
) -> dict:
    """
    Evaluate baseline heuristics without machine learning.

    Parameters
    ----------
    g_train : ig.Graph
        Train graph
    test_positives : list[tuple[int, int]]
        Positive test edges
    test_negatives : list[tuple[int, int]]
        Negative test edges

    Returns
    -------
    dict
        Metrics for each heuristic (CN, Jaccard, AA)
    """
    logger.info("Evaluating baseline heuristics")

    all_test_edges = test_positives + test_negatives
    y_true = np.array([1] * len(test_positives) + [0] * len(test_negatives))

    # Handle empty test set
    if len(all_test_edges) == 0:
        logger.warning("No test edges available for evaluation")
        feature_names = ["common_neighbors", "jaccard", "adamic_adar", "preferential_attachment"]
        return {name: {"auc": 0.0, "avg_precision": 0.0} for name in feature_names}

    # Compute features
    feats = compute_heuristic_features(g_train, all_test_edges)

    metrics = {}
    feature_names = ["common_neighbors", "jaccard", "adamic_adar", "preferential_attachment"]

    for i, name in enumerate(feature_names):
        scores = feats[:, i]
        # Handle case where all scores are the same
        if len(np.unique(scores)) > 1:
            auc = roc_auc_score(y_true, scores)
            avg_prec = average_precision_score(y_true, scores)
        else:
            auc = 0.5
            avg_prec = y_true.mean()

        metrics[name] = {"auc": float(auc), "avg_precision": float(avg_prec)}

    logger.info(f"Baseline metrics: {metrics}")
    return metrics

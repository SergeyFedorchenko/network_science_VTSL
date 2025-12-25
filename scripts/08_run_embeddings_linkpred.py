#!/usr/bin/env python
"""
WS4 Script 08: Run embeddings and link prediction.

Consumes WS1 outputs (airport network) and cleaned flight data.
Trains node2vec embeddings on train graph (time-split).
Evaluates link prediction (baseline heuristics + embedding-based classifier).
Writes results under results/analysis and results/tables.
Writes run manifest JSON.

Usage:
    python scripts/08_run_embeddings_linkpred.py
"""

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis.embeddings import (
    find_similar_airports,
    generate_node2vec_walks,
    get_embedding_pair_features,
    load_airport_graph_from_parquet,
    train_skipgram,
    write_embeddings,
)
from analysis.link_prediction import (
    build_igraph_from_edges,
    build_month_split_graphs,
    combine_features,
    compute_heuristic_features,
    evaluate_baseline_heuristics,
    evaluate_classifier,
    negative_sample_non_edges,
    rank_candidate_edges,
)
from utils.logging import setup_logging
from utils.paths import get_project_root
from utils.seeds import set_global_seed


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def load_config() -> dict:
    """Load config from config/config.yaml."""
    root = get_project_root()
    config_path = root / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    """Main execution."""
    # Setup
    root = get_project_root()
    log_dir = root / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(log_dir / "08_run_embeddings_linkpred.log")
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("WS4 Script 08: Embeddings + Link Prediction")
    logger.info("=" * 80)

    # Load config
    config = load_config()
    logger.info(f"Loaded config: seed={config.get('seed')}")

    # Set seed
    seed = config.get("seed", 42)
    set_global_seed(seed)
    logger.info(f"Set global seed: {seed}")

    # Extract config parameters
    emb_cfg = config.get("analysis", {}).get("embeddings", {})
    lp_cfg = config.get("analysis", {}).get("link_prediction", {})
    filters = config.get("filters", {})

    # Paths
    cleaned_path = root / config["data"]["cleaned_path"]
    networks_dir = root / "results" / "networks"
    analysis_dir = root / "results" / "analysis"
    tables_dir = root / "results" / "tables"

    analysis_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # PART 1: Build time-split graphs
    # ========================================================================
    logger.info("-" * 80)
    logger.info("PART 1: Building time-split graphs")
    logger.info("-" * 80)

    train_months = lp_cfg.get("time_split", {}).get("train_months", [1, 2, 3, 4, 5, 6, 7, 8, 9])
    test_months = lp_cfg.get("time_split", {}).get("test_months", [10, 11, 12])

    train_edges, test_edges, code_to_id = build_month_split_graphs(
        cleaned_path=cleaned_path,
        train_months=train_months,
        test_months=test_months,
        filters=filters,
    )

    n_nodes = len(code_to_id)
    id_to_code = {v: k for k, v in code_to_id.items()}

    logger.info(f"Train edges: {len(train_edges)}, Test new edges: {len(test_edges)}")

    # Build train graph
    g_train = build_igraph_from_edges(train_edges, n_nodes, directed=True)

    # ========================================================================
    # PART 2: Train embeddings on train graph
    # ========================================================================
    logger.info("-" * 80)
    logger.info("PART 2: Training node2vec embeddings")
    logger.info("-" * 80)

    num_walks = emb_cfg.get("num_walks", 10)
    walk_length = emb_cfg.get("walk_length", 80)
    p = emb_cfg.get("p", 1.0)
    q = emb_cfg.get("q", 1.0)
    dimensions = emb_cfg.get("dimensions", 128)
    window_size = emb_cfg.get("window_size", 10)

    walks = generate_node2vec_walks(
        g=g_train,
        num_walks=num_walks,
        walk_length=walk_length,
        p=p,
        q=q,
        seed=seed,
        weight_col=None,  # Unweighted for link prediction
    )

    model = train_skipgram(
        walks=walks,
        dimensions=dimensions,
        window_size=window_size,
        seed=seed,
    )

    # Write embeddings
    embeddings_path = analysis_dir / "airport_embeddings.parquet"
    overwrite = config.get("outputs", {}).get("overwrite", False)
    if not embeddings_path.exists() or overwrite:
        write_embeddings(model, id_to_code, embeddings_path)
    else:
        logger.warning(f"Skipped (exists): {embeddings_path}")

    # Find similar airports for major hubs
    major_hubs = ["ATL", "ORD", "DFW", "DEN", "LAX", "SFO", "JFK", "LAS", "PHX", "IAH"]
    major_hubs = [h for h in major_hubs if h in code_to_id]

    if major_hubs:
        logger.info(f"Finding similar airports for {len(major_hubs)} major hubs")
        neighbors_df = find_similar_airports(
            embeddings_path=embeddings_path,
            query_codes=major_hubs,
            top_k=10,
        )
        neighbors_path = tables_dir / "airport_embedding_neighbors.csv"
        if not neighbors_path.exists() or overwrite:
            neighbors_df.write_csv(neighbors_path)
            logger.info(f"Wrote embedding neighbors: {neighbors_path}")

    # ========================================================================
    # PART 3: Link prediction evaluation
    # ========================================================================
    logger.info("-" * 80)
    logger.info("PART 3: Link prediction evaluation")
    logger.info("-" * 80)

    # Check if we have test edges
    if len(test_edges) == 0:
        logger.warning("No new test edges found. Link prediction evaluation will be skipped.")
        logger.warning("This may happen if test months don't have new routes or data is incomplete.")
        
        # Write empty outputs
        metrics = {
            "baseline_heuristics": {},
            "embedding_classifier": {},
            "note": "No test edges available for evaluation"
        }
        
        metrics_path = analysis_dir / "linkpred_metrics.json"
        if not metrics_path.exists() or overwrite:
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Wrote empty metrics: {metrics_path}")
        
        # Write empty predictions
        import polars as pl
        top_predictions_df = pl.DataFrame({"origin": [], "dest": [], "score": [], "rank": []})
        predictions_path = tables_dir / "linkpred_top_predictions.csv"
        if not predictions_path.exists() or overwrite:
            top_predictions_df.write_csv(predictions_path)
            logger.info(f"Wrote empty predictions: {predictions_path}")
        
        # Write manifest and exit
        manifest = {
            "script": "08_run_embeddings_linkpred.py",
            "timestamp": datetime.now().isoformat(),
            "git_commit": get_git_commit(),
            "config_snapshot": {
                "seed": seed,
                "train_months": train_months,
                "test_months": test_months,
            },
            "inputs": {
                "cleaned_path": str(cleaned_path),
            },
            "outputs": {
                "embeddings": str(embeddings_path),
                "metrics": str(metrics_path),
                "predictions": str(predictions_path),
            },
            "graph_summary": {
                "n_nodes": n_nodes,
                "train_edges": len(train_edges),
                "test_edges_new": 0,
                "test_negatives": 0,
            },
            "metrics": metrics,
            "note": "Link prediction skipped due to no test edges"
        }
        
        manifest_path = log_dir / "08_run_embeddings_linkpred_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Wrote manifest: {manifest_path}")
        
        logger.info("=" * 80)
        logger.info("WS4 Script 08: Complete (embeddings only)")
        logger.info("=" * 80)
        return

    # Negative sampling
    negative_ratio = lp_cfg.get("negative_ratio", 5)
    test_negatives = negative_sample_non_edges(
        n_nodes=n_nodes,
        train_edges=train_edges,
        test_positives=test_edges,
        ratio=negative_ratio,
        seed=seed,
    )

    # Prepare test sets
    test_positives_list = list(test_edges)
    test_negatives_list = list(test_negatives)

    # Baseline heuristics
    logger.info("Evaluating baseline heuristics")
    baseline_metrics = evaluate_baseline_heuristics(
        g_train=g_train,
        test_positives=test_positives_list,
        test_negatives=test_negatives_list,
    )

    # Embedding-based classifier
    logger.info("Training embedding-based classifier")

    # Get embeddings as numpy array
    embeddings_matrix = np.array([model.wv[str(i)].tolist() for i in range(n_nodes)])

    # Compute features for test set
    all_test_pairs = test_positives_list + test_negatives_list
    y_test = np.array([1] * len(test_positives_list) + [0] * len(test_negatives_list))

    heuristic_feats_test = compute_heuristic_features(g_train, all_test_pairs)
    embedding_feats_test = get_embedding_pair_features(embeddings_matrix, all_test_pairs)
    X_test = combine_features(heuristic_feats_test, embedding_feats_test)

    # Train set: sample from train edges + negatives (not overlapping with test)
    train_positives_sample = list(train_edges)[:min(len(train_edges), 5000)]
    train_negatives_sample = negative_sample_non_edges(
        n_nodes=n_nodes,
        train_edges=train_edges,
        test_positives=test_edges,  # Exclude test positives
        ratio=1,
        seed=seed + 1,
    )
    train_negatives_sample = list(train_negatives_sample)[:len(train_positives_sample)]

    train_pairs = train_positives_sample + train_negatives_sample
    y_train = np.array([1] * len(train_positives_sample) + [0] * len(train_negatives_sample))

    heuristic_feats_train = compute_heuristic_features(g_train, train_pairs)
    embedding_feats_train = get_embedding_pair_features(embeddings_matrix, train_pairs)
    X_train = combine_features(heuristic_feats_train, embedding_feats_train)

    # Train classifier
    classifier, classifier_metrics = evaluate_classifier(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        seed=seed,
    )

    # ========================================================================
    # PART 4: Rank top predictions
    # ========================================================================
    logger.info("-" * 80)
    logger.info("PART 4: Ranking top predicted routes")
    logger.info("-" * 80)

    # Sample candidate routes (non-edges in train, excluding test positives)
    candidate_sample = negative_sample_non_edges(
        n_nodes=n_nodes,
        train_edges=train_edges,
        test_positives=test_edges,
        ratio=10,
        seed=seed + 2,
    )
    candidate_list = list(candidate_sample)[:1000]  # Limit for efficiency

    candidate_heuristic = compute_heuristic_features(g_train, candidate_list)
    candidate_embedding = get_embedding_pair_features(embeddings_matrix, candidate_list)
    candidate_features = combine_features(candidate_heuristic, candidate_embedding)

    top_predictions_df = rank_candidate_edges(
        model=classifier,
        candidate_edges=candidate_list,
        features=candidate_features,
        id_to_code=id_to_code,
        top_k=100,
    )

    predictions_path = tables_dir / "linkpred_top_predictions.csv"
    if not predictions_path.exists() or overwrite:
        top_predictions_df.write_csv(predictions_path)
        logger.info(f"Wrote top predictions: {predictions_path}")

    # ========================================================================
    # PART 5: Write outputs
    # ========================================================================
    logger.info("-" * 80)
    logger.info("PART 5: Writing outputs")
    logger.info("-" * 80)

    # Metrics JSON
    metrics = {
        "baseline_heuristics": baseline_metrics,
        "embedding_classifier": classifier_metrics,
    }

    metrics_path = analysis_dir / "linkpred_metrics.json"
    if not metrics_path.exists() or overwrite:
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Wrote metrics: {metrics_path}")

    # Write manifest
    manifest = {
        "script": "08_run_embeddings_linkpred.py",
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "config_snapshot": {
            "seed": seed,
            "train_months": train_months,
            "test_months": test_months,
            "negative_ratio": negative_ratio,
            "num_walks": num_walks,
            "walk_length": walk_length,
            "p": p,
            "q": q,
            "dimensions": dimensions,
            "window_size": window_size,
        },
        "inputs": {
            "cleaned_path": str(cleaned_path),
        },
        "outputs": {
            "embeddings": str(embeddings_path),
            "metrics": str(metrics_path),
            "predictions": str(predictions_path),
        },
        "graph_summary": {
            "n_nodes": n_nodes,
            "train_edges": len(train_edges),
            "test_edges_new": len(test_edges),
            "test_negatives": len(test_negatives),
        },
        "metrics": metrics,
    }

    manifest_path = log_dir / "08_run_embeddings_linkpred_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Wrote manifest: {manifest_path}")

    logger.info("=" * 80)
    logger.info("WS4 Script 08: Complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

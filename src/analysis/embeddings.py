"""
Graph embeddings for airport network using node2vec.

This module implements:
- Node2vec random walks on igraph Graph
- Skip-gram training using gensim Word2Vec
- Embedding persistence to parquet
- Nearest neighbor similarity search

Follows WS4 contract: consumes WS1 network outputs (airport_nodes, airport_edges)
and produces embedding vectors for downstream link prediction and visualization.
"""

import logging
from pathlib import Path
from typing import Optional

import igraph as ig
import numpy as np
import polars as pl
from gensim.models import Word2Vec

logger = logging.getLogger(__name__)


def load_airport_graph_from_parquet(
    nodes_path: str | Path,
    edges_path: str | Path,
    directed: bool = True,
    weight_col: Optional[str] = "flight_count",
) -> ig.Graph:
    """
    Load an igraph Graph from WS1 parquet outputs (nodes and edges).

    Parameters
    ----------
    nodes_path : str or Path
        Path to airport_nodes.parquet (columns: vertex_id, code, ...)
    edges_path : str or Path
        Path to airport_edges.parquet (columns: src_id, dst_id, weight, ...)
    directed : bool
        Whether to construct a directed graph
    weight_col : str, optional
        Name of edge weight column; if None, graph is unweighted

    Returns
    -------
    ig.Graph
        An igraph Graph with vertex attribute 'code' and optional edge attribute 'weight'
    """
    nodes_df = pl.read_parquet(nodes_path)
    edges_df = pl.read_parquet(edges_path)

    # Ensure vertex_id is sequential 0..N-1
    n_nodes = len(nodes_df)
    logger.info(f"Loading graph with {n_nodes} nodes, {len(edges_df)} edges")

    # Create graph
    edge_tuples = list(
        zip(
            edges_df["src_id"].to_list(),
            edges_df["dst_id"].to_list(),
        )
    )

    g = ig.Graph(n=n_nodes, edges=edge_tuples, directed=directed)

    # Add vertex attributes
    for col in nodes_df.columns:
        g.vs[col] = nodes_df[col].to_list()

    # Add edge weight
    if weight_col and weight_col in edges_df.columns:
        g.es["weight"] = edges_df[weight_col].to_list()

    logger.info(f"Built graph: N={g.vcount()}, E={g.ecount()}, directed={g.is_directed()}")
    return g


def generate_node2vec_walks(
    g: ig.Graph,
    num_walks: int,
    walk_length: int,
    p: float,
    q: float,
    seed: int,
    weight_col: Optional[str] = "weight",
) -> list[list[int]]:
    """
    Generate node2vec biased random walks.

    Implements the node2vec algorithm (Grover & Leskovec, KDD 2016):
    - p controls return probability (DFS vs BFS)
    - q controls in-out exploration

    Parameters
    ----------
    g : ig.Graph
        The input graph
    num_walks : int
        Number of walks per node
    walk_length : int
        Length of each walk
    p : float
        Return parameter (1/p for returning to previous node)
    q : float
        In-out parameter (1/q for moving outward)
    seed : int
        Random seed for reproducibility
    weight_col : str, optional
        Edge weight attribute; if None, uniform transitions

    Returns
    -------
    list[list[int]]
        List of walks, each walk is a list of vertex IDs
    """
    logger.info(
        f"Generating node2vec walks: num_walks={num_walks}, "
        f"walk_length={walk_length}, p={p}, q={q}, seed={seed}"
    )

    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    n = g.vcount()
    walks = []

    # Precompute neighbor lists and weights
    neighbors = [g.neighbors(v, mode="out") for v in range(n)]
    if weight_col and weight_col in g.es.attributes():
        weights = [
            [g.es[eid][weight_col] for eid in g.incident(v, mode="out")]
            for v in range(n)
        ]
    else:
        weights = [None] * n

    def sample_neighbor(current: int, prev: Optional[int]) -> Optional[int]:
        """Sample next node based on node2vec bias."""
        nbs = neighbors[current]
        if not nbs:
            return None

        wts = weights[current]

        if prev is None or p == 1.0 and q == 1.0:
            # Uniform or first step
            if wts:
                probs = np.array(wts, dtype=float)
                probs /= probs.sum()
                return rng.choice(nbs, p=probs)
            else:
                return rng.choice(nbs)

        # Compute biased probabilities
        prev_nbs = set(neighbors[prev])
        biases = []
        for nb in nbs:
            if nb == prev:
                bias = 1.0 / p
            elif nb in prev_nbs:
                bias = 1.0
            else:
                bias = 1.0 / q
            biases.append(bias)

        probs = np.array(biases, dtype=float)
        if wts:
            probs *= np.array(wts, dtype=float)
        probs /= probs.sum()

        return rng.choice(nbs, p=probs)

    # Generate walks
    for walk_idx in range(num_walks):
        if walk_idx > 0 and walk_idx % 5 == 0:
            logger.info(f"  Generated {walk_idx * n} walks...")

        for start_node in range(n):
            walk = [start_node]
            prev = None

            for _ in range(walk_length - 1):
                current = walk[-1]
                nxt = sample_neighbor(current, prev)
                if nxt is None:
                    break
                walk.append(nxt)
                prev = current

            walks.append(walk)

    logger.info(f"Generated {len(walks)} walks (avg length {np.mean([len(w) for w in walks]):.1f})")
    return walks


def train_skipgram(
    walks: list[list[int]],
    dimensions: int,
    window_size: int,
    seed: int,
    epochs: int = 5,
    workers: int = 4,
    min_count: int = 0,
) -> Word2Vec:
    """
    Train skip-gram embeddings using gensim Word2Vec.

    Parameters
    ----------
    walks : list[list[int]]
        List of random walks (each walk is a list of vertex IDs)
    dimensions : int
        Embedding dimensionality
    window_size : int
        Context window size
    seed : int
        Random seed
    epochs : int
        Training epochs
    workers : int
        Number of parallel workers
    min_count : int
        Minimum word frequency (0 to include all nodes)

    Returns
    -------
    Word2Vec
        Trained gensim Word2Vec model
    """
    logger.info(
        f"Training skip-gram: dimensions={dimensions}, window={window_size}, "
        f"epochs={epochs}, seed={seed}"
    )

    # Convert walks to strings (gensim expects sequences of strings)
    walks_str = [[str(node) for node in walk] for walk in walks]

    model = Word2Vec(
        sentences=walks_str,
        vector_size=dimensions,
        window=window_size,
        min_count=min_count,
        sg=1,  # skip-gram
        workers=workers,
        epochs=epochs,
        seed=seed,
        negative=5,
        ns_exponent=0.75,
    )

    logger.info(f"Trained embeddings for {len(model.wv)} nodes")
    return model


def write_embeddings(
    model: Word2Vec,
    node_id_to_code: dict[int, str],
    output_path: str | Path,
) -> None:
    """
    Write node embeddings to parquet.

    Parameters
    ----------
    model : Word2Vec
        Trained Word2Vec model
    node_id_to_code : dict
        Mapping from vertex_id to airport code
    output_path : str or Path
        Output parquet path

    Writes
    ------
    Parquet file with columns:
        - vertex_id (int)
        - code (str)
        - embedding (list[float])
    """
    logger.info(f"Writing embeddings to {output_path}")

    records = []
    for node_id_str in model.wv.index_to_key:
        node_id = int(node_id_str)
        code = node_id_to_code.get(node_id, "UNKNOWN")
        embedding = model.wv[node_id_str].tolist()
        records.append({"vertex_id": node_id, "code": code, "embedding": embedding})

    df = pl.DataFrame(records)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    logger.info(f"Wrote {len(df)} embeddings")


def find_similar_airports(
    embeddings_path: str | Path,
    query_codes: list[str],
    top_k: int = 10,
) -> pl.DataFrame:
    """
    Find most similar airports based on cosine similarity.

    Parameters
    ----------
    embeddings_path : str or Path
        Path to airport_embeddings.parquet
    query_codes : list[str]
        Airport codes to query (e.g., ['ATL', 'ORD', 'LAX'])
    top_k : int
        Number of nearest neighbors to return

    Returns
    -------
    pl.DataFrame
        Columns: query_code, neighbor_code, similarity, rank
    """
    logger.info(f"Finding top-{top_k} neighbors for {len(query_codes)} airports")

    df = pl.read_parquet(embeddings_path)

    # Build embedding matrix
    codes = df["code"].to_list()
    embeddings = np.array([emb for emb in df["embedding"].to_list()])

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / (norms + 1e-8)

    code_to_idx = {code: idx for idx, code in enumerate(codes)}

    results = []
    for query_code in query_codes:
        if query_code not in code_to_idx:
            logger.warning(f"Airport {query_code} not in embeddings")
            continue

        query_idx = code_to_idx[query_code]
        query_vec = embeddings_norm[query_idx]

        # Cosine similarity
        sims = embeddings_norm @ query_vec

        # Top-k (excluding self)
        top_indices = np.argsort(sims)[::-1][1 : top_k + 1]

        for rank, idx in enumerate(top_indices, start=1):
            results.append(
                {
                    "query_code": query_code,
                    "neighbor_code": codes[idx],
                    "similarity": float(sims[idx]),
                    "rank": rank,
                }
            )

    return pl.DataFrame(results)


def get_embedding_pair_features(
    embeddings: np.ndarray,
    node_pairs: list[tuple[int, int]],
) -> np.ndarray:
    """
    Compute embedding-based features for node pairs.

    Features:
        - Hadamard (element-wise product)
        - L1 distance
        - L2 distance
        - Cosine similarity

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (N, D) node embeddings
    node_pairs : list[tuple[int, int]]
        List of (u, v) node pairs

    Returns
    -------
    np.ndarray
        Shape (len(node_pairs), feature_dim) feature matrix
    """
    features = []
    for u, v in node_pairs:
        emb_u = embeddings[u]
        emb_v = embeddings[v]

        # Hadamard
        hadamard = emb_u * emb_v

        # Distances
        l1_dist = np.abs(emb_u - emb_v).sum()
        l2_dist = np.linalg.norm(emb_u - emb_v)

        # Cosine similarity
        cos_sim = (emb_u @ emb_v) / (np.linalg.norm(emb_u) * np.linalg.norm(emb_v) + 1e-8)

        # Concatenate: hadamard + 3 scalar features
        feat = np.concatenate([hadamard, [l1_dist, l2_dist, cos_sim]])
        features.append(feat)

    return np.array(features)

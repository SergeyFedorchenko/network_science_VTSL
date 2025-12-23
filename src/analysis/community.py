"""
Community detection beyond modularity using Leiden algorithm with CPM objective.

This module implements:
- Leiden CPM with multiple runs and best partition selection
- Community summarization for airport and flight networks
- Optional SBM alternative (if dependencies exist and approved)

Follows WS2 contract: applies to airport graph (full) and flight graph (scoped).
"""

import logging
import time
from typing import Optional

import igraph as ig
import leidenalg
import polars as pl

logger = logging.getLogger(__name__)


def run_leiden_cpm(
    g: ig.Graph,
    resolution: float = 0.01,
    seed: Optional[int] = None,
    n_iterations: int = -1,
    weights: Optional[str] = "weight",
    convert_to_undirected: bool = False,
) -> tuple[list[int], float]:
    """
    Run Leiden algorithm with CPM (Constant Potts Model) objective.

    Parameters
    ----------
    g : ig.Graph
        Input graph
    resolution : float
        Resolution parameter for CPM (higher = more communities)
    seed : int, optional
        Random seed for reproducibility
    n_iterations : int
        Maximum iterations (-1 = until convergence)
    weights : str, optional
        Name of edge weight attribute; None for unweighted
    convert_to_undirected : bool
        If True, convert directed graph to undirected (faster for community detection)

    Returns
    -------
    membership : list of int
        Community assignment for each vertex
    quality : float
        CPM quality score
    """
    # Convert to undirected if requested (significantly faster for large graphs)
    if convert_to_undirected and g.is_directed():
        logger.info(f"Converting directed graph to undirected for community detection")
        g = g.as_undirected(mode="collapse", combine_edges="sum")
    
    # Prepare edge weights
    edge_weights = None
    if weights and weights in g.es.attributes():
        edge_weights = g.es[weights]
        logger.debug(f"Using edge weights: {weights}")
    
    # Create partition with seed
    # Note: leidenalg uses igraph's RNG, so we seed via find_partition's seed parameter
    partition = leidenalg.find_partition(
        g,
        leidenalg.CPMVertexPartition,
        weights=edge_weights,
        n_iterations=n_iterations,
        resolution_parameter=resolution,
        seed=seed,
    )
    
    membership = partition.membership
    quality = partition.quality()
    
    logger.debug(f"Leiden CPM: {len(set(membership))} communities, quality={quality:.4f}")
    
    return membership, quality


def select_best_partition(
    g: ig.Graph,
    config: dict,
    weights: Optional[str] = "weight",
    convert_to_undirected: bool = False,
) -> tuple[list[int], dict]:
    """
    Run Leiden CPM multiple times with different seeds and select best partition.
    
    Automatically adjusts parameters for large graphs to optimize performance.

    Parameters
    ----------
    g : ig.Graph
        Input graph
    config : dict
        Config dict with keys:
        - analysis.communities.leiden.resolution (float)
        - analysis.communities.leiden.n_runs (int)
        - analysis.communities.leiden.max_iterations (int, optional)
        - seed (int, base seed for runs)
    weights : str, optional
        Name of edge weight attribute
    convert_to_undirected : bool
        If True, convert directed graph to undirected before community detection
        (significantly faster for large graphs, preserves community structure)

    Returns
    -------
    best_membership : list of int
        Best community assignment
    run_log : dict
        Log with keys: runs (list of dicts), best_seed, best_quality, n_communities
    """
    resolution = config.get("analysis", {}).get("communities", {}).get("leiden", {}).get("resolution", 0.01)
    n_runs = config.get("analysis", {}).get("communities", {}).get("leiden", {}).get("n_runs", 10)
    max_iterations = config.get("analysis", {}).get("communities", {}).get("leiden", {}).get("max_iterations", None)
    base_seed = config.get("seed", 42)
    
    n_vertices = g.vcount()
    n_edges = g.ecount()
    
    # Adaptive parameters for large graphs
    if n_vertices > 1_000_000:
        # Very large graph: reduce runs and set iteration limit
        if n_runs > 3:
            original_runs = n_runs
            n_runs = 3
            logger.info(f"Large graph detected (N={n_vertices:,}): reducing runs from {original_runs} to {n_runs}")
        
        if max_iterations is None or max_iterations < 0:
            max_iterations = 5
            logger.info(f"Large graph: setting max_iterations={max_iterations} for faster convergence")
    elif n_vertices > 100_000:
        # Large graph: moderate optimization
        if max_iterations is None or max_iterations < 0:
            max_iterations = 10
            logger.info(f"Large graph (N={n_vertices:,}): setting max_iterations={max_iterations}")
    else:
        # Small/medium graph: use config or unlimited
        if max_iterations is None:
            max_iterations = -1
    
    logger.info(f"Running Leiden CPM {n_runs} times on graph with N={n_vertices:,}, E={n_edges:,}")
    logger.info(f"Parameters: resolution={resolution}, max_iterations={max_iterations}")
    
    runs = []
    best_membership = None
    best_quality = -float("inf")
    best_seed = None
    
    for i in range(n_runs):
        run_seed = base_seed + i
        start_time = time.time()
        
        logger.info(f"Run {i+1}/{n_runs} (seed={run_seed})...")
        
        membership, quality = run_leiden_cpm(
            g,
            resolution=resolution,
            seed=run_seed,
            n_iterations=max_iterations,
            weights=weights,
            convert_to_undirected=convert_to_undirected,
        )
        runtime = time.time() - start_time
        
        n_communities = len(set(membership))
        logger.info(f"Run {i+1} complete: {n_communities} communities, quality={quality:.2f}, time={runtime:.1f}s")
        
        runs.append({
            "run_id": i,
            "seed": run_seed,
            "quality": quality,
            "n_communities": n_communities,
            "runtime_seconds": runtime,
        })
        
        if quality > best_quality:
            best_quality = quality
            best_membership = membership
            best_seed = run_seed
    
    run_log = {
        "runs": runs,
        "best_seed": best_seed,
        "best_quality": best_quality,
        "n_communities": len(set(best_membership)),
        "resolution": resolution,
        "max_iterations": max_iterations,
        "adaptive_params_used": n_vertices > 100_000,
    }
    
    logger.info(f"Best partition: seed={best_seed}, quality={best_quality:.4f}, "
                f"n_communities={run_log['n_communities']}")
    
    return best_membership, run_log


def summarize_communities_airport(
    nodes_df: pl.DataFrame,
    membership_df: pl.DataFrame,
    centrality_df: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """
    Summarize airport communities.

    Parameters
    ----------
    nodes_df : pl.DataFrame
        Airport nodes (columns: vertex_id, code, city, state, ...)
    membership_df : pl.DataFrame
        Community membership (columns: vertex_id, community_id)
    centrality_df : pl.DataFrame, optional
        Centrality results for ranking top airports per community

    Returns
    -------
    pl.DataFrame
        Community summary with columns:
        - community_id
        - size (number of airports)
        - top_airports (comma-separated list of top 5 by PageRank if available)
        - top_states (if state column exists)
    """
    # Determine vertex ID column and create alias if needed
    if "vertex_id" not in nodes_df.columns:
        if "node_id" in nodes_df.columns:
            nodes_df = nodes_df.with_columns(pl.col("node_id").alias("vertex_id"))
        else:
            raise ValueError(f"No vertex_id or node_id column found in nodes. Columns: {nodes_df.columns}")
    
    # Join membership with nodes
    df = nodes_df.join(membership_df, on="vertex_id", how="inner")
    
    # Community sizes
    community_sizes = (
        df.group_by("community_id")
        .agg(pl.col("vertex_id").count().alias("size"))
        .sort("size", descending=True)
    )
    
    # Top airports per community (if centrality available)
    if centrality_df is not None:
        df = df.join(centrality_df.select(["vertex_id", "pagerank"]), on="vertex_id", how="left")
        
        top_airports = (
            df.sort(["community_id", "pagerank"], descending=[False, True])
            .group_by("community_id")
            .agg(
                pl.col("code").head(5).str.join(", ").alias("top_airports")
            )
        )
        community_sizes = community_sizes.join(top_airports, on="community_id", how="left")
    
    # Top states per community (if available)
    if "ORIGIN_STATE_NM" in df.columns:
        top_states = (
            df.group_by(["community_id", "ORIGIN_STATE_NM"])
            .agg(pl.col("vertex_id").count().alias("state_count"))
            .sort(["community_id", "state_count"], descending=[False, True])
            .group_by("community_id")
            .agg(
                pl.col("ORIGIN_STATE_NM").head(3).str.join(", ").alias("top_states")
            )
        )
        community_sizes = community_sizes.join(top_states, on="community_id", how="left")
    
    logger.info(f"Summarized {len(community_sizes)} airport communities")
    return community_sizes


def summarize_communities_flight(
    flight_nodes_df: pl.DataFrame,
    membership_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Summarize flight communities.

    Parameters
    ----------
    flight_nodes_df : pl.DataFrame
        Flight nodes (columns: vertex_id, carrier, origin, dest, dep_ts, ...)
    membership_df : pl.DataFrame
        Community membership (columns: vertex_id, community_id)

    Returns
    -------
    pl.DataFrame
        Community summary with columns:
        - community_id
        - size (number of flights)
        - dominant_carrier (most frequent)
        - dominant_origin (most frequent)
        - dominant_dest (most frequent)
    """
    # Determine vertex ID column and create alias if needed
    if "vertex_id" not in flight_nodes_df.columns:
        if "node_id" in flight_nodes_df.columns:
            flight_nodes_df = flight_nodes_df.with_columns(pl.col("node_id").alias("vertex_id"))
        elif "flight_id" in flight_nodes_df.columns:
            flight_nodes_df = flight_nodes_df.with_columns(pl.col("flight_id").alias("vertex_id"))
        else:
            raise ValueError(f"No vertex_id, node_id, or flight_id column found. Columns: {flight_nodes_df.columns}")
    
    # Join membership with flight nodes
    df = flight_nodes_df.join(membership_df, on="vertex_id", how="inner")
    
    # Community sizes
    community_sizes = (
        df.group_by("community_id")
        .agg(pl.col("vertex_id").count().alias("size"))
        .sort("size", descending=True)
    )
    
    # Dominant carrier per community
    if "carrier" in df.columns or "OP_UNIQUE_CARRIER" in df.columns:
        carrier_col = "carrier" if "carrier" in df.columns else "OP_UNIQUE_CARRIER"
        dominant_carrier = (
            df.group_by(["community_id", carrier_col])
            .agg(pl.col("vertex_id").count().alias("carrier_count"))
            .sort(["community_id", "carrier_count"], descending=[False, True])
            .group_by("community_id")
            .agg(pl.col(carrier_col).first().alias("dominant_carrier"))
        )
        community_sizes = community_sizes.join(dominant_carrier, on="community_id", how="left")
    
    # Dominant origin/dest per community
    if "origin" in df.columns or "ORIGIN" in df.columns:
        origin_col = "origin" if "origin" in df.columns else "ORIGIN"
        dominant_origin = (
            df.group_by(["community_id", origin_col])
            .agg(pl.col("vertex_id").count().alias("origin_count"))
            .sort(["community_id", "origin_count"], descending=[False, True])
            .group_by("community_id")
            .agg(pl.col(origin_col).first().alias("dominant_origin"))
        )
        community_sizes = community_sizes.join(dominant_origin, on="community_id", how="left")
    
    if "dest" in df.columns or "DEST" in df.columns:
        dest_col = "dest" if "dest" in df.columns else "DEST"
        dominant_dest = (
            df.group_by(["community_id", dest_col])
            .agg(pl.col("vertex_id").count().alias("dest_count"))
            .sort(["community_id", "dest_count"], descending=[False, True])
            .group_by("community_id")
            .agg(pl.col(dest_col).first().alias("dominant_dest"))
        )
        community_sizes = community_sizes.join(dominant_dest, on="community_id", how="left")
    
    logger.info(f"Summarized {len(community_sizes)} flight communities")
    return community_sizes


def write_community_outputs(
    membership_df: pl.DataFrame,
    summary_df: pl.DataFrame,
    run_log: dict,
    output_dir: str,
    graph_type: str,
    overwrite: bool = False,
) -> dict:
    """
    Write community detection outputs.

    Parameters
    ----------
    membership_df : pl.DataFrame
        Vertex membership (vertex_id, community_id)
    summary_df : pl.DataFrame
        Community summary table
    run_log : dict
        Leiden run log
    output_dir : str
        Base output directory (results/)
    graph_type : str
        'airport' or 'flight'
    overwrite : bool
        Whether to overwrite existing files

    Returns
    -------
    dict
        Paths to written files
    """
    from pathlib import Path
    
    output_dir = Path(output_dir)
    analysis_dir = output_dir / "analysis"
    tables_dir = output_dir / "tables"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    # Membership parquet
    membership_path = analysis_dir / f"{graph_type}_leiden_membership.parquet"
    if not membership_path.exists() or overwrite:
        membership_df.write_parquet(membership_path)
        logger.info(f"Wrote {membership_path}")
        paths["membership"] = str(membership_path)
    else:
        logger.warning(f"{membership_path} exists; skipping (overwrite=False)")
    
    # Community summary CSV
    summary_path = tables_dir / f"community_summary_{graph_type}.csv"
    if not summary_path.exists() or overwrite:
        summary_df.write_csv(summary_path)
        logger.info(f"Wrote {summary_path}")
        paths["summary"] = str(summary_path)
    
    # Table for report (tbl02)
    if graph_type == "airport":
        report_table_path = tables_dir / "tbl02_airport_communities_summary.csv"
        if not report_table_path.exists() or overwrite:
            summary_df.write_csv(report_table_path)
            logger.info(f"Wrote {report_table_path}")
            paths["report_table"] = str(report_table_path)
    
    return paths

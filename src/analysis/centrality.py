"""
Centrality analysis for airport and flight networks.

This module computes:
- Degree (in/out) and strength (weighted in/out)
- PageRank (optionally weighted)
- Betweenness centrality (exact or approximate)
- Component structure (weak/strong, LCC)
- Degree distributions

Follows WS2 contract: consumes WS1 network outputs from parquet.
"""

import logging
from pathlib import Path
from typing import Optional

import igraph as ig
import polars as pl

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

    logger.info(f"Loaded {len(nodes_df)} nodes from {nodes_path}")
    logger.info(f"Loaded {len(edges_df)} edges from {edges_path}")

    # Build vertex list: igraph expects vertices 0..N-1
    # WS1 outputs may have vertex_id or node_id; check which exists
    if "vertex_id" in nodes_df.columns:
        vertex_id_col = "vertex_id"
    elif "node_id" in nodes_df.columns:
        vertex_id_col = "node_id"
    else:
        raise ValueError("No vertex_id or node_id column found in nodes file")
    
    vertex_ids = nodes_df[vertex_id_col].to_list()
    codes = nodes_df["code"].to_list()
    
    # Check contiguity
    if vertex_ids != list(range(len(vertex_ids))):
        logger.warning("vertex_id not contiguous 0..N-1; remapping required")
        # Build a mapping
        id_map = {old: new for new, old in enumerate(vertex_ids)}
        # Remap edges
        edges_df = edges_df.with_columns([
            pl.col("src_id").map_elements(lambda x: id_map[x], return_dtype=pl.Int64).alias("src_id"),
            pl.col("dst_id").map_elements(lambda x: id_map[x], return_dtype=pl.Int64).alias("dst_id"),
        ])
        vertex_ids = list(range(len(vertex_ids)))

    n_vertices = len(vertex_ids)
    
    # Build edge list
    edge_list = list(zip(edges_df["src_id"].to_list(), edges_df["dst_id"].to_list()))
    
    # Create graph
    g = ig.Graph(n=n_vertices, edges=edge_list, directed=directed)
    
    # Add vertex attribute 'code'
    g.vs["code"] = codes
    
    # Add edge weight if requested
    if weight_col and weight_col in edges_df.columns:
        g.es["weight"] = edges_df[weight_col].to_list()
        logger.info(f"Added edge weights from column '{weight_col}'")
    
    logger.info(f"Created igraph: N={g.vcount()}, E={g.ecount()}, directed={g.is_directed()}")
    
    return g


def compute_graph_summary(g: ig.Graph) -> dict:
    """
    Compute basic graph structure summary.

    Parameters
    ----------
    g : ig.Graph
        Input graph

    Returns
    -------
    dict
        Summary with keys: n_vertices, n_edges, directed, n_components_weak, 
        n_components_strong (if directed), lcc_size_weak, lcc_size_strong (if directed)
    """
    summary = {
        "n_vertices": g.vcount(),
        "n_edges": g.ecount(),
        "directed": g.is_directed(),
    }
    
    if g.is_directed():
        components_weak = g.connected_components(mode="weak")
        components_strong = g.connected_components(mode="strong")
        summary["n_components_weak"] = len(components_weak)
        summary["n_components_strong"] = len(components_strong)
        summary["lcc_size_weak"] = max(components_weak.sizes()) if components_weak.sizes() else 0
        summary["lcc_size_strong"] = max(components_strong.sizes()) if components_strong.sizes() else 0
    else:
        components = g.connected_components()
        summary["n_components"] = len(components)
        summary["lcc_size"] = max(components.sizes()) if components.sizes() else 0
    
    logger.info(f"Graph summary: {summary}")
    return summary


def compute_degree_distribution(g: ig.Graph, mode: str = "all") -> pl.DataFrame:
    """
    Compute degree distribution table.

    Parameters
    ----------
    g : ig.Graph
        Input graph
    mode : str
        'in', 'out', or 'all'

    Returns
    -------
    pl.DataFrame
        Columns: degree, count
    """
    degrees = g.degree(mode=mode)
    degree_counts = {}
    for d in degrees:
        degree_counts[d] = degree_counts.get(d, 0) + 1
    
    df = pl.DataFrame({
        "degree": list(degree_counts.keys()),
        "count": list(degree_counts.values()),
    }).sort("degree")
    
    return df


def compute_airport_centrality(
    g: ig.Graph,
    weight_col: Optional[str] = "weight",
    config: Optional[dict] = None,
) -> pl.DataFrame:
    """
    Compute centrality measures for an airport graph.

    Computes:
    - in_degree, out_degree (or degree if undirected)
    - in_strength, out_strength (weighted degree)
    - pagerank (optionally weighted)
    - betweenness (exact if small, approximate if large)

    Parameters
    ----------
    g : ig.Graph
        Airport graph with vertex attribute 'code'
    weight_col : str, optional
        Name of edge weight attribute; None if unweighted
    config : dict, optional
        Config dict with optional keys:
        - analysis.centrality.betweenness_approx_cutoff: int (default 20000)
          If n_vertices > cutoff, use approximation

    Returns
    -------
    pl.DataFrame
        Columns: vertex_id, code, in_degree, out_degree, in_strength, out_strength,
                 pagerank, betweenness
    """
    if config is None:
        config = {}
    
    betweenness_cutoff = config.get("analysis", {}).get("centrality", {}).get("betweenness_approx_cutoff", 20000)
    
    n = g.vcount()
    codes = g.vs["code"]
    
    logger.info(f"Computing centrality for {n} vertices")
    
    # Degree
    if g.is_directed():
        in_degree = g.degree(mode="in")
        out_degree = g.degree(mode="out")
    else:
        in_degree = g.degree(mode="all")
        out_degree = in_degree
    
    # Strength (weighted degree)
    has_weights = weight_col and weight_col in g.es.attributes()
    if has_weights:
        if g.is_directed():
            in_strength = g.strength(weights=weight_col, mode="in")
            out_strength = g.strength(weights=weight_col, mode="out")
        else:
            in_strength = g.strength(weights=weight_col, mode="all")
            out_strength = in_strength
        logger.info("Computed weighted strength")
    else:
        # Fallback: strength = degree
        in_strength = in_degree
        out_strength = out_degree
        logger.info("No weights found; strength = degree")
    
    # PageRank
    if has_weights:
        pagerank = g.pagerank(weights=weight_col, directed=g.is_directed())
        logger.info("Computed weighted PageRank")
    else:
        pagerank = g.pagerank(directed=g.is_directed())
        logger.info("Computed unweighted PageRank")
    
    # Betweenness: exact or approximate
    if n > betweenness_cutoff:
        # Approximate: use top 10% nodes by out_degree as sources
        logger.warning(f"Graph size {n} > {betweenness_cutoff}; using approximate betweenness")
        top_k = max(int(0.1 * n), 100)
        sources = sorted(range(n), key=lambda i: out_degree[i], reverse=True)[:top_k]
        betweenness = g.betweenness(vertices=sources, directed=g.is_directed())
        logger.info(f"Computed approximate betweenness using {len(sources)} sources")
    else:
        betweenness = g.betweenness(directed=g.is_directed())
        logger.info("Computed exact betweenness")
    
    # Build DataFrame
    df = pl.DataFrame({
        "vertex_id": list(range(n)),
        "code": codes,
        "in_degree": in_degree,
        "out_degree": out_degree,
        "in_strength": in_strength,
        "out_strength": out_strength,
        "pagerank": pagerank,
        "betweenness": betweenness,
    })
    
    logger.info("Centrality computation complete")
    return df


def write_centrality_outputs(
    centrality_df: pl.DataFrame,
    degree_dist_in: pl.DataFrame,
    degree_dist_out: pl.DataFrame,
    graph_summary: dict,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict:
    """
    Write centrality outputs to disk.

    Parameters
    ----------
    centrality_df : pl.DataFrame
        Centrality results
    degree_dist_in : pl.DataFrame
        In-degree distribution
    degree_dist_out : pl.DataFrame
        Out-degree distribution
    graph_summary : dict
        Graph structure summary
    output_dir : str or Path
        Base output directory (results/)
    overwrite : bool
        Whether to overwrite existing files

    Returns
    -------
    dict
        Paths to written files
    """
    output_dir = Path(output_dir)
    analysis_dir = output_dir / "analysis"
    tables_dir = output_dir / "tables"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    # Centrality parquet
    centrality_path = analysis_dir / "airport_centrality.parquet"
    if not centrality_path.exists() or overwrite:
        centrality_df.write_parquet(centrality_path)
        logger.info(f"Wrote {centrality_path}")
        paths["centrality"] = str(centrality_path)
    else:
        logger.warning(f"{centrality_path} exists; skipping (overwrite=False)")
    
    # Degree distributions
    degree_dist_in_path = tables_dir / "airport_degree_dist_in.csv"
    if not degree_dist_in_path.exists() or overwrite:
        degree_dist_in.write_csv(degree_dist_in_path)
        logger.info(f"Wrote {degree_dist_in_path}")
        paths["degree_dist_in"] = str(degree_dist_in_path)
    
    degree_dist_out_path = tables_dir / "airport_degree_dist_out.csv"
    if not degree_dist_out_path.exists() or overwrite:
        degree_dist_out.write_csv(degree_dist_out_path)
        logger.info(f"Wrote {degree_dist_out_path}")
        paths["degree_dist_out"] = str(degree_dist_out_path)
    
    # Top centrality table (for report)
    top_centrality_path = tables_dir / "tbl01_top_airports_by_centrality.csv"
    if not top_centrality_path.exists() or overwrite:
        # Top 20 by each metric
        top_df = centrality_df.sort("pagerank", descending=True).head(20).select([
            "code", "in_degree", "out_degree", "pagerank", "betweenness"
        ])
        top_df.write_csv(top_centrality_path)
        logger.info(f"Wrote {top_centrality_path}")
        paths["top_centrality"] = str(top_centrality_path)
    
    return paths

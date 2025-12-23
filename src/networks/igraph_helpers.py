"""
Helper utilities for igraph operations.

Provides common functions for building and exporting igraph graphs.
"""
import igraph as ig
import polars as pl
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging


logger = logging.getLogger(__name__)


def build_graph_from_edges(
    nodes_df: pl.DataFrame,
    edges_df: pl.DataFrame,
    node_id_col: str = "node_id",
    node_label_col: str = "code",
    src_col: str = "src_id",
    dst_col: str = "dst_id",
    directed: bool = True,
    edge_attributes: List[str] = None,
    node_attributes: List[str] = None
) -> ig.Graph:
    """
    Build an igraph Graph from node and edge DataFrames.
    
    Args:
        nodes_df: DataFrame with node information
        edges_df: DataFrame with edge information
        node_id_col: Column name for node integer IDs
        node_label_col: Column name for node labels (e.g., airport codes)
        src_col: Column name for source node ID in edges
        dst_col: Column name for destination node ID in edges
        directed: Whether graph is directed
        edge_attributes: List of edge attribute column names to include
        node_attributes: List of node attribute column names to include
        
    Returns:
        igraph Graph object
    """
    n_nodes = len(nodes_df)
    
    # Create graph
    g = ig.Graph(directed=directed)
    g.add_vertices(n_nodes)
    
    # Add node attributes
    if node_label_col in nodes_df.columns:
        g.vs[node_label_col] = nodes_df[node_label_col].to_list()
    
    if node_attributes:
        for attr in node_attributes:
            if attr in nodes_df.columns and attr != node_id_col:
                g.vs[attr] = nodes_df[attr].to_list()
    
    # Add edges
    edge_list = list(zip(
        edges_df[src_col].to_list(),
        edges_df[dst_col].to_list()
    ))
    g.add_edges(edge_list)
    
    # Add edge attributes
    if edge_attributes:
        for attr in edge_attributes:
            if attr in edges_df.columns:
                g.es[attr] = edges_df[attr].to_list()
    
    logger.info(f"Built graph: {n_nodes} nodes, {g.ecount()} edges, directed={directed}")
    
    return g


def get_graph_summary(g: ig.Graph) -> Dict[str, Any]:
    """
    Get summary statistics for a graph.
    
    Args:
        g: igraph Graph
        
    Returns:
        Dictionary with graph statistics
    """
    summary = {
        "n_nodes": g.vcount(),
        "n_edges": g.ecount(),
        "directed": g.is_directed(),
        "density": g.density(),
    }
    
    # Component analysis
    if g.is_directed():
        components = g.connected_components(mode="weak")
        summary["n_weak_components"] = len(components)
        summary["lcc_size"] = max(components.sizes())
        
        strong_components = g.connected_components(mode="strong")
        summary["n_strong_components"] = len(strong_components)
        summary["largest_scc_size"] = max(strong_components.sizes())
    else:
        components = g.connected_components()
        summary["n_components"] = len(components)
        summary["lcc_size"] = max(components.sizes())
    
    # Degree statistics (in/out for directed)
    if g.is_directed():
        summary["mean_in_degree"] = sum(g.indegree()) / g.vcount()
        summary["mean_out_degree"] = sum(g.outdegree()) / g.vcount()
        summary["max_in_degree"] = max(g.indegree())
        summary["max_out_degree"] = max(g.outdegree())
    else:
        summary["mean_degree"] = sum(g.degree()) / g.vcount()
        summary["max_degree"] = max(g.degree())
    
    return summary


def save_graph(
    g: ig.Graph,
    output_path: Path,
    format: str = "graphml"
) -> None:
    """
    Save graph to file.
    
    Args:
        g: igraph Graph
        output_path: Output file path
        format: Format ("graphml", "gml", "edgelist", "pajek")
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "graphml":
        g.write_graphml(str(output_path))
    elif format == "gml":
        g.write_gml(str(output_path))
    elif format == "edgelist":
        g.write_edgelist(str(output_path))
    elif format == "pajek":
        g.write_pajek(str(output_path))
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Saved graph to {output_path}")


def create_node_mapping(
    unique_ids: List[str]
) -> Tuple[Dict[str, int], pl.DataFrame]:
    """
    Create mapping from string IDs to integer node IDs.
    
    Args:
        unique_ids: List of unique string identifiers
        
    Returns:
        Tuple of (id_to_int dict, DataFrame with mapping)
    """
    sorted_ids = sorted(set(unique_ids))
    id_to_int = {id_val: idx for idx, id_val in enumerate(sorted_ids)}
    
    mapping_df = pl.DataFrame({
        "node_id": list(range(len(sorted_ids))),
        "code": sorted_ids
    })
    
    return id_to_int, mapping_df


def export_degree_distribution(
    g: ig.Graph,
    output_path: Path,
    directed: bool = True
) -> None:
    """
    Export degree distribution to CSV.
    
    Args:
        g: igraph Graph
        output_path: Output CSV path
        directed: Whether to compute in/out degree separately
    """
    if directed and g.is_directed():
        degree_df = pl.DataFrame({
            "node_id": list(range(g.vcount())),
            "in_degree": g.indegree(),
            "out_degree": g.outdegree(),
            "total_degree": [i + o for i, o in zip(g.indegree(), g.outdegree())]
        })
    else:
        degree_df = pl.DataFrame({
            "node_id": list(range(g.vcount())),
            "degree": g.degree()
        })
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    degree_df.write_csv(output_path)
    logger.info(f"Exported degree distribution to {output_path}")

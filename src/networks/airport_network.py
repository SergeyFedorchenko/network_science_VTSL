"""
Airport network construction.

Builds airport-centric network where nodes are airports and edges are routes.
"""
import polars as pl
import igraph as ig
from pathlib import Path
from typing import Dict, Any, List
import logging

from src.networks.igraph_helpers import (
    create_node_mapping,
    build_graph_from_edges,
    get_graph_summary,
    save_graph
)


logger = logging.getLogger(__name__)


def build_airport_nodes(lf: pl.LazyFrame) -> pl.DataFrame:
    """
    Extract unique airports from flight data.
    
    Args:
        lf: LazyFrame with flight data
        
    Returns:
        DataFrame with airport nodes and attributes
    """
    logger.info("Building airport nodes table")
    
    # Get unique airports from both ORIGIN and DEST
    origins = lf.select([
        pl.col("ORIGIN").alias("code"),
        pl.col("ORIGIN_CITY_NAME").alias("city"),
        pl.col("ORIGIN_STATE_NM").alias("state"),
        pl.col("ORIGIN_AIRPORT_ID").alias("airport_id")
    ])
    
    dests = lf.select([
        pl.col("DEST").alias("code"),
        pl.col("DEST_CITY_NAME").alias("city"),
        pl.col("DEST_STATE_NM").alias("state"),
        pl.col("ORIGIN_AIRPORT_ID").alias("airport_id")  # Note: using DEST info if available
    ])
    
    # Combine and get unique
    airports = pl.concat([origins, dests]).unique(subset=["code"]).collect()
    
    # Sort by code for stability
    airports = airports.sort("code")
    
    # Add integer node IDs
    airports = airports.with_columns([
        pl.arange(0, len(airports)).alias("node_id")
    ])
    
    logger.info(f"Created {len(airports)} airport nodes")
    
    return airports


def build_airport_edges(
    lf: pl.LazyFrame,
    airport_nodes: pl.DataFrame,
    edge_metrics: List[str] = None
) -> pl.DataFrame:
    """
    Build airport network edges from flight data.
    
    Aggregates flights by route (ORIGIN -> DEST) and computes metrics.
    
    Args:
        lf: LazyFrame with flight data
        airport_nodes: DataFrame with airport nodes and IDs
        edge_metrics: List of metrics to compute (from config)
        
    Returns:
        DataFrame with edges and metrics
    """
    logger.info("Building airport edges table")
    
    if edge_metrics is None:
        edge_metrics = ["mean_dep_delay", "mean_arr_delay", "cancel_rate", "mean_distance"]
    
    # Build aggregation expressions
    agg_exprs = [
        pl.col("FLIGHTS").sum().alias("flight_count")
    ]
    
    if "mean_dep_delay" in edge_metrics:
        agg_exprs.append(pl.col("DEP_DELAY").mean().alias("mean_dep_delay"))
    
    if "mean_arr_delay" in edge_metrics:
        agg_exprs.append(pl.col("ARR_DELAY").mean().alias("mean_arr_delay"))
    
    if "cancel_rate" in edge_metrics:
        # This only works if cancelled flights are included in the data
        # Otherwise will be 0
        agg_exprs.append(pl.col("CANCELLED").mean().alias("cancel_rate"))
    
    if "mean_distance" in edge_metrics:
        agg_exprs.append(pl.col("DISTANCE").mean().alias("mean_distance"))
    
    # Aggregate by route
    edges = lf.group_by(["ORIGIN", "DEST"]).agg(agg_exprs).collect()
    
    logger.info(f"Aggregated {len(edges)} unique routes")
    
    # Join with node IDs
    # Create mapping dict for faster lookup
    node_map = dict(zip(airport_nodes["code"], airport_nodes["node_id"]))
    
    edges = edges.with_columns([
        pl.col("ORIGIN").map_elements(lambda x: node_map.get(x), return_dtype=pl.Int64).alias("src_id"),
        pl.col("DEST").map_elements(lambda x: node_map.get(x), return_dtype=pl.Int64).alias("dst_id")
    ])
    
    # Remove any edges with missing node mappings (shouldn't happen but safety check)
    edges = edges.filter(
        pl.col("src_id").is_not_null() & pl.col("dst_id").is_not_null()
    )
    
    # Sort for stability
    edges = edges.sort(["src_id", "dst_id"])
    
    logger.info(f"Created {len(edges)} edges with node IDs")
    
    return edges


def build_airport_network(
    lf: pl.LazyFrame,
    config: Dict[str, Any],
    output_dir: Path
) -> Dict[str, Any]:
    """
    Build complete airport network and save outputs.
    
    Args:
        lf: LazyFrame with flight data
        config: Configuration dictionary
        output_dir: Output directory for results
        
    Returns:
        Dictionary with network summary statistics
    """
    airport_config = config["airport_network"]
    
    # Build nodes
    nodes = build_airport_nodes(lf)
    
    # Build edges
    edges = build_airport_edges(
        lf,
        nodes,
        edge_metrics=airport_config.get("edge_metrics", [])
    )
    
    # Save node and edge tables
    networks_dir = output_dir / "networks"
    networks_dir.mkdir(parents=True, exist_ok=True)
    
    nodes_path = networks_dir / "airport_nodes.parquet"
    edges_path = networks_dir / "airport_edges.parquet"
    
    nodes.write_parquet(nodes_path)
    edges.write_parquet(edges_path)
    
    logger.info(f"Saved airport nodes to {nodes_path}")
    logger.info(f"Saved airport edges to {edges_path}")
    
    # Build igraph Graph
    edge_attrs = [col for col in edges.columns if col not in ["ORIGIN", "DEST", "src_id", "dst_id"]]
    node_attrs = [col for col in nodes.columns if col not in ["node_id"]]
    
    g = build_graph_from_edges(
        nodes_df=nodes,
        edges_df=edges,
        node_id_col="node_id",
        node_label_col="code",
        src_col="src_id",
        dst_col="dst_id",
        directed=airport_config.get("directed", True),
        edge_attributes=edge_attrs,
        node_attributes=node_attrs
    )
    
    # Get graph summary
    summary = get_graph_summary(g)
    summary["n_airports"] = len(nodes)
    summary["n_routes"] = len(edges)
    
    # Save graph (optional, for inspection)
    graphml_path = networks_dir / "airport_graph.graphml"
    save_graph(g, graphml_path, format="graphml")
    
    # Save summary JSON
    from src.utils.manifests import save_json
    summary_path = output_dir / "logs" / "airport_network_summary.json"
    save_json(summary, summary_path)
    
    logger.info(f"Airport network summary: {summary['n_airports']} airports, {summary['n_routes']} routes")
    logger.info(f"LCC size: {summary['lcc_size']} ({100*summary['lcc_size']/summary['n_airports']:.1f}%)")
    
    return summary

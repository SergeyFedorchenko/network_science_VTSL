"""
Multilayer network construction.

Builds airline-layered network representation where each layer is an airline's
sub-network of flights/routes. Supports both airport-level and flight-level layers.

Per IMPLEMENTATION_PLAN Section 6.3:
- Store a single edge table with layer columns (src_id, dst_id, src_layer, dst_layer)
- Layers are OP_UNIQUE_CARRIER
- Within-layer edges: src_layer == dst_layer
- Optional inter-layer edges at shared airports (disabled by default for scale)
"""
import polars as pl
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


def build_multilayer_airport_edges(
    lf: pl.LazyFrame,
    airport_nodes: pl.DataFrame,
    config: Dict[str, Any],
) -> pl.DataFrame:
    """
    Build multilayer airport network edges with airline layers.
    
    Each edge has src_layer and dst_layer indicating the airline.
    Intra-layer edges connect airports within an airline's network.
    
    Args:
        lf: LazyFrame with flight data
        airport_nodes: DataFrame with airport nodes and IDs
        config: Multilayer configuration dict
        
    Returns:
        DataFrame with multilayer edges
    """
    logger.info("Building multilayer airport edges (airline layers)")
    
    layer_key = config.get("layer_key", "OP_UNIQUE_CARRIER")
    
    # Aggregate by (carrier, origin, dest) to get per-airline route weights
    edges = (
        lf.group_by([layer_key, "ORIGIN", "DEST"])
        .agg([
            pl.col("FLIGHTS").sum().alias("flight_count"),
            pl.col("DEP_DELAY").mean().alias("mean_dep_delay"),
            pl.col("ARR_DELAY").mean().alias("mean_arr_delay"),
            pl.col("DISTANCE").mean().alias("mean_distance"),
        ])
        .collect()
    )
    
    logger.info(f"Aggregated {len(edges)} airline-route combinations")
    
    # Create node mapping
    node_map = dict(zip(airport_nodes["code"], airport_nodes["node_id"]))
    
    # Add node IDs and layer columns
    edges = edges.with_columns([
        pl.col("ORIGIN").map_elements(lambda x: node_map.get(x), return_dtype=pl.Int64).alias("src_id"),
        pl.col("DEST").map_elements(lambda x: node_map.get(x), return_dtype=pl.Int64).alias("dst_id"),
        pl.col(layer_key).alias("src_layer"),
        pl.col(layer_key).alias("dst_layer"),  # Same layer for intra-layer edges
        pl.lit("intra_layer").alias("edge_type"),
    ])
    
    # Filter out edges with missing mappings
    edges = edges.filter(
        pl.col("src_id").is_not_null() & pl.col("dst_id").is_not_null()
    )
    
    # Select final columns in standard order
    edges = edges.select([
        "src_id", "dst_id", "src_layer", "dst_layer", "edge_type",
        "ORIGIN", "DEST", "flight_count", "mean_dep_delay", "mean_arr_delay", "mean_distance"
    ])
    
    logger.info(f"Created {len(edges)} multilayer edges")
    
    return edges


def build_interlayer_transfer_edges(
    lf: pl.LazyFrame,
    airport_nodes: pl.DataFrame,
    config: Dict[str, Any],
) -> pl.DataFrame:
    """
    Build inter-layer transfer edges between airlines at shared airports.
    
    These represent potential interline passenger connections.
    Disabled by default for scale.
    
    Args:
        lf: LazyFrame with flight data
        airport_nodes: DataFrame with airport nodes and IDs
        config: Multilayer configuration dict
        
    Returns:
        DataFrame with inter-layer edges (may be empty if disabled)
    """
    if not config.get("include_interlayer_transfer_edges", False):
        logger.info("Inter-layer transfer edges disabled in config")
        return pl.DataFrame({
            "src_id": [], "dst_id": [], "src_layer": [], "dst_layer": [],
            "edge_type": [], "airport": [], "transfer_count": []
        }).cast({
            "src_id": pl.Int64, "dst_id": pl.Int64,
            "src_layer": pl.Utf8, "dst_layer": pl.Utf8,
            "edge_type": pl.Utf8, "airport": pl.Utf8, "transfer_count": pl.Int64
        })
    
    logger.info("Building inter-layer transfer edges at shared airports")
    
    layer_key = config.get("layer_key", "OP_UNIQUE_CARRIER")
    
    # Find airlines present at each airport (as origin)
    airport_carriers = (
        lf.group_by(["ORIGIN", layer_key])
        .agg(pl.len().alias("presence"))
        .collect()
    )
    
    # Self-join to find carrier pairs at same airport
    carrier_pairs = (
        airport_carriers.join(
            airport_carriers,
            on="ORIGIN",
            suffix="_other"
        )
        .filter(pl.col(layer_key) < pl.col(f"{layer_key}_other"))  # Avoid duplicates
        .group_by(["ORIGIN", layer_key, f"{layer_key}_other"])
        .agg(pl.lit(1).alias("transfer_count"))
    )
    
    node_map = dict(zip(airport_nodes["code"], airport_nodes["node_id"]))
    
    # Create transfer edges (airport node to itself, different layers)
    interlayer = carrier_pairs.with_columns([
        pl.col("ORIGIN").map_elements(lambda x: node_map.get(x), return_dtype=pl.Int64).alias("src_id"),
        pl.col("ORIGIN").map_elements(lambda x: node_map.get(x), return_dtype=pl.Int64).alias("dst_id"),
        pl.col(layer_key).alias("src_layer"),
        pl.col(f"{layer_key}_other").alias("dst_layer"),
        pl.lit("inter_layer_transfer").alias("edge_type"),
        pl.col("ORIGIN").alias("airport"),
    ]).select([
        "src_id", "dst_id", "src_layer", "dst_layer", "edge_type", "airport", "transfer_count"
    ])
    
    logger.info(f"Created {len(interlayer)} inter-layer transfer edges")
    
    return interlayer


def compute_layer_summary(
    multilayer_edges: pl.DataFrame,
    layer_key: str = "src_layer"
) -> pl.DataFrame:
    """
    Compute summary statistics per layer (airline).
    
    Args:
        multilayer_edges: DataFrame with multilayer edges
        layer_key: Column name for layer identifier
        
    Returns:
        DataFrame with per-layer summary
    """
    logger.info("Computing layer summary statistics")
    
    # Edges per layer
    layer_edges = (
        multilayer_edges
        .filter(pl.col("edge_type") == "intra_layer")
        .group_by(layer_key)
        .agg([
            pl.len().alias("edge_count"),
            pl.col("flight_count").sum().alias("total_flights"),
            pl.col("mean_dep_delay").mean().alias("avg_dep_delay"),
            pl.col("mean_arr_delay").mean().alias("avg_arr_delay"),
        ])
    )
    
    # Unique nodes per layer
    layer_nodes = (
        multilayer_edges
        .filter(pl.col("edge_type") == "intra_layer")
        .select([layer_key, "src_id", "dst_id"])
        .unpivot(index=[layer_key], on=["src_id", "dst_id"], value_name="node_id")
        .group_by(layer_key)
        .agg(pl.col("node_id").n_unique().alias("node_count"))
    )
    
    # Join summaries
    summary = layer_edges.join(layer_nodes, on=layer_key, how="left")
    summary = summary.rename({layer_key: "layer"})
    summary = summary.sort("total_flights", descending=True)
    
    logger.info(f"Computed summary for {len(summary)} layers")
    
    return summary


def build_multilayer_network(
    lf: pl.LazyFrame,
    airport_nodes: pl.DataFrame,
    config: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Build complete multilayer network and save outputs.
    
    Args:
        lf: LazyFrame with flight data
        airport_nodes: DataFrame with airport nodes
        config: Multilayer configuration dict
        output_dir: Output directory for results
        
    Returns:
        Dictionary with network summary statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build intra-layer edges
    multilayer_edges = build_multilayer_airport_edges(lf, airport_nodes, config)
    
    # Build inter-layer edges if enabled
    interlayer_edges = build_interlayer_transfer_edges(lf, airport_nodes, config)
    
    # Combine if inter-layer edges exist
    if len(interlayer_edges) > 0:
        # Need to align schemas before concat
        multilayer_edges = multilayer_edges.with_columns([
            pl.lit(None).cast(pl.Utf8).alias("airport"),
            pl.lit(None).cast(pl.Int64).alias("transfer_count"),
        ])
        interlayer_edges = interlayer_edges.with_columns([
            pl.lit(None).cast(pl.Utf8).alias("ORIGIN"),
            pl.lit(None).cast(pl.Utf8).alias("DEST"),
            pl.lit(None).cast(pl.Int64).alias("flight_count"),
            pl.lit(None).cast(pl.Float64).alias("mean_dep_delay"),
            pl.lit(None).cast(pl.Float64).alias("mean_arr_delay"),
            pl.lit(None).cast(pl.Float64).alias("mean_distance"),
        ])
        all_edges = pl.concat([multilayer_edges, interlayer_edges], how="diagonal")
    else:
        all_edges = multilayer_edges
    
    # Compute layer summary
    layer_summary = compute_layer_summary(multilayer_edges)
    
    # Save outputs
    edges_path = output_dir / "multilayer_edges.parquet"
    summary_path = output_dir / "layer_summary.parquet"
    
    all_edges.write_parquet(edges_path)
    layer_summary.write_parquet(summary_path)
    
    logger.info(f"Saved multilayer edges to {edges_path}")
    logger.info(f"Saved layer summary to {summary_path}")
    
    # Return summary stats
    n_layers = layer_summary.height
    n_edges = len(all_edges)
    n_interlayer = len(interlayer_edges) if len(interlayer_edges) > 0 else 0
    
    return {
        "n_layers": n_layers,
        "n_edges": n_edges,
        "n_interlayer_edges": n_interlayer,
        "top_layers": layer_summary.head(10).to_dicts(),
    }

"""
Flight network construction.

Builds flight-centric network where nodes are individual flights.
Implements tail sequence edges and route kNN edges (scalable, no cliques).
"""
import polars as pl
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging
import hashlib


logger = logging.getLogger(__name__)


def create_flight_key(row_expr: pl.Expr) -> pl.Expr:
    """
    Create stable composite key for a flight.
    
    Uses: FL_DATE, OP_UNIQUE_CARRIER, OP_CARRIER_FL_NUM, ORIGIN, DEST, DEP_TIME
    
    Args:
        row_expr: Polars expression (typically pl.struct with relevant columns)
        
    Returns:
        String expression with composite key
    """
    # Create deterministic string key
    return (
        pl.col("FL_DATE").cast(pl.Utf8) + "|" +
        pl.col("OP_UNIQUE_CARRIER") + "|" +
        pl.col("OP_CARRIER_FL_NUM").cast(pl.Utf8) + "|" +
        pl.col("ORIGIN") + "|" +
        pl.col("DEST") + "|" +
        pl.col("DEP_TIME").cast(pl.Utf8)
    )


def scope_flights(
    lf: pl.LazyFrame,
    scope_config: Dict[str, Any]
) -> pl.LazyFrame:
    """
    Apply scoping to reduce flight graph size.
    
    Modes:
    - "full": All flights (use with caution)
    - "top_airports": Flights touching top K airports by volume
    - "sample": Deterministic sample of flights
    
    Args:
        lf: LazyFrame with flight data
        scope_config: Scope configuration dict
        
    Returns:
        Scoped LazyFrame
    """
    mode = scope_config.get("mode", "top_airports")
    
    if mode == "full":
        logger.info("Scope mode: FULL (no filtering)")
        return lf
    
    elif mode == "top_airports":
        k = scope_config.get("top_airports_k", 50)
        logger.info(f"Scope mode: TOP_AIRPORTS (k={k})")
        
        # Find top K airports by flight volume
        airport_volumes = pl.concat([
            lf.select([pl.col("ORIGIN").alias("airport")]),
            lf.select([pl.col("DEST").alias("airport")])
        ]).group_by("airport").agg(
            pl.count().alias("volume")
        ).collect().sort("volume", descending=True).head(k)
        
        top_airports = set(airport_volumes["airport"].to_list())
        logger.info(f"Top {k} airports: {len(top_airports)} unique codes")
        
        # Filter flights touching these airports
        lf = lf.filter(
            pl.col("ORIGIN").is_in(top_airports) | pl.col("DEST").is_in(top_airports)
        )
        
        return lf
    
    elif mode == "sample":
        frac = scope_config.get("sample_frac", 0.10)
        logger.info(f"Scope mode: SAMPLE (frac={frac})")
        
        # Deterministic sample using hash-based selection
        # Hash the flight key and take modulo
        lf = lf.with_columns([
            create_flight_key(pl.struct(["FL_DATE", "OP_UNIQUE_CARRIER", "OP_CARRIER_FL_NUM", "ORIGIN", "DEST", "DEP_TIME"])).alias("_temp_key")
        ])
        
        # Use hash to sample
        sample_threshold = int(frac * 2**31)
        lf = lf.filter(
            pl.col("_temp_key").str.len_bytes().hash() % (2**31) < sample_threshold
        ).drop("_temp_key")
        
        return lf
    
    else:
        raise ValueError(f"Unknown scope mode: {mode}")


def build_flight_nodes(
    lf: pl.LazyFrame,
    scope_config: Dict[str, Any]
) -> pl.DataFrame:
    """
    Build flight node table with scoping applied.
    
    Args:
        lf: LazyFrame with flight data (with time features)
        scope_config: Scope configuration
        
    Returns:
        DataFrame with flight nodes
    """
    logger.info("Building flight nodes table")
    
    # Apply scoping
    lf = scope_flights(lf, scope_config)
    
    # Create flight key
    lf = lf.with_columns([
        create_flight_key(pl.struct(["FL_DATE", "OP_UNIQUE_CARRIER", "OP_CARRIER_FL_NUM", "ORIGIN", "DEST", "DEP_TIME"])).alias("flight_key")
    ])
    
    # Select relevant columns for nodes
    nodes = lf.select([
        pl.col("flight_key"),
        pl.col("FL_DATE"),
        pl.col("MONTH"),
        pl.col("OP_UNIQUE_CARRIER").alias("carrier"),
        pl.col("OP_CARRIER_FL_NUM").alias("flight_num"),
        pl.col("TAIL_NUM").alias("tail"),
        pl.col("ORIGIN").alias("origin"),
        pl.col("DEST").alias("dest"),
        pl.col("dep_ts"),
        pl.col("arr_ts"),
        pl.col("DEP_DELAY").alias("dep_delay"),
        pl.col("ARR_DELAY").alias("arr_delay"),
        pl.col("CANCELLED").alias("cancelled"),
        pl.col("DISTANCE").alias("distance"),
        pl.col("AIR_TIME").alias("air_time"),
    ]).collect()
    
    # Remove duplicates (if any) and sort for stability
    nodes = nodes.unique(subset=["flight_key"]).sort("dep_ts")
    
    # Add integer flight IDs
    nodes = nodes.with_columns([
        pl.arange(0, len(nodes)).alias("flight_id")
    ])
    
    logger.info(f"Created {len(nodes)} flight nodes")
    
    return nodes


def build_tail_sequence_edges(
    nodes: pl.DataFrame
) -> pl.DataFrame:
    """
    Build edges connecting consecutive flights by the same aircraft (TAIL_NUM).
    
    This captures aircraft rotation and is critical for delay propagation.
    
    Args:
        nodes: DataFrame with flight nodes
        
    Returns:
        DataFrame with tail sequence edges
    """
    logger.info("Building tail sequence edges")
    
    # Filter to flights with non-null tail and timestamp
    tail_flights = nodes.filter(
        pl.col("tail").is_not_null() & 
        pl.col("dep_ts").is_not_null() &
        pl.col("arr_ts").is_not_null()
    ).sort(["tail", "dep_ts"])
    
    # Create shifted columns to get next flight within each tail
    tail_edges = tail_flights.with_columns([
        pl.col("flight_id").shift(-1).over("tail").alias("next_flight_id"),
        pl.col("dep_ts").shift(-1).over("tail").alias("next_dep_ts"),
        pl.col("carrier").shift(-1).over("tail").alias("next_carrier"),
    ])
    
    # Filter to valid edges (where next flight exists and is later)
    tail_edges = tail_edges.filter(
        pl.col("next_flight_id").is_not_null() &
        (pl.col("next_dep_ts") > pl.col("arr_ts"))
    )
    
    # Calculate ground time
    tail_edges = tail_edges.with_columns([
        ((pl.col("next_dep_ts") - pl.col("arr_ts")).dt.total_minutes()).alias("ground_time_minutes"),
        (pl.col("carrier") == pl.col("next_carrier")).alias("same_carrier")
    ])
    
    # Select edge columns
    edges = tail_edges.select([
        pl.col("flight_id").alias("src_id"),
        pl.col("next_flight_id").alias("dst_id"),
        pl.lit("tail_next_leg").alias("edge_type"),
        pl.col("ground_time_minutes"),
        pl.col("same_carrier"),
        pl.col("tail")
    ])
    
    logger.info(f"Created {len(edges)} tail sequence edges")
    
    return edges


def build_route_knn_edges(
    nodes: pl.DataFrame,
    k: int = 3
) -> pl.DataFrame:
    """
    Build edges connecting each flight to next k flights on same route.
    
    This avoids creating route cliques while capturing temporal route patterns.
    
    Args:
        nodes: DataFrame with flight nodes
        k: Number of next flights to connect to
        
    Returns:
        DataFrame with route kNN edges
    """
    logger.info(f"Building route kNN edges (k={k})")
    
    # Filter to flights with valid timestamps
    route_flights = nodes.filter(
        pl.col("dep_ts").is_not_null()
    ).sort(["origin", "dest", "dep_ts"])
    
    # Create k shifted columns for next k flights
    shift_exprs = []
    for i in range(1, k + 1):
        shift_exprs.extend([
            pl.col("flight_id").shift(-i).over(["origin", "dest"]).alias(f"next_id_{i}"),
            pl.col("dep_ts").shift(-i).over(["origin", "dest"]).alias(f"next_dep_ts_{i}"),
        ])
    
    route_edges = route_flights.with_columns(shift_exprs)
    
    # Melt to long format (create one row per edge)
    edge_dfs = []
    for i in range(1, k + 1):
        edge_df = route_edges.select([
            pl.col("flight_id").alias("src_id"),
            pl.col(f"next_id_{i}").alias("dst_id"),
            pl.col("dep_ts"),
            pl.col(f"next_dep_ts_{i}").alias("next_dep_ts"),
        ]).filter(
            pl.col("dst_id").is_not_null()
        )
        edge_dfs.append(edge_df)
    
    edges = pl.concat(edge_dfs)
    
    # Calculate time delta
    edges = edges.with_columns([
        ((pl.col("next_dep_ts") - pl.col("dep_ts")).dt.total_minutes()).alias("delta_dep_minutes"),
        pl.lit("route_knn").alias("edge_type")
    ])
    
    # Select final columns (match tail edge schema)
    edges = edges.select([
        pl.col("src_id"),
        pl.col("dst_id"),
        pl.col("edge_type"),
        pl.col("delta_dep_minutes").alias("ground_time_minutes"),  # Use same column name
        pl.lit(None, dtype=pl.Boolean).alias("same_carrier"),
        pl.lit(None, dtype=pl.String).alias("tail")
    ])
    
    logger.info(f"Created {len(edges)} route kNN edges")
    
    return edges


def build_flight_network(
    lf: pl.LazyFrame,
    config: Dict[str, Any],
    output_dir: Path
) -> Dict[str, Any]:
    """
    Build complete flight network and save outputs.
    
    Args:
        lf: LazyFrame with flight data (must have time features)
        config: Configuration dictionary
        output_dir: Output directory
        
    Returns:
        Dictionary with network summary
    """
    flight_config = config["flight_graph"]
    scope_config = flight_config["scope"]
    edges_config = flight_config["edges"]
    
    # Build nodes (with scoping)
    nodes = build_flight_nodes(lf, scope_config)
    
    # Build edges
    edge_dfs = []
    
    # Tail sequence edges
    if edges_config.get("include_tail_sequence", True):
        tail_edges = build_tail_sequence_edges(nodes)
        edge_dfs.append(tail_edges)
    
    # Route kNN edges
    if edges_config.get("include_same_route_knn", True):
        route_k = edges_config.get("route_knn_k", 3)
        route_edges = build_route_knn_edges(nodes, k=route_k)
        edge_dfs.append(route_edges)
    
    # Combine all edges
    if not edge_dfs:
        raise ValueError("No edge types enabled in configuration")
    
    all_edges = pl.concat(edge_dfs)
    
    # Remove duplicate edges (can happen if same pair connected by multiple mechanisms)
    all_edges = all_edges.unique(subset=["src_id", "dst_id"], keep="first")
    
    logger.info(f"Total unique edges: {len(all_edges)}")
    
    # Count by edge type
    edge_type_counts = all_edges.group_by("edge_type").agg(
        pl.count().alias("count")
    )
    logger.info("Edge type distribution:")
    for row in edge_type_counts.iter_rows(named=True):
        logger.info(f"  {row['edge_type']}: {row['count']}")
    
    # Save outputs
    networks_dir = output_dir / "networks"
    networks_dir.mkdir(parents=True, exist_ok=True)
    
    nodes_path = networks_dir / "flight_nodes.parquet"
    edges_path = networks_dir / "flight_edges.parquet"
    
    nodes.write_parquet(nodes_path)
    all_edges.write_parquet(edges_path)
    
    logger.info(f"Saved flight nodes to {nodes_path}")
    logger.info(f"Saved flight edges to {edges_path}")
    
    # Create summary
    summary = {
        "n_flights": len(nodes),
        "n_edges": len(all_edges),
        "scope_mode": scope_config.get("mode"),
        "scope_params": scope_config,
        "edge_types": edge_type_counts.to_dicts(),
        "date_range": {
            "min": str(nodes["FL_DATE"].min()),
            "max": str(nodes["FL_DATE"].max())
        },
        "carriers": nodes["carrier"].n_unique(),
        "airports": nodes["origin"].n_unique() + nodes["dest"].n_unique(),  # approx unique
    }
    
    # Save summary
    from src.utils.manifests import save_json
    summary_path = output_dir / "logs" / "flight_graph_summary.json"
    save_json(summary, summary_path)
    
    logger.info(f"Flight network summary: {summary['n_flights']} flights, {summary['n_edges']} edges")
    
    return summary

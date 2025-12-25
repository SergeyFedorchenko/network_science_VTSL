#!/usr/bin/env python
"""
WS2 Script 05: Run community detection (Leiden CPM) on airport and flight networks.

Consumes WS1 outputs (airport and flight network parquet files).
Runs Leiden with CPM objective, selects best partition from multiple runs.
Produces membership and community summary tables.
Writes run manifest JSON.

Usage:
    python scripts/05_run_communities.py
"""

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import polars as pl
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis.centrality import load_airport_graph_from_parquet
from analysis.community import (
    is_sbm_available,
    run_sbm_community_detection,
    select_best_partition,
    summarize_communities_airport,
    summarize_communities_flight,
    write_community_outputs,
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


def load_flight_graph_from_parquet(
    nodes_path: Path,
    edges_path: Path,
    directed: bool = True,
    weight_col: str | None = None,
):
    """
    Load flight graph from WS1 parquet outputs.
    
    Similar to load_airport_graph_from_parquet but handles flight node schema.
    """
    import igraph as ig
    
    nodes_df = pl.read_parquet(nodes_path)
    edges_df = pl.read_parquet(edges_path)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Loaded {len(nodes_df)} flight nodes from {nodes_path}")
    logger.info(f"Loaded {len(edges_df)} flight edges from {edges_path}")
    
    # Build vertex mapping (WS1 uses node_id for airports, flight_id for flights)
    if "vertex_id" in nodes_df.columns:
        vertex_id_col = "vertex_id"
    elif "node_id" in nodes_df.columns:
        vertex_id_col = "node_id"
    elif "flight_id" in nodes_df.columns:
        vertex_id_col = "flight_id"
    else:
        raise ValueError(f"No vertex_id, node_id, or flight_id column found in flight nodes. Columns: {nodes_df.columns}")
    
    vertex_ids = nodes_df[vertex_id_col].to_list()
    
    # Check if vertex_ids are contiguous
    if vertex_ids != list(range(len(vertex_ids))):
        logger.warning("vertex_id not contiguous; remapping")
        id_map = {old: new for new, old in enumerate(vertex_ids)}
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
    
    # Add vertex attributes (flight metadata)
    for col in nodes_df.columns:
        if col != "vertex_id":
            g.vs[col] = nodes_df[col].to_list()
    
    # Add edge weight if requested
    if weight_col and weight_col in edges_df.columns:
        g.es["weight"] = edges_df[weight_col].to_list()
        logger.info(f"Added edge weights from column '{weight_col}'")
    
    logger.info(f"Created flight graph: N={g.vcount()}, E={g.ecount()}, directed={g.is_directed()}")
    
    return g


def main():
    """Main execution."""
    # Setup
    root = get_project_root()
    log_dir = root / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(log_dir / "05_run_communities.log")
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("WS2 Script 05: Community Detection (Leiden CPM)")
    logger.info("=" * 80)
    
    # Load config
    config = load_config()
    logger.info(f"Loaded config: seed={config.get('seed')}")
    
    # Set seed
    seed = config.get("seed", 42)
    set_global_seed(seed)
    logger.info(f"Set global seed: {seed}")
    
    # Paths
    networks_dir = root / "results" / "networks"
    analysis_dir = root / "results" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    overwrite = config.get("outputs", {}).get("overwrite", False)
    
    manifest = {
        "script": "05_run_communities.py",
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "config_snapshot": {
            "seed": seed,
            "leiden_resolution": config.get("analysis", {}).get("communities", {}).get("leiden", {}).get("resolution", 0.01),
            "leiden_n_runs": config.get("analysis", {}).get("communities", {}).get("leiden", {}).get("n_runs", 10),
        },
        "outputs": {},
    }
    
    # ========================================
    # Part 1: Airport network communities
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("Part 1: Airport Network Communities")
    logger.info("=" * 80)
    
    airport_nodes_path = networks_dir / "airport_nodes.parquet"
    airport_edges_path = networks_dir / "airport_edges.parquet"
    
    if airport_nodes_path.exists() and airport_edges_path.exists():
        logger.info(f"Loading airport graph from WS1 outputs:")
        logger.info(f"  Nodes: {airport_nodes_path}")
        logger.info(f"  Edges: {airport_edges_path}")
        
        directed = config.get("airport_network", {}).get("directed", True)
        weight_col = config.get("airport_network", {}).get("edge_weight", "flight_count")
        
        airport_g = load_airport_graph_from_parquet(
            nodes_path=airport_nodes_path,
            edges_path=airport_edges_path,
            directed=directed,
            weight_col=weight_col,
        )
        
        logger.info(f"Loaded airport graph: N={airport_g.vcount()}, E={airport_g.ecount()}")
        
        # Run Leiden CPM
        logger.info("Running Leiden CPM on airport network...")
        airport_membership, airport_run_log = select_best_partition(
            g=airport_g,
            config=config,
            weights=weight_col,
        )
        
        logger.info(f"Airport communities: {airport_run_log['n_communities']} communities")
        logger.info(f"Best quality: {airport_run_log['best_quality']:.4f}")
        
        # Build membership DataFrame
        airport_membership_df = pl.DataFrame({
            "vertex_id": list(range(airport_g.vcount())),
            "community_id": airport_membership,
        })
        
        # Load centrality results for summarization (if available)
        centrality_df = None
        centrality_path = analysis_dir / "airport_centrality.parquet"
        if centrality_path.exists():
            centrality_df = pl.read_parquet(centrality_path)
            logger.info(f"Loaded centrality results from {centrality_path}")
        
        # Summarize communities
        logger.info("Summarizing airport communities...")
        airport_nodes_df = pl.read_parquet(airport_nodes_path)
        airport_summary_df = summarize_communities_airport(
            nodes_df=airport_nodes_df,
            membership_df=airport_membership_df,
            centrality_df=centrality_df,
        )
        
        logger.info(f"Airport community summary: {len(airport_summary_df)} communities")
        
        # Write outputs
        logger.info("Writing airport community outputs...")
        airport_output_paths = write_community_outputs(
            membership_df=airport_membership_df,
            summary_df=airport_summary_df,
            run_log=airport_run_log,
            output_dir=str(root / "results"),
            graph_type="airport",
            overwrite=overwrite,
        )
        
        manifest["outputs"]["airport"] = {
            "paths": airport_output_paths,
            "n_communities": airport_run_log["n_communities"],
            "best_quality": airport_run_log["best_quality"],
            "best_seed": airport_run_log["best_seed"],
            "run_log": airport_run_log,
        }
        
        logger.info(f"Wrote {len(airport_output_paths)} airport output files")
        
        # ========================================
        # Optional: SBM community detection
        # ========================================
        sbm_config = config.get("analysis", {}).get("communities", {}).get("sbm_optional", {})
        if sbm_config.get("enabled", False):
            if is_sbm_available():
                logger.info("\n" + "-" * 40)
                logger.info("Running optional SBM community detection on airport network...")
                logger.info("-" * 40)
                
                try:
                    sbm_membership, sbm_run_log = run_sbm_community_detection(
                        g=airport_g,
                        config=config,
                        weights=weight_col,
                        seed=seed,
                    )
                    
                    logger.info(f"SBM communities: {sbm_run_log['n_communities']} communities")
                    
                    # Build membership DataFrame
                    sbm_membership_df = pl.DataFrame({
                        "vertex_id": list(range(airport_g.vcount())),
                        "community_id": sbm_membership,
                    })
                    
                    # Write SBM outputs
                    sbm_output_path = analysis_dir / "airport_sbm_membership.parquet"
                    if not sbm_output_path.exists() or overwrite:
                        sbm_membership_df.write_parquet(sbm_output_path)
                        logger.info(f"Wrote SBM membership: {sbm_output_path}")
                    
                    manifest["outputs"]["airport_sbm"] = {
                        "paths": {"membership": str(sbm_output_path)},
                        "n_communities": sbm_run_log["n_communities"],
                        "run_log": sbm_run_log,
                    }
                except Exception as e:
                    logger.error(f"SBM community detection failed: {e}")
                    manifest["outputs"]["airport_sbm"] = {"error": str(e)}
            else:
                logger.warning("SBM enabled in config but graspologic not installed")
                manifest["outputs"]["airport_sbm"] = {"error": "graspologic not installed"}
    else:
        logger.warning("Airport network outputs not found; skipping airport communities")
        logger.warning(f"Expected: {airport_nodes_path}, {airport_edges_path}")
    
    # ========================================
    # Part 2: Flight network communities
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("Part 2: Flight Network Communities")
    logger.info("=" * 80)
    
    flight_nodes_path = networks_dir / "flight_nodes.parquet"
    flight_edges_path = networks_dir / "flight_edges.parquet"
    
    if flight_nodes_path.exists() and flight_edges_path.exists():
        logger.info(f"Loading flight graph from WS1 outputs:")
        logger.info(f"  Nodes: {flight_nodes_path}")
        logger.info(f"  Edges: {flight_edges_path}")
        
        # Flight graph parameters (use defaults)
        flight_g = load_flight_graph_from_parquet(
            nodes_path=flight_nodes_path,
            edges_path=flight_edges_path,
            directed=True,
            weight_col=None,  # flight edges typically unweighted or binary
        )
        
        logger.info(f"Loaded flight graph: N={flight_g.vcount()}, E={flight_g.ecount()}")
        
        # Run Leiden CPM (convert to undirected for faster computation)
        logger.info("Running Leiden CPM on flight network...")
        logger.info("Note: Converting to undirected for community detection (faster, preserves community structure)")
        flight_membership, flight_run_log = select_best_partition(
            g=flight_g,
            config=config,
            weights=None,
            convert_to_undirected=True,  # Significantly faster for large graphs
        )
        
        logger.info(f"Flight communities: {flight_run_log['n_communities']} communities")
        logger.info(f"Best quality: {flight_run_log['best_quality']:.4f}")
        
        # Build membership DataFrame
        flight_membership_df = pl.DataFrame({
            "vertex_id": list(range(flight_g.vcount())),
            "community_id": flight_membership,
        })
        
        # Summarize communities
        logger.info("Summarizing flight communities...")
        flight_nodes_df = pl.read_parquet(flight_nodes_path)
        flight_summary_df = summarize_communities_flight(
            flight_nodes_df=flight_nodes_df,
            membership_df=flight_membership_df,
        )
        
        logger.info(f"Flight community summary: {len(flight_summary_df)} communities")
        
        # Write outputs
        logger.info("Writing flight community outputs...")
        flight_output_paths = write_community_outputs(
            membership_df=flight_membership_df,
            summary_df=flight_summary_df,
            run_log=flight_run_log,
            output_dir=str(root / "results"),
            graph_type="flight",
            overwrite=overwrite,
        )
        
        manifest["outputs"]["flight"] = {
            "paths": flight_output_paths,
            "n_communities": flight_run_log["n_communities"],
            "best_quality": flight_run_log["best_quality"],
            "best_seed": flight_run_log["best_seed"],
            "run_log": flight_run_log,
        }
        
        logger.info(f"Wrote {len(flight_output_paths)} flight output files")
    else:
        logger.warning("Flight network outputs not found; skipping flight communities")
        logger.warning(f"Expected: {flight_nodes_path}, {flight_edges_path}")
    
    # ========================================
    # Write manifest
    # ========================================
    manifest_path = log_dir / "05_run_communities_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"\nWrote manifest: {manifest_path}")
    
    logger.info("=" * 80)
    logger.info("WS2 Script 05: Complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

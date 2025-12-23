#!/usr/bin/env python
"""
WS2 Script 04: Run centrality analysis on airport network.

Consumes WS1 outputs (airport_nodes.parquet, airport_edges.parquet).
Computes degree, strength, PageRank, betweenness, and graph structure summary.
Writes results under results/analysis and results/tables.
Writes run manifest JSON.

Usage:
    python scripts/04_run_centrality.py
"""

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis.centrality import (
    compute_airport_centrality,
    compute_degree_distribution,
    compute_graph_summary,
    load_airport_graph_from_parquet,
    write_centrality_outputs,
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
    
    setup_logging(log_dir / "04_run_centrality.log")
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("WS2 Script 04: Centrality Analysis")
    logger.info("=" * 80)
    
    # Load config
    config = load_config()
    logger.info(f"Loaded config: seed={config.get('seed')}")
    
    # Set seed
    seed = config.get("seed", 42)
    set_global_seed(seed)
    logger.info(f"Set global seed: {seed}")
    
    # Input paths from WS1
    networks_dir = root / "results" / "networks"
    nodes_path = networks_dir / "airport_nodes.parquet"
    edges_path = networks_dir / "airport_edges.parquet"
    
    if not nodes_path.exists() or not edges_path.exists():
        logger.error(f"WS1 outputs not found: {nodes_path}, {edges_path}")
        logger.error("Run scripts/01_build_airport_network.py first")
        sys.exit(1)
    
    logger.info(f"Loading airport graph from WS1 outputs:")
    logger.info(f"  Nodes: {nodes_path}")
    logger.info(f"  Edges: {edges_path}")
    
    # Load graph
    directed = config.get("airport_network", {}).get("directed", True)
    weight_col = config.get("airport_network", {}).get("edge_weight", "flight_count")
    
    g = load_airport_graph_from_parquet(
        nodes_path=nodes_path,
        edges_path=edges_path,
        directed=directed,
        weight_col=weight_col,
    )
    
    logger.info(f"Loaded graph: N={g.vcount()}, E={g.ecount()}, directed={g.is_directed()}")
    
    # Compute graph summary
    logger.info("Computing graph structure summary...")
    graph_summary = compute_graph_summary(g)
    logger.info(f"Graph summary: {graph_summary}")
    
    # Compute centrality
    logger.info("Computing centrality metrics...")
    centrality_df = compute_airport_centrality(
        g=g,
        weight_col=weight_col,
        config=config,
    )
    logger.info(f"Computed centrality for {len(centrality_df)} airports")
    
    # Compute degree distributions
    logger.info("Computing degree distributions...")
    if g.is_directed():
        degree_dist_in = compute_degree_distribution(g, mode="in")
        degree_dist_out = compute_degree_distribution(g, mode="out")
    else:
        degree_dist_in = compute_degree_distribution(g, mode="all")
        degree_dist_out = degree_dist_in
    
    logger.info(f"Degree distribution: {len(degree_dist_in)} unique in-degrees, "
                f"{len(degree_dist_out)} unique out-degrees")
    
    # Write outputs
    logger.info("Writing outputs...")
    overwrite = config.get("outputs", {}).get("overwrite", False)
    output_paths = write_centrality_outputs(
        centrality_df=centrality_df,
        degree_dist_in=degree_dist_in,
        degree_dist_out=degree_dist_out,
        graph_summary=graph_summary,
        output_dir=root / "results",
        overwrite=overwrite,
    )
    
    logger.info(f"Wrote {len(output_paths)} output files")
    
    # Write run manifest
    manifest = {
        "script": "04_run_centrality.py",
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "config_snapshot": {
            "seed": seed,
            "directed": directed,
            "weight_col": weight_col,
            "betweenness_approx_cutoff": config.get("analysis", {}).get("centrality", {}).get("betweenness_approx_cutoff", 20000),
        },
        "inputs": {
            "nodes": str(nodes_path),
            "edges": str(edges_path),
        },
        "outputs": output_paths,
        "graph_summary": graph_summary,
        "centrality_summary": {
            "n_airports": len(centrality_df),
            "max_in_degree": int(centrality_df["in_degree"].max()),
            "max_out_degree": int(centrality_df["out_degree"].max()),
            "max_pagerank": float(centrality_df["pagerank"].max()),
            "max_betweenness": float(centrality_df["betweenness"].max()),
        },
    }
    
    manifest_path = log_dir / "04_run_centrality_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Wrote manifest: {manifest_path}")
    
    logger.info("=" * 80)
    logger.info("WS2 Script 04: Complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

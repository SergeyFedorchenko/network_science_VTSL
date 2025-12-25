#!/usr/bin/env python
"""
WS4 Script 09: Run business analysis module.

Computes airline-level operational, network, and disruption metrics.
Optionally merges with WS2 centrality results for hub analysis.
Writes results under results/business and results/tables.
Writes run manifest JSON.

Usage:
    python scripts/09_run_business_module.py
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

from business.airline_metrics import (
    compute_airline_operational_metrics,
    compute_disruption_cost_proxy,
    compute_hub_concentration,
    merge_airline_metrics,
    write_business_outputs,
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

    setup_logging(log_dir / "09_run_business_module.log")
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("WS4 Script 09: Business Analysis Module")
    logger.info("=" * 80)

    # Load config
    config = load_config()
    logger.info(f"Loaded config: seed={config.get('seed')}")

    # Set seed (for reproducibility)
    seed = config.get("seed", 42)
    set_global_seed(seed)
    logger.info(f"Set global seed: {seed}")

    # Paths
    cleaned_path = root / config["data"]["cleaned_path"]
    analysis_dir = root / "results" / "analysis"
    business_dir = root / "results" / "business"
    tables_dir = root / "results" / "tables"

    business_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Filters
    filters = config.get("filters", {})

    # Check if cleaned data exists
    if not cleaned_path.exists():
        logger.error(f"Cleaned data not found: {cleaned_path}")
        logger.error("Run data cleaning script first")
        sys.exit(1)

    # ========================================================================
    # PART 1: Operational metrics
    # ========================================================================
    logger.info("-" * 80)
    logger.info("PART 1: Computing operational metrics")
    logger.info("-" * 80)

    operational_df = compute_airline_operational_metrics(
        cleaned_path=cleaned_path,
        filters=filters,
    )

    logger.info(f"Computed operational metrics for {len(operational_df)} carriers")
    logger.info(f"Sample:\n{operational_df.head()}")

    # ========================================================================
    # PART 2: Hub concentration
    # ========================================================================
    logger.info("-" * 80)
    logger.info("PART 2: Computing hub concentration")
    logger.info("-" * 80)

    hub_concentration_df = compute_hub_concentration(
        cleaned_path=cleaned_path,
        filters=filters,
        top_k=[1, 3],
    )

    logger.info(f"Computed hub concentration for {len(hub_concentration_df)} carriers")
    logger.info(f"Sample:\n{hub_concentration_df.head()}")

    # ========================================================================
    # PART 3: Disruption cost proxy
    # ========================================================================
    logger.info("-" * 80)
    logger.info("PART 3: Computing disruption cost proxy")
    logger.info("-" * 80)

    # Get cost parameters from config (with defaults)
    business_cfg = config.get("business", {})
    cost_per_delay_minute = business_cfg.get("cost_per_delay_minute", 75.0)
    cost_per_cancellation = business_cfg.get("cost_per_cancellation", 10000.0)

    disruption_cost_df = compute_disruption_cost_proxy(
        cleaned_path=cleaned_path,
        filters=filters,
        cost_per_delay_minute=cost_per_delay_minute,
        cost_per_cancellation=cost_per_cancellation,
    )

    logger.info(f"Computed disruption cost for {len(disruption_cost_df)} carriers")
    logger.info(f"Sample:\n{disruption_cost_df.head()}")

    # ========================================================================
    # PART 4: Merge with centrality (optional)
    # ========================================================================
    logger.info("-" * 80)
    logger.info("PART 4: Merging airline metrics")
    logger.info("-" * 80)

    # Try to load airport centrality from WS2
    centrality_path = analysis_dir / "airport_centrality.parquet"
    if centrality_path.exists():
        logger.info(f"Loading centrality from WS2: {centrality_path}")
        centrality_df = pl.read_parquet(centrality_path)
    else:
        logger.warning(f"Centrality not found: {centrality_path}")
        centrality_df = None

    airline_summary_df = merge_airline_metrics(
        operational_df=operational_df,
        hub_concentration_df=hub_concentration_df,
        disruption_cost_df=disruption_cost_df,
        centrality_df=centrality_df,
    )

    logger.info(f"Merged metrics for {len(airline_summary_df)} carriers")
    logger.info(f"Columns: {airline_summary_df.columns}")

    # ========================================================================
    # PART 5: Write outputs
    # ========================================================================
    logger.info("-" * 80)
    logger.info("PART 5: Writing outputs")
    logger.info("-" * 80)

    overwrite = config.get("outputs", {}).get("overwrite", False)
    output_paths = write_business_outputs(
        airline_summary=airline_summary_df,
        hub_concentration=hub_concentration_df,
        disruption_cost=disruption_cost_df,
        output_dir=root / "results",
        overwrite=overwrite,
    )

    logger.info(f"Wrote {len(output_paths)} output files")

    # ========================================================================
    # PART 6: Summary statistics
    # ========================================================================
    logger.info("-" * 80)
    logger.info("PART 6: Summary statistics")
    logger.info("-" * 80)

    summary_stats = {
        "n_carriers": len(airline_summary_df),
        "total_flights": int(airline_summary_df["flight_count"].sum()),
        "avg_cancellation_rate": float(airline_summary_df["cancellation_rate"].mean()),
        "avg_arr_delay": float(airline_summary_df["mean_arr_delay"].mean()),
        "total_disruption_cost": float(airline_summary_df["total_cost"].sum()),
        "top_carrier_by_flights": airline_summary_df[0, "carrier"],
        "top_carrier_by_cost": disruption_cost_df[0, "carrier"],
    }

    logger.info(f"Summary: {summary_stats}")

    # Write manifest
    manifest = {
        "script": "09_run_business_module.py",
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "config_snapshot": {
            "seed": seed,
            "filters": filters,
            "cost_per_delay_minute": cost_per_delay_minute,
            "cost_per_cancellation": cost_per_cancellation,
        },
        "inputs": {
            "cleaned_path": str(cleaned_path),
            "centrality_path": str(centrality_path) if centrality_path.exists() else None,
        },
        "outputs": output_paths,
        "summary_stats": summary_stats,
    }

    manifest_path = log_dir / "09_run_business_module_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Wrote manifest: {manifest_path}")

    logger.info("=" * 80)
    logger.info("WS4 Script 09: Complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

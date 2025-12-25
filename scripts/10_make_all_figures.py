#!/usr/bin/env python
"""
WS4 Script 10: Make all final figures.

Reads existing outputs from WS1-WS4 and produces report-ready figures.
Does NOT recompute any analyses.
Writes figures to results/figures/.
Writes run manifest JSON.

Usage:
    python scripts/10_make_all_figures.py
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

from utils.logging import setup_logging
from utils.paths import get_project_root
from viz.plotting import (
    plot_connectivity_vs_delay_scatter,
    plot_hub_dependence_by_airline,
    plot_link_prediction_performance,
    plot_top_route_predictions,
)


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


def check_file_exists(path: Path, name: str, logger) -> bool:
    """Check if file exists and log warning if not."""
    if not path.exists():
        logger.warning(f"{name} not found: {path}")
        return False
    logger.info(f"Found {name}: {path}")
    return True


def main():
    """Main execution."""
    # Setup
    root = get_project_root()
    log_dir = root / "results" / "logs"
    figures_dir = root / "results" / "figures"
    log_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(log_dir / "10_make_all_figures.log")
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("WS4 Script 10: Make All Final Figures")
    logger.info("=" * 80)

    # Load config
    config = load_config()
    overwrite = config.get("outputs", {}).get("overwrite", False)

    # Paths
    analysis_dir = root / "results" / "analysis"
    business_dir = root / "results" / "business"
    tables_dir = root / "results" / "tables"

    output_paths = []

    # ========================================================================
    # Figure 06: Hub dependence by airline
    # ========================================================================
    logger.info("-" * 80)
    logger.info("Figure 06: Hub dependence by airline")
    logger.info("-" * 80)

    hub_conc_path = business_dir / "hub_concentration.parquet"
    fig06_path = figures_dir / "fig06_hub_dependence_by_airline.png"

    if check_file_exists(hub_conc_path, "Hub concentration data", logger):
        if not fig06_path.exists() or overwrite:
            hub_df = pl.read_parquet(hub_conc_path)
            plot_hub_dependence_by_airline(
                hub_concentration_df=hub_df,
                output_path=fig06_path,
                top_n=15,
            )
            output_paths.append(str(fig06_path))
        else:
            logger.info(f"Skipped (exists): {fig06_path}")
            output_paths.append(str(fig06_path))

    # ========================================================================
    # Figure 07: Connectivity vs delay scatter
    # ========================================================================
    logger.info("-" * 80)
    logger.info("Figure 07: Connectivity vs delay scatter")
    logger.info("-" * 80)

    airline_summary_path = business_dir / "airline_summary_metrics.parquet"
    fig07_path = figures_dir / "fig07_connectivity_vs_delay_scatter.png"

    if check_file_exists(airline_summary_path, "Airline summary data", logger):
        if not fig07_path.exists() or overwrite:
            summary_df = pl.read_parquet(airline_summary_path)
            plot_connectivity_vs_delay_scatter(
                airline_summary_df=summary_df,
                output_path=fig07_path,
                x_metric="hub_top1_pct",
                y_metric="mean_arr_delay",
            )
            output_paths.append(str(fig07_path))
        else:
            logger.info(f"Skipped (exists): {fig07_path}")
            output_paths.append(str(fig07_path))

    # ========================================================================
    # Figure 08: Link prediction performance
    # ========================================================================
    logger.info("-" * 80)
    logger.info("Figure 08: Link prediction performance")
    logger.info("-" * 80)

    linkpred_metrics_path = analysis_dir / "linkpred_metrics.json"
    fig08_path = figures_dir / "fig08_link_prediction_performance.png"

    if check_file_exists(linkpred_metrics_path, "Link prediction metrics", logger):
        if not fig08_path.exists() or overwrite:
            with open(linkpred_metrics_path) as f:
                linkpred_metrics = json.load(f)

            # Restructure for plotting
            plot_metrics = {}

            # Add baseline heuristics
            for heuristic, metrics in linkpred_metrics.get("baseline_heuristics", {}).items():
                plot_metrics[heuristic] = metrics

            # Add embedding classifier
            if "embedding_classifier" in linkpred_metrics:
                plot_metrics["embedding_classifier"] = linkpred_metrics["embedding_classifier"]

            plot_link_prediction_performance(
                metrics_dict=plot_metrics,
                output_path=fig08_path,
            )
            output_paths.append(str(fig08_path))
        else:
            logger.info(f"Skipped (exists): {fig08_path}")
            output_paths.append(str(fig08_path))

    # ========================================================================
    # Figure 09: Top route predictions
    # ========================================================================
    logger.info("-" * 80)
    logger.info("Figure 09: Top route predictions")
    logger.info("-" * 80)

    predictions_path = tables_dir / "linkpred_top_predictions.csv"
    fig09_path = figures_dir / "fig09_top_route_predictions.png"

    if check_file_exists(predictions_path, "Top route predictions", logger):
        if not fig09_path.exists() or overwrite:
            predictions_df = pl.read_csv(predictions_path)
            plot_top_route_predictions(
                predictions_df=predictions_df,
                output_path=fig09_path,
                top_k=20,
            )
            output_paths.append(str(fig09_path))
        else:
            logger.info(f"Skipped (exists): {fig09_path}")
            output_paths.append(str(fig09_path))

    # ========================================================================
    # Summary
    # ========================================================================
    logger.info("-" * 80)
    logger.info("Summary")
    logger.info("-" * 80)

    logger.info(f"Generated {len(output_paths)} figures")
    for path in output_paths:
        logger.info(f"  - {path}")

    # Write manifest
    manifest = {
        "script": "10_make_all_figures.py",
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "config_snapshot": {
            "overwrite": overwrite,
        },
        "inputs": {
            "hub_concentration": str(hub_conc_path) if hub_conc_path.exists() else None,
            "airline_summary": str(airline_summary_path) if airline_summary_path.exists() else None,
            "linkpred_metrics": str(linkpred_metrics_path) if linkpred_metrics_path.exists() else None,
            "predictions": str(predictions_path) if predictions_path.exists() else None,
        },
        "outputs": {
            "figures": output_paths,
        },
    }

    manifest_path = log_dir / "10_make_all_figures_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Wrote manifest: {manifest_path}")

    logger.info("=" * 80)
    logger.info("WS4 Script 10: Complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

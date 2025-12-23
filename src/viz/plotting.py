"""
Lightweight plotting helpers for WS2 (centrality + community detection).

These functions prepare data or generate simple plots.
Do NOT recompute analysis here; read from results/analysis outputs only.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import polars as pl

logger = logging.getLogger(__name__)


def plot_degree_distribution(
    degree_dist_df: pl.DataFrame,
    output_path: str | Path,
    mode: str = "in",
    log_scale: bool = True,
) -> None:
    """
    Plot degree distribution (log-log for power law visualization).

    Parameters
    ----------
    degree_dist_df : pl.DataFrame
        Columns: degree, count
    output_path : str or Path
        Path to save figure
    mode : str
        'in', 'out', or 'all' (for title)
    log_scale : bool
        Whether to use log-log scale
    """
    degrees = degree_dist_df["degree"].to_list()
    counts = degree_dist_df["count"].to_list()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(degrees, counts, alpha=0.6, edgecolors="k", linewidth=0.5)
    
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    
    ax.set_xlabel("Degree", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Airport Network Degree Distribution ({mode})", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"Saved degree distribution plot: {output_path}")


def plot_community_size_distribution(
    community_summary_df: pl.DataFrame,
    output_path: str | Path,
    graph_type: str = "airport",
) -> None:
    """
    Plot community size distribution.

    Parameters
    ----------
    community_summary_df : pl.DataFrame
        Columns: community_id, size, ...
    output_path : str or Path
        Path to save figure
    graph_type : str
        'airport' or 'flight' (for title)
    """
    sizes = community_summary_df["size"].to_list()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(sizes, bins=50, alpha=0.7, edgecolor="k", linewidth=0.5)
    
    ax.set_xlabel("Community Size", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"{graph_type.capitalize()} Network Community Size Distribution", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"Saved community size distribution plot: {output_path}")


def plot_centrality_rankings(
    centrality_df: pl.DataFrame,
    output_path: str | Path,
    top_k: int = 20,
    metric: str = "pagerank",
) -> None:
    """
    Plot top-k airports by centrality metric.

    Parameters
    ----------
    centrality_df : pl.DataFrame
        Columns: code, pagerank, betweenness, ...
    output_path : str or Path
        Path to save figure
    top_k : int
        Number of top airports to show
    metric : str
        'pagerank', 'betweenness', 'out_degree', etc.
    """
    top_df = centrality_df.sort(metric, descending=True).head(top_k)
    codes = top_df["code"].to_list()
    values = top_df[metric].to_list()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(codes)), values, alpha=0.7, edgecolor="k", linewidth=0.5)
    ax.set_yticks(range(len(codes)))
    ax.set_yticklabels(codes)
    ax.invert_yaxis()
    
    ax.set_xlabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel("Airport Code", fontsize=12)
    ax.set_title(f"Top {top_k} Airports by {metric.replace('_', ' ').title()}", fontsize=14)
    ax.grid(True, alpha=0.3, axis="x")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"Saved centrality rankings plot: {output_path}")


def prepare_degree_distribution_for_plot(
    centrality_df: pl.DataFrame,
    mode: str = "in",
) -> pl.DataFrame:
    """
    Prepare degree distribution from centrality results.

    Parameters
    ----------
    centrality_df : pl.DataFrame
        Columns: in_degree, out_degree, ...
    mode : str
        'in' or 'out'

    Returns
    -------
    pl.DataFrame
        Columns: degree, count
    """
    degree_col = f"{mode}_degree" if mode in ["in", "out"] else "degree"
    
    degree_counts = (
        centrality_df
        .group_by(degree_col)
        .agg(pl.col("vertex_id").count().alias("count"))
        .sort(degree_col)
        .rename({degree_col: "degree"})
    )
    
    return degree_counts

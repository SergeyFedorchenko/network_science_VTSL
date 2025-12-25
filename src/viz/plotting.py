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


# ============================================================================
# WS4 Plotting Functions
# ============================================================================


def plot_hub_dependence_by_airline(
    hub_concentration_df: pl.DataFrame,
    output_path: str | Path,
    top_n: int = 15,
) -> None:
    """
    Plot hub concentration (top-1, top-3) for major airlines.

    Parameters
    ----------
    hub_concentration_df : pl.DataFrame
        Columns: carrier, hub_top1_pct, hub_top3_pct, total_flights
    output_path : str or Path
        Path to save figure
    top_n : int
        Number of airlines to show
    """
    logger.info(f"Plotting hub dependence for top {top_n} airlines")

    # Select top airlines by total flights
    df = hub_concentration_df.sort("total_flights", descending=True).head(top_n)

    carriers = df["carrier"].to_list()
    top1 = df["hub_top1_pct"].to_list()
    top3 = df["hub_top3_pct"].to_list()

    x = range(len(carriers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([i - width / 2 for i in x], top1, width, label="Top-1 Hub", alpha=0.8)
    ax.bar([i + width / 2 for i in x], top3, width, label="Top-3 Hubs", alpha=0.8)

    ax.set_xlabel("Carrier", fontsize=12)
    ax.set_ylabel("% of Flights", fontsize=12)
    ax.set_title("Hub Dependence by Airline", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(carriers, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.info(f"Saved hub dependence plot: {output_path}")


def plot_connectivity_vs_delay_scatter(
    airline_summary_df: pl.DataFrame,
    output_path: str | Path,
    x_metric: str = "hub_top1_pct",
    y_metric: str = "mean_arr_delay",
) -> None:
    """
    Scatter plot of network connectivity vs operational performance.

    Parameters
    ----------
    airline_summary_df : pl.DataFrame
        Merged airline metrics
    output_path : str or Path
        Path to save figure
    x_metric : str
        X-axis metric (e.g., 'hub_top1_pct')
    y_metric : str
        Y-axis metric (e.g., 'mean_arr_delay')
    """
    logger.info(f"Plotting {x_metric} vs {y_metric}")

    # Filter out null values
    df = airline_summary_df.filter(
        pl.col(x_metric).is_not_null() & pl.col(y_metric).is_not_null()
    )

    x = df[x_metric].to_list()
    y = df[y_metric].to_list()
    carriers = df["carrier"].to_list()
    sizes = df["flight_count"].to_list()

    # Normalize sizes for plotting
    max_size = max(sizes) if sizes else 1
    sizes_norm = [100 + 900 * (s / max_size) for s in sizes]

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        x, y, s=sizes_norm, alpha=0.6, edgecolors="k", linewidth=0.5
    )

    # Annotate major carriers
    for i, carrier in enumerate(carriers):
        if sizes[i] > max_size * 0.1:  # Only annotate large carriers
            ax.annotate(
                carrier,
                (x[i], y[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

    ax.set_xlabel(x_metric.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel(y_metric.replace("_", " ").title(), fontsize=12)
    ax.set_title("Network Connectivity vs Operational Performance", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.info(f"Saved connectivity vs delay scatter: {output_path}")


def plot_link_prediction_performance(
    metrics_dict: dict,
    output_path: str | Path,
) -> None:
    """
    Plot link prediction model comparison (AUC and average precision).

    Parameters
    ----------
    metrics_dict : dict
        Nested dict: {model_name: {'auc': ..., 'avg_precision': ...}}
    output_path : str or Path
        Path to save figure
    """
    logger.info("Plotting link prediction performance")

    # Handle empty metrics case (no test edges)
    models = [m for m in metrics_dict.keys() if metrics_dict[m]]
    if not models:
        logger.warning("No link prediction metrics available - creating placeholder figure")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No test data available\nfor link prediction evaluation",
                ha='center', va='center', fontsize=16, color='gray')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Saved placeholder link prediction plot: {output_path}")
        return

    auc_scores = [metrics_dict[m]["auc"] for m in models]
    ap_scores = [metrics_dict[m]["avg_precision"] for m in models]

    x = range(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([i - width / 2 for i in x], auc_scores, width, label="AUC", alpha=0.8)
    ax.bar([i + width / 2 for i in x], ap_scores, width, label="Avg Precision", alpha=0.8)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Link Prediction Performance", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylim([0, 1.0])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.info(f"Saved link prediction performance plot: {output_path}")


def plot_top_route_predictions(
    predictions_df: pl.DataFrame,
    output_path: str | Path,
    top_k: int = 20,
) -> None:
    """
    Plot top predicted routes with scores.

    Parameters
    ----------
    predictions_df : pl.DataFrame
        Columns: origin, dest, score, rank
    output_path : str or Path
        Path to save figure
    top_k : int
        Number of routes to show
    """
    logger.info(f"Plotting top {top_k} route predictions")

    # Handle empty predictions case
    if predictions_df.is_empty():
        logger.warning("No route predictions available - creating placeholder figure")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "No predictions available\nfor new routes",
                ha='center', va='center', fontsize=16, color='gray')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Saved placeholder route predictions plot: {output_path}")
        return

    df = predictions_df.head(top_k)
    routes = [f"{row['origin']}-{row['dest']}" for row in df.iter_rows(named=True)]
    scores = df["score"].to_list()

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(routes)), scores, alpha=0.7, edgecolor="k", linewidth=0.5)
    ax.set_yticks(range(len(routes)))
    ax.set_yticklabels(routes)
    ax.invert_yaxis()

    ax.set_xlabel("Prediction Score", fontsize=12)
    ax.set_ylabel("Route (Origin-Dest)", fontsize=12)
    ax.set_title(f"Top {top_k} Predicted New Routes", fontsize=14)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.info(f"Saved top route predictions plot: {output_path}")

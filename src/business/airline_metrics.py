"""
Airline-level business metrics computation.

This module computes operational, network, and disruption metrics per airline:
- Operational reliability: delays, cancellation rate, flight volume
- Network strategy: hub concentration, centralization
- Disruption cost proxy: delay + cancellation costs

Follows WS4 contract: consumes cleaned flight data and optionally WS1-WS3 outputs
to produce airline-facing business insights.
"""

import logging
from pathlib import Path
from typing import Optional

import polars as pl

logger = logging.getLogger(__name__)


def compute_airline_operational_metrics(
    cleaned_path: str | Path,
    filters: dict,
) -> pl.DataFrame:
    """
    Compute operational metrics per airline.

    Metrics:
        - flight_count: Total flights
        - mean_dep_delay: Mean departure delay (minutes)
        - mean_arr_delay: Mean arrival delay (minutes)
        - cancellation_rate: Fraction of cancelled flights
        - mean_distance: Mean distance flown

    Parameters
    ----------
    cleaned_path : str or Path
        Path to cleaned flights parquet
    filters : dict
        Data filters (year, include_cancelled, carriers)

    Returns
    -------
    pl.DataFrame
        Columns: carrier, flight_count, mean_dep_delay, mean_arr_delay,
                 cancellation_rate, mean_distance
    """
    logger.info("Computing airline operational metrics")

    lf = pl.scan_parquet(cleaned_path)

    # Apply year filter
    year = filters.get("year")
    if year:
        lf = lf.filter(pl.col("YEAR") == year)

    # Note: We include cancelled flights to compute cancellation rate
    # but exclude them from delay computations

    carriers = filters.get("carriers", "ALL")
    if carriers != "ALL":
        lf = lf.filter(pl.col("OP_UNIQUE_CARRIER").is_in(carriers))

    # Compute metrics per carrier
    df = (
        lf.group_by("OP_UNIQUE_CARRIER")
        .agg(
            [
                pl.len().alias("flight_count"),
                # Delays (exclude cancelled flights)
                pl.when(pl.col("CANCELLED") == 0)
                .then(pl.col("DEP_DELAY"))
                .mean()
                .alias("mean_dep_delay"),
                pl.when(pl.col("CANCELLED") == 0)
                .then(pl.col("ARR_DELAY"))
                .mean()
                .alias("mean_arr_delay"),
                # Cancellation rate
                (pl.col("CANCELLED").sum() / pl.len()).alias("cancellation_rate"),
                # Distance
                pl.col("DISTANCE").mean().alias("mean_distance"),
            ]
        )
        .collect()
    )

    df = df.rename({"OP_UNIQUE_CARRIER": "carrier"})
    df = df.sort("flight_count", descending=True)

    logger.info(f"Computed metrics for {len(df)} carriers")
    return df


def compute_hub_concentration(
    cleaned_path: str | Path,
    filters: dict,
    top_k: list[int] = [1, 3],
) -> pl.DataFrame:
    """
    Compute hub concentration metrics per airline.

    Hub concentration = share of flights through top-K airports (by flight count).

    Parameters
    ----------
    cleaned_path : str or Path
        Path to cleaned flights parquet
    filters : dict
        Data filters
    top_k : list[int]
        List of K values (e.g., [1, 3] for top-1 and top-3)

    Returns
    -------
    pl.DataFrame
        Columns: carrier, hub_top1_pct, hub_top3_pct, primary_hub, ...
    """
    logger.info("Computing hub concentration")

    lf = pl.scan_parquet(cleaned_path)

    year = filters.get("year")
    if year:
        lf = lf.filter(pl.col("YEAR") == year)

    include_cancelled = filters.get("include_cancelled", False)
    if not include_cancelled:
        lf = lf.filter(pl.col("CANCELLED") == 0)

    carriers = filters.get("carriers", "ALL")
    if carriers != "ALL":
        lf = lf.filter(pl.col("OP_UNIQUE_CARRIER").is_in(carriers))

    # Count flights per carrier-origin pair
    carrier_origin = (
        lf.group_by(["OP_UNIQUE_CARRIER", "ORIGIN"])
        .agg(pl.len().alias("flights"))
        .collect()
    )

    # Compute total flights per carrier
    carrier_totals = (
        carrier_origin.group_by("OP_UNIQUE_CARRIER")
        .agg(pl.col("flights").sum().alias("total_flights"))
    )

    # For each carrier, find top-K hubs
    results = []
    for row in carrier_totals.iter_rows(named=True):
        carrier = row["OP_UNIQUE_CARRIER"]
        total = row["total_flights"]

        carrier_hubs = (
            carrier_origin.filter(pl.col("OP_UNIQUE_CARRIER") == carrier)
            .sort("flights", descending=True)
        )

        record = {"carrier": carrier, "total_flights": total}

        for k in top_k:
            top_k_flights = carrier_hubs.head(k)["flights"].sum()
            pct = (top_k_flights / total * 100) if total > 0 else 0.0
            record[f"hub_top{k}_pct"] = pct

        # Primary hub
        if len(carrier_hubs) > 0:
            primary = carrier_hubs[0, "ORIGIN"]
            primary_flights = carrier_hubs[0, "flights"]
            record["primary_hub"] = primary
            record["primary_hub_flights"] = primary_flights
        else:
            record["primary_hub"] = None
            record["primary_hub_flights"] = 0

        results.append(record)

    df = pl.DataFrame(results)
    df = df.sort("total_flights", descending=True)

    logger.info(f"Computed hub concentration for {len(df)} carriers")
    return df


def compute_disruption_cost_proxy(
    cleaned_path: str | Path,
    filters: dict,
    cost_per_delay_minute: float = 75.0,
    cost_per_cancellation: float = 10000.0,
) -> pl.DataFrame:
    """
    Compute disruption cost proxy per airline.

    Cost proxy:
        - delay_cost = sum(max(ARR_DELAY, 0)) * cost_per_delay_minute
        - cancellation_cost = sum(CANCELLED) * cost_per_cancellation
        - total_cost = delay_cost + cancellation_cost

    Parameters
    ----------
    cleaned_path : str or Path
        Path to cleaned flights parquet
    filters : dict
        Data filters
    cost_per_delay_minute : float
        Cost per minute of arrival delay (dollars)
    cost_per_cancellation : float
        Cost per cancelled flight (dollars)

    Returns
    -------
    pl.DataFrame
        Columns: carrier, delay_cost, cancellation_cost, total_cost,
                 total_delay_minutes, total_cancellations
    """
    logger.info(
        f"Computing disruption cost: delay=${cost_per_delay_minute}/min, "
        f"cancel=${cost_per_cancellation}"
    )

    lf = pl.scan_parquet(cleaned_path)

    year = filters.get("year")
    if year:
        lf = lf.filter(pl.col("YEAR") == year)

    carriers = filters.get("carriers", "ALL")
    if carriers != "ALL":
        lf = lf.filter(pl.col("OP_UNIQUE_CARRIER").is_in(carriers))

    df = (
        lf.group_by("OP_UNIQUE_CARRIER")
        .agg(
            [
                # Total positive delay minutes
                pl.when(pl.col("CANCELLED") == 0)
                .then(pl.col("ARR_DELAY").clip(lower_bound=0))
                .sum()
                .alias("total_delay_minutes"),
                # Total cancellations
                pl.col("CANCELLED").sum().alias("total_cancellations"),
            ]
        )
        .collect()
    )

    df = df.with_columns(
        [
            (pl.col("total_delay_minutes") * cost_per_delay_minute).alias("delay_cost"),
            (pl.col("total_cancellations") * cost_per_cancellation).alias(
                "cancellation_cost"
            ),
        ]
    )

    df = df.with_columns(
        (pl.col("delay_cost") + pl.col("cancellation_cost")).alias("total_cost")
    )

    df = df.rename({"OP_UNIQUE_CARRIER": "carrier"})
    df = df.sort("total_cost", descending=True)

    logger.info(f"Computed disruption cost for {len(df)} carriers")
    return df


def merge_airline_metrics(
    operational_df: pl.DataFrame,
    hub_concentration_df: pl.DataFrame,
    disruption_cost_df: pl.DataFrame,
    centrality_df: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """
    Merge all airline metrics into a single table.

    Parameters
    ----------
    operational_df : pl.DataFrame
        Operational metrics
    hub_concentration_df : pl.DataFrame
        Hub concentration metrics
    disruption_cost_df : pl.DataFrame
        Disruption cost metrics
    centrality_df : pl.DataFrame, optional
        Airport centrality (if available, can join with primary hub)

    Returns
    -------
    pl.DataFrame
        Merged metrics table
    """
    logger.info("Merging airline metrics")

    # Start with operational
    df = operational_df

    # Join hub concentration
    df = df.join(
        hub_concentration_df.select(
            [
                "carrier",
                "hub_top1_pct",
                "hub_top3_pct",
                "primary_hub",
                "primary_hub_flights",
            ]
        ),
        on="carrier",
        how="left",
    )

    # Join disruption cost
    df = df.join(
        disruption_cost_df.select(
            [
                "carrier",
                "delay_cost",
                "cancellation_cost",
                "total_cost",
                "total_delay_minutes",
                "total_cancellations",
            ]
        ),
        on="carrier",
        how="left",
    )

    # Optionally join centrality for primary hub
    if centrality_df is not None:
        # Assume centrality_df has columns: code, pagerank, betweenness, ...
        hub_centrality = centrality_df.select(
            [
                pl.col("code").alias("primary_hub"),
                pl.col("pagerank").alias("primary_hub_pagerank"),
                pl.col("betweenness").alias("primary_hub_betweenness"),
            ]
        )
        df = df.join(hub_centrality, on="primary_hub", how="left")

    logger.info(f"Merged metrics for {len(df)} carriers")
    return df


def write_business_outputs(
    airline_summary: pl.DataFrame,
    hub_concentration: pl.DataFrame,
    disruption_cost: pl.DataFrame,
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, str]:
    """
    Write business analysis outputs to parquet and CSV.

    Parameters
    ----------
    airline_summary : pl.DataFrame
        Merged airline metrics
    hub_concentration : pl.DataFrame
        Hub concentration metrics
    disruption_cost : pl.DataFrame
        Disruption cost metrics
    output_dir : Path
        Root output directory (results/)
    overwrite : bool
        Whether to overwrite existing files

    Returns
    -------
    dict[str, str]
        Mapping of output names to paths
    """
    logger.info("Writing business outputs")

    business_dir = output_dir / "business"
    tables_dir = output_dir / "tables"
    business_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Parquet outputs
    parquet_outputs = {
        "airline_summary_metrics.parquet": airline_summary,
        "hub_concentration.parquet": hub_concentration,
        "disruption_cost_proxy.parquet": disruption_cost,
    }

    for filename, df in parquet_outputs.items():
        path = business_dir / filename
        if not path.exists() or overwrite:
            df.write_parquet(path)
            logger.info(f"Wrote {path}")
        else:
            logger.warning(f"Skipped (exists): {path}")
        paths[filename] = str(path)

    # CSV table for report
    csv_path = tables_dir / "airline_business_metrics.csv"
    if not csv_path.exists() or overwrite:
        airline_summary.write_csv(csv_path)
        logger.info(f"Wrote {csv_path}")
    else:
        logger.warning(f"Skipped (exists): {csv_path}")
    paths["airline_business_metrics.csv"] = str(csv_path)

    return paths

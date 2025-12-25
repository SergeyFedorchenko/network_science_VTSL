"""
Test business metrics module with toy data.

Validates:
- Operational metrics computation
- Hub concentration calculation
- Disruption cost proxy
- Metric merging
"""

import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import polars as pl

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from business.airline_metrics import (
    compute_airline_operational_metrics,
    compute_disruption_cost_proxy,
    compute_hub_concentration,
    merge_airline_metrics,
)


def test_operational_metrics():
    """Test operational metrics computation."""
    # Create toy flight data
    data = {
        "YEAR": [2025] * 10,
        "MONTH": [1] * 10,
        "OP_UNIQUE_CARRIER": ["AA", "AA", "AA", "DL", "DL", "DL", "UA", "UA", "UA", "UA"],
        "DEP_DELAY": [10, 20, -5, 15, 0, 30, 5, 10, 15, 20],
        "ARR_DELAY": [15, 25, 0, 20, 5, 35, 10, 15, 20, 25],
        "CANCELLED": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "DISTANCE": [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400],
    }
    df = pl.DataFrame(data)

    with NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        df.write_parquet(tmp.name)

        filters = {"year": 2025, "include_cancelled": False}
        metrics_df = compute_airline_operational_metrics(
            cleaned_path=tmp.name,
            filters=filters,
        )

        # Check that all carriers are present
        carriers = set(metrics_df["carrier"].to_list())
        assert carriers == {"AA", "DL", "UA"}, f"Expected AA, DL, UA; got {carriers}"

        # Check that cancellation rate is computed (including cancelled flights)
        # DL has 1 cancelled out of 3 flights = 33.3%
        dl_metrics = metrics_df.filter(pl.col("carrier") == "DL")
        assert len(dl_metrics) == 1, "Should have exactly one row for DL"
        dl_cancel_rate = dl_metrics[0, "cancellation_rate"]
        assert abs(dl_cancel_rate - 1 / 3) < 0.01, f"DL cancellation rate should be ~0.33, got {dl_cancel_rate}"

        # Check that mean delays exclude cancelled flights
        # AA: 3 flights, delays = [10, 20, -5], mean_dep = 8.33
        aa_metrics = metrics_df.filter(pl.col("carrier") == "AA")
        aa_mean_dep = aa_metrics[0, "mean_dep_delay"]
        expected_mean = (10 + 20 - 5) / 3
        assert abs(aa_mean_dep - expected_mean) < 0.1, \
            f"AA mean_dep_delay should be ~{expected_mean}, got {aa_mean_dep}"

        print("✓ Operational metrics computed correctly")


def test_hub_concentration():
    """Test hub concentration calculation."""
    # Create toy flight data with clear hubs
    data = {
        "YEAR": [2025] * 12,
        "MONTH": [1] * 12,
        "OP_UNIQUE_CARRIER": ["AA"] * 6 + ["DL"] * 6,
        "ORIGIN": ["ATL"] * 4 + ["ORD", "DFW"] + ["DEN"] * 5 + ["SFO"],
        "DEST": ["ORD", "DFW", "LAX", "SFO", "LAX", "SFO", "SFO", "LAX", "ORD", "DFW", "ATL", "ATL"],
        "CANCELLED": [0] * 12,
    }
    df = pl.DataFrame(data)

    with NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        df.write_parquet(tmp.name)

        filters = {"year": 2025, "include_cancelled": False}
        hub_df = compute_hub_concentration(
            cleaned_path=tmp.name,
            filters=filters,
            top_k=[1, 3],
        )

        # AA: 4 flights from ATL, 1 from ORD, 1 from DFW -> 4/6 = 66.7% top-1
        aa_hub = hub_df.filter(pl.col("carrier") == "AA")
        aa_top1_pct = aa_hub[0, "hub_top1_pct"]
        assert abs(aa_top1_pct - 66.67) < 1.0, f"AA top-1 hub should be ~66.67%, got {aa_top1_pct}"

        # AA primary hub should be ATL
        aa_primary = aa_hub[0, "primary_hub"]
        assert aa_primary == "ATL", f"AA primary hub should be ATL, got {aa_primary}"

        # DL: 5 flights from DEN, 1 from SFO -> 5/6 = 83.3% top-1
        dl_hub = hub_df.filter(pl.col("carrier") == "DL")
        dl_top1_pct = dl_hub[0, "hub_top1_pct"]
        assert abs(dl_top1_pct - 83.33) < 1.0, f"DL top-1 hub should be ~83.33%, got {dl_top1_pct}"

        print("✓ Hub concentration calculated correctly")


def test_disruption_cost_proxy():
    """Test disruption cost proxy computation."""
    # Create toy flight data
    data = {
        "YEAR": [2025] * 8,
        "MONTH": [1] * 8,
        "OP_UNIQUE_CARRIER": ["AA", "AA", "AA", "AA", "DL", "DL", "DL", "DL"],
        "ARR_DELAY": [10, 20, -5, 30, 15, 0, 50, -10],  # Negatives should be clipped to 0
        "CANCELLED": [0, 0, 0, 1, 0, 0, 0, 1],
    }
    df = pl.DataFrame(data)

    with NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        df.write_parquet(tmp.name)

        filters = {"year": 2025}
        cost_per_minute = 100.0
        cost_per_cancel = 5000.0

        cost_df = compute_disruption_cost_proxy(
            cleaned_path=tmp.name,
            filters=filters,
            cost_per_delay_minute=cost_per_minute,
            cost_per_cancellation=cost_per_cancel,
        )

        # AA: delays = [10, 20, 0, 0] (cancelled excluded, negatives clipped)
        # Total delay = 30 minutes, 1 cancellation
        # delay_cost = 30 * 100 = 3000, cancel_cost = 1 * 5000 = 5000, total = 8000
        aa_cost = cost_df.filter(pl.col("carrier") == "AA")
        aa_total = aa_cost[0, "total_cost"]
        expected_aa = 30 * cost_per_minute + 1 * cost_per_cancel
        assert abs(aa_total - expected_aa) < 1.0, \
            f"AA total cost should be ~{expected_aa}, got {aa_total}"

        # DL: delays = [15, 0, 50, 0] = 65 minutes, 1 cancellation
        # delay_cost = 65 * 100 = 6500, cancel_cost = 5000, total = 11500
        dl_cost = cost_df.filter(pl.col("carrier") == "DL")
        dl_total = dl_cost[0, "total_cost"]
        expected_dl = 65 * cost_per_minute + 1 * cost_per_cancel
        assert abs(dl_total - expected_dl) < 1.0, \
            f"DL total cost should be ~{expected_dl}, got {dl_total}"

        print("✓ Disruption cost proxy computed correctly")


def test_merge_airline_metrics():
    """Test metric merging."""
    # Create toy dataframes
    operational = pl.DataFrame({
        "carrier": ["AA", "DL"],
        "flight_count": [100, 150],
        "mean_dep_delay": [10.5, 8.2],
        "mean_arr_delay": [12.0, 9.5],
        "cancellation_rate": [0.02, 0.01],
        "mean_distance": [800, 1000],
    })

    hub_concentration = pl.DataFrame({
        "carrier": ["AA", "DL"],
        "hub_top1_pct": [60.0, 75.0],
        "hub_top3_pct": [85.0, 90.0],
        "primary_hub": ["ATL", "DEN"],
        "primary_hub_flights": [60, 112],
        "total_flights": [100, 150],
    })

    disruption_cost = pl.DataFrame({
        "carrier": ["AA", "DL"],
        "delay_cost": [50000.0, 30000.0],
        "cancellation_cost": [20000.0, 15000.0],
        "total_cost": [70000.0, 45000.0],
        "total_delay_minutes": [500, 300],
        "total_cancellations": [2, 1],
    })

    merged = merge_airline_metrics(
        operational_df=operational,
        hub_concentration_df=hub_concentration,
        disruption_cost_df=disruption_cost,
        centrality_df=None,
    )

    # Check that all columns are present
    expected_cols = {
        "carrier", "flight_count", "mean_dep_delay", "mean_arr_delay",
        "cancellation_rate", "mean_distance", "hub_top1_pct", "hub_top3_pct",
        "primary_hub", "primary_hub_flights", "delay_cost", "cancellation_cost",
        "total_cost", "total_delay_minutes", "total_cancellations",
    }
    actual_cols = set(merged.columns)
    assert expected_cols.issubset(actual_cols), \
        f"Missing columns: {expected_cols - actual_cols}"

    # Check that data is correctly merged
    aa_row = merged.filter(pl.col("carrier") == "AA")
    assert aa_row[0, "flight_count"] == 100
    assert aa_row[0, "hub_top1_pct"] == 60.0
    assert aa_row[0, "total_cost"] == 70000.0

    print("✓ Airline metrics merged correctly")


if __name__ == "__main__":
    test_operational_metrics()
    test_hub_concentration()
    test_disruption_cost_proxy()
    test_merge_airline_metrics()
    print("\n✅ All business metrics tests passed!")

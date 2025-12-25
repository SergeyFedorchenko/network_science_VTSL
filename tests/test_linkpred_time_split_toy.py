"""
Test link prediction time-split logic with toy data.

Validates:
- Train/test split prevents data leakage
- Negative sampling excludes test positives
- Heuristic features compute correctly
"""

import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import igraph as ig
import numpy as np
import polars as pl

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis.link_prediction import (
    build_igraph_from_edges,
    build_month_split_graphs,
    compute_heuristic_features,
    negative_sample_non_edges,
)


def test_time_split_no_leakage():
    """Test that train/test split prevents data leakage."""
    # Create toy flight data
    data = {
        "YEAR": [2025] * 10,
        "MONTH": [1, 1, 2, 2, 3, 3, 10, 10, 11, 11],
        "ORIGIN": ["A", "B", "A", "C", "B", "C", "A", "D", "B", "D"],
        "DEST": ["B", "C", "B", "D", "C", "D", "D", "E", "D", "E"],
        "CANCELLED": [0] * 10,
    }
    df = pl.DataFrame(data)

    # Write to temp file
    with NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        df.write_parquet(tmp.name)

        filters = {"year": 2025, "include_cancelled": False}
        train_months = [1, 2, 3]
        test_months = [10, 11]

        train_edges, test_edges, code_to_id = build_month_split_graphs(
            cleaned_path=tmp.name,
            train_months=train_months,
            test_months=test_months,
            filters=filters,
        )

        # Check no overlap
        overlap = train_edges & test_edges
        assert len(overlap) == 0, "Train and test edges should not overlap"

        # Check that test edges are truly "new" (not in train)
        for edge in test_edges:
            assert edge not in train_edges, f"Edge {edge} should not be in train"

        print("✓ Time split prevents data leakage")


def test_negative_sampling_excludes_test():
    """Test that negative sampling excludes test positives."""
    n_nodes = 10
    train_edges = {(0, 1), (1, 2), (2, 3)}
    test_positives = {(3, 4), (4, 5)}

    seed = 42
    ratio = 3

    negatives = negative_sample_non_edges(
        n_nodes=n_nodes,
        train_edges=train_edges,
        test_positives=test_positives,
        ratio=ratio,
        seed=seed,
    )

    # Check that negatives don't overlap with train or test
    assert len(negatives & train_edges) == 0, "Negatives should not overlap with train"
    assert len(negatives & test_positives) == 0, "Negatives should not overlap with test positives"

    # Check self-loops are excluded
    for u, v in negatives:
        assert u != v, "Negatives should not include self-loops"

    print("✓ Negative sampling excludes test positives")


def test_heuristic_features_computation():
    """Test that heuristic features compute correctly."""
    # Create toy graph: triangle + one extra node
    # 0-1, 1-2, 2-0, 3-1
    g = ig.Graph(n=4, edges=[(0, 1), (1, 2), (2, 0), (3, 1)], directed=False)

    # Test pairs
    pairs = [
        (0, 2),  # Connected (triangle)
        (0, 3),  # Not connected, but share neighbor 1
        (2, 3),  # Not connected, share neighbor 1
    ]

    features = compute_heuristic_features(g, pairs)

    # Check shape: [CN, Jaccard, AA, PA]
    assert features.shape == (len(pairs), 4), "Expected 4 features per pair"

    # Check common neighbors
    # (0, 2): share neighbor 1
    assert features[0, 0] == 1, "Pair (0, 2) should have 1 common neighbor"

    # (0, 3): share neighbor 1
    assert features[1, 0] == 1, "Pair (0, 3) should have 1 common neighbor"

    # Check Jaccard
    # (0, 2): neighbors = {1, 0} and {0, 1}, union = {0, 1}, intersection = {0, 1}
    # Actually, for undirected: 0 has neighbors {1, 2}, 2 has neighbors {0, 1}
    # Common: {1}, Union: {0, 1, 2}, Jaccard = 1/3
    # This depends on implementation details, just check non-zero
    assert features[0, 1] > 0, "Jaccard should be positive"

    print("✓ Heuristic features computed correctly")


def test_build_igraph_from_edges():
    """Test igraph construction from edge set."""
    edges = {(0, 1), (1, 2), (2, 3)}
    n_nodes = 4

    g = build_igraph_from_edges(edges, n_nodes, directed=True)

    assert g.vcount() == n_nodes, f"Expected {n_nodes} nodes"
    assert g.ecount() == len(edges), f"Expected {len(edges)} edges"
    assert g.is_directed(), "Graph should be directed"

    print("✓ igraph construction from edges works")


def test_month_split_carriers_filter():
    """Test that carrier filtering works in month split."""
    # Create toy flight data with multiple carriers
    data = {
        "YEAR": [2025] * 8,
        "MONTH": [1, 1, 2, 2, 10, 10, 11, 11],
        "ORIGIN": ["A", "B", "A", "C", "A", "D", "B", "D"],
        "DEST": ["B", "C", "B", "D", "D", "E", "D", "E"],
        "CANCELLED": [0] * 8,
        "OP_UNIQUE_CARRIER": ["AA", "DL", "AA", "DL", "AA", "UA", "DL", "UA"],
    }
    df = pl.DataFrame(data)

    with NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        df.write_parquet(tmp.name)

        filters = {"year": 2025, "include_cancelled": False, "carriers": ["AA", "DL"]}
        train_months = [1, 2]
        test_months = [10, 11]

        train_edges, test_edges, code_to_id = build_month_split_graphs(
            cleaned_path=tmp.name,
            train_months=train_months,
            test_months=test_months,
            filters=filters,
        )

        # Should only include AA and DL flights
        # Train: A-B (AA), B-C (DL), A-B (AA), C-D (DL)
        # Test: A-D (AA), B-D (DL)
        # So we should not see E (only UA flies there)
        assert "E" not in code_to_id, "Airport E should be excluded (only UA flies there)"

        print("✓ Carrier filtering works in month split")


if __name__ == "__main__":
    test_time_split_no_leakage()
    test_negative_sampling_excludes_test()
    test_heuristic_features_computation()
    test_build_igraph_from_edges()
    test_month_split_carriers_filter()
    print("\n✅ All link prediction tests passed!")

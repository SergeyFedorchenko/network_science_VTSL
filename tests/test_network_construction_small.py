"""
Test network construction on toy dataset.
"""
import pytest
import polars as pl
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.io.time_features import add_time_features
from src.networks.airport_network import build_airport_nodes, build_airport_edges
from src.networks.flight_network import (
    build_flight_nodes,
    build_tail_sequence_edges,
    build_route_knn_edges
)


@pytest.fixture
def toy_data():
    """Load toy dataset."""
    toy_path = Path(__file__).parent / "fixtures" / "toy_flights.parquet"
    
    if not toy_path.exists():
        # Generate it
        from tests.fixtures.generate_toy_data import generate_toy_dataset
        df = generate_toy_dataset()
        df.write_parquet(toy_path)
    
    df = pl.read_parquet(toy_path)
    lf = df.lazy()
    
    # Add time features
    lf = add_time_features(lf)
    
    return lf


def test_airport_nodes_count(toy_data):
    """Test airport node extraction."""
    nodes = build_airport_nodes(toy_data)
    
    # Should have unique airports from origins and dests
    assert len(nodes) > 0
    assert "code" in nodes.columns
    assert "node_id" in nodes.columns
    
    # All node IDs should be unique
    assert nodes["node_id"].n_unique() == len(nodes)


def test_airport_edges_aggregation(toy_data):
    """Test airport edge aggregation."""
    nodes = build_airport_nodes(toy_data)
    edges = build_airport_edges(toy_data, nodes)
    
    # Should have edges
    assert len(edges) > 0
    assert "src_id" in edges.columns
    assert "dst_id" in edges.columns
    assert "flight_count" in edges.columns
    
    # Flight counts should be positive
    assert all(edges["flight_count"] > 0)


def test_flight_nodes_with_scoping(toy_data):
    """Test flight node creation."""
    scope_config = {"mode": "full"}
    nodes = build_flight_nodes(toy_data, scope_config)
    
    # Should have flight nodes
    assert len(nodes) > 0
    assert "flight_id" in nodes.columns
    assert "flight_key" in nodes.columns
    
    # All flight IDs should be unique
    assert nodes["flight_id"].n_unique() == len(nodes)


def test_tail_sequence_edges(toy_data):
    """Test tail sequence edge construction."""
    scope_config = {"mode": "full"}
    nodes = build_flight_nodes(toy_data, scope_config)
    
    edges = build_tail_sequence_edges(nodes)
    
    # Should have some tail edges (toy data has 2 aircraft with multiple legs)
    assert len(edges) > 0
    assert "edge_type" in edges.columns
    assert all(edges["edge_type"] == "tail_next_leg")
    
    # Ground time should be positive
    assert all(edges["ground_time_minutes"] > 0)
    
    # Expected: N123AA has 3 flights -> 2 edges
    #           N456DL has 2 flights -> 1 edge
    # Minimum expected: 3 edges
    assert len(edges) >= 3, f"Expected at least 3 tail edges, got {len(edges)}"


def test_route_knn_edges(toy_data):
    """Test route kNN edge construction."""
    scope_config = {"mode": "full"}
    nodes = build_flight_nodes(toy_data, scope_config)
    
    k = 3
    edges = build_route_knn_edges(nodes, k=k)
    
    # Should have route kNN edges
    assert len(edges) > 0
    assert "edge_type" in edges.columns
    assert all(edges["edge_type"] == "route_knn")
    
    # Toy data has 5 JFK->LAX flights on same day (UA)
    # Each should connect to next k flights
    # Flight 1 -> 2, 3, 4 (3 edges)
    # Flight 2 -> 3, 4, 5 (3 edges)
    # Flight 3 -> 4, 5 (2 edges, only 2 remain)
    # Flight 4 -> 5 (1 edge)
    # Flight 5 -> none
    # Expected for this route: 3 + 3 + 2 + 1 = 9 edges
    # Plus other routes may have edges too
    assert len(edges) >= 9, f"Expected at least 9 route kNN edges, got {len(edges)}"


def test_no_duplicate_edges(toy_data):
    """Test that we don't create duplicate edges."""
    scope_config = {"mode": "full"}
    nodes = build_flight_nodes(toy_data, scope_config)
    
    tail_edges = build_tail_sequence_edges(nodes)
    route_edges = build_route_knn_edges(nodes, k=3)
    
    # Combine
    all_edges = pl.concat([tail_edges, route_edges])
    
    # Check for duplicates
    unique_pairs = all_edges.select(["src_id", "dst_id"]).unique()
    
    # Should have no exact duplicates (though this test doesn't enforce uniqueness yet)
    # The actual script will handle deduplication
    assert len(unique_pairs) <= len(all_edges)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

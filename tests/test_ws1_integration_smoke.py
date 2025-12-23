"""
WS1 integration smoke test for WS2 pipeline.

Tests that WS2 can load WS1 parquet outputs and produce valid results.
Uses toy fixture data if available, or creates minimal test data.
"""

import sys
from pathlib import Path

import igraph as ig
import polars as pl
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis.centrality import (
    compute_airport_centrality,
    load_airport_graph_from_parquet,
)
from analysis.community import (
    run_leiden_cpm,
    summarize_communities_airport,
)


@pytest.fixture
def toy_airport_network(tmp_path):
    """
    Create minimal toy airport network parquet files for testing.
    
    Returns paths to nodes and edges parquet files.
    """
    # Create toy nodes
    nodes_df = pl.DataFrame({
        "vertex_id": [0, 1, 2, 3],
        "code": ["LAX", "SFO", "ORD", "JFK"],
        "ORIGIN_CITY_NAME": ["Los Angeles", "San Francisco", "Chicago", "New York"],
        "ORIGIN_STATE_NM": ["California", "California", "Illinois", "New York"],
    })
    
    # Create toy edges (4 airports, 5 routes)
    edges_df = pl.DataFrame({
        "src_id": [0, 0, 1, 2, 3],
        "dst_id": [1, 2, 2, 3, 0],
        "flight_count": [100.0, 50.0, 75.0, 60.0, 40.0],
        "mean_dep_delay": [5.2, 3.1, 7.5, 2.0, 10.3],
        "mean_arr_delay": [4.8, 2.9, 8.1, 1.5, 9.7],
        "cancel_rate": [0.01, 0.005, 0.02, 0.008, 0.015],
        "mean_distance": [337.0, 1744.0, 1846.0, 740.0, 2475.0],
    })
    
    # Write to tmp directory
    nodes_path = tmp_path / "airport_nodes.parquet"
    edges_path = tmp_path / "airport_edges.parquet"
    
    nodes_df.write_parquet(nodes_path)
    edges_df.write_parquet(edges_path)
    
    return nodes_path, edges_path


def test_load_airport_graph_from_toy_parquet(toy_airport_network):
    """Test loading airport graph from toy parquet files."""
    nodes_path, edges_path = toy_airport_network
    
    # Load graph
    g = load_airport_graph_from_parquet(
        nodes_path=nodes_path,
        edges_path=edges_path,
        directed=True,
        weight_col="flight_count",
    )
    
    # Validate graph structure
    assert g.vcount() == 4
    assert g.ecount() == 5
    assert g.is_directed()
    
    # Check vertex attribute
    assert "code" in g.vs.attributes()
    codes = g.vs["code"]
    assert "LAX" in codes
    assert "SFO" in codes
    
    # Check edge weights
    assert "weight" in g.es.attributes()
    weights = g.es["weight"]
    assert len(weights) == 5
    assert max(weights) == 100.0


def test_compute_centrality_on_toy_graph(toy_airport_network):
    """Test centrality computation on toy airport graph."""
    nodes_path, edges_path = toy_airport_network
    
    g = load_airport_graph_from_parquet(
        nodes_path=nodes_path,
        edges_path=edges_path,
        directed=True,
        weight_col="flight_count",
    )
    
    # Compute centrality
    df = compute_airport_centrality(g, weight_col="weight", config={})
    
    # Validate output
    assert len(df) == 4
    assert "vertex_id" in df.columns
    assert "code" in df.columns
    assert "pagerank" in df.columns
    assert "betweenness" in df.columns
    
    # Check that codes match
    codes = df["code"].to_list()
    assert set(codes) == {"LAX", "SFO", "ORD", "JFK"}


def test_leiden_on_toy_graph(toy_airport_network):
    """Test Leiden community detection on toy airport graph."""
    nodes_path, edges_path = toy_airport_network
    
    g = load_airport_graph_from_parquet(
        nodes_path=nodes_path,
        edges_path=edges_path,
        directed=True,
        weight_col="flight_count",
    )
    
    # Run Leiden
    membership, quality = run_leiden_cpm(g, resolution=0.01, seed=42, weights="weight")
    
    # Validate output
    assert len(membership) == 4
    assert isinstance(quality, (int, float))
    
    # Check that membership is valid
    n_communities = len(set(membership))
    assert n_communities >= 1
    assert n_communities <= 4


def test_summarize_communities_toy(toy_airport_network):
    """Test community summarization on toy airport graph."""
    nodes_path, edges_path = toy_airport_network
    
    g = load_airport_graph_from_parquet(
        nodes_path=nodes_path,
        edges_path=edges_path,
        directed=True,
        weight_col="flight_count",
    )
    
    # Run Leiden
    membership, quality = run_leiden_cpm(g, resolution=0.01, seed=42, weights="weight")
    
    # Create membership DataFrame
    membership_df = pl.DataFrame({
        "vertex_id": list(range(4)),
        "community_id": membership,
    })
    
    # Load nodes
    nodes_df = pl.read_parquet(nodes_path)
    
    # Summarize communities
    summary_df = summarize_communities_airport(
        nodes_df=nodes_df,
        membership_df=membership_df,
        centrality_df=None,
    )
    
    # Validate output
    assert len(summary_df) > 0
    assert "community_id" in summary_df.columns
    assert "size" in summary_df.columns
    
    # Check that total size equals number of nodes
    assert summary_df["size"].sum() == 4


def test_ws2_pipeline_end_to_end_smoke(toy_airport_network):
    """
    End-to-end smoke test: load graph, compute centrality, run communities, summarize.
    
    This simulates the minimal WS2 workflow consuming WS1 outputs.
    """
    nodes_path, edges_path = toy_airport_network
    
    # Step 1: Load graph
    g = load_airport_graph_from_parquet(
        nodes_path=nodes_path,
        edges_path=edges_path,
        directed=True,
        weight_col="flight_count",
    )
    
    assert g.vcount() == 4
    
    # Step 2: Compute centrality
    centrality_df = compute_airport_centrality(g, weight_col="weight", config={})
    assert len(centrality_df) == 4
    
    # Step 3: Run Leiden
    membership, quality = run_leiden_cpm(g, resolution=0.01, seed=42, weights="weight")
    assert len(membership) == 4
    
    # Step 4: Summarize communities
    membership_df = pl.DataFrame({
        "vertex_id": list(range(4)),
        "community_id": membership,
    })
    nodes_df = pl.read_parquet(nodes_path)
    summary_df = summarize_communities_airport(
        nodes_df=nodes_df,
        membership_df=membership_df,
        centrality_df=centrality_df,
    )
    
    assert len(summary_df) > 0
    assert "community_id" in summary_df.columns
    
    # If everything passes, WS2 can consume WS1 outputs successfully
    print("✓ WS2 pipeline smoke test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

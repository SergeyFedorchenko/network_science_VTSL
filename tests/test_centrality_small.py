"""
Test centrality computation on small toy graphs.

Validates that centrality functions produce expected outputs with correct schemas.
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
    compute_degree_distribution,
    compute_graph_summary,
)


def test_compute_centrality_small_directed():
    """Test centrality computation on a small directed graph."""
    # Create a simple directed graph: 0 -> 1 -> 2 -> 0 (triangle)
    g = ig.Graph(directed=True)
    g.add_vertices(3)
    g.add_edges([(0, 1), (1, 2), (2, 0)])
    g.vs["code"] = ["A", "B", "C"]
    g.es["weight"] = [1.0, 2.0, 3.0]
    
    # Compute centrality
    df = compute_airport_centrality(g, weight_col="weight", config={})
    
    # Check schema
    assert "vertex_id" in df.columns
    assert "code" in df.columns
    assert "in_degree" in df.columns
    assert "out_degree" in df.columns
    assert "in_strength" in df.columns
    assert "out_strength" in df.columns
    assert "pagerank" in df.columns
    assert "betweenness" in df.columns
    
    # Check row count
    assert len(df) == 3
    
    # Check degrees (each vertex has in=1, out=1 in this triangle)
    assert df["in_degree"].to_list() == [1, 1, 1]
    assert df["out_degree"].to_list() == [1, 1, 1]
    
    # Check strengths (weighted degrees)
    assert df["in_strength"].to_list() == [3.0, 1.0, 2.0]  # A receives 3, B receives 1, C receives 2
    assert df["out_strength"].to_list() == [1.0, 2.0, 3.0]  # A sends 1, B sends 2, C sends 3
    
    # Check PageRank exists and sums to ~1
    assert abs(sum(df["pagerank"]) - 1.0) < 0.01


def test_compute_centrality_small_undirected():
    """Test centrality computation on a small undirected graph."""
    # Create a simple undirected graph: 0 - 1 - 2
    g = ig.Graph(directed=False)
    g.add_vertices(3)
    g.add_edges([(0, 1), (1, 2)])
    g.vs["code"] = ["X", "Y", "Z"]
    
    # Compute centrality (no weights)
    df = compute_airport_centrality(g, weight_col=None, config={})
    
    # Check schema
    assert len(df) == 3
    
    # Check degrees
    degrees = df["in_degree"].to_list()
    assert degrees == [1, 2, 1]  # X and Z have degree 1, Y has degree 2
    
    # Check that in_degree == out_degree for undirected
    assert df["in_degree"].to_list() == df["out_degree"].to_list()


def test_degree_distribution():
    """Test degree distribution computation."""
    # Create a graph with known degree distribution
    g = ig.Graph(directed=True)
    g.add_vertices(5)
    # 0 has out-degree 2, 1 and 2 have out-degree 1, 3 and 4 have out-degree 0
    g.add_edges([(0, 1), (0, 2), (1, 3), (2, 4)])
    
    # In-degree distribution
    degree_dist = compute_degree_distribution(g, mode="in")
    
    # Check schema
    assert "degree" in degree_dist.columns
    assert "count" in degree_dist.columns
    
    # Check counts
    # Vertices 1,2,3,4 have in-degree 1 (count=4), vertex 0 has in-degree 0 (count=1)
    degree_dict = dict(zip(degree_dist["degree"].to_list(), degree_dist["count"].to_list()))
    assert degree_dict[0] == 1
    assert degree_dict[1] == 4


def test_graph_summary_directed():
    """Test graph summary computation for directed graph."""
    g = ig.Graph(directed=True)
    g.add_vertices(5)
    g.add_edges([(0, 1), (1, 2), (3, 4)])  # Two components
    
    summary = compute_graph_summary(g)
    
    assert summary["n_vertices"] == 5
    assert summary["n_edges"] == 3
    assert summary["directed"] is True
    assert summary["n_components_weak"] == 2
    assert summary["lcc_size_weak"] == 3  # Component {0,1,2}


def test_graph_summary_undirected():
    """Test graph summary computation for undirected graph."""
    g = ig.Graph(directed=False)
    g.add_vertices(4)
    g.add_edges([(0, 1), (1, 2), (2, 0)])  # One connected triangle, vertex 3 isolated
    
    summary = compute_graph_summary(g)
    
    assert summary["n_vertices"] == 4
    assert summary["n_edges"] == 3
    assert summary["directed"] is False
    assert summary["n_components"] == 2
    assert summary["lcc_size"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

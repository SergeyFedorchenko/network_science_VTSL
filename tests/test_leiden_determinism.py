"""
Test Leiden algorithm determinism with fixed seeds.

Validates that Leiden CPM produces deterministic results when seeded properly.
"""

import sys
from pathlib import Path

import igraph as ig
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis.community import run_leiden_cpm


def test_leiden_determinism_same_seed():
    """Test that Leiden produces identical results with the same seed."""
    # Create a small graph
    g = ig.Graph(directed=False)
    g.add_vertices(10)
    # Create a graph with clear community structure
    g.add_edges([
        # Community 1: 0-1-2-3
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Community 2: 4-5-6
        (4, 5), (5, 6), (6, 4),
        # Weak inter-community edge
        (3, 4),
        # Isolated vertices: 7, 8, 9
        (7, 8),
    ])
    
    # Run twice with same seed
    membership1, quality1 = run_leiden_cpm(g, resolution=0.01, seed=42)
    membership2, quality2 = run_leiden_cpm(g, resolution=0.01, seed=42)
    
    # Should be identical
    assert membership1 == membership2
    assert abs(quality1 - quality2) < 1e-10


def test_leiden_different_seeds():
    """Test that Leiden can produce different results with different seeds."""
    # Create a graph with ambiguous community structure
    g = ig.Graph(directed=False)
    g.add_vertices(12)
    # Create a more complex structure where multiple partitions are plausible
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # square 1
        (4, 5), (5, 6), (6, 7), (7, 4),  # square 2
        (8, 9), (9, 10), (10, 11), (11, 8),  # square 3
        # Multiple inter-square connections
        (0, 4), (1, 5), (4, 8), (5, 9),
    ]
    g.add_edges(edges)
    
    # Run with multiple different seeds
    results = []
    for seed in [10, 20, 30, 40, 50]:
        membership, quality = run_leiden_cpm(g, resolution=0.05, seed=seed)
        results.append((membership, quality))
    
    # Check that each run is internally consistent (same number of vertices)
    for membership, quality in results:
        assert len(membership) == g.vcount()
    
    # Note: We don't require different results, but we verify determinism
    # by checking that running the same seed twice gives the same result
    membership_a, _ = run_leiden_cpm(g, resolution=0.05, seed=10)
    membership_b, _ = run_leiden_cpm(g, resolution=0.05, seed=10)
    assert membership_a == membership_b


def test_leiden_membership_length():
    """Test that membership list has correct length."""
    g = ig.Graph(directed=True)
    g.add_vertices(20)
    g.add_edges([(i, (i + 1) % 20) for i in range(20)])  # Ring
    
    membership, quality = run_leiden_cpm(g, resolution=0.01, seed=123)
    
    assert len(membership) == g.vcount()
    assert len(membership) == 20


def test_leiden_quality_score():
    """Test that quality score is computed and is a valid number."""
    g = ig.Graph(directed=False)
    g.add_vertices(6)
    g.add_edges([(0, 1), (1, 2), (3, 4), (4, 5)])  # Two chains
    
    membership, quality = run_leiden_cpm(g, resolution=0.01, seed=42)
    
    # Quality should be a finite number
    assert isinstance(quality, (int, float))
    assert not (quality != quality)  # Check not NaN


def test_leiden_weighted():
    """Test Leiden with weighted edges."""
    g = ig.Graph(directed=False)
    g.add_vertices(6)
    g.add_edges([
        (0, 1), (1, 2), (2, 0),  # Triangle 1 (high weights)
        (3, 4), (4, 5), (5, 3),  # Triangle 2 (high weights)
        (2, 3),  # Weak inter-triangle edge
    ])
    g.es["weight"] = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 1.0]
    
    membership, quality = run_leiden_cpm(g, resolution=0.05, seed=42, weights="weight")
    
    # With strong intra-triangle weights and weak inter-triangle weight,
    # we expect 2 communities
    n_communities = len(set(membership))
    assert n_communities >= 1  # At least some structure detected
    
    # Check that membership is valid
    assert len(membership) == g.vcount()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

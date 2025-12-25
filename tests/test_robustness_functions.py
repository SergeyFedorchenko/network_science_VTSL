"""
Test robustness analysis functions.

Tests the core robustness functions from the robustness script.
"""
import pytest
import igraph as ig
import numpy as np
import sys
import logging
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

# Import functions from the robustness script
from importlib.util import spec_from_file_location, module_from_spec

# Load the robustness module
script_path = Path(__file__).parent.parent / "scripts" / "06_run_robustness.py"
spec = spec_from_file_location("robustness", script_path)
assert spec is not None, f"Could not load spec from {script_path}"
assert spec.loader is not None, f"Spec has no loader"
robustness = module_from_spec(spec)
spec.loader.exec_module(robustness)


@pytest.fixture
def simple_graph():
    """Create a simple test graph."""
    # Star graph: center node (0) connected to 5 peripheral nodes
    g = ig.Graph(n=6, directed=False)
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
    g.add_edges(edges)
    return g


@pytest.fixture
def directed_graph():
    """Create a simple directed graph."""
    g = ig.Graph(n=5, directed=True)
    # Linear chain: 0 -> 1 -> 2 -> 3 -> 4
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    g.add_edges(edges)
    return g


@pytest.fixture
def weighted_graph():
    """Create a weighted graph."""
    g = ig.Graph(n=4, directed=False)
    edges = [(0, 1), (0, 2), (1, 2), (2, 3)]
    g.add_edges(edges)
    g.es["weight"] = [1.0, 2.0, 3.0, 4.0]
    return g


class TestLargestComponentSize:
    """Tests for largest_component_size function."""
    
    def test_connected_graph(self, simple_graph):
        """Test LCC size for a connected graph."""
        size = robustness.largest_component_size(simple_graph)
        assert size == 6
    
    def test_disconnected_graph(self):
        """Test LCC size for a disconnected graph."""
        g = ig.Graph(n=6, directed=False)
        # Two components: (0,1,2) and (3,4,5)
        g.add_edges([(0, 1), (1, 2), (3, 4), (4, 5)])
        size = robustness.largest_component_size(g)
        assert size == 3
    
    def test_empty_graph(self):
        """Test LCC size for an empty graph."""
        g = ig.Graph(n=0)
        size = robustness.largest_component_size(g)
        assert size == 0
    
    def test_single_node(self):
        """Test LCC size for a single node graph."""
        g = ig.Graph(n=1)
        size = robustness.largest_component_size(g)
        assert size == 1


class TestLccFraction:
    """Tests for lcc_fraction function."""
    
    def test_full_connectivity(self, simple_graph):
        """Test LCC fraction for a fully connected graph."""
        frac = robustness.lcc_fraction(simple_graph, n_original=6)
        assert frac == 1.0
    
    def test_partial_connectivity(self):
        """Test LCC fraction for a disconnected graph."""
        g = ig.Graph(n=6, directed=False)
        g.add_edges([(0, 1), (1, 2)])  # Only 3 nodes connected
        frac = robustness.lcc_fraction(g, n_original=6)
        assert frac == 0.5
    
    def test_zero_original(self):
        """Test handling of zero original nodes."""
        g = ig.Graph(n=5)
        frac = robustness.lcc_fraction(g, n_original=0)
        assert frac == 0.0


class TestRankNodesByStrategy:
    """Tests for rank_nodes_by_strategy function."""
    
    def test_degree_strategy(self, simple_graph):
        """Test ranking by degree."""
        logger = logging.getLogger(__name__)
        
        ranking = robustness.rank_nodes_by_strategy(simple_graph, "degree", logger)
        
        # Node 0 (center) should be first (highest degree)
        assert ranking[0] == 0
    
    def test_random_strategy_determinism(self, simple_graph):
        """Test that random strategy is deterministic with same seed."""
        logger = logging.getLogger(__name__)
        
        rank1 = robustness.rank_nodes_by_strategy(simple_graph, "random", logger, seed=42)
        rank2 = robustness.rank_nodes_by_strategy(simple_graph, "random", logger, seed=42)
        
        assert rank1 == rank2
    
    def test_random_strategy_different_seeds(self, simple_graph):
        """Test that different seeds give different orders."""
        logger = logging.getLogger(__name__)
        
        rank1 = robustness.rank_nodes_by_strategy(simple_graph, "random", logger, seed=42)
        rank2 = robustness.rank_nodes_by_strategy(simple_graph, "random", logger, seed=123)
        
        # Very unlikely to be the same
        assert rank1 != rank2
    
    def test_strength_strategy_with_weights(self, weighted_graph):
        """Test ranking by strength (weighted degree)."""
        logger = logging.getLogger(__name__)
        
        ranking = robustness.rank_nodes_by_strategy(weighted_graph, "strength", logger)
        
        # All nodes should be in the ranking
        assert len(ranking) == 4
        assert set(ranking) == {0, 1, 2, 3}
    
    def test_betweenness_strategy(self, simple_graph):
        """Test ranking by betweenness."""
        logger = logging.getLogger(__name__)
        
        ranking = robustness.rank_nodes_by_strategy(simple_graph, "betweenness", logger)
        
        # Node 0 (center) has highest betweenness in star graph
        assert ranking[0] == 0


class TestSimulateRandomRemoval:
    """Tests for random removal simulation."""
    
    def test_returns_arrays(self, simple_graph):
        """Test that simulate_random_removal returns correct structure."""
        logger = logging.getLogger(__name__)
        
        x_removed, mean_lcc, std_lcc = robustness.simulate_random_removal(
            simple_graph,
            n_runs=5,
            seed=42,
            connectivity_mode="weak",
            logger=logger,
            sample_points=10,
        )
        
        # Should return numpy arrays
        assert isinstance(x_removed, np.ndarray)
        assert isinstance(mean_lcc, np.ndarray)
        assert isinstance(std_lcc, np.ndarray)
        
        # LCC should start at 1.0 (full connectivity)
        assert mean_lcc[0] == 1.0
    
    def test_lcc_decreases_with_removal(self, simple_graph):
        """Test that LCC decreases as nodes are removed."""
        logger = logging.getLogger(__name__)
        
        x_removed, mean_lcc, std_lcc = robustness.simulate_random_removal(
            simple_graph,
            n_runs=10,
            seed=42,
            connectivity_mode="weak",
            logger=logger,
            sample_points=10,
        )
        
        # LCC should decrease (first value >= last value)
        assert mean_lcc[0] >= mean_lcc[-1]


class TestSimulateTargetedRemoval:
    """Tests for targeted removal simulation."""
    
    def test_returns_arrays(self, simple_graph):
        """Test that simulate_targeted_removal returns correct structure."""
        logger = logging.getLogger(__name__)
        
        # Get removal order first
        removal_order = robustness.rank_nodes_by_strategy(simple_graph, "degree", logger)
        
        x_removed, lcc_frac = robustness.simulate_targeted_removal(
            simple_graph,
            removal_order=removal_order,
            connectivity_mode="weak",
            logger=logger,
        )
        
        # Should return numpy arrays
        assert isinstance(x_removed, np.ndarray)
        assert isinstance(lcc_frac, np.ndarray)
        
        # LCC should start at 1.0
        assert lcc_frac[0] == 1.0
    
    def test_targeted_affects_star_center(self, simple_graph):
        """Test that removing star center (highest degree) causes fragmentation."""
        logger = logging.getLogger(__name__)
        
        # Get removal order by degree (center node first)
        removal_order = robustness.rank_nodes_by_strategy(simple_graph, "degree", logger)
        
        x_removed, lcc_frac = robustness.simulate_targeted_removal(
            simple_graph,
            removal_order=removal_order,
            connectivity_mode="weak",
            logger=logger,
        )
        
        # After removing center (node 0), LCC should drop significantly
        # At step 1, removing the hub should leave only isolated nodes
        assert lcc_frac[1] < lcc_frac[0]

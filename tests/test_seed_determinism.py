"""
Test seed determinism.

Ensures that running the same code with the same seed produces identical results.
"""
import pytest
import polars as pl
import sys
from pathlib import Path
import hashlib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seeds import set_global_seed
from src.io.time_features import add_time_features
from src.networks.flight_network import build_flight_nodes, build_tail_sequence_edges


@pytest.fixture
def toy_data():
    """Load toy dataset."""
    toy_path = Path(__file__).parent / "fixtures" / "toy_flights.parquet"
    
    if not toy_path.exists():
        from tests.fixtures.generate_toy_data import generate_toy_dataset
        df = generate_toy_dataset()
        df.write_parquet(toy_path)
    
    return pl.scan_parquet(toy_path)


def hash_dataframe(df: pl.DataFrame) -> str:
    """
    Compute hash of a DataFrame for comparison.
    
    Args:
        df: DataFrame to hash
        
    Returns:
        Hex string hash
    """
    # Convert to dict and hash
    data_str = str(df.to_dicts())
    return hashlib.sha256(data_str.encode()).hexdigest()


def test_seed_determinism_flight_nodes(toy_data):
    """Test that flight node creation is deterministic."""
    scope_config = {"mode": "full"}
    
    # Add time features
    lf = add_time_features(toy_data)
    
    # Run 1
    set_global_seed(42)
    nodes1 = build_flight_nodes(lf, scope_config)
    hash1 = hash_dataframe(nodes1)
    
    # Run 2 with same seed
    set_global_seed(42)
    nodes2 = build_flight_nodes(lf, scope_config)
    hash2 = hash_dataframe(nodes2)
    
    assert hash1 == hash2, "Flight nodes should be identical with same seed"


def test_seed_determinism_tail_edges(toy_data):
    """Test that tail edge creation is deterministic."""
    scope_config = {"mode": "full"}
    
    lf = add_time_features(toy_data)
    nodes = build_flight_nodes(lf, scope_config)
    
    # Run 1
    set_global_seed(42)
    edges1 = build_tail_sequence_edges(nodes)
    hash1 = hash_dataframe(edges1)
    
    # Run 2 with same seed
    set_global_seed(42)
    edges2 = build_tail_sequence_edges(nodes)
    hash2 = hash_dataframe(edges2)
    
    assert hash1 == hash2, "Tail edges should be identical with same seed"


def test_different_seeds_may_differ():
    """Test that different seeds can produce different results (if randomness involved)."""
    # This is more of a sanity check
    # For deterministic operations, results should be the same regardless of seed
    # But having different seeds available is important for Monte Carlo analyses later
    
    set_global_seed(42)
    import random
    val1 = random.random()
    
    set_global_seed(99)
    val2 = random.random()
    
    # These should be different
    assert val1 != val2, "Different seeds should produce different random values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

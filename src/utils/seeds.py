"""
Deterministic seeding utility for reproducibility.

All stochastic operations must call set_global_seed at the start of execution.
"""
import random
import numpy as np


def set_global_seed(seed: int) -> None:
    """
    Set random seeds for all libraries to ensure deterministic execution.
    
    Args:
        seed: Integer seed value (e.g., 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    # igraph uses numpy's random state
    # leidenalg also uses numpy's random state when available
    print(f"[SEED] Global seed set to {seed}")


def get_derived_seed(base_seed: int, offset: int) -> int:
    """
    Generate a derived seed for sub-processes or parallel tasks.
    
    Args:
        base_seed: Base seed value
        offset: Integer offset to create variation
        
    Returns:
        Derived seed value
    """
    return (base_seed + offset) % (2**31 - 1)

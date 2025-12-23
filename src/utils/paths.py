"""
Path management utilities for consistent file access across the project.
"""
from pathlib import Path
from typing import Optional


def get_project_root(start_path: Optional[Path] = None) -> Path:
    """
    Find the project root directory (contains config/ and src/ directories).
    
    Args:
        start_path: Starting path for search (default: current file's directory)
        
    Returns:
        Path to project root
    """
    if start_path is None:
        start_path = Path(__file__).resolve().parent
    
    current = start_path
    for _ in range(10):  # Limit search depth
        if (current / "config").exists() and (current / "src").exists():
            return current
        if current.parent == current:  # Reached filesystem root
            break
        current = current.parent
    
    # If not found, assume we're in src/utils and go up two levels
    return Path(__file__).resolve().parent.parent.parent


def get_config_path() -> Path:
    """Get path to config.yaml."""
    return get_project_root() / "config" / "config.yaml"


def get_results_dir() -> Path:
    """Get path to results directory."""
    return get_project_root() / "results"


def get_data_dir() -> Path:
    """Get path to data directory."""
    return get_project_root() / "data"

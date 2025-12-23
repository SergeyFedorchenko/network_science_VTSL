"""
Run manifest generation for reproducibility tracking.

Each script writes a manifest capturing inputs, config, and outputs.
"""
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import hashlib


def get_git_commit() -> Optional[str]:
    """
    Get current git commit hash if available.
    
    Returns:
        Commit hash string or None if not in a git repo
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def hash_file(file_path: Path) -> str:
    """
    Compute SHA256 hash of a file.
    
    Args:
        file_path: Path to file
        
    Returns:
        Hexadecimal hash string
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_schema_hash(columns: List[str]) -> str:
    """
    Compute hash of dataset schema (column names).
    
    Args:
        columns: List of column names
        
    Returns:
        Hexadecimal hash string
    """
    schema_str = ",".join(sorted(columns))
    return hashlib.sha256(schema_str.encode()).hexdigest()[:16]


def create_run_manifest(
    script_name: str,
    config: Dict[str, Any],
    input_files: List[Path],
    output_files: List[Path],
    metadata: Optional[Dict[str, Any]] = None,
    manifest_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Create a run manifest capturing execution context.
    
    Args:
        script_name: Name of the script (e.g., "00_validate_inputs")
        config: Configuration dictionary
        input_files: List of input file paths
        output_files: List of output file paths
        metadata: Optional additional metadata (e.g., row counts, statistics)
        manifest_path: Optional path to save manifest JSON
        
    Returns:
        Manifest dictionary
    """
    manifest = {
        "script": script_name,
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "config_snapshot": config,
        "inputs": [],
        "outputs": [],
        "metadata": metadata or {}
    }
    
    # Add input file info
    for input_file in input_files:
        if input_file.exists():
            manifest["inputs"].append({
                "path": str(input_file),
                "size_bytes": input_file.stat().st_size,
                "hash": hash_file(input_file)[:16]  # truncate for readability
            })
    
    # Add output file info
    for output_file in output_files:
        if output_file.exists():
            manifest["outputs"].append({
                "path": str(output_file),
                "size_bytes": output_file.stat().st_size
            })
    
    # Save to file if path provided
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
    
    return manifest


def save_json(data: Dict[str, Any], file_path: Path) -> None:
    """
    Save dictionary to JSON file with pretty formatting.
    
    Args:
        data: Dictionary to save
        file_path: Output file path
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

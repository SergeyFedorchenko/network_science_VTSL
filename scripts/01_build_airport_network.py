"""
Script 01: Build airport network.

Constructs airport-centric network from flight data.
"""
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seeds import set_global_seed
from src.utils.logging import get_script_logger
from src.utils.paths import get_project_root, get_config_path, get_results_dir
from src.utils.manifests import create_run_manifest
from src.io.load_data import load_from_config
from src.io.time_features import add_time_features
from src.networks.airport_network import build_airport_network


def main():
    """Build airport network."""
    # Setup
    project_root = get_project_root()
    results_dir = get_results_dir()
    logger = get_script_logger("01_build_airport_network", results_dir)
    
    logger.info("="*80)
    logger.info("SCRIPT 01: BUILD AIRPORT NETWORK")
    logger.info("="*80)
    
    # Load config
    config_path = get_config_path()
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded config from {config_path}")
    
    # Set seed
    set_global_seed(config["seed"])
    
    # Check if outputs exist
    output_nodes = results_dir / "networks" / "airport_nodes.parquet"
    output_edges = results_dir / "networks" / "airport_edges.parquet"
    
    if output_nodes.exists() and not config["outputs"].get("overwrite", False):
        logger.warning(f"Output already exists: {output_nodes}")
        logger.warning("Set config.outputs.overwrite=true to regenerate")
        logger.info("Skipping (idempotent behavior)")
        sys.exit(0)
    
    # Load data
    logger.info("Loading flight data with filters")
    lf = load_from_config(config)
    
    # Add time features (optional for airport network, but useful for consistency)
    logger.info("Adding time features")
    lf = add_time_features(lf)
    
    # Build airport network
    logger.info("Building airport network")
    summary = build_airport_network(lf, config, results_dir)
    
    # Get input path
    data_path = Path(config["data"]["cleaned_path"])
    if not data_path.is_absolute():
        data_path = project_root / data_path
    
    # Create run manifest
    manifest = create_run_manifest(
        script_name="01_build_airport_network",
        config=config,
        input_files=[data_path],
        output_files=[
            output_nodes,
            output_edges,
            results_dir / "networks" / "airport_graph.graphml",
            results_dir / "logs" / "airport_network_summary.json"
        ],
        metadata=summary,
        manifest_path=results_dir / "logs" / "01_build_airport_network_manifest.json"
    )
    
    logger.info("="*80)
    logger.info("Airport network construction complete!")
    logger.info(f"Nodes: {summary['n_airports']}")
    logger.info(f"Edges: {summary['n_routes']}")
    logger.info(f"LCC: {summary['lcc_size']} nodes ({100*summary['lcc_size']/summary['n_airports']:.1f}%)")
    logger.info("="*80)


if __name__ == "__main__":
    main()

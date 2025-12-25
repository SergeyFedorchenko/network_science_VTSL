"""
Script 03: Build multilayer network.

Constructs airline-layered multilayer network representation.
Each layer corresponds to an airline's sub-network of routes.

Per IMPLEMENTATION_PLAN Section 6.3:
- Layers are OP_UNIQUE_CARRIER
- Intra-layer edges connect airports within an airline's network
- Optional inter-layer edges at shared airports (disabled by default)

Outputs:
- results/networks/multilayer_edges.parquet
- results/networks/layer_summary.parquet
"""
import sys
from pathlib import Path
import yaml
import polars as pl

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seeds import set_global_seed
from src.utils.logging import get_script_logger
from src.utils.paths import get_project_root, get_config_path, get_results_dir
from src.utils.manifests import create_run_manifest
from src.io.load_data import load_from_config
from src.networks.multilayer_network import build_multilayer_network


def main():
    """Build multilayer network."""
    # Setup
    project_root = get_project_root()
    results_dir = get_results_dir()
    logger = get_script_logger("03_build_multilayer", results_dir)
    
    logger.info("="*80)
    logger.info("SCRIPT 03: BUILD MULTILAYER NETWORK")
    logger.info("="*80)
    
    # Load config
    config_path = get_config_path()
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded config from {config_path}")
    
    # Set seed
    set_global_seed(config["seed"])
    
    # Check if outputs exist
    output_edges = results_dir / "networks" / "multilayer_edges.parquet"
    output_summary = results_dir / "networks" / "layer_summary.parquet"
    
    if output_edges.exists() and not config["outputs"].get("overwrite", False):
        logger.warning(f"Output already exists: {output_edges}")
        logger.warning("Set config.outputs.overwrite=true to regenerate")
        logger.info("Skipping (idempotent behavior)")
        sys.exit(0)
    
    # Check that airport network exists (required input)
    airport_nodes_path = results_dir / "networks" / "airport_nodes.parquet"
    if not airport_nodes_path.exists():
        logger.error(f"Airport nodes not found: {airport_nodes_path}")
        logger.error("Run script 01_build_airport_network.py first")
        sys.exit(1)
    
    # Load airport nodes
    logger.info(f"Loading airport nodes from {airport_nodes_path}")
    airport_nodes = pl.read_parquet(airport_nodes_path)
    logger.info(f"Loaded {len(airport_nodes)} airport nodes")
    
    # Load flight data
    logger.info("Loading flight data with filters")
    lf = load_from_config(config)
    
    # Build multilayer network
    logger.info("Building multilayer network")
    multilayer_config = config.get("multilayer", {})
    summary = build_multilayer_network(
        lf=lf,
        airport_nodes=airport_nodes,
        config=multilayer_config,
        output_dir=results_dir / "networks"
    )
    
    # Get input path
    data_path = Path(config["data"]["cleaned_path"])
    if not data_path.is_absolute():
        data_path = project_root / data_path
    
    # Create run manifest
    manifest = create_run_manifest(
        script_name="03_build_multilayer",
        config=config,
        input_files=[data_path, airport_nodes_path],
        output_files=[output_edges, output_summary],
        metadata=summary,
        manifest_path=results_dir / "logs" / "03_build_multilayer_manifest.json"
    )
    
    logger.info("="*80)
    logger.info("Multilayer network construction complete!")
    logger.info(f"Layers (airlines): {summary['n_layers']}")
    logger.info(f"Total edges: {summary['n_edges']}")
    logger.info(f"Inter-layer edges: {summary['n_interlayer_edges']}")
    logger.info("="*80)
    
    # Log top layers
    logger.info("Top 5 layers by flight volume:")
    for layer in summary["top_layers"][:5]:
        logger.info(f"  {layer['layer']}: {layer['total_flights']} flights, "
                   f"{layer['node_count']} airports, {layer['edge_count']} routes")


if __name__ == "__main__":
    main()

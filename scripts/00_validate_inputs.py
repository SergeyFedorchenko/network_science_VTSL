"""
Script 00: Validate input data.

Validates schema, constraints, and data quality of the cleaned dataset.
"""
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seeds import set_global_seed
from src.utils.logging import get_script_logger
from src.utils.paths import get_project_root, get_config_path, get_results_dir
from src.utils.manifests import create_run_manifest, compute_schema_hash
from src.io.load_data import load_flights_data, get_schema_summary, get_row_count
from src.io.validate_data import run_full_validation


def main():
    """Run data validation."""
    # Setup
    project_root = get_project_root()
    results_dir = get_results_dir()
    logger = get_script_logger("00_validate_inputs", results_dir)
    
    logger.info("="*80)
    logger.info("SCRIPT 00: VALIDATE INPUT DATA")
    logger.info("="*80)
    
    # Load config
    config_path = get_config_path()
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded config from {config_path}")
    
    # Set seed
    set_global_seed(config["seed"])
    
    # Load data (no filters, validate raw)
    data_path = Path(config["data"]["cleaned_path"])
    if not data_path.is_absolute():
        data_path = project_root / data_path
    
    logger.info(f"Loading data from {data_path}")
    
    lf = load_flights_data(
        data_path=data_path,
        year=None,  # No filter for validation
        include_cancelled=True,  # Include all for validation
        format=config["data"].get("format", "parquet")
    )
    
    # Get basic info
    schema_info = get_schema_summary(lf)
    row_count = get_row_count(lf)
    
    logger.info(f"Dataset has {row_count:,} rows and {schema_info['n_columns']} columns")
    logger.info(f"Columns: {', '.join(schema_info['columns'][:10])}...")
    
    # Run validation
    year = config["filters"]["year"]
    validation_results = run_full_validation(lf, year=year, output_dir=results_dir)
    
    # Save data fingerprint
    fingerprint = {
        "file_path": str(data_path),
        "row_count": row_count,
        "n_columns": schema_info['n_columns'],
        "columns": schema_info['columns'],
        "schema_hash": compute_schema_hash(schema_info['columns']),
        "validation_passed": validation_results["schema_valid"]
    }
    
    from src.utils.manifests import save_json
    fingerprint_path = results_dir / "logs" / "data_fingerprint.json"
    save_json(fingerprint, fingerprint_path)
    logger.info(f"Saved data fingerprint to {fingerprint_path}")
    
    # Create run manifest
    manifest = create_run_manifest(
        script_name="00_validate_inputs",
        config=config,
        input_files=[data_path],
        output_files=[
            results_dir / "tables" / "data_validation_summary.csv",
            fingerprint_path
        ],
        metadata=validation_results,
        manifest_path=results_dir / "logs" / "00_validate_inputs_manifest.json"
    )
    
    logger.info("Validation complete!")
    
    # Exit with error if validation failed
    if not validation_results["schema_valid"]:
        logger.error("VALIDATION FAILED - see errors above")
        sys.exit(1)
    else:
        logger.info("VALIDATION PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()

"""
Data validation module for flight dataset.

Validates schema, constraints, and data quality.
"""
import polars as pl
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging


logger = logging.getLogger(__name__)


# Required columns with expected types (polars dtype names)
REQUIRED_COLUMNS = {
    "YEAR": ["Int64", "Int32", "Int16"],
    "MONTH": ["Int64", "Int32", "Int16"],
    "FL_DATE": ["Date", "Utf8", "String"],  # Can be string initially
    "OP_UNIQUE_CARRIER": ["Utf8", "String"],
    "TAIL_NUM": ["Utf8", "String"],  # nullable
    "OP_CARRIER_FL_NUM": ["Int64", "Int32", "Int16"],
    "ORIGIN_AIRPORT_ID": ["Int64", "Int32", "Int16"],
    "ORIGIN": ["Utf8", "String"],
    "DEST": ["Utf8", "String"],
    "DEP_TIME": ["Float64", "Float32", "Int64", "Int32"],  # nullable
    "ARR_TIME": ["Float64", "Float32", "Int64", "Int32"],  # nullable
    "DEP_DELAY": ["Float64", "Float32"],
    "ARR_DELAY": ["Float64", "Float32"],
    "CANCELLED": ["Float64", "Float32", "Int64", "Int32", "Int16"],
    "AIR_TIME": ["Float64", "Float32"],  # nullable
    "FLIGHTS": ["Float64", "Float32", "Int64", "Int32"],
    "DISTANCE": ["Float64", "Float32"],
}


def validate_schema(lf: pl.LazyFrame) -> Tuple[bool, List[str]]:
    """
    Validate that all required columns exist with compatible types.
    
    Args:
        lf: Polars LazyFrame to validate
        
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    schema = lf.collect_schema()
    
    # Check for missing columns
    missing = set(REQUIRED_COLUMNS.keys()) - set(schema.names())
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")
    
    # Check type compatibility
    for col, allowed_types in REQUIRED_COLUMNS.items():
        if col in schema:
            actual_type = str(schema[col])
            if actual_type not in allowed_types:
                errors.append(
                    f"Column {col} has type {actual_type}, "
                    f"expected one of {allowed_types}"
                )
    
    is_valid = len(errors) == 0
    return is_valid, errors


def validate_constraints(lf: pl.LazyFrame, year: int) -> Dict[str, Any]:
    """
    Validate data constraints and compute quality metrics.
    
    Args:
        lf: Polars LazyFrame to validate
        year: Expected year value
        
    Returns:
        Dictionary with validation results and statistics
    """
    # Compute validation metrics efficiently
    validation = lf.select([
        pl.count().alias("total_rows"),
        
        # Year validation
        (pl.col("YEAR") == year).sum().alias("year_matches"),
        pl.col("YEAR").min().alias("year_min"),
        pl.col("YEAR").max().alias("year_max"),
        
        # Month validation
        ((pl.col("MONTH") >= 1) & (pl.col("MONTH") <= 12)).sum().alias("month_valid"),
        pl.col("MONTH").min().alias("month_min"),
        pl.col("MONTH").max().alias("month_max"),
        
        # Cancelled validation
        (pl.col("CANCELLED").is_in([0, 1])).sum().alias("cancelled_valid"),
        pl.col("CANCELLED").sum().alias("n_cancelled"),
        
        # Non-cancelled flights checks
        ((pl.col("CANCELLED") == 0) & (pl.col("ORIGIN").str.len_chars() > 0)).sum().alias("origin_non_empty"),
        ((pl.col("CANCELLED") == 0) & (pl.col("DEST").str.len_chars() > 0)).sum().alias("dest_non_empty"),
        
        # Null rates
        pl.col("TAIL_NUM").is_null().sum().alias("tail_num_nulls"),
        pl.col("DEP_TIME").is_null().sum().alias("dep_time_nulls"),
        pl.col("ARR_TIME").is_null().sum().alias("arr_time_nulls"),
        pl.col("AIR_TIME").is_null().sum().alias("air_time_nulls"),
        
        # Delay extremes
        pl.col("DEP_DELAY").min().alias("dep_delay_min"),
        pl.col("DEP_DELAY").max().alias("dep_delay_max"),
        pl.col("ARR_DELAY").min().alias("arr_delay_min"),
        pl.col("ARR_DELAY").max().alias("arr_delay_max"),
        
        # Unique counts
        pl.col("ORIGIN").n_unique().alias("n_unique_origins"),
        pl.col("DEST").n_unique().alias("n_unique_dests"),
        pl.col("OP_UNIQUE_CARRIER").n_unique().alias("n_unique_carriers"),
    ]).collect()
    
    result = validation.to_dicts()[0]
    
    # Compute derived metrics
    total = result["total_rows"]
    result["year_mismatch_pct"] = 100.0 * (1 - result["year_matches"] / total) if total > 0 else 0
    result["month_invalid_pct"] = 100.0 * (1 - result["month_valid"] / total) if total > 0 else 0
    result["cancelled_invalid_pct"] = 100.0 * (1 - result["cancelled_valid"] / total) if total > 0 else 0
    result["cancellation_rate"] = 100.0 * result["n_cancelled"] / total if total > 0 else 0
    
    # Null rates as percentages
    result["tail_num_null_pct"] = 100.0 * result["tail_num_nulls"] / total if total > 0 else 0
    result["dep_time_null_pct"] = 100.0 * result["dep_time_nulls"] / total if total > 0 else 0
    result["arr_time_null_pct"] = 100.0 * result["arr_time_nulls"] / total if total > 0 else 0
    result["air_time_null_pct"] = 100.0 * result["air_time_nulls"] / total if total > 0 else 0
    
    return result


def validate_air_time_logic(lf: pl.LazyFrame) -> Dict[str, Any]:
    """
    Validate AIR_TIME nullability logic (null when cancelled, positive when not).
    
    Args:
        lf: Polars LazyFrame
        
    Returns:
        Dictionary with AIR_TIME validation results
    """
    air_time_check = lf.select([
        pl.count().alias("total_rows"),
        
        # Cancelled flights should have null AIR_TIME
        ((pl.col("CANCELLED") == 1) & pl.col("AIR_TIME").is_null()).sum().alias("cancelled_airtime_null"),
        ((pl.col("CANCELLED") == 1) & pl.col("AIR_TIME").is_not_null()).sum().alias("cancelled_airtime_not_null"),
        
        # Non-cancelled flights should have positive AIR_TIME
        ((pl.col("CANCELLED") == 0) & (pl.col("AIR_TIME") > 0)).sum().alias("active_airtime_positive"),
        ((pl.col("CANCELLED") == 0) & (pl.col("AIR_TIME").is_null())).sum().alias("active_airtime_null"),
        ((pl.col("CANCELLED") == 0) & (pl.col("AIR_TIME") <= 0)).sum().alias("active_airtime_nonpositive"),
        
        (pl.col("CANCELLED") == 1).sum().alias("n_cancelled"),
        (pl.col("CANCELLED") == 0).sum().alias("n_not_cancelled"),
    ]).collect()
    
    result = air_time_check.to_dicts()[0]
    
    # Compute percentages
    n_cancelled = result["n_cancelled"]
    n_not_cancelled = result["n_not_cancelled"]
    
    if n_cancelled > 0:
        result["cancelled_null_pct"] = 100.0 * result["cancelled_airtime_null"] / n_cancelled
    else:
        result["cancelled_null_pct"] = None
    
    if n_not_cancelled > 0:
        result["active_positive_pct"] = 100.0 * result["active_airtime_positive"] / n_not_cancelled
        result["active_null_pct"] = 100.0 * result["active_airtime_null"] / n_not_cancelled
    else:
        result["active_positive_pct"] = None
        result["active_null_pct"] = None
    
    return result


def run_full_validation(
    lf: pl.LazyFrame,
    year: int,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Run complete validation suite and save results.
    
    Args:
        lf: Polars LazyFrame to validate
        year: Expected year
        output_dir: Directory to save validation outputs
        
    Returns:
        Combined validation results dictionary
    """
    logger.info("Starting data validation")
    
    # Schema validation
    schema_valid, schema_errors = validate_schema(lf)
    logger.info(f"Schema validation: {'PASS' if schema_valid else 'FAIL'}")
    if schema_errors:
        for error in schema_errors:
            logger.error(f"  - {error}")
    
    if not schema_valid:
        return {
            "schema_valid": False,
            "schema_errors": schema_errors
        }
    
    # Constraint validation
    logger.info("Validating constraints")
    constraints = validate_constraints(lf, year)
    
    # AIR_TIME logic validation
    logger.info("Validating AIR_TIME logic")
    air_time = validate_air_time_logic(lf)
    
    # Combine results
    results = {
        "schema_valid": schema_valid,
        "schema_errors": schema_errors,
        "constraints": constraints,
        "air_time_logic": air_time
    }
    
    # Save summary table
    summary_path = output_dir / "tables" / "data_validation_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary_df = pl.DataFrame([{
        "metric": k,
        "value": str(v)
    } for k, v in constraints.items()])
    summary_df.write_csv(summary_path)
    logger.info(f"Saved validation summary to {summary_path}")
    
    # Log key findings
    logger.info(f"Total rows: {constraints['total_rows']:,}")
    logger.info(f"Date range: {constraints.get('year_min')} to {constraints.get('year_max')}")
    logger.info(f"Unique airports: {constraints['n_unique_origins']} origins, {constraints['n_unique_dests']} destinations")
    logger.info(f"Unique carriers: {constraints['n_unique_carriers']}")
    logger.info(f"Cancellation rate: {constraints['cancellation_rate']:.2f}%")
    logger.info(f"TAIL_NUM null rate: {constraints['tail_num_null_pct']:.2f}%")
    logger.info(f"DEP_TIME null rate: {constraints['dep_time_null_pct']:.2f}%")
    
    return results

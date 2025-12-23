"""
Data loading utilities for flight dataset.

Provides consistent interface for loading and filtering data via polars LazyFrame.
"""
import polars as pl
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging


logger = logging.getLogger(__name__)


def load_flights_data(
    data_path: Path,
    year: Optional[int] = None,
    include_cancelled: bool = False,
    carriers: Optional[List[str]] = None,
    format: str = "parquet"
) -> pl.LazyFrame:
    """
    Load flight data with optional filters applied.
    
    Uses LazyFrame for memory efficiency - filters are applied lazily.
    
    Args:
        data_path: Path to data file
        year: Filter to specific year (default: None = no filter)
        include_cancelled: Whether to include cancelled flights (default: False)
        carriers: List of carrier codes to include, or None for all
        format: File format ("parquet" or "csv")
        
    Returns:
        Polars LazyFrame with filters applied
    """
    # Load data lazily
    if format == "parquet":
        lf = pl.scan_parquet(data_path)
    elif format == "csv":
        lf = pl.scan_csv(data_path)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Loaded data from {data_path}")
    
    # Apply filters
    filters = []
    
    if year is not None:
        filters.append(pl.col("YEAR") == year)
        logger.info(f"Filter: YEAR == {year}")
    
    if not include_cancelled:
        filters.append(pl.col("CANCELLED") == 0)
        logger.info("Filter: CANCELLED == 0")
    
    if carriers is not None and carriers != "ALL":
        filters.append(pl.col("OP_UNIQUE_CARRIER").is_in(carriers))
        logger.info(f"Filter: carriers in {carriers}")
    
    if filters:
        lf = lf.filter(pl.all_horizontal(filters))
    
    return lf


def get_schema_summary(lf: pl.LazyFrame) -> Dict[str, Any]:
    """
    Get schema information from a LazyFrame (requires small collect).
    
    Args:
        lf: Polars LazyFrame
        
    Returns:
        Dictionary with schema info (columns, types)
    """
    # Get schema without full collect
    schema = lf.collect_schema()
    
    return {
        "columns": list(schema.names()),
        "dtypes": {name: str(dtype) for name, dtype in schema.items()},
        "n_columns": len(schema)
    }


def get_row_count(lf: pl.LazyFrame) -> int:
    """
    Get row count efficiently from LazyFrame.
    
    Args:
        lf: Polars LazyFrame
        
    Returns:
        Number of rows
    """
    return lf.select(pl.count()).collect().item()


def get_data_summary(lf: pl.LazyFrame) -> Dict[str, Any]:
    """
    Get comprehensive summary statistics for the dataset.
    
    Args:
        lf: Polars LazyFrame
        
    Returns:
        Dictionary with summary statistics
    """
    # Compute basic stats efficiently
    summary = lf.select([
        pl.count().alias("n_rows"),
        pl.col("FL_DATE").min().alias("min_date"),
        pl.col("FL_DATE").max().alias("max_date"),
        pl.col("ORIGIN").n_unique().alias("n_unique_origins"),
        pl.col("DEST").n_unique().alias("n_unique_dests"),
        pl.col("OP_UNIQUE_CARRIER").n_unique().alias("n_unique_carriers"),
        pl.col("TAIL_NUM").n_unique().alias("n_unique_tails"),
    ]).collect()
    
    return summary.to_dicts()[0]


def load_from_config(config: Dict[str, Any]) -> pl.LazyFrame:
    """
    Load flight data using configuration dictionary.
    
    Args:
        config: Configuration dictionary with data and filters sections
        
    Returns:
        Filtered LazyFrame
    """
    data_config = config["data"]
    filters_config = config["filters"]
    
    # Handle carriers config
    carriers = filters_config.get("carriers", "ALL")
    if carriers == "ALL":
        carriers = None
    
    return load_flights_data(
        data_path=Path(data_config["cleaned_path"]),
        year=filters_config.get("year"),
        include_cancelled=filters_config.get("include_cancelled", False),
        carriers=carriers,
        format=data_config.get("format", "parquet")
    )

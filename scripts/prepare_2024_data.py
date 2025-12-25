"""
Concatenate all 2024 monthly CSV files into a single cleaned Parquet dataset.

This script:
1. Reads all monthly CSV files from data/2024/
2. Concatenates them into a single dataset
3. Performs data cleaning and column selection
4. Saves the result as a Parquet file

Usage:
    python scripts/prepare_2024_data.py
"""

import logging
import sys
from pathlib import Path

import polars as pl

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging import setup_logger

# Setup logger
project_root = Path(__file__).resolve().parents[1]
log_file = project_root / "results" / "logs" / "prepare_2024_data.log"
logger = setup_logger(__name__, log_file=log_file, console=True)


def find_monthly_csv_files(data_dir: Path) -> list[Path]:
    """
    Find all monthly CSV files in the data directory.
    
    Parameters
    ----------
    data_dir : Path
        Path to data/2024 directory
        
    Returns
    -------
    list[Path]
        Sorted list of CSV file paths
    """
    csv_files = []
    
    # Iterate through month directories (1_january, 2_february, etc.)
    for month_dir in sorted(data_dir.iterdir()):
        if month_dir.is_dir():
            # Look for CSV file in month directory
            csv_file = month_dir / "T_ONTIME_REPORTING.csv"
            if csv_file.exists():
                csv_files.append(csv_file)
                logger.info(f"Found: {csv_file.relative_to(data_dir.parent)}")
            else:
                logger.warning(f"No CSV file found in {month_dir}")
    
    return csv_files


def read_and_concatenate_monthly_files(csv_files: list[Path]) -> pl.DataFrame:
    """
    Read all monthly CSV files and concatenate them.
    
    Parameters
    ----------
    csv_files : list[Path]
        List of CSV file paths
        
    Returns
    -------
    pl.DataFrame
        Concatenated DataFrame
    """
    logger.info(f"Reading {len(csv_files)} monthly CSV files...")
    
    dfs = []
    total_rows = 0
    
    for csv_file in csv_files:
        logger.info(f"  Reading {csv_file.parent.name}...")
        
        # Read CSV with polars (much faster than pandas)
        # Use infer_schema_length=None to scan all rows for type inference
        # This helps with columns that might have mixed types
        df = pl.read_csv(
            csv_file,
            infer_schema_length=None,
            ignore_errors=True,  # Skip problematic rows
        )
        rows = len(df)
        total_rows += rows
        dfs.append(df)
        
        logger.info(f"    Loaded {rows:,} rows")
    
    logger.info(f"Concatenating {len(dfs)} DataFrames...")
    
    # Use diagonal concatenation to handle schema mismatches
    # This will align by column names and fill missing columns with nulls
    combined = pl.concat(dfs, how="diagonal")
    
    logger.info(f"Total rows: {len(combined):,}")
    return combined


def clean_and_select_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Clean data and select relevant columns matching existing schema.
    
    The existing cleaned dataset uses these columns:
    - YEAR, MONTH, FL_DATE
    - OP_UNIQUE_CARRIER, TAIL_NUM (note: not in new data), OP_CARRIER_FL_NUM
    - ORIGIN_AIRPORT_ID, ORIGIN, ORIGIN_CITY_NAME, ORIGIN_STATE_NM
    - DEST, DEST_CITY_NAME, DEST_STATE_NM
    - DEP_TIME, DEP_DELAY, ARR_TIME, ARR_DELAY
    - CANCELLED, AIR_TIME, FLIGHTS, DISTANCE
    
    Parameters
    ----------
    df : pl.DataFrame
        Raw concatenated DataFrame
        
    Returns
    -------
    pl.DataFrame
        Cleaned DataFrame with selected columns
    """
    logger.info("Cleaning and selecting columns...")
    
    # Check for TAIL_NUM - not in documentation, might not exist
    has_tail_num = "TAIL_NUM" in df.columns
    if not has_tail_num:
        logger.warning("TAIL_NUM column not found - will be set to null")
    
    # Parse FL_DATE to proper date format (currently "1/1/2024 12:00:00 AM")
    logger.info("Parsing FL_DATE to date format...")
    df = df.with_columns(
        pl.col("FL_DATE").str.strptime(pl.Date, "%m/%d/%Y %I:%M:%S %p")
    )
    
    # Select columns matching existing schema
    # Note: We're excluding many columns that are not needed:
    # - Sequence IDs, City Market IDs, State abbreviations, FIPS codes, WAC codes
    # - CRS (scheduled) times - we keep actual times
    # - Delay breakdown columns (CARRIER_DELAY, WEATHER_DELAY, etc.) - high missing rate
    # - Diversion columns - mostly empty
    # - Ground time columns - not needed
    # - QUARTER, DAY_OF_MONTH, DAY_OF_WEEK - can be derived from FL_DATE
    # - DEP_DELAY_NEW, ARR_DELAY_NEW - redundant with DEP_DELAY, ARR_DELAY
    # - CANCELLATION_CODE - too granular, CANCELLED flag is sufficient
    # - DIVERTED - low occurrence rate
    # - CRS_ELAPSED_TIME, ACTUAL_ELAPSED_TIME - redundant with AIR_TIME
    # - DISTANCE_GROUP - redundant with DISTANCE
    
    columns_to_keep = [
        "YEAR",
        "MONTH",
        "FL_DATE",
        "OP_UNIQUE_CARRIER",
        "OP_CARRIER_FL_NUM",
        "ORIGIN_AIRPORT_ID",
        "ORIGIN",
        "ORIGIN_CITY_NAME",
        "ORIGIN_STATE_NM",
        "DEST",
        "DEST_CITY_NAME",
        "DEST_STATE_NM",
        "DEP_TIME",
        "DEP_DELAY",
        "ARR_TIME",
        "ARR_DELAY",
        "CANCELLED",
        "AIR_TIME",
        "FLIGHTS",
        "DISTANCE",
    ]
    
    # Add TAIL_NUM if it exists, otherwise create null column
    if has_tail_num:
        columns_to_keep.insert(4, "TAIL_NUM")
        df_cleaned = df.select(columns_to_keep)
    else:
        df_cleaned = df.select(columns_to_keep[:4] + columns_to_keep[4:]).with_columns(
            pl.lit(None, dtype=pl.Utf8).alias("TAIL_NUM")
        )
        # Reorder to match expected schema
        df_cleaned = df_cleaned.select(
            columns_to_keep[:4] + ["TAIL_NUM"] + columns_to_keep[4:]
        )
    
    logger.info(f"Selected {len(df_cleaned.columns)} columns from {len(df.columns)} original columns")
    
    return df_cleaned


def report_data_quality(df: pl.DataFrame) -> None:
    """
    Report data quality metrics.
    
    Parameters
    ----------
    df : pl.DataFrame
        Cleaned DataFrame
    """
    logger.info("=" * 80)
    logger.info("Data Quality Report")
    logger.info("=" * 80)
    
    logger.info(f"Total rows: {len(df):,}")
    logger.info(f"Total columns: {len(df.columns)}")
    logger.info(f"Date range: {df['FL_DATE'].min()} to {df['FL_DATE'].max()}")
    logger.info(f"Months covered: {sorted(df['MONTH'].unique().to_list())}")
    logger.info(f"Unique carriers: {df['OP_UNIQUE_CARRIER'].n_unique()}")
    logger.info(f"Unique origin airports: {df['ORIGIN'].n_unique()}")
    logger.info(f"Unique destination airports: {df['DEST'].n_unique()}")
    
    logger.info("\nMissing data by column:")
    for col in df.columns:
        null_count = df[col].null_count()
        null_pct = 100 * null_count / len(df)
        if null_count > 0:
            logger.info(f"  {col}: {null_count:,} ({null_pct:.2f}%)")
    
    logger.info("\nCancellation statistics:")
    cancelled_count = df["CANCELLED"].sum()
    cancelled_pct = 100 * cancelled_count / len(df)
    logger.info(f"  Total cancelled: {cancelled_count:,} ({cancelled_pct:.2f}%)")
    
    logger.info("\nDelay statistics (non-cancelled flights):")
    non_cancelled = df.filter(pl.col("CANCELLED") == 0)
    logger.info(f"  Mean departure delay: {non_cancelled['DEP_DELAY'].mean():.2f} min")
    logger.info(f"  Mean arrival delay: {non_cancelled['ARR_DELAY'].mean():.2f} min")
    logger.info(f"  Median departure delay: {non_cancelled['DEP_DELAY'].median():.2f} min")
    logger.info(f"  Median arrival delay: {non_cancelled['ARR_DELAY'].median():.2f} min")
    
    logger.info("=" * 80)


def main() -> None:
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("Data Preparation: 2024 Monthly CSV to Parquet")
    logger.info("=" * 80)
    
    # Paths
    project_root = Path(__file__).resolve().parents[1]
    data_2024_dir = project_root / "data" / "2024"
    output_file = project_root / "data" / "cleaned" / "flights_2024.parquet"
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Find all monthly CSV files
    logger.info("Step 1: Finding monthly CSV files...")
    csv_files = find_monthly_csv_files(data_2024_dir)
    
    if not csv_files:
        logger.error("No CSV files found!")
        return
    
    logger.info(f"Found {len(csv_files)} monthly CSV files")
    
    # Step 2: Read and concatenate
    logger.info("\nStep 2: Reading and concatenating monthly files...")
    df_raw = read_and_concatenate_monthly_files(csv_files)
    
    # Step 3: Clean and select columns
    logger.info("\nStep 3: Cleaning and selecting columns...")
    df_cleaned = clean_and_select_columns(df_raw)
    
    # Step 4: Report data quality
    logger.info("\nStep 4: Data quality report...")
    report_data_quality(df_cleaned)
    
    # Step 5: Save to Parquet
    logger.info(f"\nStep 5: Saving to Parquet: {output_file}")
    df_cleaned.write_parquet(
        output_file,
        compression="snappy",
        statistics=True,
        use_pyarrow=False,
    )
    
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info(f"Saved successfully: {file_size_mb:.2f} MB")
    
    logger.info("=" * 80)
    logger.info("Data preparation complete!")
    logger.info("=" * 80)
    logger.info(f"Output file: {output_file}")
    logger.info(f"Rows: {len(df_cleaned):,}")
    logger.info(f"Columns: {len(df_cleaned.columns)}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

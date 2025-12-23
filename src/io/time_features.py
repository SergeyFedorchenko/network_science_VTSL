"""
Time feature engineering for flight data.

Converts HHMM format to timestamps with proper midnight roll handling.
"""
import polars as pl
from typing import List
import logging


logger = logging.getLogger(__name__)


def hhmm_to_minutes(hhmm: pl.Expr) -> pl.Expr:
    """
    Convert HHMM format (e.g., 1430 for 14:30) to minutes since midnight.
    
    Handles both integer and float inputs. Null values remain null.
    
    Args:
        hhmm: Polars expression with HHMM values
        
    Returns:
        Polars expression with minutes since midnight (0-1439)
    """
    # Cast to int (handles both int and float inputs)
    # For null values, this will remain null
    hhmm_int = hhmm.cast(pl.Int32)
    
    # Extract hours and minutes
    hours = hhmm_int // 100
    minutes = hhmm_int % 100
    
    # Convert to total minutes
    total_minutes = hours * 60 + minutes
    
    return total_minutes


def add_time_features(
    lf: pl.LazyFrame,
    hhmm_columns: List[str] = ["DEP_TIME", "ARR_TIME"],
    apply_midnight_roll: bool = True
) -> pl.LazyFrame:
    """
    Add time features to flight data.
    
    Creates:
    - dep_minutes: Minutes since midnight for departure
    - arr_minutes: Minutes since midnight for arrival
    - dep_ts: Full departure timestamp
    - arr_ts: Full arrival timestamp (with midnight roll correction)
    
    Midnight roll rule: If ARR_TIME < DEP_TIME and AIR_TIME > 0,
    then arrival is next day (add 24 hours to arr_ts).
    
    Args:
        lf: Polars LazyFrame with flight data
        hhmm_columns: List of columns in HHMM format to process
        apply_midnight_roll: Whether to apply midnight roll correction
        
    Returns:
        LazyFrame with added time features
    """
    logger.info(f"Adding time features for columns: {hhmm_columns}")
    
    # Convert HHMM to minutes
    expressions = []
    
    if "DEP_TIME" in hhmm_columns:
        expressions.append(
            hhmm_to_minutes(pl.col("DEP_TIME")).alias("dep_minutes")
        )
    
    if "ARR_TIME" in hhmm_columns:
        expressions.append(
            hhmm_to_minutes(pl.col("ARR_TIME")).alias("arr_minutes")
        )
    
    lf = lf.with_columns(expressions)
    
    # Convert FL_DATE to datetime (handles both Date and String types)
    # First check schema and convert appropriately
    schema = lf.collect_schema()
    fl_date_dtype = schema["FL_DATE"]
    
    if fl_date_dtype == pl.String:
        lf = lf.with_columns(pl.col("FL_DATE").str.to_datetime().alias("FL_DATE"))
    elif fl_date_dtype == pl.Date:
        lf = lf.with_columns(pl.col("FL_DATE").cast(pl.Datetime).alias("FL_DATE"))
    # else: already Datetime
    
    # Create timestamps
    timestamp_expressions = []
    
    # Departure timestamp: FL_DATE + dep_minutes
    if "DEP_TIME" in hhmm_columns:
        timestamp_expressions.append(
            (
                pl.col("FL_DATE") +
                pl.duration(minutes=pl.col("dep_minutes"))
            ).alias("dep_ts")
        )
    
    # Arrival timestamp with midnight roll
    if "ARR_TIME" in hhmm_columns:
        if apply_midnight_roll:
            # Detect midnight roll: ARR_TIME < DEP_TIME and AIR_TIME > 0
            # Add 1 day in these cases
            timestamp_expressions.append(
                pl.when(
                    (pl.col("arr_minutes") < pl.col("dep_minutes")) & 
                    (pl.col("AIR_TIME") > 0)
                ).then(
                    # Arrival is next day
                    pl.col("FL_DATE").dt.offset_by("1d") +
                    pl.duration(minutes=pl.col("arr_minutes"))
                ).otherwise(
                    # Arrival is same day
                    pl.col("FL_DATE") +
                    pl.duration(minutes=pl.col("arr_minutes"))
                ).alias("arr_ts")
            )
        else:
            # No midnight roll correction
            timestamp_expressions.append(
                (
                    pl.col("FL_DATE") +
                    pl.duration(minutes=pl.col("arr_minutes"))
                ).alias("arr_ts")
            )
    
    lf = lf.with_columns(timestamp_expressions)
    
    logger.info("Time features added successfully")
    return lf


def validate_time_features(lf: pl.LazyFrame) -> dict:
    """
    Validate time feature correctness.
    
    Checks:
    - Minutes are in valid range (0-1439)
    - Midnight roll cases detected correctly
    - Timestamps are reasonable
    
    Args:
        lf: LazyFrame with time features
        
    Returns:
        Dictionary with validation statistics
    """
    validation = lf.select([
        pl.count().alias("total_rows"),
        
        # Minutes validation
        (pl.col("dep_minutes").is_between(0, 1439)).sum().alias("dep_minutes_valid"),
        (pl.col("arr_minutes").is_between(0, 1439)).sum().alias("arr_minutes_valid"),
        
        # Midnight roll detection
        ((pl.col("arr_minutes") < pl.col("dep_minutes")) & (pl.col("AIR_TIME") > 0)).sum().alias("midnight_roll_cases"),
        
        # Timestamp checks
        (pl.col("arr_ts") > pl.col("dep_ts")).sum().alias("arr_after_dep"),
        (pl.col("arr_ts") < pl.col("dep_ts")).sum().alias("arr_before_dep"),
        
        # Null counts
        pl.col("dep_minutes").is_null().sum().alias("dep_minutes_null"),
        pl.col("arr_minutes").is_null().sum().alias("arr_minutes_null"),
        pl.col("dep_ts").is_null().sum().alias("dep_ts_null"),
        pl.col("arr_ts").is_null().sum().alias("arr_ts_null"),
    ]).collect()
    
    result = validation.to_dicts()[0]
    
    logger.info(f"Midnight roll cases detected: {result['midnight_roll_cases']}")
    logger.info(f"Arr after dep: {result['arr_after_dep']}, Arr before dep: {result['arr_before_dep']}")
    
    return result


# Unit test helper functions
def test_hhmm_conversion():
    """Test HHMM to minutes conversion."""
    test_data = pl.DataFrame({
        "time": [0, 100, 1430, 2359, None],
        "expected": [0, 60, 14*60+30, 23*60+59, None]
    })
    
    result = test_data.with_columns(
        hhmm_to_minutes(pl.col("time")).alias("minutes")
    )
    
    # Check if conversion is correct
    mismatches = result.filter(
        pl.col("minutes") != pl.col("expected")
    ).filter(
        pl.col("expected").is_not_null()  # Exclude null comparisons
    )
    
    if len(mismatches) > 0:
        print("HHMM conversion test FAILED:")
        print(mismatches)
        return False
    else:
        print("HHMM conversion test PASSED")
        return True


def test_midnight_roll():
    """Test midnight roll logic."""
    from datetime import date
    
    test_data = pl.DataFrame({
        "FL_DATE": [date(2025, 1, 1), date(2025, 1, 1), date(2025, 1, 1)],
        "DEP_TIME": [2300, 2330, 1000],
        "ARR_TIME": [100, 2400, 1200],
        "AIR_TIME": [120.0, 30.0, 120.0],
        "expected_next_day": [True, False, False]  # First case crosses midnight
    })
    
    result = add_time_features(test_data.lazy()).collect()
    
    # Check if midnight roll applied correctly
    result = result.with_columns([
        (pl.col("arr_ts").dt.date() > pl.col("FL_DATE")).alias("crossed_midnight")
    ])
    
    mismatches = result.filter(
        pl.col("crossed_midnight") != pl.col("expected_next_day")
    )
    
    if len(mismatches) > 0:
        print("Midnight roll test FAILED:")
        print(mismatches)
        return False
    else:
        print("Midnight roll test PASSED")
        return True


if __name__ == "__main__":
    # Run unit tests
    print("Running time feature unit tests...")
    test_hhmm_conversion()
    test_midnight_roll()

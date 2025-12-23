"""
Test data validation module.
"""
import pytest
import polars as pl
from datetime import date
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.io.validate_data import (
    validate_schema,
    validate_constraints,
    validate_air_time_logic
)


def test_validate_schema_valid():
    """Test schema validation with valid data."""
    df = pl.DataFrame({
        "YEAR": [2025],
        "MONTH": [1],
        "FL_DATE": [date(2025, 1, 1)],
        "OP_UNIQUE_CARRIER": ["AA"],
        "TAIL_NUM": ["N123AA"],
        "OP_CARRIER_FL_NUM": [100],
        "ORIGIN_AIRPORT_ID": [12478],
        "ORIGIN": ["JFK"],
        "DEST": ["LAX"],
        "DEP_TIME": [800.0],
        "ARR_TIME": [1100.0],
        "DEP_DELAY": [5.0],
        "ARR_DELAY": [10.0],
        "CANCELLED": [0.0],
        "AIR_TIME": [180.0],
        "FLIGHTS": [1.0],
        "DISTANCE": [2475.0],
    })
    
    is_valid, errors = validate_schema(df.lazy())
    assert is_valid, f"Schema should be valid, but got errors: {errors}"
    assert len(errors) == 0


def test_validate_schema_missing_columns():
    """Test schema validation with missing columns."""
    df = pl.DataFrame({
        "YEAR": [2025],
        "MONTH": [1],
        # Missing required columns
    })
    
    is_valid, errors = validate_schema(df.lazy())
    assert not is_valid
    assert len(errors) > 0
    assert any("Missing" in err for err in errors)


def test_validate_constraints():
    """Test constraint validation."""
    df = pl.DataFrame({
        "YEAR": [2025, 2025, 2024],  # One wrong year
        "MONTH": [1, 2, 13],  # One invalid month
        "FL_DATE": [date(2025, 1, 1), date(2025, 1, 2), date(2024, 12, 31)],
        "OP_UNIQUE_CARRIER": ["AA", "DL", "UA"],
        "TAIL_NUM": ["N123", "N456", None],
        "OP_CARRIER_FL_NUM": [100, 200, 300],
        "ORIGIN_AIRPORT_ID": [1, 2, 3],
        "ORIGIN": ["JFK", "LAX", "ORD"],
        "DEST": ["LAX", "JFK", "ATL"],
        "DEP_TIME": [800.0, 900.0, 1000.0],
        "ARR_TIME": [1100.0, 1200.0, 1300.0],
        "DEP_DELAY": [5.0, -10.0, 0.0],
        "ARR_DELAY": [10.0, -5.0, 2.0],
        "CANCELLED": [0.0, 0.0, 0.0],
        "AIR_TIME": [180.0, 190.0, 200.0],
        "FLIGHTS": [1.0, 1.0, 1.0],
        "DISTANCE": [2475.0, 2475.0, 600.0],
    })
    
    results = validate_constraints(df.lazy(), year=2025)
    
    assert results["total_rows"] == 3
    assert results["year_matches"] == 2  # 2 out of 3 match
    assert results["month_valid"] == 2  # 2 out of 3 valid


def test_validate_air_time_logic():
    """Test AIR_TIME nullability logic."""
    df = pl.DataFrame({
        "CANCELLED": [0.0, 0.0, 1.0, 1.0],
        "AIR_TIME": [180.0, None, None, 150.0],  # Last one is wrong
    })
    
    results = validate_air_time_logic(df.lazy())
    
    assert results["n_cancelled"] == 2
    assert results["n_not_cancelled"] == 2
    assert results["cancelled_airtime_null"] == 1  # Should be 1
    assert results["active_airtime_positive"] == 1  # Should be 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

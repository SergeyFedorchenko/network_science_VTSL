"""
Test time feature engineering.
"""
import pytest
import polars as pl
from datetime import date
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.io.time_features import hhmm_to_minutes, add_time_features


def test_hhmm_to_minutes():
    """Test HHMM to minutes conversion."""
    df = pl.DataFrame({
        "time": [0, 100, 1430, 2359, None],
    })
    
    result = df.with_columns(
        hhmm_to_minutes(pl.col("time")).alias("minutes")
    )
    
    expected = [0, 60, 14*60+30, 23*60+59, None]
    actual = result["minutes"].to_list()
    
    assert actual == expected, f"Expected {expected}, got {actual}"


def test_midnight_roll_detection():
    """Test midnight roll logic."""
    df = pl.DataFrame({
        "FL_DATE": [date(2025, 1, 1), date(2025, 1, 1), date(2025, 1, 1)],
        "DEP_TIME": [2300, 2330, 1000],  # Late evening, late evening, morning
        "ARR_TIME": [100, 2400, 1200],   # Next morning, same day, same day
        "AIR_TIME": [120.0, 30.0, 120.0],
    })
    
    result = add_time_features(df.lazy()).collect()
    
    # First flight should cross midnight (arr < dep, positive air time)
    # Second should not (arr >= dep when normalized)
    # Third should not (arr > dep)
    
    crossed_midnight = (result["arr_ts"].dt.date() > result["FL_DATE"]).to_list()
    
    # First case: 2300 -> 0100 next day (crossed)
    assert crossed_midnight[0] == True, "Flight 1 should cross midnight"
    
    # Second case: 2330 -> 2400 (00:00), edge case but AIR_TIME is only 30min
    # This shouldn't cross based on our logic
    
    # Third case: 1000 -> 1200 same day (not crossed)
    assert crossed_midnight[2] == False, "Flight 3 should not cross midnight"


def test_time_features_with_nulls():
    """Test time feature handling with null values."""
    df = pl.DataFrame({
        "FL_DATE": [date(2025, 1, 1), date(2025, 1, 1)],
        "DEP_TIME": [800, None],
        "ARR_TIME": [1100, None],
        "AIR_TIME": [180.0, None],
    })
    
    result = add_time_features(df.lazy()).collect()
    
    # First row should have valid timestamps
    assert result["dep_ts"][0] is not None
    assert result["arr_ts"][0] is not None
    
    # Second row should have null timestamps
    assert result["dep_ts"][1] is None
    assert result["arr_ts"][1] is None


def test_time_ordering():
    """Test that arrival is after departure for normal cases."""
    df = pl.DataFrame({
        "FL_DATE": [date(2025, 1, 1)],
        "DEP_TIME": [800],
        "ARR_TIME": [1100],
        "AIR_TIME": [180.0],
    })
    
    result = add_time_features(df.lazy()).collect()
    
    assert result["arr_ts"][0] > result["dep_ts"][0], \
        "Arrival timestamp should be after departure"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

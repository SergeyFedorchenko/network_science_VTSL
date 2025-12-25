"""
Generate synthetic toy dataset for testing network construction and determinism.

This module creates a small, controlled flight dataset for unit testing without
requiring large real datasets.
"""

from datetime import date, datetime
import polars as pl


def generate_toy_dataset() -> pl.DataFrame:
    """
    Generate a small synthetic flight dataset for testing.
    
    Creates a dataset with:
    - 3 airports (JFK, LAX, ORD)
    - 2 carriers (AA, DL)
    - 20 flights over 2 days
    - Mix of on-time, delayed, and cancelled flights
    
    Returns
    -------
    pl.DataFrame
        Synthetic flight data matching the expected schema
    """
    # Create synthetic flight records
    data = {
        "YEAR": [2024] * 20,
        "MONTH": [1] * 20,
        "FL_DATE": [
            date(2024, 1, 1), date(2024, 1, 1), date(2024, 1, 1), date(2024, 1, 1),
            date(2024, 1, 1), date(2024, 1, 1), date(2024, 1, 1), date(2024, 1, 1),
            date(2024, 1, 1), date(2024, 1, 1),
            date(2024, 1, 2), date(2024, 1, 2), date(2024, 1, 2), date(2024, 1, 2),
            date(2024, 1, 2), date(2024, 1, 2), date(2024, 1, 2), date(2024, 1, 2),
            date(2024, 1, 2), date(2024, 1, 2),
        ],
        "OP_UNIQUE_CARRIER": [
            "AA", "AA", "AA", "DL", "DL", "DL", "AA", "AA", "DL", "DL",
            "AA", "AA", "DL", "DL", "AA", "DL", "AA", "DL", "AA", "DL",
        ],
        "TAIL_NUM": [
            "N101AA", "N101AA", "N102AA", "N201DL", "N201DL", "N202DL",
            "N103AA", "N104AA", "N203DL", "N204DL",
            "N101AA", "N102AA", "N201DL", "N202DL", "N103AA", "N203DL",
            "N104AA", "N204DL", "N101AA", "N201DL",
        ],
        "OP_CARRIER_FL_NUM": [
            100, 101, 102, 200, 201, 202, 103, 104, 203, 204,
            100, 102, 200, 202, 103, 203, 104, 204, 101, 201,
        ],
        "ORIGIN_AIRPORT_ID": [10001] * 20,
        "ORIGIN": [
            "JFK", "JFK", "LAX", "LAX", "ORD", "ORD", "JFK", "LAX", "ORD", "JFK",
            "JFK", "LAX", "LAX", "ORD", "JFK", "ORD", "LAX", "JFK", "ORD", "LAX",
        ],
        "ORIGIN_CITY_NAME": [
            "New York, NY", "New York, NY", "Los Angeles, CA", "Los Angeles, CA",
            "Chicago, IL", "Chicago, IL", "New York, NY", "Los Angeles, CA",
            "Chicago, IL", "New York, NY",
            "New York, NY", "Los Angeles, CA", "Los Angeles, CA", "Chicago, IL",
            "New York, NY", "Chicago, IL", "Los Angeles, CA", "New York, NY",
            "Chicago, IL", "Los Angeles, CA",
        ],
        "ORIGIN_STATE_NM": [
            "New York", "New York", "California", "California",
            "Illinois", "Illinois", "New York", "California",
            "Illinois", "New York",
            "New York", "California", "California", "Illinois",
            "New York", "Illinois", "California", "New York",
            "Illinois", "California",
        ],
        "DEST": [
            "LAX", "ORD", "ORD", "JFK", "JFK", "LAX", "LAX", "ORD", "LAX", "ORD",
            "ORD", "JFK", "JFK", "LAX", "ORD", "JFK", "JFK", "LAX", "JFK", "ORD",
        ],
        "DEST_CITY_NAME": [
            "Los Angeles, CA", "Chicago, IL", "Chicago, IL", "New York, NY",
            "New York, NY", "Los Angeles, CA", "Los Angeles, CA", "Chicago, IL",
            "Los Angeles, CA", "Chicago, IL",
            "Chicago, IL", "New York, NY", "New York, NY", "Los Angeles, CA",
            "Chicago, IL", "New York, NY", "New York, NY", "Los Angeles, CA",
            "New York, NY", "Chicago, IL",
        ],
        "DEST_STATE_NM": [
            "California", "Illinois", "Illinois", "New York",
            "New York", "California", "California", "Illinois",
            "California", "Illinois",
            "Illinois", "New York", "New York", "California",
            "Illinois", "New York", "New York", "California",
            "New York", "Illinois",
        ],
        "DEP_TIME": [
            800.0, 900.0, 1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, None,
            830.0, 1030.0, 1130.0, 1230.0, 1430.0, 1630.0, 1530.0, 1730.0, 1830.0, 930.0,
        ],
        "DEP_DELAY": [
            0.0, 5.0, -3.0, 15.0, 0.0, 10.0, 2.0, 20.0, -5.0, None,
            5.0, 0.0, 10.0, -2.0, 3.0, 8.0, 0.0, 12.0, 6.0, 4.0,
        ],
        "ARR_TIME": [
            1100.0, 1130.0, 1300.0, 1400.0, 1430.0, 1600.0, 1700.0, 1800.0, 1900.0, None,
            1100.0, 1330.0, 1430.0, 1530.0, 1730.0, 1930.0, 1830.0, 2030.0, 2130.0, 1230.0,
        ],
        "ARR_DELAY": [
            -5.0, 0.0, -8.0, 10.0, -3.0, 5.0, 0.0, 15.0, -10.0, None,
            0.0, -5.0, 5.0, -5.0, 0.0, 3.0, -3.0, 8.0, 2.0, 0.0,
        ],
        "CANCELLED": [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        "AIR_TIME": [
            180.0, 120.0, 180.0, 240.0, 120.0, 180.0, 180.0, 180.0, 180.0, None,
            175.0, 180.0, 180.0, 180.0, 175.0, 175.0, 180.0, 180.0, 180.0, 175.0,
        ],
        "FLIGHTS": [1.0] * 20,
        "DISTANCE": [
            2475.0, 740.0, 1740.0, 2475.0, 740.0, 2475.0, 2475.0, 1740.0, 2475.0, 740.0,
            740.0, 2475.0, 2475.0, 2475.0, 740.0, 740.0, 2475.0, 2475.0, 740.0, 1740.0,
        ],
    }
    
    df = pl.DataFrame(data)
    
    return df

# WS1 Execution Summary

## Date: 2025-12-23

## Environment Setup
- **Python Version**: 3.11.14 (conda-forge)
- **Conda Environment**: network_science
- **Key Dependencies**:
  - polars==1.36.1
  - python-igraph==1.0.0
  - leidenalg==0.11.0
  - numpy==1.26.4
  - scipy==1.16.3
  - matplotlib==3.10.8
  - seaborn==0.13.2
  - scikit-learn==1.8.0
  - gensim==4.4.0
  - pytest==9.0.2

**Note**: Initially created with Python 3.13, but encountered NumPy compatibility issues. Recreated environment with Python 3.11 for stable operation.

## Data Processing
### CSV to Parquet Conversion
- **Input**: `data/united_flights/united_flights.csv` (865.43 MB)
- **Output**: `data/cleaned/flights_2025.parquet` (120.47 MB)
- **Compression Ratio**: 7.18x
- **Rows**: 6,408,906
- **Columns**: 21
  - YEAR, MONTH, FL_DATE, OP_UNIQUE_CARRIER, TAIL_NUM, OP_CARRIER_FL_NUM
  - ORIGIN_AIRPORT_ID, ORIGIN, ORIGIN_CITY_NAME, ORIGIN_STATE_NM
  - DEST, DEST_CITY_NAME, DEST_STATE_NM
  - DEP_TIME, DEP_DELAY, ARR_TIME, ARR_DELAY
  - CANCELLED, AIR_TIME, FLIGHTS, DISTANCE

## Script Execution Results

### Script 00: Validate Input Data
**Status**: ✅ PASSED

**Validation Checks**:
- Schema validation (all 21 required columns present with correct types)
- Data type compatibility (supports both Utf8 and String for text columns)
- Constraint validation (year, month, cancelled values)
- AIR_TIME logic (null when cancelled)

**Fingerprint**: Saved to `results/logs/data_fingerprint.json`

### Script 01: Build Airport Network
**Status**: ✅ SUCCESS

**Results**:
- **Nodes (Airports)**: 349
- **Edges (Routes)**: 6,721
- **Largest Connected Component**: 349 nodes (100.0%)
- **Graph Type**: Directed
- **Edge Metrics**: flight_count, mean_dep_delay, mean_arr_delay, cancel_rate, mean_distance

**Outputs**:
- `results/networks/airport_nodes.parquet`
- `results/networks/airport_edges.parquet`
- `results/networks/airport_network_graphml.xml`

### Script 02: Build Flight Network
**Status**: ✅ SUCCESS

**Results**:
- **Nodes (Flights)**: 5,074,460
- **Edges**: 20,230,470
- **Scope Mode**: top_airports (k=50)
- **Edge Types**:
  - tail_next_leg: 5,044,160 (aircraft rotation)
  - route_knn: 15,186,310 (temporal route patterns, k=3)

**Outputs**:
- `results/networks/flight_nodes.parquet`
- `results/networks/flight_edges.parquet`
- `results/networks/flight_network_graphml.xml`

## Test Suite Results
**Status**: ✅ ALL PASSED

**Test Summary**: 17 passed, 0 failed, 4 warnings

**Test Categories**:
1. **Network Construction** (6 tests)
   - Airport nodes count and properties
   - Airport edges aggregation
   - Flight nodes with scoping
   - Tail sequence edges
   - Route kNN edges
   - No duplicate edges

2. **Seed Determinism** (3 tests)
   - Flight nodes determinism
   - Tail edges determinism
   - Different seeds produce different results

3. **Time Features** (4 tests)
   - HHMM to minutes conversion
   - Midnight roll detection
   - Handling null values
   - Time ordering (arr > dep)

4. **Data Validation** (4 tests)
   - Schema validation (valid & invalid)
   - Constraint validation
   - AIR_TIME logic

## Issues Resolved

### 1. NumPy Compatibility (Python 3.13)
- **Problem**: NumPy 2.4.0 failed to import on Python 3.13 (DLL dependency issue)
- **Solution**: Downgraded to Python 3.11, resulting in NumPy 1.26.4

### 2. Polars Type Names
- **Problem**: Validation expected "Utf8" but Polars reported "String" (newer naming)
- **Solution**: Updated REQUIRED_COLUMNS to accept both "Utf8" and "String"

### 3. Duration API
- **Problem**: `pl.Duration(time_unit="m")` invalid (expects 'ns', 'us', 'ms')
- **Solution**: Used `pl.duration(minutes=...)` constructor instead

### 4. LazyFrame Operations
- **Problem**: `.vstack()` not available on LazyFrame
- **Solution**: Used `pl.concat([...])` instead

### 5. Datetime Parsing
- **Problem**: `cast(pl.Datetime)` failed for string dates
- **Solution**: Used `str.to_datetime()` for String type, `cast(pl.Datetime)` for Date type

### 6. Edge Schema Alignment
- **Problem**: tail_edges (6 columns) vs route_edges (4 columns) couldn't concat
- **Solution**: Standardized to 6 columns with nulls for route edges

### 7. Date Type Handling in Tests
- **Problem**: Test data used `date` type, but time_features expected String
- **Solution**: Added dtype detection and appropriate conversion (Date → cast, String → str.to_datetime)

## Deprecation Warnings
- `pl.replace(..., default=...)` → use `replace_strict()`
- `pl.count()` → use `pl.len()`

These are non-breaking and can be addressed in future refactoring.

## Performance Notes
- **Validation**: < 1 second
- **Airport Network**: < 2 seconds
- **Flight Network**: ~15 seconds (5M nodes, 20M edges)
- **Test Suite**: ~1 second

## Reproducibility
- Global seed set to 42
- All runs generate identical manifests with git commit tracking
- LazyFrame evaluation ensures consistent processing order

## Next Steps (WS2+)
1. **Community Detection**: Run Leiden algorithm on both networks
2. **Delay Propagation**: Analyze tail sequence edges for cascading delays
3. **Route Analysis**: Examine route kNN patterns for operational insights
4. **Visualization**: Generate network plots and delay heatmaps
5. **Business Metrics**: Calculate operational KPIs from network structure

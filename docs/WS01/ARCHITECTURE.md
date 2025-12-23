# WS1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    NETWORK SCIENCE PROJECT - WS1                 │
│                 Data Validation & Network Construction           │
└─────────────────────────────────────────────────────────────────┘

INPUT DATA
══════════
┌──────────────────────────┐
│  data/cleaned/           │
│  flights_2025.parquet    │◄────── Place your dataset here
│  (millions of rows)      │
└──────────────────────────┘
            │
            │ polars.scan_parquet()
            ▼
┌──────────────────────────────────────────────────────────────────┐
│                          SCRIPT 00                                │
│                     Data Validation                               │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  • Schema check (columns, types)                           │  │
│  │  • Constraint validation (year, month, cancelled)          │  │
│  │  • AIR_TIME logic check                                    │  │
│  │  • Null rate analysis                                      │  │
│  │  • Extreme value detection                                 │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  Modules: src.io.validate_data                                   │
│  Output:  results/tables/data_validation_summary.csv             │
│           results/logs/data_fingerprint.json                     │
└──────────────────────────────────────────────────────────────────┘
            │
            │ Validation PASSED
            ▼
┌──────────────────────────────────────────────────────────────────┐
│                     TIME FEATURE ENGINEERING                      │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  HHMM Conversion:                                          │  │
│  │    DEP_TIME (828) → dep_minutes (8*60+28 = 488)           │  │
│  │    ARR_TIME (1617) → arr_minutes (16*60+17 = 977)         │  │
│  │                                                            │  │
│  │  Timestamp Creation:                                       │  │
│  │    dep_ts = FL_DATE + dep_minutes                         │  │
│  │    arr_ts = FL_DATE + arr_minutes                         │  │
│  │                                                            │  │
│  │  Midnight Roll Detection:                                  │  │
│  │    IF arr_minutes < dep_minutes AND AIR_TIME > 0:         │  │
│  │       arr_ts += 1 day                                     │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  Modules: src.io.time_features                                   │
└──────────────────────────────────────────────────────────────────┘
            │
            ├──────────────────┬──────────────────┐
            ▼                  ▼                  ▼
    ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
    │   SCRIPT 01   │  │   SCRIPT 02   │  │   SCRIPT 03   │
    │    Airport    │  │    Flight     │  │  Multilayer   │
    │    Network    │  │    Network    │  │   (future)    │
    └───────────────┘  └───────────────┘  └───────────────┘


SCRIPT 01: AIRPORT NETWORK
═══════════════════════════

┌──────────────────────────────────────────────────────────────────┐
│                   Node Extraction                                 │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  • Collect unique airports from ORIGIN and DEST            │  │
│  │  • Assign integer node_id (0, 1, 2, ...)                  │  │
│  │  • Store: code, city, state, airport_id                   │  │
│  └────────────────────────────────────────────────────────────┘  │
│           │                                                       │
│           ▼                                                       │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Edge Aggregation (by ORIGIN → DEST)                       │  │
│  │                                                            │  │
│  │  Metrics computed:                                         │  │
│  │    • flight_count = sum(FLIGHTS)                          │  │
│  │    • mean_dep_delay = mean(DEP_DELAY)                     │  │
│  │    • mean_arr_delay = mean(ARR_DELAY)                     │  │
│  │    • cancel_rate = mean(CANCELLED)                        │  │
│  │    • mean_distance = mean(DISTANCE)                       │  │
│  └────────────────────────────────────────────────────────────┘  │
│           │                                                       │
│           ▼                                                       │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  igraph Graph Construction                                 │  │
│  │    • Directed graph                                        │  │
│  │    • Vertex attributes: code, city, state                 │  │
│  │    • Edge attributes: metrics                             │  │
│  │    • Export: parquet + graphml                            │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  Output:                                                          │
│    ├─ results/networks/airport_nodes.parquet                     │
│    ├─ results/networks/airport_edges.parquet                     │
│    ├─ results/networks/airport_graph.graphml                     │
│    └─ results/logs/airport_network_summary.json                  │
└──────────────────────────────────────────────────────────────────┘


SCRIPT 02: FLIGHT NETWORK
══════════════════════════

┌──────────────────────────────────────────────────────────────────┐
│                      Scoping Strategy                             │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Mode: top_airports (default K=50)                         │  │
│  │                                                            │  │
│  │  Process:                                                  │  │
│  │    1. Count flights per airport (ORIGIN + DEST)           │  │
│  │    2. Select top K airports by volume                     │  │
│  │    3. Filter flights where ORIGIN or DEST in top set      │  │
│  │                                                            │  │
│  │  Alternative modes: "sample" (10%), "full" (all)          │  │
│  └────────────────────────────────────────────────────────────┘  │
│           │                                                       │
│           ▼                                                       │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Flight Node Creation                                      │  │
│  │                                                            │  │
│  │  Composite key:                                            │  │
│  │    FL_DATE | CARRIER | FLIGHT_NUM | ORIGIN | DEST | DEP_TIME│
│  │                                                            │  │
│  │  Attributes: carrier, tail, origin, dest, timestamps,     │  │
│  │              delays, cancelled, distance, air_time        │  │
│  └────────────────────────────────────────────────────────────┘  │
│           │                                                       │
│           ├──────────────────┬────────────────────┐              │
│           ▼                  ▼                    ▼              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐      │
│  │ Tail Sequence│  │  Route kNN   │  │ Airport Transfer │      │
│  │    Edges     │  │    Edges     │  │  (optional)      │      │
│  └──────────────┘  └──────────────┘  └──────────────────┘      │
│           │                  │                    │              │
│           └──────────┬───────┴────────────────────┘              │
│                      ▼                                           │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Combined Edge Table                                       │  │
│  │    • Deduplicate (src_id, dst_id) pairs                   │  │
│  │    • Store edge_type + type-specific attributes           │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  Output:                                                          │
│    ├─ results/networks/flight_nodes.parquet                      │
│    ├─ results/networks/flight_edges.parquet                      │
│    └─ results/logs/flight_graph_summary.json                     │
└──────────────────────────────────────────────────────────────────┘


EDGE TYPES DETAIL
═════════════════

┌──────────────────────────────────────────────────────────────────┐
│  TAIL SEQUENCE EDGES (tail_next_leg)                             │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                                                            │  │
│  │  For each TAIL_NUM (aircraft):                            │  │
│  │    1. Sort flights by dep_ts                              │  │
│  │    2. Connect consecutive legs: flight[i] → flight[i+1]   │  │
│  │                                                            │  │
│  │  Example:                                                  │  │
│  │    N123AA: JFK→LAX (08:00) → LAX→SFO (13:00) → ...       │  │
│  │            └─────────────────┘                            │  │
│  │            ground_time = 2 hours                          │  │
│  │                                                            │  │
│  │  Attributes:                                               │  │
│  │    • ground_time_minutes                                  │  │
│  │    • same_carrier (bool)                                  │  │
│  │    • tail (string)                                        │  │
│  │                                                            │  │
│  │  Implementation: window function with shift(-1)           │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  ROUTE KNN EDGES (route_knn)                                     │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                                                            │  │
│  │  For each route (ORIGIN, DEST):                           │  │
│  │    1. Sort flights by dep_ts                              │  │
│  │    2. Connect each flight to next K flights (default K=3) │  │
│  │                                                            │  │
│  │  Example (K=3):                                            │  │
│  │    JFK→LAX flights sorted by time:                        │  │
│  │    F1 (06:00) ──┬──> F2 (08:00)                          │  │
│  │                 ├──> F3 (10:00)                          │  │
│  │                 └──> F4 (12:00)                          │  │
│  │    F2 (08:00) ──┬──> F3 (10:00)                          │  │
│  │                 ├──> F4 (12:00)                          │  │
│  │                 └──> F5 (14:00)                          │  │
│  │                                                            │  │
│  │  Attributes:                                               │  │
│  │    • delta_dep_minutes (time between departures)          │  │
│  │                                                            │  │
│  │  Avoids cliques: O(N*K) edges, not O(N²)                 │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘


UTILITIES & INFRASTRUCTURE
═══════════════════════════

┌──────────────────────────────────────────────────────────────────┐
│  Deterministic Seeding (src.utils.seeds)                         │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  • set_global_seed(42) at start of every script           │  │
│  │  • Seeds: random, numpy.random                            │  │
│  │  • Ensures reproducible outputs                           │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  Logging (src.utils.logging)                                     │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  • Console + file output                                   │  │
│  │  • Timestamp, level, module name                          │  │
│  │  • Logs saved to results/logs/<script>.log                │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  Run Manifests (src.utils.manifests)                             │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Every script writes JSON manifest with:                   │  │
│  │    • Timestamp                                             │  │
│  │    • Git commit hash                                       │  │
│  │    • Config snapshot                                       │  │
│  │    • Input file fingerprints (hash, size)                 │  │
│  │    • Output file list                                      │  │
│  │    • Metadata (row counts, network stats)                 │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘


OUTPUT DIRECTORY STRUCTURE
══════════════════════════

results/
  ├── networks/              # Network parquet tables & graphml
  │   ├── airport_nodes.parquet
  │   ├── airport_edges.parquet
  │   ├── airport_graph.graphml
  │   ├── flight_nodes.parquet
  │   └── flight_edges.parquet
  │
  ├── tables/                # Summary CSVs
  │   └── data_validation_summary.csv
  │
  └── logs/                  # Manifests and log files
      ├── data_fingerprint.json
      ├── airport_network_summary.json
      ├── flight_graph_summary.json
      ├── 00_validate_inputs_manifest.json
      ├── 01_build_airport_network_manifest.json
      ├── 02_build_flight_network_manifest.json
      ├── 00_validate_inputs.log
      ├── 01_build_airport_network.log
      └── 02_build_flight_network.log


TESTING INFRASTRUCTURE
══════════════════════

tests/
  ├── fixtures/
  │   ├── generate_toy_data.py       # Creates 15-flight test dataset
  │   └── toy_flights.parquet        # Generated toy data
  │
  ├── test_validate_data.py          # Schema & constraint tests
  ├── test_time_features.py          # HHMM conversion & midnight roll
  ├── test_network_construction_small.py  # Edge correctness
  └── test_seed_determinism.py       # Reproducibility verification

Toy dataset includes:
  • 2 aircraft with 3 & 2 consecutive flights (tail edges)
  • 5 flights on JFK→LAX route (route kNN edges)
  • 1 midnight roll case (ARR_TIME < DEP_TIME)
  • 1 cancelled flight


PERFORMANCE CHARACTERISTICS
════════════════════════════

Memory Efficiency:
  ✓ LazyFrame evaluation (no full data in memory)
  ✓ Windowed operations (no Cartesian products)
  ✓ Columnar storage (parquet compression)

Scalability:
  ✓ Scoping strategies (top K airports)
  ✓ No O(n²) edge creation
  ✓ Streaming aggregations

Expected Runtimes (millions of rows):
  • Validation:      30-60 seconds
  • Airport network: 1-2 minutes
  • Flight network:  2-5 minutes (scoped to top 50 airports)


KEY DESIGN DECISIONS
════════════════════

1. Polars over Pandas
   → 10-100x faster, lazy evaluation, better memory

2. Scoping for flight graph
   → Top 50 airports captures 80%+ traffic, avoids explosion

3. Window functions for edges
   → O(N*K) complexity instead of O(N²)

4. Parquet storage
   → 5-10x compression, fast I/O, schema preservation

5. Composite flight keys
   → Stable IDs without external database


HANDOFF TO WS2-4
════════════════

Stable Output Schemas:
  ✓ airport_nodes.parquet (node_id, code, city, state)
  ✓ airport_edges.parquet (src_id, dst_id, metrics)
  ✓ flight_nodes.parquet  (flight_id, timestamps, delays)
  ✓ flight_edges.parquet  (src_id, dst_id, edge_type, attrs)

Available Utilities:
  ✓ src.io.load_data
  ✓ src.networks.igraph_helpers
  ✓ src.utils (seeds, logging, manifests)

Ready for:
  → WS2: Centrality & Communities
  → WS3: Robustness & Delay Propagation
  → WS4: Embeddings, Business, Figures


═══════════════════════════════════════════════════════════════════
                          WS1 COMPLETE ✅
═══════════════════════════════════════════════════════════════════
```

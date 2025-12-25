# AI Coding Agent Instructions

## Project Overview
US flight network analysis using **polars** (data) + **python-igraph** (graphs) + **leidenalg** (community detection).
Data: `data/cleaned/flights_2024.parquet` (~7M flights). Config: `config/config.yaml`.

## Critical Patterns

### 1. Always Use Polars LazyFrame
```python
# CORRECT - memory efficient
lf = pl.scan_parquet(path).filter(...).select(...)
df = lf.collect()  # Only at final step

# WRONG - loads entire dataset
df = pl.read_parquet(path)
```

### 2. Script Structure (see scripts/01_build_airport_network.py)
Every pipeline script must:
1. Import from `src/` modules, add path: `sys.path.insert(0, str(Path(__file__).parent.parent))`
2. Call `set_global_seed(config["seed"])` immediately after loading config
3. Check `config["outputs"]["overwrite"]` before regenerating outputs
4. Write run manifest to `results/logs/` via `create_run_manifest()`
5. Write all outputs under `results/` (never modify `data/`)

### 3. Network Construction Pattern
```python
# Airport network: aggregate flights → routes
edges = lf.group_by(["ORIGIN", "DEST"]).agg([...])

# Flight network: use windowed operations, NOT O(n²)
# For tail sequences: sort by [TAIL_NUM, dep_ts], use shift(-1).over("TAIL_NUM")
# For kNN: use shift(-j).over(["ORIGIN", "DEST"]) for j in 1..k
```

### 4. Graph Building (src/networks/igraph_helpers.py)
```python
g = ig.Graph(directed=True)
g.add_vertices(n)
g.add_edges(edge_pairs)  # list of (src_id, dst_id) tuples
g.vs["code"] = codes     # vertex attributes
g.es["weight"] = weights # edge attributes
```

### 5. Midnight Roll for Overnight Flights (src/io/time_features.py)
```python
# Flight departs 2300, arrives 0130 next day with AIR_TIME=150 min
# arr_minutes (90) < dep_minutes (1380) AND AIR_TIME > 0 → add 1 day to arr_ts
arr_ts = pl.when((pl.col("arr_minutes") < pl.col("dep_minutes")) & (pl.col("AIR_TIME") > 0))
    .then(pl.col("FL_DATE") + pl.duration(days=1, minutes=pl.col("arr_minutes")))
    .otherwise(pl.col("FL_DATE") + pl.duration(minutes=pl.col("arr_minutes")))
```

### 6. Leiden Community Detection (src/analysis/community.py)
```python
# Use CPM objective (not modularity) with these config-driven params:
partition = leidenalg.find_partition(
    g, leidenalg.CPMVertexPartition,
    resolution_parameter=0.01,  # config.analysis.communities.leiden.resolution
    seed=42,                    # determinism
    weights=g.es["weight"]
)
# Run n_runs=10 times, select partition with best quality score
```

### 7. Analysis Patterns
**Delay Propagation (IC model):** Seeds are top-k outdegree flights. Transmission: `p_tail=0.60` for aircraft rotations, `p_pax=0.25` for passenger connections. Run 200 Monte Carlo simulations.

**Node2vec Embeddings:** `dimensions=128`, `walk_length=80`, `num_walks=10`, `p=1.0`, `q=1.0`. Train on airport graph with gensim Word2Vec.

## Commands
```bash
make all          # Full pipeline: validate → networks → analysis → figures
make tests        # pytest tests/ -v
make validate     # python scripts/00_validate_inputs.py
make networks     # Build airport + flight networks
```

## Data Schema (required columns)
`FL_DATE`, `OP_UNIQUE_CARRIER`, `TAIL_NUM`, `ORIGIN`, `DEST`, `DEP_TIME`, `ARR_TIME`, 
`DEP_DELAY`, `ARR_DELAY`, `CANCELLED`, `AIR_TIME`, `DISTANCE`

Time features added by `src/io/time_features.py`: `dep_minutes`, `arr_minutes`, `dep_ts`, `arr_ts`

## Key Directories
- `src/io/` - Data loading, validation, time features
- `src/networks/` - Airport/flight graph construction
- `src/analysis/` - Centrality, community, embeddings, link prediction
- `src/business/` - Airline metrics, hub strategy
- `results/networks/*.parquet` - Graph node/edge tables
- `results/analysis/` - Analysis outputs
- `results/logs/*_manifest.json` - Reproducibility manifests

## Non-Negotiables
- **Determinism**: Fixed seed (42), run manifests for every script
- **Idempotent**: Check existing outputs before regenerating
- **Config-driven**: All parameters from `config/config.yaml`, no hardcoded values
- **No O(n²)**: Flight graph edges via windowed polars ops only

## Pipeline Order
`00_validate_inputs` → `01_build_airport_network` → `02_build_flight_network` → `03_build_multilayer` →
`04_run_centrality` → `05_run_communities` → `06_run_robustness` → 
`07_run_delay_propagation` → `08_run_embeddings_linkpred` → `09_run_business_module` → `10_make_all_figures`

## Testing
- Toy data: `tests/fixtures/toy_flights.parquet`
- Test patterns: `test_*_small.py` for integration, `test_*_toy.py` for unit tests
- Required coverage: validation, time parsing, network construction, seed determinism

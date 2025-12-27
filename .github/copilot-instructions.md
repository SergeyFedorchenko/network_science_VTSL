# AI Coding Agent Instructions

## Project Overview
US flight network analysis: **polars** (data) + **python-igraph** (graphs) + **leidenalg** (community detection).  
Data: `data/cleaned/flights_2024.parquet` (~7M flights). All params in `config/config.yaml`.

## Architecture
Three network representations built in sequence:
1. **Airport Network** – nodes=airports, edges=routes with aggregated metrics
2. **Flight Network** – nodes=individual flights, edges=tail sequences + route kNN (scoped to top-50 airports)
3. **Multilayer Network** – carrier-specific layers

**Key modules:** `src/io/` (load/validate), `src/networks/` (graph construction), `src/analysis/` (metrics), `src/viz/` (figures)

## Critical Patterns

### Polars LazyFrame Always
```python
lf = pl.scan_parquet(path).filter(...).select(...)  # CORRECT
df = lf.collect()  # Only at final step
```

### Script Template (see [scripts/01_build_airport_network.py](../scripts/01_build_airport_network.py))
```python
sys.path.insert(0, str(Path(__file__).parent.parent))  # Import src/
set_global_seed(config["seed"])                        # Immediately after loading config
if output.exists() and not config["outputs"]["overwrite"]: sys.exit(0)  # Idempotent
create_run_manifest(...)                               # Write to results/logs/
```

### Flight Graph Edges – No O(n²)
```python
# Tail sequences: shift(-1).over("TAIL_NUM") after sorting by [TAIL_NUM, dep_ts]
# Route kNN: shift(-j).over(["ORIGIN", "DEST"]) for j in 1..k
```

### Midnight Roll for Overnight Flights
```python
# If arr_minutes < dep_minutes AND AIR_TIME > 0 → arrival is next day
arr_ts = pl.when((col("arr_minutes") < col("dep_minutes")) & (col("AIR_TIME") > 0))
    .then(col("FL_DATE") + duration(days=1, minutes=col("arr_minutes")))
```

### Leiden Community Detection
```python
partition = leidenalg.find_partition(g, leidenalg.CPMVertexPartition,
    resolution_parameter=config["analysis"]["communities"]["leiden"]["resolution"],
    seed=config["seed"], weights=g.es["weight"])
# Run n_runs=10, select best quality score
```

## Commands
```powershell
conda activate network_science  # Environment from environment.yml
make all       # Full pipeline (validate → networks → analysis → figures)
make tests     # pytest tests/ -v
make validate  # python scripts/00_validate_inputs.py
```

Analysis hyperparameters (IC model, node2vec, robustness strategies) are in `config/config.yaml` under `analysis.*`.

## Pipeline Order
`00_validate_inputs` → `01_build_airport_network` → `02_build_flight_network` → `03_build_multilayer` → `04_run_centrality` → `05_run_communities` → `06_run_robustness` → `07_run_delay_propagation` → `08_run_embeddings_linkpred` → `09_run_business_module` → `10_make_all_figures`

## Data Schema
Required: `FL_DATE`, `OP_UNIQUE_CARRIER`, `TAIL_NUM`, `ORIGIN`, `DEST`, `DEP_TIME`, `ARR_TIME`, `DEP_DELAY`, `ARR_DELAY`, `CANCELLED`, `AIR_TIME`, `DISTANCE`  
Generated: `dep_minutes`, `arr_minutes`, `dep_ts`, `arr_ts`

## Non-Negotiables
- **Determinism**: `set_global_seed(42)` + run manifests for every script
- **Idempotent**: Check `config["outputs"]["overwrite"]` before regenerating
- **Config-driven**: All parameters from `config/config.yaml`, never hardcode
- **No O(n²)**: Flight edges via polars windowed ops only
- **Outputs**: Write to `results/`, never modify `data/`

## Testing
- Toy fixtures: `tests/fixtures/toy_flights.parquet`
- Naming: `test_*_small.py` (integration), `test_*_toy.py` (unit)
- Run: `pytest tests/ -v`

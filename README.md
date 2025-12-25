# Network Science: US Flight Network Analysis (2025)

**Team Size:** 4 members  
**Workstream (WS1):** Data Validation & Network Construction  
**Stack:** Python 3.11 + Polars + python-igraph + leidenalg

---

## Project Overview

This project analyzes one year (2025) of US flight data to build and compare three network representations:

1. **Airport Network** - Nodes are airports, edges are routes
2. **Flight Network** - Nodes are individual flights, edges capture temporal/operational dependencies
3. **Multilayer Network** - Airline-specific layers with optional inter-layer connections

### Advanced Methods

- **Community Detection:** Leiden algorithm with CPM objective
- **Robustness Analysis:** Percolation under targeted/random node removal
- **Delay Propagation:** SIR-based contagion modeling on flight graph
- **Graph Embeddings:** node2vec + link prediction
- **Business Metrics:** Hub concentration, disruption costs

---

## Quick Start (WS1)

### 1. Environment Setup

```powershell
conda env create -f environment.yml
conda activate network_science
```

### 2. Data Preparation

Place your dataset at: `data/cleaned/flights_2024.parquet`

See [data/README.md](data/README.md) for schema requirements.

### 3. Run Pipeline

```powershell
# Validate data
python scripts/00_validate_inputs.py

# Build airport network
python scripts/01_build_airport_network.py

# Build flight network
python scripts/02_build_flight_network.py
```

---

## Repository Structure

```
.
├── config/config.yaml       # Central configuration
├── src/
│   ├── utils/              # Seeds, logging, manifests
│   ├── io/                 # Data loading, validation, time features
│   └── networks/           # Airport & flight network construction
├── scripts/                # Executable pipeline scripts
├── tests/                  # Unit tests + toy dataset
└── results/                # All outputs (networks, logs, tables)
```

---

## Testing

```powershell
# Run all tests
pytest tests/ -v
```

---

## Output Files

**After script 01:**
- `results/networks/airport_nodes.parquet` - Airport nodes
- `results/networks/airport_edges.parquet` - Route edges with metrics

**After script 02:**
- `results/networks/flight_nodes.parquet` - Flight nodes (scoped)
- `results/networks/flight_edges.parquet` - Tail sequence + route kNN edges

All scripts write run manifests to `results/logs/` for reproducibility.

---

## Key Features

### Performance
- Polars LazyFrame for memory efficiency
- No O(n²) edge creation
- Configurable scoping for flight graph

### Reproducibility
- Global seed control
- Run manifests with git commit tracking
- Idempotent scripts

### Scalability
- Handles millions of flights
- Top-K airport scoping
- Parquet columnar storage

---

## Configuration

Key settings in [config/config.yaml](config/config.yaml):

```yaml
seed: 42
data:
  cleaned_path: "data/cleaned/flights_2024.parquet"
filters:
  year: 2024
  include_cancelled: false
flight_graph:
  scope:
    mode: "top_airports"
    top_airports_k: 50
  edges:
    include_tail_sequence: true
    route_knn_k: 3
```

---

## How to Run WS2 (Centrality & Community Detection)

### Prerequisites

WS1 outputs must exist:
- `results/networks/airport_nodes.parquet`
- `results/networks/airport_edges.parquet`
- (Optional) `results/networks/flight_nodes.parquet` and `flight_edges.parquet`

### Commands

```powershell
# Step 1: Compute centrality metrics (degree, PageRank, betweenness)
python scripts/04_run_centrality.py

# Step 2: Run Leiden CPM community detection
python scripts/05_run_communities.py
```

### WS2 Outputs

**After script 04:**
- `results/analysis/airport_centrality.parquet` - Centrality metrics for all airports
- `results/tables/airport_degree_dist_in.csv` - In-degree distribution
- `results/tables/airport_degree_dist_out.csv` - Out-degree distribution
- `results/tables/tbl01_top_airports_by_centrality.csv` - Top 20 airports by PageRank
- `results/logs/04_run_centrality_manifest.json` - Run manifest with graph summary

**After script 05:**
- `results/analysis/airport_leiden_membership.parquet` - Community assignments for airports
- `results/analysis/flight_leiden_membership.parquet` - Community assignments for flights (if available)
- `results/tables/community_summary_airport.csv` - Airport community summary (sizes, top airports)
- `results/tables/community_summary_flight.csv` - Flight community summary (dominant carriers)
- `results/tables/tbl02_airport_communities_summary.csv` - Report-ready table
- `results/logs/05_run_communities_manifest.json` - Run manifest with Leiden run logs

### Testing WS2

```powershell
# Run WS2-specific tests
pytest tests/test_centrality_small.py -v
pytest tests/test_leiden_determinism.py -v
pytest tests/test_ws1_integration_smoke.py -v

# Or run all tests
pytest tests/ -v
```

### Key Configuration (config.yaml)

```yaml
analysis:
  centrality:
    measures: ["degree", "betweenness", "pagerank"]
    betweenness_approx_cutoff: 20000  # Use approximation if N > 20000
  communities:
    method: "leiden_cpm"
    leiden:
      objective: "CPM"
      resolution: 0.01
      n_runs: 10  # Number of runs with different seeds
```

### Performance Notes

- **Betweenness centrality:** Automatically switches to approximation (top 10% nodes as sources) for graphs with more than 20,000 vertices
- **Leiden algorithm:** Runs multiple times (default 10) and selects partition with best CPM quality score
- **Determinism:** All randomness is seeded; rerunning with same config produces identical results

### Figures Enabled by WS2

The centrality and community outputs enable these report figures:
- `fig01_airport_degree_distribution.png` - Power law visualization
- `fig02_airport_centrality_rankings.png` - Top airports by centrality metrics
- `fig03_leiden_community_sizes_airport.png` - Community size distribution
- `fig07_connectivity_vs_delay_scatter.png` - Centrality vs operational metrics (with WS3 delay data)

These will be generated by script 10 (WS4).

---

## Next Steps

WS3-4 will implement:
- Robustness & delay propagation (scripts 06-07)
- Embeddings, link prediction, business metrics, figures (scripts 08-10)

---

## How to Run WS4 (Embeddings, Link Prediction, Business Analysis)

### Prerequisites

WS1-WS3 outputs must exist:
- `results/networks/airport_nodes.parquet`, `airport_edges.parquet`
- `results/analysis/airport_centrality.parquet` (WS2)
- `data/cleaned/flights_2024.parquet` (complete 2024 data, 7.08M flights)
- `data/cleaned/flights_2025.parquet` (legacy data, missing October)

### Commands

```powershell
# Step 1: Train embeddings and run link prediction
python scripts/08_run_embeddings_linkpred.py

# Step 2: Compute business metrics (airline-level)
python scripts/09_run_business_module.py

# Step 3: Generate all final figures
python scripts/10_make_all_figures.py
```

### WS4 Outputs

**After script 08 (embeddings + link prediction):**
- `results/analysis/airport_embeddings.parquet` - Node2vec embeddings (128-dim)
- `results/analysis/linkpred_metrics.json` - AUC and avg precision for all models
- `results/tables/airport_embedding_neighbors.csv` - Similar airports for major hubs
- `results/tables/linkpred_top_predictions.csv` - Top 100 predicted new routes
- `results/logs/08_run_embeddings_linkpred_manifest.json` - Run manifest

**After script 09 (business analysis):**
- `results/business/airline_summary_metrics.parquet` - Merged airline metrics
- `results/business/hub_concentration.parquet` - Hub dependence per airline
- `results/business/disruption_cost_proxy.parquet` - Delay/cancellation costs
- `results/tables/airline_business_metrics.csv` - Report-ready table
- `results/logs/09_run_business_module_manifest.json` - Run manifest

**After script 10 (figures):**
- `results/figures/fig06_hub_dependence_by_airline.png` - Top-1 vs top-3 hub concentration
- `results/figures/fig07_connectivity_vs_delay_scatter.png` - Network centralization vs delays
- `results/figures/fig08_link_prediction_performance.png` - Model comparison (AUC, avg precision)
- `results/figures/fig09_top_route_predictions.png` - Top predicted new routes
- `results/logs/10_make_all_figures_manifest.json` - Run manifest

### Testing WS4

```powershell
# Run WS4-specific tests
pytest tests/test_embeddings_small.py -v
pytest tests/test_linkpred_time_split_toy.py -v
pytest tests/test_business_metrics_toy.py -v
```

### Key Configuration (config.yaml)

```yaml
analysis:
  embeddings:
    method: "node2vec"
    dimensions: 128
    walk_length: 80
    num_walks: 10
    window_size: 10
    p: 1.0  # Return parameter
    q: 1.0  # In-out parameter
  link_prediction:
    time_split:
      train_months: [1, 2, 3, 4, 5, 6, 7, 8, 9]
      test_months: [10, 11, 12]
    negative_ratio: 5
    classifier: "logreg"

business:
  cost_per_delay_minute: 75.0  # USD per minute
  cost_per_cancellation: 10000.0  # USD per cancelled flight
```

### WS4 Data Leakage Prevention

**Link Prediction Time-Split:**
- Train graph: flights from months 1-9
- Test positives: new routes appearing in months 10-12 (not in train)
- Test negatives: sampled non-edges excluding all test positives
- Embeddings trained only on train graph
- No information from test period leaks into features or training

### Business Metrics Explained

**Operational Reliability:**
- `mean_dep_delay`, `mean_arr_delay`: Average delays (excluding cancelled flights)
- `cancellation_rate`: Fraction of cancelled flights
- `flight_count`: Total flights operated

**Network Strategy:**
- `hub_top1_pct`: % of flights through primary hub
- `hub_top3_pct`: % of flights through top-3 hubs
- `primary_hub`: Airport with most flights
- `primary_hub_pagerank`: PageRank centrality of primary hub (if WS2 available)

**Disruption Cost Proxy:**
- `delay_cost`: Sum of positive arrival delays × cost per minute
- `cancellation_cost`: Count of cancellations × cost per cancellation
- `total_cost`: Sum of delay and cancellation costs

### Performance Notes

- **Embeddings:** Node2vec walk generation is parallelizable; expect ~1-2 min for 300 airports
- **Link prediction:** Negative sampling is efficient; candidate ranking limited to 1000 edges for speed
- **Business metrics:** Polars lazy evaluation makes airline aggregation fast even on millions of flights

---

## References

- Project instructions: `.vscode/copilot-instructions.md`
- Polars: https://pola-rs.github.io/polars/
- python-igraph: https://igraph.org/python/

---

**Version:** 1.2.0 (WS1-WS4 Complete)  
**Last Updated:** December 2025

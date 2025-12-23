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

Place your dataset at: `data/cleaned/flights_2025.parquet`

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
# Generate toy dataset
python tests/fixtures/generate_toy_data.py

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
  cleaned_path: "data/cleaned/flights_2025.parquet"
filters:
  year: 2025
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

## Next Steps

WS2-4 will implement:
- Centrality & communities (scripts 04-05)
- Robustness & delay propagation (scripts 06-07)
- Embeddings, link prediction, business metrics, figures (scripts 08-10)

---

## References

- Project instructions: `.vscode/copilot-instructions.md`
- Polars: https://pola-rs.github.io/polars/
- python-igraph: https://igraph.org/python/

---

**Version:** 1.0.0 (WS1 Complete)  
**Last Updated:** December 2025

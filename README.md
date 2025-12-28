# Network Science: US Flight Network Analysis (2024)

**Status:** ✅ Complete (WS1–WS4 + Analysis Notebooks)  
**Stack:** Python 3.11 + Polars + python-igraph + leidenalg

---

## Project Overview

This project analyzes one year (2024) of US flight data (~7M flights) to build and compare three network representations:

1. **Airport Network** - Nodes are airports, edges are routes
2. **Flight Network** - Nodes are individual flights, edges capture temporal/operational dependencies
3. **Multilayer Network** - Airline-specific layers with optional inter-layer connections

### Advanced Methods

- **Community Detection:** Leiden algorithm with CPM objective (+ optional SBM)
- **Robustness Analysis:** Percolation under targeted/random node removal
- **Delay Propagation:** Independent Cascade (IC) model on flight graph
- **Graph Embeddings:** node2vec + link prediction
- **Business Metrics:** Hub concentration, disruption costs

### Research-Grade Analysis

A complete set of 10 Jupyter notebooks provides comprehensive interpretation of all pipeline outputs, generating 60+ report-ready figures and 30+ evidence tables with full reproducibility tracking.

---

## Quick Start

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
# Run full pipeline (validate → networks → analysis → figures)
make all

# Or run individual scripts
python scripts/00_validate_inputs.py   # Validate data
python scripts/01_build_airport_network.py
python scripts/02_build_flight_network.py
python scripts/03_build_multilayer.py
python scripts/04_run_centrality.py
python scripts/05_run_communities.py
python scripts/06_run_robustness.py
python scripts/07_run_delay_propagation.py
python scripts/08_run_embeddings_linkpred.py
python scripts/09_run_business_module.py
python scripts/10_make_all_figures.py
```

### 4. Run Analysis Notebooks

After the pipeline completes, run the analysis notebooks in `analysis/notebooks/` to generate research-grade interpretations (see [Analysis Notebooks](#analysis-notebooks) section below).

---

## Repository Structure

```
.
├── config/config.yaml       # Central configuration
├── src/
│   ├── utils/              # Seeds, logging, manifests
│   ├── io/                 # Data loading, validation, time features
│   ├── networks/           # Airport & flight network construction
│   ├── analysis/           # Centrality, communities, robustness, delay propagation
│   ├── business/           # Airline metrics and hub concentration
│   └── viz/                # Visualization utilities
├── scripts/                # Executable pipeline scripts (00-10)
├── tests/                  # Unit tests + toy dataset
├── analysis/
│   ├── notebooks/          # 10 research-grade Jupyter notebooks
│   └── RESULT_REPORT.md    # Analysis guide and interpretation framework
└── results/                # All outputs
    ├── networks/           # Graph artifacts (.parquet, .graphml)
    ├── analysis/           # Centrality, communities, robustness, embeddings
    ├── business/           # Airline metrics and cost proxies
    ├── tables/             # CSV tables (+ report/ subfolder)
    ├── figures/            # PNG figures (+ report/ subfolder)
    └── logs/               # Run manifests and provenance
```

---

## Testing

```powershell
# Run all tests
pytest tests/ -v
```

---

## Analysis Notebooks

The `analysis/notebooks/` directory contains 10 research-grade Jupyter notebooks that interpret all pipeline outputs. These notebooks follow a strict evidence-first methodology outlined in [analysis/RESULT_REPORT.md](analysis/RESULT_REPORT.md).

### Notebook Overview

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | `01_run_inventory__manifest_reconciliation.ipynb` | Pipeline provenance, manifest reconciliation, missing artifact detection |
| 02 | `02_network_construction__structure_and_sanity.ipynb` | Network structure summaries, top routes, degree distributions |
| 03 | `03_centrality__rankings_and_mechanisms.ipynb` | Centrality rankings, connector vs hub airports, distributions |
| 04 | `04_communities__structure_and_attributes.ipynb` | Leiden/SBM community sizes, airline dominance, geographic composition |
| 05 | `05_robustness__percolation_and_hub_dependence.ipynb` | Robustness curves, targeted vs random attack, critical nodes |
| 06 | `06_delay_propagation__cascades_and_superspreaders.ipynb` | Cascade size distributions, superspreader rankings, centrality overlap |
| 07 | `07_embeddings_linkpred__evaluation_and_plausibility.ipynb` | Embedding sanity, link prediction metrics, top route predictions |
| 08 | `08_business__hub_strategy_and_resilience.ipynb` | Airline KPIs, hub concentration, disruption cost proxies |
| 09 | `09_synthesis__integrated_findings.ipynb` | Cross-domain synthesis, evidence index, centrality-delay joins |
| 10 | `10_appendix__assumptions_limitations_reproducibility.ipynb` | Assumptions table, limitations, reproducibility checklist |

### Report Evidence Outputs

All notebooks write report-ready artifacts to:
- `results/tables/report/` - CSV tables with `nb##_` prefixes
- `results/figures/report/` - PNG figures with `nb##_` prefixes
- `results/tables/report/_warnings.log` - Consolidated warnings

### Key Generated Artifacts

**Tables (30+ files):**
- `nb01_run_index.csv`, `nb01_manifest_reconciliation.csv`, `nb01_missing_artifacts.csv`
- `nb03_centrality_top20_by_metric.csv`
- `nb04_community_sizes.csv`, `nb04_sbm_geographic_composition.csv`
- `nb05_robustness_summary_metrics.csv`
- `nb06_cascade_size_distribution.csv`, `delay_superspreaders_top20__delay_cascades.csv`
- `nb07_linkpred_metrics_flat.csv`, `nb07_top_predictions_annotated.csv`
- `nb08_airline_kpi_summary.csv`
- `nb09_master_evidence_index.csv`, `nb09_synthesis_findings.csv`
- `nb10_assumptions_table.csv`, `nb10_limitations_table.csv`, `nb10_final_checklist.csv`

**Figures (60+ files):**
- Centrality histograms and top-20 bar charts per metric
- Community size distributions (Leiden and SBM)
- Robustness curves by attack strategy
- Cascade size distribution and superspreader rankings
- Airline KPI comparisons (delays, cancellation rates, costs)
- Hub concentration vs disruption cost scatter plots
- Synthesis visualizations

---

## Output Files

### After Pipeline Completion

**Networks (`results/networks/`):**
- `airport_nodes.parquet`, `airport_edges.parquet`, `airport_graph.graphml`
- `flight_nodes.parquet`, `flight_edges.parquet`
- `multilayer_edges.parquet`, `layer_summary.parquet`

**Analysis (`results/analysis/`):**
- `airport_centrality.parquet` - Degree, betweenness, PageRank
- `airport_leiden_membership.parquet`, `airport_sbm_membership.parquet`
- `flight_leiden_membership.parquet`
- `robustness_curves.parquet`, `robustness_summary.json`
- `delay_cascades.parquet`, `delay_propagation_summary.json`
- `airport_embeddings.parquet`, `linkpred_metrics.json`

**Business (`results/business/`):**
- `airline_summary_metrics.parquet`
- `hub_concentration.parquet`
- `disruption_cost_proxy.parquet`

**Figures (`results/figures/`):**
- `fig01_airport_degree_distribution.png` through `fig09_top_route_predictions.png`

**Logs (`results/logs/`):**
- Run manifests for all scripts (`*_manifest.json`)
- Execution logs (`*.log`)
- Summary JSONs (`airport_network_summary.json`, etc.)

---

## Key Features

### Performance
- Polars LazyFrame for memory efficiency
- No O(n²) edge creation
- Configurable scoping for flight graph

### Reproducibility
- Global seed control (`seed: 42`)
- Run manifests with git commit tracking
- Idempotent scripts (check `outputs.overwrite` before regenerating)
- Evidence-first analysis notebooks with artifact traceability

### Scalability
- Handles millions of flights
- Top-K airport scoping for flight graph
- Parquet columnar storage

### Research Quality
- 10 analysis notebooks following strict evidence standards
- All claims backed by concrete artifacts (tables + figures)
- Mechanistic explanations grounded in network science
- Explicit assumptions, limitations, and sensitivity notes

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

analysis:
  centrality:
    measures: ["degree", "betweenness", "pagerank"]
    betweenness_approx_cutoff: 20000
  communities:
    method: "leiden_cpm"
    leiden:
      objective: "CPM"
      resolution: 0.01
      n_runs: 10
    sbm_optional:
      enabled: true
  robustness:
    strategies: ["random", "highest_degree", "highest_betweenness"]
    random_trials: 30
  delay_propagation:
    model: "IC"
    beta: 0.25
    p_tail: 0.60
    monte_carlo_runs: 200
  embeddings:
    method: "node2vec"
    dimensions: 128
  link_prediction:
    time_split:
      train_months: [1-9]
      test_months: [10-12]
    classifier: "logreg"

business:
  cost_per_delay_minute: 75.0
  cost_per_cancellation: 10000.0
```

---

## Pipeline Details

### WS1: Data Validation & Network Construction (Scripts 00-03)

**Prerequisites:** `data/cleaned/flights_2024.parquet`

```powershell
python scripts/00_validate_inputs.py   # Data validation
python scripts/01_build_airport_network.py
python scripts/02_build_flight_network.py
python scripts/03_build_multilayer.py
```

**Outputs:** Network artifacts in `results/networks/`

---

### WS2: Centrality & Community Detection (Scripts 04-05)

**Prerequisites:** WS1 outputs

```powershell
python scripts/04_run_centrality.py    # Degree, betweenness, PageRank
python scripts/05_run_communities.py   # Leiden CPM + optional SBM
```

**Key Features:**
- Betweenness approximation for large graphs (>20k vertices)
- Multi-run Leiden with best quality selection
- Optional Stochastic Block Model (SBM)

**Outputs:** `results/analysis/airport_centrality.parquet`, `*_leiden_membership.parquet`, `*_sbm_membership.parquet`

---

### WS3: Robustness & Delay Propagation (Scripts 06-07)

**Prerequisites:** WS1-WS2 outputs

```powershell
python scripts/06_run_robustness.py         # Percolation analysis
python scripts/07_run_delay_propagation.py  # IC cascade model
```

**Robustness Analysis:**
- Attack strategies: random, highest-degree, highest-betweenness
- Tracks largest connected component vs fraction removed
- Multiple random trials (default 30) for statistical robustness

**Delay Propagation (Independent Cascade):**
- Transmission probabilities: `p_tail=0.60` (aircraft rotations), `beta=0.25` (passenger connections)
- Seeds: top-K out-degree nodes with initial delays
- Monte Carlo runs (default 200) for cascade distribution
- Identifies "superspreader" airports

**Outputs:** `robustness_curves.parquet`, `delay_cascades.parquet`, `delay_propagation_summary.json`

---

### WS4: Embeddings, Link Prediction & Business (Scripts 08-10)

**Prerequisites:** WS1-WS3 outputs

```powershell
python scripts/08_run_embeddings_linkpred.py
python scripts/09_run_business_module.py
python scripts/10_make_all_figures.py
```

**Embeddings & Link Prediction:**
- Node2vec (128-dim) trained on months 1-9
- Time-split evaluation: test on new routes in months 10-12
- Leakage-free: embeddings use only training period
- Models: Common Neighbors, Jaccard, Adamic-Adar, Embedding-based

**Business Metrics (per airline):**
- Hub concentration: `hub_top1_pct`, `hub_top3_pct`
- Operational reliability: delays, cancellation rates
- Disruption cost proxy: `$75/delay-minute + $10k/cancellation`

**Outputs:** `airport_embeddings.parquet`, `linkpred_metrics.json`, `results/business/*.parquet`

---

## Generated Figures

After running the full pipeline and script 10:

| Figure | Description |
|--------|-------------|
| `fig01_airport_degree_distribution.png` | In/out degree distributions (power-law) |
| `fig02_airport_centrality_rankings.png` | Top airports by PageRank, betweenness |
| `fig03_leiden_community_sizes_airport.png` | Community size distribution |
| `fig04_robustness_curves.png` | Targeted vs random attack comparison |
| `fig05_delay_cascade_distribution.png` | Cascade size distribution |
| `fig06_hub_dependence_by_airline.png` | Airline hub concentration |
| `fig07_connectivity_vs_delay_scatter.png` | Centrality vs operational metrics |
| `fig08_link_prediction_performance.png` | Model comparison (AUC, avg precision) |
| `fig09_top_route_predictions.png` | Top predicted new routes |

---

## References

- Analysis guide: [analysis/RESULT_REPORT.md](analysis/RESULT_REPORT.md)
- Project instructions: `.github/copilot-instructions.md`
- Polars: https://pola-rs.github.io/polars/
- python-igraph: https://igraph.org/python/
- leidenalg: https://leidenalg.readthedocs.io/

---

**Version:** 2.0.0 (Complete with Analysis Notebooks)  
**Last Updated:** December 2025  
**Data Year:** 2024 (~7.08M flights)

# WS4 Implementation Summary

## Overview

Workstream 04 (WS4) implements:
1. **Graph embeddings** (node2vec) for airport similarity analysis
2. **Link prediction** with time-split evaluation to prevent data leakage
3. **Business analysis module** computing airline-level operational and network metrics
4. **Final figure generation** for report integration

All modules follow WS1-WS3 patterns: deterministic execution, manifests, config-driven parameters, and parquet/JSON outputs.

---

## File-by-File Implementation Plan

### Modules

#### `src/analysis/embeddings.py`
- `load_airport_graph_from_parquet()`: Load igraph from WS1 parquet outputs
- `generate_node2vec_walks()`: Biased random walks with p/q parameters
- `train_skipgram()`: Train Word2Vec skip-gram model using gensim
- `write_embeddings()`: Save embeddings to parquet
- `find_similar_airports()`: Cosine similarity-based nearest neighbors
- `get_embedding_pair_features()`: Compute Hadamard, L1, L2, cosine for edge pairs

**Performance:** Walks are generated in Python with numpy random state for determinism. For larger graphs, consider using igraph's C-based random walk (but harder to control p/q bias).

#### `src/analysis/link_prediction.py`
- `build_month_split_graphs()`: Split flights by month, build train/test edge sets
- `build_igraph_from_edges()`: Construct igraph from edge set
- `negative_sample_non_edges()`: Sample negatives excluding train/test positives
- `compute_heuristic_features()`: Common neighbors, Jaccard, Adamic-Adar, preferential attachment
- `evaluate_classifier()`: Train logistic regression, compute AUC and avg precision
- `rank_candidate_edges()`: Rank edges by predicted probability
- `evaluate_baseline_heuristics()`: Baseline metrics without ML

**Data leakage prevention:**
- Train edges: routes in months 1-9
- Test positives: routes in months 10-12 NOT in train
- Test negatives: sampled from non-edges, excluding all test positives
- Embeddings trained only on train graph

#### `src/business/airline_metrics.py`
- `compute_airline_operational_metrics()`: Delays, cancellation rate, flight count per airline
- `compute_hub_concentration()`: % of flights through top-K airports per airline
- `compute_disruption_cost_proxy()`: Delay cost + cancellation cost
- `merge_airline_metrics()`: Join all metrics, optionally with WS2 centrality
- `write_business_outputs()`: Write parquet + CSV outputs

**Polars usage:** All operations use LazyFrame scans for memory efficiency.

#### `src/viz/plotting.py` (extended)
- `plot_hub_dependence_by_airline()`: Bar chart of top-1 vs top-3 hub concentration
- `plot_connectivity_vs_delay_scatter()`: Scatter plot (hub concentration vs delay)
- `plot_link_prediction_performance()`: Bar chart comparing model metrics
- `plot_top_route_predictions()`: Horizontal bar chart of top predicted routes

### Scripts

#### `scripts/08_run_embeddings_linkpred.py`
**Purpose:** Train embeddings and evaluate link prediction.

**Steps:**
1. Build time-split graphs (train/test)
2. Train node2vec embeddings on train graph
3. Write embeddings to parquet
4. Find similar airports for major hubs
5. Evaluate baseline heuristics (CN, Jaccard, AA)
6. Train embedding-based classifier (logreg)
7. Rank candidate routes
8. Write outputs and manifest

**Outputs:**
- `results/analysis/airport_embeddings.parquet`
- `results/analysis/linkpred_metrics.json`
- `results/tables/airport_embedding_neighbors.csv`
- `results/tables/linkpred_top_predictions.csv`
- `results/logs/08_run_embeddings_linkpred_manifest.json`

#### `scripts/09_run_business_module.py`
**Purpose:** Compute airline-level business metrics.

**Steps:**
1. Compute operational metrics (delays, cancellations, flight count)
2. Compute hub concentration (top-1, top-3 hubs)
3. Compute disruption cost proxy (delay + cancellation costs)
4. Merge with WS2 centrality (if available)
5. Write outputs and manifest

**Outputs:**
- `results/business/airline_summary_metrics.parquet`
- `results/business/hub_concentration.parquet`
- `results/business/disruption_cost_proxy.parquet`
- `results/tables/airline_business_metrics.csv`
- `results/logs/09_run_business_module_manifest.json`

#### `scripts/10_make_all_figures.py`
**Purpose:** Generate report-ready figures from existing outputs.

**Steps:**
1. Read WS4 outputs (hub concentration, airline summary, link prediction metrics)
2. Generate 4 figures (hub dependence, connectivity vs delay, link prediction performance, top predictions)
3. Write manifest

**Outputs:**
- `results/figures/fig06_hub_dependence_by_airline.png`
- `results/figures/fig07_connectivity_vs_delay_scatter.png`
- `results/figures/fig08_link_prediction_performance.png`
- `results/figures/fig09_top_route_predictions.png`
- `results/logs/10_make_all_figures_manifest.json`

### Tests

#### `tests/test_embeddings_small.py`
- Test node2vec walks are deterministic with fixed seed
- Test skip-gram training produces correct embedding shape
- Test embedding pair features compute correctly
- Test p/q parameters affect walk behavior

#### `tests/test_linkpred_time_split_toy.py`
- Test time-split prevents data leakage (no train/test overlap)
- Test negative sampling excludes test positives
- Test heuristic features compute correctly
- Test igraph construction from edge sets
- Test carrier filtering in month split

#### `tests/test_business_metrics_toy.py`
- Test operational metrics computation (delays, cancellations)
- Test hub concentration calculation (top-1, top-3 hubs)
- Test disruption cost proxy (delay + cancellation costs)
- Test metric merging (joins all dataframes)

---

## Performance and Scalability

### Embeddings
- **Node2vec walks:** O(walks × nodes × walk_length). For 300 airports, 10 walks, length 80: ~240K steps
- **Skip-gram training:** gensim is efficient; ~5-10 seconds for 100K sentences
- **Bottleneck:** Walk generation (CPU-bound, single-threaded numpy random)

**Optimization:** Use igraph's random walk methods or parallelize walk generation across nodes.

### Link Prediction
- **Time-split:** O(flights) single pass through data (polars lazy scan)
- **Negative sampling:** O(ratio × positives) random edge sampling
- **Feature computation:** O(edges × avg_degree²) for common neighbors; use sparse adjacency
- **Classifier training:** O(features × samples) for logistic regression

**Optimization:** Limit candidate set for ranking (default 1000 edges).

### Business Metrics
- **Airline aggregation:** O(flights) with polars group_by (efficient)
- **Hub concentration:** O(carriers × airports) sorting per carrier
- **Cost computation:** O(flights) single pass

**Optimization:** Already efficient; polars lazy evaluation minimizes memory.

---

## Data Leakage Prevention

### Link Prediction
**Problem:** Using future information to predict future edges.

**Solution:**
1. **Time-split:** Train graph uses only months 1-9; test edges from months 10-12
2. **Test positives:** Routes in test period NOT in train period
3. **Negative sampling:** Exclude all test positives from negatives
4. **Embeddings:** Trained only on train graph
5. **Features:** Computed using only train graph topology

**Validation:** Tests verify train/test edge sets have zero overlap.

---

## Final Outputs Reference

### Embeddings + Link Prediction
| File | Content | Format |
|------|---------|--------|
| `airport_embeddings.parquet` | Node embeddings (vertex_id, code, embedding) | parquet |
| `linkpred_metrics.json` | AUC, avg precision for all models | JSON |
| `airport_embedding_neighbors.csv` | Top-10 similar airports for major hubs | CSV |
| `linkpred_top_predictions.csv` | Top 100 predicted new routes (origin, dest, score) | CSV |

### Business Analysis
| File | Content | Format |
|------|---------|--------|
| `airline_summary_metrics.parquet` | All airline metrics merged | parquet |
| `hub_concentration.parquet` | Hub dependence per airline | parquet |
| `disruption_cost_proxy.parquet` | Delay/cancellation costs per airline | parquet |
| `airline_business_metrics.csv` | Report-ready table | CSV |

### Figures
| File | Content | Purpose |
|------|---------|---------|
| `fig06_hub_dependence_by_airline.png` | Top-1 vs top-3 hub concentration | Network strategy comparison |
| `fig07_connectivity_vs_delay_scatter.png` | Hub concentration vs arrival delay | Operational efficiency |
| `fig08_link_prediction_performance.png` | Model comparison (AUC, avg precision) | Method evaluation |
| `fig09_top_route_predictions.png` | Top 20 predicted new routes | Business recommendations |

---

## Report Narrative Support

### Embeddings + Link Prediction
**Narrative:** "We use node2vec to learn airport embeddings capturing network structure. Link prediction with time-split evaluation (months 1-9 for training, 10-12 for testing) achieves AUC=X.XX. The embedding-based classifier outperforms baseline heuristics, suggesting learned representations capture route formation patterns."

**Tables:**
- Table: Top-10 similar airports for ATL, ORD, DFW (nearest neighbors)
- Table: Link prediction performance (all models)
- Table: Top 20 predicted new routes with scores

**Figures:**
- Fig 8: Model comparison (bar chart)
- Fig 9: Top predictions (horizontal bar chart)

### Business Analysis
**Narrative:** "Hub concentration varies widely: legacy carriers (AA, DL, UA) show 60-80% of flights through top-3 hubs, indicating hub-and-spoke strategies. Low-cost carriers (WN, B6) show lower concentration. Disruption cost correlates with hub dependence: airlines with higher concentration face amplified delay cascades."

**Tables:**
- Table: Airline business metrics (operational, network, cost)

**Figures:**
- Fig 6: Hub dependence by airline (grouped bar chart)
- Fig 7: Connectivity vs delay scatter (bubble chart, size = flight count)

---

## Testing Strategy

### Unit Tests
- **Embeddings:** Validate determinism, output shapes, feature computation
- **Link prediction:** Validate time-split logic, negative sampling, heuristic features
- **Business metrics:** Validate aggregations, joins, cost calculations

### Integration Tests
- Run full pipeline on toy dataset (5 airports, 100 flights)
- Verify outputs exist and have expected schemas

### Smoke Tests
- Run scripts 08-10 on real data with `top_airports_k=10` for speed
- Check manifests, output file counts, log messages

---

## How to Run WS4

```powershell
# Ensure WS1-WS3 outputs exist
ls results/networks/airport_*.parquet
ls results/analysis/airport_centrality.parquet

# Run WS4 pipeline
python scripts/08_run_embeddings_linkpred.py
python scripts/09_run_business_module.py
python scripts/10_make_all_figures.py

# Verify outputs
ls results/analysis/airport_embeddings.parquet
ls results/business/airline_summary_metrics.parquet
ls results/figures/fig06_*.png

# Check manifests
cat results/logs/08_run_embeddings_linkpred_manifest.json
cat results/logs/09_run_business_module_manifest.json
cat results/logs/10_make_all_figures_manifest.json
```

---

## Configuration Parameters

### Embeddings
```yaml
embeddings:
  dimensions: 128        # Embedding dimensionality
  walk_length: 80        # Random walk length
  num_walks: 10          # Walks per node
  window_size: 10        # Skip-gram context window
  p: 1.0                 # Return parameter (1/p for returning to previous node)
  q: 1.0                 # In-out parameter (1/q for moving outward)
```

**Tuning:**
- Increase `dimensions` for richer representations (but slower training)
- Increase `num_walks` for more stable embeddings (but longer runtime)
- Adjust `p` and `q` to bias walks: low p = DFS-like, low q = BFS-like

### Link Prediction
```yaml
link_prediction:
  time_split:
    train_months: [1, 2, 3, 4, 5, 6, 7, 8, 9]
    test_months: [10, 11, 12]
  negative_ratio: 5      # Negatives per positive
  classifier: "logreg"   # Logistic regression
```

**Tuning:**
- Adjust `negative_ratio` to balance class distribution (5:1 is standard)
- Change `train_months` / `test_months` for different time splits

### Business Metrics
```yaml
business:
  cost_per_delay_minute: 75.0    # USD per minute of arrival delay
  cost_per_cancellation: 10000.0 # USD per cancelled flight
```

**Tuning:**
- Adjust costs based on industry estimates (FAA: $74/min, DOT: $8K/cancellation)

---

## Dependencies

### New for WS4
- `gensim`: Word2Vec skip-gram training
- `scikit-learn`: Logistic regression, metrics

### Already installed (WS1-WS3)
- `polars`: Data operations
- `igraph`: Graph construction
- `numpy`: Numerical operations
- `matplotlib`: Plotting

---

## Troubleshooting

### Issue: Embeddings training is slow
**Solution:** Reduce `num_walks` or `walk_length`; use smaller subset for testing.

### Issue: Link prediction AUC is low
**Diagnosis:** Check if train/test split is too small; verify features are non-zero.
**Solution:** Increase train months; add more heuristic features.

### Issue: Business metrics don't match expectations
**Diagnosis:** Check `include_cancelled` filter; verify carrier codes.
**Solution:** Review config filters; check cleaned data schema.

### Issue: Figures are empty
**Diagnosis:** Check if input files exist; verify column names.
**Solution:** Run scripts 08-09 first; check log files for errors.

---

## Future Enhancements

### Embeddings
- **DeepWalk:** Simpler alternative (p=1, q=1)
- **struc2vec:** Role-based embeddings (similar degree nodes)
- **Graph neural networks:** More expressive representations

### Link Prediction
- **Matrix factorization:** SVD-based methods
- **Graph neural networks:** GCN, GraphSAGE for edge prediction
- **Temporal dynamics:** Incorporate time-varying features

### Business Metrics
- **Real-time monitoring:** Integrate with live flight data
- **Causality analysis:** Disentangle hub dependence from delay propagation
- **Optimization:** Recommend hub rebalancing or route additions

---

**Document Version:** 1.0  
**Last Updated:** December 24, 2025  
**Author:** WS4 Team

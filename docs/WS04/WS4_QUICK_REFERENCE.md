# WS4 Quick Reference

## Commands

```powershell
# Run WS4 pipeline
python scripts/08_run_embeddings_linkpred.py  # ~2-5 min
python scripts/09_run_business_module.py       # ~30 sec
python scripts/10_make_all_figures.py          # ~10 sec

# Run tests
pytest tests/test_embeddings_small.py -v
pytest tests/test_linkpred_time_split_toy.py -v
pytest tests/test_business_metrics_toy.py -v
```

---

## Key Files

### Modules
- `src/analysis/embeddings.py` - Node2vec + skip-gram
- `src/analysis/link_prediction.py` - Time-split + classifiers
- `src/business/airline_metrics.py` - Airline-level metrics
- `src/viz/plotting.py` - WS4 figures (extended)

### Scripts
- `scripts/08_run_embeddings_linkpred.py` - Embeddings + link prediction
- `scripts/09_run_business_module.py` - Business analysis
- `scripts/10_make_all_figures.py` - Final figures

### Outputs
- `results/analysis/airport_embeddings.parquet`
- `results/analysis/linkpred_metrics.json`
- `results/business/airline_summary_metrics.parquet`
- `results/figures/fig06_*.png` - fig09

---

## Configuration Snippets

### Embeddings (config.yaml)
```yaml
analysis:
  embeddings:
    dimensions: 128
    walk_length: 80
    num_walks: 10
    p: 1.0  # Return parameter
    q: 1.0  # In-out parameter
```

### Link Prediction
```yaml
  link_prediction:
    time_split:
      train_months: [1, 2, 3, 4, 5, 6, 7, 8, 9]
      test_months: [10, 11, 12]
    negative_ratio: 5
```

### Business Costs
```yaml
business:
  cost_per_delay_minute: 75.0
  cost_per_cancellation: 10000.0
```

---

## API Quick Reference

### Embeddings
```python
from analysis.embeddings import (
    generate_node2vec_walks,
    train_skipgram,
    write_embeddings,
    find_similar_airports,
)

# Generate walks
walks = generate_node2vec_walks(g, num_walks=10, walk_length=80, p=1.0, q=1.0, seed=42)

# Train embeddings
model = train_skipgram(walks, dimensions=128, window_size=10, seed=42)

# Write to parquet
write_embeddings(model, node_id_to_code, "embeddings.parquet")

# Find similar airports
neighbors = find_similar_airports("embeddings.parquet", ["ATL", "ORD"], top_k=10)
```

### Link Prediction
```python
from analysis.link_prediction import (
    build_month_split_graphs,
    negative_sample_non_edges,
    compute_heuristic_features,
    evaluate_classifier,
)

# Build time-split graphs
train_edges, test_edges, code_to_id = build_month_split_graphs(
    cleaned_path="flights.parquet",
    train_months=[1, 2, 3],
    test_months=[10, 11, 12],
    filters={"year": 2025},
)

# Negative sampling
negatives = negative_sample_non_edges(
    n_nodes=len(code_to_id),
    train_edges=train_edges,
    test_positives=test_edges,
    ratio=5,
    seed=42,
)

# Compute features
features = compute_heuristic_features(g_train, edge_pairs)

# Train classifier
model, metrics = evaluate_classifier(X_train, y_train, X_test, y_test, seed=42)
```

### Business Metrics
```python
from business.airline_metrics import (
    compute_airline_operational_metrics,
    compute_hub_concentration,
    compute_disruption_cost_proxy,
    merge_airline_metrics,
)

# Operational metrics
operational = compute_airline_operational_metrics("flights.parquet", filters)

# Hub concentration
hubs = compute_hub_concentration("flights.parquet", filters, top_k=[1, 3])

# Disruption cost
cost = compute_disruption_cost_proxy(
    "flights.parquet",
    filters,
    cost_per_delay_minute=75.0,
    cost_per_cancellation=10000.0,
)

# Merge all metrics
summary = merge_airline_metrics(operational, hubs, cost, centrality_df)
```

---

## Common Tasks

### Adjust embedding parameters
Edit `config/config.yaml`:
```yaml
embeddings:
  dimensions: 64     # Smaller = faster
  num_walks: 5       # Fewer walks = faster
```

### Change time split
```yaml
link_prediction:
  time_split:
    train_months: [1, 2, 3, 4, 5, 6]  # First half
    test_months: [7, 8, 9, 10, 11, 12]  # Second half
```

### Filter to specific carriers
```yaml
filters:
  carriers: ["AA", "DL", "UA"]  # Only majors
```

### Adjust cost parameters
```yaml
business:
  cost_per_delay_minute: 100.0   # Higher cost estimate
  cost_per_cancellation: 15000.0  # Higher cancellation cost
```

---

## Output Schema Reference

### airport_embeddings.parquet
| Column | Type | Description |
|--------|------|-------------|
| vertex_id | int | Node ID (0..N-1) |
| code | str | Airport code (e.g., "ATL") |
| embedding | list[float] | Embedding vector (length=dimensions) |

### linkpred_metrics.json
```json
{
  "baseline_heuristics": {
    "common_neighbors": {"auc": 0.85, "avg_precision": 0.78},
    "jaccard": {"auc": 0.82, "avg_precision": 0.75},
    "adamic_adar": {"auc": 0.87, "avg_precision": 0.80}
  },
  "embedding_classifier": {"auc": 0.92, "avg_precision": 0.88}
}
```

### airline_summary_metrics.parquet
| Column | Type | Description |
|--------|------|-------------|
| carrier | str | Airline code (e.g., "AA") |
| flight_count | int | Total flights |
| mean_dep_delay | float | Mean departure delay (minutes) |
| mean_arr_delay | float | Mean arrival delay (minutes) |
| cancellation_rate | float | Fraction of cancelled flights |
| hub_top1_pct | float | % flights through primary hub |
| hub_top3_pct | float | % flights through top-3 hubs |
| primary_hub | str | Airport code of primary hub |
| delay_cost | float | Total delay cost (USD) |
| cancellation_cost | float | Total cancellation cost (USD) |
| total_cost | float | Total disruption cost (USD) |

---

## Troubleshooting Checklist

- [ ] WS1-WS3 outputs exist (check `results/networks/`, `results/analysis/`)
- [ ] Config parameters are correct (check `config/config.yaml`)
- [ ] Cleaned data exists (check `data/cleaned/flights_2024.parquet`)
- [ ] Python environment activated (`conda activate network_science`)
- [ ] Dependencies installed (`gensim`, `scikit-learn`)
- [ ] Logs checked for errors (check `results/logs/08_*.log`)
- [ ] Manifests verified (check `results/logs/*_manifest.json`)

---

## Performance Expectations

| Operation | Time (300 airports) | Time (3000 airports) |
|-----------|---------------------|----------------------|
| Node2vec walks | 30 sec | 5 min |
| Skip-gram training | 10 sec | 1 min |
| Link prediction eval | 1 min | 10 min |
| Business metrics | 30 sec | 2 min |
| Figure generation | 10 sec | 30 sec |
| **Total WS4** | **~3 min** | **~20 min** |

---

## Version History

- **1.0** (Dec 24, 2025) - Initial WS4 implementation

---

**For full details, see:** `docs/WS04/WS4_IMPLEMENTATION_SUMMARY.md`

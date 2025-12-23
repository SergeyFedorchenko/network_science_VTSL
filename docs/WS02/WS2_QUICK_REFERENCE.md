# WS2 Quick Reference Guide

**For:** Team members running or extending WS2  
**Updated:** December 23, 2025

---

## 🚀 Quick Start

```powershell
# Run WS2 pipeline (requires WS1 outputs)
python scripts/04_run_centrality.py
python scripts/05_run_communities.py

# Verify outputs
ls results/analysis/airport_*.parquet
ls results/tables/tbl01*.csv
ls results/tables/tbl02*.csv
```

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `src/analysis/centrality.py` | Centrality computation logic |
| `src/analysis/community.py` | Leiden CPM community detection |
| `scripts/04_run_centrality.py` | Centrality pipeline script |
| `scripts/05_run_communities.py` | Community detection pipeline |
| `results/analysis/airport_centrality.parquet` | Centrality metrics output |
| `results/analysis/airport_leiden_membership.parquet` | Community memberships |

---

## ⚙️ Configuration

Edit `config/config.yaml`:

```yaml
# Global seed (affects all randomness)
seed: 42

# Centrality settings
analysis:
  centrality:
    betweenness_approx_cutoff: 20000  # Switch to approx if N > 20000
  
  # Community detection settings
  communities:
    leiden:
      resolution: 0.01    # Higher = more/smaller communities
      n_runs: 10          # Number of runs (more = better quality)
```

---

## 🔍 Output Schemas

### Centrality Output
`results/analysis/airport_centrality.parquet`

| Column | Type | Description |
|--------|------|-------------|
| vertex_id | int | Vertex index (0..N-1) |
| code | str | Airport IATA code |
| in_degree | int | Number of incoming edges |
| out_degree | int | Number of outgoing edges |
| in_strength | float | Weighted in-degree |
| out_strength | float | Weighted out-degree |
| pagerank | float | PageRank score |
| betweenness | float | Betweenness centrality |

### Community Membership
`results/analysis/airport_leiden_membership.parquet`

| Column | Type | Description |
|--------|------|-------------|
| vertex_id | int | Vertex index |
| community_id | int | Community assignment |

### Community Summary
`results/tables/community_summary_airport.csv`

| Column | Type | Description |
|--------|------|-------------|
| community_id | int | Community ID |
| size | int | Number of airports |
| top_airports | str | Top 5 airports (by PageRank) |
| top_states | str | Top 3 states (if available) |

---

## 🧪 Testing

```powershell
# Run all WS2 tests
pytest tests/test_centrality_small.py tests/test_leiden_determinism.py tests/test_ws1_integration_smoke.py -v

# Run specific test
pytest tests/test_centrality_small.py::test_compute_centrality_small_directed -v
```

---

## 🐛 Troubleshooting

### Issue: "WS1 outputs not found"
**Solution:** Run WS1 scripts first:
```powershell
python scripts/01_build_airport_network.py
```

### Issue: Betweenness takes too long
**Solution:** Lower `betweenness_approx_cutoff` in config:
```yaml
analysis:
  centrality:
    betweenness_approx_cutoff: 10000  # Use 10k instead of 20k
```

### Issue: Too many/too few communities
**Solution:** Adjust Leiden resolution:
```yaml
analysis:
  communities:
    leiden:
      resolution: 0.05  # Higher = more communities
```

### Issue: Non-deterministic results
**Solution:** Check that seed is set consistently:
```yaml
seed: 42  # Same seed = same results
```

---

## 📊 Performance Tips

1. **Betweenness:** Most expensive operation
   - Small graphs (N < 1000): Exact is fine
   - Medium graphs (1000-20000): Exact feasible but slow
   - Large graphs (N > 20000): Approximation recommended

2. **Leiden runs:** More runs = better quality but slower
   - Default 10 runs: Good balance
   - For quick testing: Use 3-5 runs
   - For final analysis: Use 10+ runs

3. **Flight graph:** Much larger than airport graph
   - Leiden on flight graph can be slow
   - Ensure flight graph is scoped (WS1 config)

---

## 🔗 Integration Points

### Inputs (from WS1)
- `results/networks/airport_nodes.parquet`
- `results/networks/airport_edges.parquet`
- `results/networks/flight_nodes.parquet` (optional)
- `results/networks/flight_edges.parquet` (optional)

### Outputs (to WS3/WS4)
- `results/analysis/airport_centrality.parquet` → WS3 robustness, WS4 business
- `results/analysis/*_leiden_membership.parquet` → WS4 figures
- `results/tables/tbl01*.csv` → Report
- `results/tables/tbl02*.csv` → Report

---

## 📝 Common Tasks

### Add a new centrality metric

1. Edit `src/analysis/centrality.py`:
```python
def compute_airport_centrality(...):
    # ... existing code ...
    
    # Add new metric
    closeness = g.closeness(mode="all")
    
    # Add to DataFrame
    df = pl.DataFrame({
        # ... existing columns ...
        "closeness": closeness,
    })
```

2. Update output schema documentation

3. Add test in `tests/test_centrality_small.py`

### Change community algorithm

1. Edit `src/analysis/community.py`:
```python
# Replace run_leiden_cpm() with your algorithm
def run_custom_algorithm(g, config):
    # Your implementation
    return membership, quality
```

2. Update `scripts/05_run_communities.py` to call new function

3. Update tests

### Add a new plot

1. Edit `src/viz/plotting.py`:
```python
def plot_new_visualization(data_df, output_path):
    # Your plotting code
    plt.savefig(output_path)
```

2. Use in script 10 (WS4)

---

## 📚 Key Functions Reference

### centrality.py

```python
# Load graph from WS1 parquet
g = load_airport_graph_from_parquet(nodes_path, edges_path, directed=True, weight_col="flight_count")

# Compute centrality
df = compute_airport_centrality(g, weight_col="weight", config=config)

# Graph structure summary
summary = compute_graph_summary(g)

# Degree distribution
degree_dist = compute_degree_distribution(g, mode="in")
```

### community.py

```python
# Single Leiden run
membership, quality = run_leiden_cpm(g, resolution=0.01, seed=42, weights="weight")

# Multi-run with best selection
best_membership, run_log = select_best_partition(g, config, weights="weight")

# Summarize airport communities
summary_df = summarize_communities_airport(nodes_df, membership_df, centrality_df)

# Summarize flight communities
summary_df = summarize_communities_flight(flight_nodes_df, membership_df)
```

---

## 🎯 Best Practices

1. **Always set seed** in config before running
2. **Check logs** in `results/logs/` for debugging
3. **Use manifests** to verify reproducibility
4. **Run tests** before pushing changes
5. **Don't modify WS1 schemas** (breaks integration)
6. **Document config changes** in commit messages

---

## 📞 Need Help?

- See full docs: `docs/WS02/WS2_IMPLEMENTATION_SUMMARY.md`
- See copilot instructions: `.vscode/copilot-instructions.md`
- Check tests for usage examples
- Review manifests in `results/logs/` for debugging

---

**Quick Ref Version:** 1.0  
**WS2 Status:** Complete

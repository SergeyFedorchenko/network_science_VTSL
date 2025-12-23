# WS2 Performance Optimization Guide

## Overview

This document describes performance optimizations for Leiden community detection on large-scale graphs (millions of nodes/edges).

## Problem

The flight graph has **5M+ nodes and 20M+ edges**, causing the Leiden algorithm to take excessive time with default parameters. The airport network (349 nodes) runs in ~1 second, but the flight network stalls.

## Solutions Implemented

### 1. **Adaptive Parameters Based on Graph Size**

The `select_best_partition()` function now automatically adjusts parameters for large graphs:

| Graph Size | Adjustments |
|------------|-------------|
| **N > 1,000,000** | Reduce runs: 10 → 3<br>Set max_iterations: unlimited → 5 |
| **N > 100,000** | Set max_iterations: unlimited → 10 |
| **N ≤ 100,000** | Use config defaults |

**Rationale**: Large graphs require more computation per iteration. Limiting iterations and runs provides faster convergence while maintaining quality.

### 2. **Convert Directed → Undirected**

For community detection, direction often doesn't affect community structure. Converting to undirected reduces:
- Memory usage
- Computation time (fewer edges after collapsing)
- Algorithm complexity

**Implementation**:
```python
# In scripts/05_run_communities.py
flight_membership, flight_run_log = select_best_partition(
    g=flight_g,
    config=config,
    weights=None,
    convert_to_undirected=True,  # Enable for flight graph
)
```

**When to use**:
- ✅ Flight networks (temporal connections, direction less critical for communities)
- ✅ Social networks (often treat connections as bidirectional)
- ❌ Citation networks (direction matters: who cites whom)
- ❌ Metabolic pathways (direction = flow)

### 3. **Enhanced Logging**

Progress logging now includes:
- Graph size (N, E)
- Per-run timing
- Adaptive parameter notifications
- Quality scores and community counts

**Example output**:
```
Large graph detected (N=5,074,460): reducing runs from 10 to 3
Large graph: setting max_iterations=5 for faster convergence
Running Leiden CPM 3 times on graph with N=5,074,460, E=20,230,470
Parameters: resolution=0.01, max_iterations=5
Run 1/3 (seed=42)...
Converting directed graph to undirected for community detection
Run 1 complete: 15234 communities, quality=125678.45, time=45.2s
```

## Performance Impact

### Estimated Speedup

| Optimization | Speedup Factor |
|--------------|----------------|
| Reduce runs (10→3) | **3.3x** |
| Limit iterations (∞→5) | **2-5x** (varies by convergence) |
| Directed→Undirected | **1.5-2x** (fewer edges) |
| **Combined** | **10-30x faster** |

### Example: 5M Node Flight Graph

| Configuration | Estimated Runtime |
|---------------|-------------------|
| Original (10 runs, unlimited iterations) | 8-12 hours |
| Optimized (3 runs, 5 iterations, undirected) | 20-60 minutes |

## Configuration Options

You can customize these parameters in `config/config.yaml`:

```yaml
analysis:
  communities:
    leiden:
      resolution: 0.01       # Higher = more communities
      n_runs: 10             # Number of runs with different seeds
      max_iterations: -1     # -1 = until convergence, or set explicit limit
```

**Recommendations**:

| Graph Size | n_runs | max_iterations |
|------------|--------|----------------|
| < 10K nodes | 10 | -1 (unlimited) |
| 10K - 100K | 10 | 20 |
| 100K - 1M | 5 | 10 |
| > 1M | 3 | 5 |

## Quality vs Speed Tradeoff

### Does optimization reduce quality?

**Short answer**: Minimal impact for most networks.

1. **Multiple runs**: 3 runs still provide good coverage of solution space
2. **Iteration limits**: Leiden converges quickly; most improvement happens in first 5-10 iterations
3. **Directed→Undirected**: Communities are typically robust to edge direction

### Validation

Compare quality scores across configurations:

```python
# Example: Test on sample of flight graph
small_sample = flight_g.subgraph(sample_nodes)

# Full optimization
membership_fast, log_fast = select_best_partition(
    small_sample, config, convert_to_undirected=True
)

# No optimization  
membership_slow, log_slow = select_best_partition(
    small_sample, config, convert_to_undirected=False
)

print(f"Fast quality: {log_fast['best_quality']:.2f}")
print(f"Slow quality: {log_slow['best_quality']:.2f}")
print(f"Speedup: {log_slow['total_time'] / log_fast['total_time']:.1f}x")
```

## Alternative Approaches (Not Implemented)

If optimizations are still insufficient:

1. **Parallel Runs**: Use `multiprocessing` to run seeds in parallel
2. **Louvain Algorithm**: Faster than Leiden but lower quality
3. **Approximate Methods**: InfoMap, Label Propagation
4. **Graph Sampling**: Analyze subset, infer full communities
5. **Incremental Detection**: Update communities as graph grows

## References

- Traag, V. A., Waltman, L., & Van Eck, N. J. (2019). From Louvain to Leiden: guaranteeing well-connected communities. *Scientific Reports*, 9(1), 5233.
- Blondel, V. D., et al. (2008). Fast unfolding of communities in large networks. *Journal of Statistical Mechanics*.

## Usage Example

Run optimized community detection:

```bash
# Activate environment
conda activate network_science

# Run script (automatically applies optimizations for large graphs)
python scripts/05_run_communities.py
```

Check logs for performance metrics:
```bash
cat results/logs/05_run_communities.log | grep "time="
```

## Summary

- **Adaptive parameters** automatically optimize for large graphs
- **Directed→Undirected** conversion preserves communities while reducing computation
- **Enhanced logging** provides visibility into performance
- **10-30x speedup** expected for million-node graphs
- **Minimal quality impact** for most network types

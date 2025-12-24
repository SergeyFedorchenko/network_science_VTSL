"""
WS3 Script 06: Network Robustness & Resilience Analysis

Tests robustness of both airport-centric and flight-centric networks against:
1. Random node removal (resilience to random disruptions)
2. Targeted removal (vulnerability to hub/critical node failures)

Outputs:
- Percolation curves (LCC size vs % nodes removed)
- Critical node rankings (which airports/flights matter most)
- Network fragmentation metrics

"""

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import igraph as ig
import numpy as np
import polars as pl
import yaml
from dataclasses import dataclass, asdict

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis.centrality import load_airport_graph_from_parquet
from utils.logging import setup_logging
from utils.paths import get_project_root
from utils.seeds import set_global_seed


# ============================================================================
# Configuration & Data Classes
# ============================================================================

@dataclass(frozen=True)
class RobustnessConfig:
    """Configuration for robustness analysis."""
    n_runs_random: int = 300              # Monte Carlo runs for random removal
    random_seed: int = 42
    connectivity_mode: str = "weak"       # "weak", "strong" for directed graphs
    targeted_strategies: Tuple[str, ...] = ("degree", "strength", "betweenness")
    k_values: Tuple[int, ...] = (1, 5, 10, 20, 50, 100)
    save_dir: str = "outputs/robustness"
    plot_figs: bool = True


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


# ============================================================================
# Network Analysis Functions
# ============================================================================

def largest_component_size(g: ig.Graph, mode: str = "weak") -> int:
    """Get size of largest connected component."""
    if g.vcount() == 0:
        return 0

    if g.is_directed():
        components = g.connected_components(mode=mode)
    else:
        components = g.connected_components(mode="weak")

    return max(components.sizes()) if components.sizes() else 0


def lcc_fraction(g: ig.Graph, n_original: int, mode: str = "weak") -> float:
    """LCC size normalized by original node count."""
    if n_original <= 0:
        return 0.0
    return largest_component_size(g, mode=mode) / float(n_original)


def rank_nodes_by_strategy(
    g: ig.Graph,
    strategy: str,
    logger: logging.Logger,
    seed: int = 42,
) -> List[int]:
    """
    Rank nodes from most to least important for targeted removal.
    
    Strategies:
    - 'degree': total degree (out + in for directed)
    - 'strength': weighted degree (if weights exist)
    - 'betweenness': betweenness centrality
    - 'random': random order (for comparison)
    """
    n = g.vcount()

    if strategy == "degree":
        if g.is_directed():
            # Use total degree (in + out)
            scores = [g.indegree(v) + g.outdegree(v) for v in range(n)]
        else:
            scores = g.degree()
        return sorted(range(n), key=lambda v: scores[v], reverse=True)

    elif strategy == "strength":
        # Weighted degree
        if "weight" in g.es.attributes():
            if g.is_directed():
                scores = [g.strength(v, weights="weight", mode="in") + 
                         g.strength(v, weights="weight", mode="out") for v in range(n)]
            else:
                scores = [g.strength(v, weights="weight") for v in range(n)]
        else:
            logger.warning("No 'weight' attribute found; falling back to degree")
            return rank_nodes_by_strategy(g, "degree", logger, seed=seed)
        return sorted(range(n), key=lambda v: scores[v], reverse=True)

    elif strategy == "betweenness":
        # Compute betweenness (can be slow for large graphs)
        if n > 10000:
            logger.warning(f"Graph has {n} nodes; betweenness may be slow")

        try:
            betweenness = g.betweenness(directed=g.is_directed())
            return sorted(range(n), key=lambda v: betweenness[v], reverse=True)
        except Exception as e:
            logger.error(f"Betweenness computation failed: {e}; using degree")
            return rank_nodes_by_strategy(g, "degree", logger, seed=seed)

    elif strategy == "random":
        # Random order for comparison baseline
        rng = np.random.default_rng(seed)
        order = list(range(n))
        rng.shuffle(order)
        return order

    else:
        logger.warning(f"Unknown strategy: {strategy}; defaulting to degree")
        return rank_nodes_by_strategy(g, "degree", logger, seed=seed)


def _build_undirected_adjlist(g: ig.Graph) -> List[List[int]]:
    """Create an undirected adjacency list (used for fast weak connectivity runs)."""
    n = g.vcount()
    adj: List[List[int]] = [[] for _ in range(n)]
    for src, dst in g.get_edgelist():
        adj[src].append(dst)
        adj[dst].append(src)
    return adj


def _percolation_curve_union_find(
    removal_order: List[int],
    adjlist: List[List[int]],
    n_vertices: int,
    sample_points: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute LCC sizes after each removal using a reverse Union-Find sweep.

    This runs in O(n + m) time for a single removal sequence, avoiding repeated
    connected-component computations.
    """
    n = n_vertices
    total_steps = len(removal_order)
    if n == 0:
        return np.array([0]), np.array([0.0])

    removal_array = np.array(removal_order, dtype=int)
    active = np.ones(n, dtype=bool)
    active[removal_array] = False  # state after all removals

    parent = np.arange(n, dtype=int)
    comp_size = np.where(active, 1, 0)
    largest = int(comp_size.max(initial=0))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(u: int, v: int) -> int:
        ru, rv = find(u), find(v)
        if ru == rv:
            return comp_size[ru]
        if comp_size[ru] < comp_size[rv]:
            ru, rv = rv, ru
        parent[rv] = ru
        comp_size[ru] += comp_size[rv]
        comp_size[rv] = 0
        return comp_size[ru]

    # Build initial components with nodes that remain after all removals
    for u in range(n):
        if not active[u]:
            continue
        for v in adjlist[u]:
            if u < v and active[v]:
                merged_size = union(u, v)
                if merged_size > largest:
                    largest = merged_size

    lcc_sizes = np.zeros(total_steps + 1, dtype=float)
    lcc_sizes[total_steps] = float(largest)

    # Re-add removed nodes in reverse to reconstruct LCC sizes
    for idx in range(total_steps - 1, -1, -1):
        node = removal_array[idx]
        active[node] = True
        parent[node] = node
        comp_size[node] = 1
        if largest < 1:
            largest = 1

        for neighbor in adjlist[node]:
            if active[neighbor]:
                merged_size = union(node, neighbor)
                if merged_size > largest:
                    largest = merged_size

        lcc_sizes[idx] = float(largest)

    if sample_points and sample_points > 0:
        sample_idx = np.linspace(0, total_steps, sample_points + 1, dtype=int)
        sample_idx = np.unique(np.clip(sample_idx, 0, total_steps))
        x = sample_idx
        lcc_sizes = lcc_sizes[sample_idx]
    else:
        x = np.arange(total_steps + 1)

    return x, lcc_sizes


def simulate_random_removal(
    g: ig.Graph,
    n_runs: int,
    seed: int,
    connectivity_mode: str,
    logger: logging.Logger,
    sample_points: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate random node removal with multiple runs.
    
    For large graphs, samples removal points instead of computing every single removal.
    
    Args:
        sample_points: For large graphs, compute LCC at this many evenly-spaced removal fractions
                      (e.g., 0%, 2%, 4%, ..., 100%). Set to -1 for all points (slow).
    
    Returns:
        x_removed: array of [0, 1, 2, ..., n]
        mean_lcc: mean LCC fraction across runs
        std_lcc: std LCC fraction across runs
    """
    if connectivity_mode == "weak":
        try:
            return _simulate_random_removal_union_find(
                g, n_runs, seed, logger, sample_points
            )
        except Exception as e:
            logger.warning(f"Fast random removal failed ({e}); falling back to slow version")

    return _simulate_random_removal_naive(
        g, n_runs, seed, connectivity_mode, logger, sample_points
    )


def _simulate_random_removal_union_find(
    g: ig.Graph,
    n_runs: int,
    seed: int,
    logger: logging.Logger,
    sample_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n0 = g.vcount()
    if n0 == 0:
        return np.array([0]), np.array([0.0]), np.array([0.0])

    rng = np.random.default_rng(seed)
    adjlist = _build_undirected_adjlist(g)

    curves: List[np.ndarray] = []
    x_ref: Optional[np.ndarray] = None

    for r in range(n_runs):
        removal_order = rng.permutation(n0).tolist()
        x, lcc_sizes = _percolation_curve_union_find(
            removal_order, adjlist, n_vertices=n0, sample_points=sample_points
        )
        lcc_frac = lcc_sizes / float(n0)

        if x_ref is None:
            x_ref = x
        curves.append(lcc_frac)

        if (r + 1) % max(1, n_runs // 10) == 0:
            logger.info(f"Random removal (fast): {r + 1}/{n_runs} runs complete")

    if x_ref is None:
        return np.array([0]), np.array([0.0]), np.array([0.0])

    curves_arr = np.vstack(curves)
    mean = curves_arr.mean(axis=0)
    std = curves_arr.std(axis=0, ddof=1) if n_runs > 1 else np.zeros_like(mean)
    return x_ref, mean, std


def _simulate_random_removal_naive(
    g: ig.Graph,
    n_runs: int,
    seed: int,
    connectivity_mode: str,
    logger: logging.Logger,
    sample_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n0 = g.vcount()
    if n0 == 0:
        return np.array([0]), np.array([0.0]), np.array([0.0])

    rng = np.random.default_rng(seed)

    # For very large graphs, use sampling
    use_sampling = n0 > 100_000 and sample_points > 0

    if use_sampling:
        # Compute at evenly-spaced removal fractions
        sample_indices = np.linspace(0, n0, sample_points + 1, dtype=int)
        sample_indices = np.unique(sample_indices)  # Remove duplicates
        curves = np.zeros((n_runs, len(sample_indices)), dtype=float)

        logger.info(f"Large graph (N={n0}): using {len(sample_indices)} sampling points instead of {n0}")
    else:
        sample_indices = np.arange(n0 + 1)
        curves = np.zeros((n_runs, n0 + 1), dtype=float)

    nodes = list(range(n0))

    for r in range(n_runs):
        order = nodes.copy()
        rng.shuffle(order)

        H = g.copy()
        removed_set = set()

        if use_sampling:
            # Compute LCC only at sample points
            curves[r, 0] = lcc_fraction(H, n_original=n0, mode=connectivity_mode)
            sample_idx = 1

            for i, original_node_id in enumerate(order, start=1):
                adjusted_id = original_node_id - sum(1 for x in removed_set if x < original_node_id)

                if adjusted_id < H.vcount():
                    H.delete_vertices([adjusted_id])
                    removed_set.add(original_node_id)

                # Record LCC only at sample points
                if sample_idx < len(sample_indices) and i >= sample_indices[sample_idx]:
                    curves[r, sample_idx] = lcc_fraction(H, n_original=n0, mode=connectivity_mode)
                    sample_idx += 1
        else:
            # Compute at every removal (slow for large graphs)
            curves[r, 0] = lcc_fraction(H, n_original=n0, mode=connectivity_mode)

            for i, original_node_id in enumerate(order, start=1):
                adjusted_id = original_node_id - sum(1 for x in removed_set if x < original_node_id)

                if adjusted_id < H.vcount():
                    H.delete_vertices([adjusted_id])
                    removed_set.add(original_node_id)

                curves[r, i] = lcc_fraction(H, n_original=n0, mode=connectivity_mode)

        if (r + 1) % max(1, n_runs // 10) == 0:
            logger.info(f"Random removal: {r + 1}/{n_runs} runs complete")

    x = sample_indices if use_sampling else np.arange(n0 + 1)
    mean = curves.mean(axis=0)
    std = curves.std(axis=0, ddof=1) if n_runs > 1 else np.zeros_like(mean)

    return x, mean, std


def simulate_targeted_removal(
    g: ig.Graph,
    removal_order: List[int],
    connectivity_mode: str,
    logger: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate targeted node removal in specified order.
    
    Returns:
        x_removed: array of [0, 1, 2, ..., n]
        lcc_frac: LCC fraction after each removal
    """
    if connectivity_mode == "weak":
        try:
            return _simulate_targeted_removal_union_find(
                g, removal_order, connectivity_mode
            )
        except Exception:
            pass  # fall back to original approach if fast path fails

    return _simulate_targeted_removal_naive(
        g, removal_order, connectivity_mode, logger
    )


def _simulate_targeted_removal_union_find(
    g: ig.Graph,
    removal_order: List[int],
    connectivity_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    n0 = g.vcount()
    if n0 == 0:
        return np.array([0]), np.array([0.0])

    adjlist = _build_undirected_adjlist(g)
    x, lcc_sizes = _percolation_curve_union_find(
        removal_order, adjlist, n_vertices=n0, sample_points=-1
    )
    lcc_frac = lcc_sizes / float(n0) if n0 > 0 else lcc_sizes
    return x, lcc_frac


def _simulate_targeted_removal_naive(
    g: ig.Graph,
    removal_order: List[int],
    connectivity_mode: str,
    logger: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray]:
    n0 = g.vcount()
    if n0 == 0:
        return np.array([0]), np.array([0.0])

    H = g.copy()
    steps = len(removal_order)
    lcc_values = np.zeros(steps + 1, dtype=float)
    lcc_values[0] = lcc_fraction(H, n_original=n0, mode=connectivity_mode)

    # Sort removal order descending and track which nodes have been removed
    ordered_nodes = sorted(removal_order, reverse=True)
    removed_set = set()

    for i, original_node_id in enumerate(ordered_nodes, start=1):
        # Adjust node_id by counting how many nodes with lower IDs were already removed
        adjusted_id = original_node_id - sum(1 for x in removed_set if x < original_node_id)

        if H.vcount() > 0 and 0 <= adjusted_id < H.vcount():
            H.delete_vertices([adjusted_id])
            removed_set.add(original_node_id)

        lcc_values[i] = lcc_fraction(H, n_original=n0, mode=connectivity_mode)

        if i % max(1, max(steps, 1) // 20) == 0:
            logger.info(f"Targeted removal: {i}/{steps} nodes removed")

    return np.arange(steps + 1), lcc_values


def critical_nodes_after_removal(
    g: ig.Graph,
    removal_order: List[int],
    k_values: List[int],
    connectivity_mode: str,
) -> Dict[int, Dict[str, Any]]:
    """
    Compute metrics for "what happens when top-k nodes are removed".
    
    Returns:
        Dict mapping k -> {lcc_size, lcc_frac, disconnected_frac}
    """
    n0 = g.vcount()
    H = g.copy()
    results: Dict[int, Dict[str, Any]] = {}
    
    ordered_nodes = sorted(removal_order, reverse=True)
    removed_set = set()
    
    # Track current position in ordered removal
    removal_idx = 0
    
    for k in sorted(set(k_values)):
        # Remove nodes up to position k
        while removal_idx < min(k, len(ordered_nodes)):
            original_node_id = ordered_nodes[removal_idx]
            adjusted_id = original_node_id - sum(1 for x in removed_set if x < original_node_id)
            
            if H.vcount() > 0 and 0 <= adjusted_id < H.vcount():
                H.delete_vertices([adjusted_id])
                removed_set.add(original_node_id)
            
            removal_idx += 1
        
        lcc = largest_component_size(H, mode=connectivity_mode)
        remaining = H.vcount()
        
        results[k] = {
            "lcc_size": lcc,
            "lcc_frac_of_original": lcc / float(n0) if n0 > 0 else 0.0,
            "disconnected_frac_of_original": 1.0 - (lcc / float(n0) if n0 > 0 else 0.0),
            "nodes_removed": k,
            "nodes_remaining": remaining,
        }
    
    return results


# ============================================================================
# Visualization (Optional - requires matplotlib)
# ============================================================================

def plot_robustness_curves(
    x: np.ndarray,
    curves_dict: Dict[str, Dict[str, Any]],
    output_path: Path,
    graph_name: str,
) -> None:
    """Plot robustness curves (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"matplotlib not available; skipping plot for {graph_name}")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Random removal
    if "random" in curves_dict:
        mean = curves_dict["random"]["mean"]
        std = curves_dict["random"]["std"]
        ax1.plot(x, mean, "b-", label="Mean LCC", linewidth=2)
        ax1.fill_between(x, mean - std, mean + std, alpha=0.2, color="blue")
        ax1.set_xlabel("# Nodes Removed")
        ax1.set_ylabel("LCC Fraction")
        ax1.set_title(f"{graph_name}: Random Removal")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Targeted removals
    if "targeted" in curves_dict:
        for strategy, data in curves_dict["targeted"].items():
            ax2.plot(x, data["lcc_frac"], label=strategy, linewidth=2)
        ax2.set_xlabel("# Nodes Removed")
        ax2.set_ylabel("LCC Fraction")
        ax2.set_title(f"{graph_name}: Targeted Removal")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {output_path}")


# ============================================================================
# Main Robustness Analysis
# ============================================================================

def run_airport_robustness(
    g: ig.Graph,
    nodes_df: pl.DataFrame,
    cfg: RobustnessConfig,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Run robustness analysis on airport network."""
    logger.info(f"\n{'='*80}")
    logger.info("Airport Network Robustness Analysis")
    logger.info(f"{'='*80}")
    
    n0 = g.vcount()
    logger.info(f"Graph: N={n0} airports, E={g.ecount()} routes")
    
    results = {"airport": {}}
    
    # Random removal
    logger.info("Running random removal simulations...")
    x, mean, std = simulate_random_removal(
        g, n_runs=cfg.n_runs_random, seed=cfg.random_seed,
        connectivity_mode=cfg.connectivity_mode, logger=logger
    )
    results["airport"]["random"] = {
        "x_removed": x.tolist(),
        "mean_lcc_frac": mean.tolist(),
        "std_lcc_frac": std.tolist(),
    }
    logger.info(f"Random removal complete: LCC drops from {mean[0]:.3f} to {mean[-1]:.3f}")
    
    # Targeted removals
    results["airport"]["targeted"] = {}
    for strategy in cfg.targeted_strategies:
        logger.info(f"Computing targeted removal ({strategy})...")
        removal_order = rank_nodes_by_strategy(g, strategy, logger, seed=cfg.random_seed)
        x_t, lcc_t = simulate_targeted_removal(
            g, removal_order, cfg.connectivity_mode, logger
        )
        
        critical = critical_nodes_after_removal(
            g, removal_order, list(cfg.k_values), cfg.connectivity_mode
        )
        
        results["airport"]["targeted"][strategy] = {
            "x_removed": x_t.tolist(),
            "lcc_frac": lcc_t.tolist(),
            "critical_k": critical,
        }
        logger.info(f"Targeted {strategy}: LCC drops from {lcc_t[0]:.3f} to {lcc_t[-1]:.3f}")
        
        # Save top critical nodes
        if "code" in g.vs.attributes():
            top_nodes = removal_order[:20]
            codes = [g.vs[v]["code"] for v in top_nodes]
            logger.info(f"Top {len(codes)} critical airports ({strategy}): {codes}")
    
    return results


def load_flight_graph_from_parquet(
    nodes_path: Path,
    edges_path: Path,
    directed: bool = True,
) -> ig.Graph:
    """Load flight graph from WS1 outputs."""
    nodes_df = pl.read_parquet(nodes_path)
    edges_df = pl.read_parquet(edges_path)
    
    # Get vertex IDs
    if "vertex_id" in nodes_df.columns:
        vertex_id_col = "vertex_id"
    elif "flight_id" in nodes_df.columns:
        vertex_id_col = "flight_id"
    else:
        raise ValueError(f"No vertex_id or flight_id found")
    
    n_vertices = len(nodes_df)
    edge_list = list(zip(edges_df["src_id"].to_list(), edges_df["dst_id"].to_list()))
    
    g = ig.Graph(n=n_vertices, edges=edge_list, directed=directed)
    return g


def run_flight_robustness(
    g: ig.Graph,
    cfg: RobustnessConfig,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Run robustness analysis on flight network."""
    logger.info(f"\n{'='*80}")
    logger.info("Flight Network Robustness Analysis")
    logger.info(f"{'='*80}")
    
    n0 = g.vcount()
    logger.info(f"Graph: N={n0:,} flights, E={g.ecount():,} connections")
    
    # For very large flight networks, dramatically reduce computational load
    if n0 > 1_000_000:
        logger.warning(f"Flight graph extremely large (N={n0:,})")
        logger.warning("Using aggressive sampling to keep analysis tractable")
        
        # Use only degree for targeted (betweenness/strength too slow)
        strategies = ["degree"]
        
        # Use MUCH fewer random runs
        n_runs = min(cfg.n_runs_random, 10)
        sample_points = 30  # Only compute at 30 evenly-spaced points
        
        logger.info(f"Parameters: {n_runs} random runs, {sample_points} sample points")
    elif n0 > 100_000:
        logger.warning(f"Flight graph large (N={n0:,}); using degree only")
        strategies = ["degree"]
        n_runs = min(cfg.n_runs_random, 50)
        sample_points = 40
    else:
        strategies = list(cfg.targeted_strategies)
        n_runs = cfg.n_runs_random
        sample_points = -1  # All points
    
    results = {"flight": {}}
    
    # Random removal
    logger.info(f"Running random removal ({n_runs} runs with {sample_points if sample_points > 0 else 'all'} sample points)...")
    x, mean, std = simulate_random_removal(
        g, n_runs=n_runs, seed=cfg.random_seed,
        connectivity_mode=cfg.connectivity_mode, logger=logger,
        sample_points=sample_points
    )
    results["flight"]["random"] = {
        "x_removed": x.tolist(),
        "mean_lcc_frac": mean.tolist(),
        "std_lcc_frac": std.tolist(),
    }
    logger.info(f"Random removal complete: LCC mean {mean[0]:.3f} → {mean[-1]:.3f}")
    
    # Targeted removals (only degree, only top nodes)
    results["flight"]["targeted"] = {}
    for strategy in strategies:
        logger.info(f"Computing targeted removal ({strategy})...")
        removal_order = rank_nodes_by_strategy(g, strategy, logger, seed=cfg.random_seed)
        
        # For very large graphs, only remove top fraction
        if n0 > 1_000_000:
            removal_order = removal_order[:int(0.1 * n0)]  # Only top 10%
            logger.info(f"Only testing removal of top 10% ({len(removal_order):,} nodes)")
        
        x_t, lcc_t = simulate_targeted_removal(
            g, removal_order, cfg.connectivity_mode, logger
        )
        
        results["flight"]["targeted"][strategy] = {
            "x_removed": x_t.tolist(),
            "lcc_frac": lcc_t.tolist(),
        }
        logger.info(f"Targeted {strategy}: LCC mean {lcc_t[0]:.3f} → {lcc_t[-1]:.3f}")
    
    return results


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution."""
    root = get_project_root()
    log_dir = root / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(log_dir / "06_run_robustness.log")
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("WS3 Script 06: Network Robustness Analysis")
    logger.info("=" * 80)
    
    # Load config
    config_path = root / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded config from {config_path}")
    set_global_seed(config.get("seed", 42))
    
    # Create robustness config
    robustness_config = config.get("analysis", {}).get("robustness", {})
    
    # Map strategy names: config has "random", "highest_degree", "highest_betweenness"
    # Script uses "degree", "strength", "betweenness"
    config_strategies = robustness_config.get("strategies", ["degree", "betweenness"])
    targeted_strategies = []
    for s in config_strategies:
        if s == "random":
            continue  # random is not a targeted strategy
        elif s == "highest_degree":
            targeted_strategies.append("degree")
        elif s == "highest_betweenness":
            targeted_strategies.append("betweenness")
        elif s in ["degree", "betweenness", "strength"]:
            targeted_strategies.append(s)
    
    rob_cfg = RobustnessConfig(
        n_runs_random=robustness_config.get("random_trials", 30),  # FIX: use correct key
        random_seed=config.get("seed", 42),
        connectivity_mode=robustness_config.get("mode", "weak"),
        targeted_strategies=tuple(targeted_strategies),
        k_values=tuple(robustness_config.get("k_values", [1, 5, 10, 20, 50])),
        save_dir=str(root / "results" / "analysis"),  # FIX: use standard output dir
    )
    
    # Create output directory
    Path(rob_cfg.save_dir).mkdir(parents=True, exist_ok=True)
    
    # ========================================
    # Airport Network Robustness
    # ========================================
    networks_dir = root / "results" / "networks"
    airport_nodes_path = networks_dir / "airport_nodes.parquet"
    airport_edges_path = networks_dir / "airport_edges.parquet"
    
    summary = {
        "script": "06_run_robustness.py",
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "config": {
            "n_runs_random": rob_cfg.n_runs_random,
            "connectivity_mode": rob_cfg.connectivity_mode,
            "strategies": rob_cfg.targeted_strategies,
        },
        "results": {}
    }
    
    if airport_nodes_path.exists() and airport_edges_path.exists():
        logger.info(f"Loading airport graph from {airport_nodes_path}")
        
        directed = config.get("airport_network", {}).get("directed", True)
        weight_col = config.get("airport_network", {}).get("edge_weight", "flight_count")
        
        g_airport = load_airport_graph_from_parquet(
            airport_nodes_path, airport_edges_path,
            directed=directed, weight_col=weight_col
        )
        
        nodes_df = pl.read_parquet(airport_nodes_path)
        airport_results = run_airport_robustness(g_airport, nodes_df, rob_cfg, logger)
        summary["results"].update(airport_results)
    else:
        logger.warning(f"Airport network not found at {airport_nodes_path}")
    
    # ========================================
    # Flight Network Robustness
    # ========================================
    flight_nodes_path = networks_dir / "flight_nodes.parquet"
    flight_edges_path = networks_dir / "flight_edges.parquet"
    
    if flight_nodes_path.exists() and flight_edges_path.exists():
        logger.info(f"Loading flight graph from {flight_nodes_path}")
        
        g_flight = load_flight_graph_from_parquet(
            flight_nodes_path, flight_edges_path, directed=True
        )
        
        flight_results = run_flight_robustness(g_flight, rob_cfg, logger)
        summary["results"].update(flight_results)
    else:
        logger.warning(f"Flight network not found at {flight_nodes_path}")
    
    # ========================================
    # Save Results
    # ========================================
    
    # Create output directories
    analysis_dir = root / "results" / "analysis"
    tables_dir = root / "results" / "tables"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate robustness_curves.parquet (figure-ready table)
    logger.info("Generating robustness_curves.parquet...")
    curves_data = []
    
    for graph_name, graph_results in summary["results"].items():
        # Determine n0 for this graph
        if graph_name == "airport" and 'g_airport' in locals():
            n0 = g_airport.vcount()
        elif graph_name == "flight" and 'g_flight' in locals():
            n0 = g_flight.vcount()
        else:
            n0 = 1  # fallback
        
        # Random removal
        if "random" in graph_results:
            rand = graph_results["random"]
            x_vals = rand["x_removed"]
            mean_vals = rand["mean_lcc_frac"]
            std_vals = rand["std_lcc_frac"]
            
            for x, mean, std in zip(x_vals, mean_vals, std_vals):
                curves_data.append({
                    "graph": graph_name,
                    "strategy": "random",
                    "fraction_removed": float(x) / n0 if n0 > 0 else 0.0,
                    "lcc_fraction": float(mean),
                    "lcc_std": float(std) if std is not None else None,
                })
        
        # Targeted removal
        if "targeted" in graph_results:
            for strategy, strat_data in graph_results["targeted"].items():
                x_vals = strat_data["x_removed"]
                lcc_vals = strat_data["lcc_frac"]
                
                for x, lcc in zip(x_vals, lcc_vals):
                    curves_data.append({
                        "graph": graph_name,
                        "strategy": f"targeted_{strategy}",
                        "fraction_removed": float(x) / n0 if n0 > 0 else 0.0,
                        "lcc_fraction": float(lcc),
                        "lcc_std": None,
                    })
    
    if curves_data:
        curves_df = pl.DataFrame(curves_data)
        curves_path = analysis_dir / "robustness_curves.parquet"
        curves_df.write_parquet(curves_path)
        logger.info(f"Wrote {curves_path} ({len(curves_data)} rows)")
    
    # Generate robustness_critical_nodes.csv
    logger.info("Generating robustness_critical_nodes.csv...")
    critical_nodes_data = []
    
    for graph_name, graph_results in summary["results"].items():
        if "targeted" not in graph_results:
            continue
        
        # Get the graph object
        g = None
        if graph_name == "airport" and 'g_airport' in locals():
            g = g_airport
        elif graph_name == "flight" and 'g_flight' in locals():
            g = g_flight
        
        if g is None:
            continue
        
        for strategy, strat_data in graph_results["targeted"].items():
            if "critical_k" not in strat_data:
                continue
            
            # Get removal order for this strategy
            removal_order = rank_nodes_by_strategy(g, strategy, logger, seed=rob_cfg.random_seed)
            
            for k_str, metrics in strat_data["critical_k"].items():
                k = int(k_str)
                top_k_nodes = removal_order[:min(k, len(removal_order))]
                
                # Get node codes if available
                if "code" in g.vs.attributes():
                    codes = [g.vs[v]["code"] for v in top_k_nodes[:20]]
                else:
                    codes = [str(v) for v in top_k_nodes[:20]]
                
                critical_nodes_data.append({
                    "graph": graph_name,
                    "strategy": strategy,
                    "k": k,
                    "nodes": ",".join(codes),
                    "lcc_frac_after_removal": float(metrics["lcc_frac_of_original"]),
                })
    
    if critical_nodes_data:
        critical_df = pl.DataFrame(critical_nodes_data)
        critical_path = tables_dir / "robustness_critical_nodes.csv"
        critical_df.write_csv(critical_path)
        logger.info(f"Wrote {critical_path} ({len(critical_nodes_data)} rows)")
    
    # Save summary JSON
    output_path = analysis_dir / "robustness_summary.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved results to {output_path}")
    
    # Write manifest
    manifest = {
        "script": "06_run_robustness.py",
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "config_snapshot": {
            "seed": config.get("seed"),
            "robustness": config.get("analysis", {}).get("robustness", {}),
        },
        "inputs": {},
        "outputs": {
            "curves_parquet": str(analysis_dir / "robustness_curves.parquet"),
            "critical_nodes_csv": str(tables_dir / "robustness_critical_nodes.csv"),
            "summary_json": str(output_path),
        },
        "results_summary": {},
    }
    
    # Add input fingerprints
    if airport_nodes_path.exists():
        manifest["inputs"]["airport_network"] = {
            "nodes_path": str(airport_nodes_path),
            "edges_path": str(airport_edges_path),
            "n_nodes": g_airport.vcount() if 'g_airport' in locals() else 0,
            "n_edges": g_airport.ecount() if 'g_airport' in locals() else 0,
        }
    
    if flight_nodes_path.exists():
        manifest["inputs"]["flight_network"] = {
            "nodes_path": str(flight_nodes_path),
            "edges_path": str(flight_edges_path),
            "n_nodes": g_flight.vcount() if 'g_flight' in locals() else 0,
            "n_edges": g_flight.ecount() if 'g_flight' in locals() else 0,
        }
    
    # Add results summary
    for graph_name in summary["results"].keys():
        manifest["results_summary"][graph_name] = {
            "strategies_run": list(summary["results"][graph_name].get("targeted", {}).keys()),
        }
    
    manifest_path = log_dir / "06_run_robustness_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Wrote manifest: {manifest_path}")
    
    logger.info("=" * 80)
    logger.info("WS3 Script 06: Complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

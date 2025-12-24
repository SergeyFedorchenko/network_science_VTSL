"""
WS3 Script 07: Delay Propagation & Cascade Analysis

Models delays as a contagion spreading through the flight network.
Simulates how delays cascade through passenger connections and aircraft rotations.

Outputs:
- Cascade size distributions (heavy-tail pattern expected)
- Super-spreader flights (highest influence on downstream delays)
- Scenario analysis (e.g., weather delay at major hub)
- Example cascade trajectories

"""

import json
import logging
import subprocess
import sys
from bisect import bisect_left
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Callable, Sequence

import igraph as ig
import numpy as np
import polars as pl
import yaml
from dataclasses import dataclass, asdict

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.logging import setup_logging
from utils.paths import get_project_root
from utils.seeds import set_global_seed


# ============================================================================
# Configuration & Data Classes
# ============================================================================

@dataclass(frozen=True)
class DelayConfig:
    """Configuration for delay propagation analysis."""
    p_pax: float = 0.30              # Transmission probability for passenger connections
    p_tail: float = 0.60             # Transmission probability for same-aircraft (higher)
    min_conn_minutes: int = 30       # Minimum feasible connection time
    max_conn_minutes: int = 240      # Maximum feasible connection time
    n_runs: int = 500                # Monte Carlo simulation runs
    rng_seed: int = 123
    delay_threshold_minutes: float = 1.0  # What counts as "delayed"
    save_dir: str = "outputs/delay"


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
# Flight Connection Graph Construction
# ============================================================================

def build_flight_connection_graph(
    flights_df: pl.DataFrame,
    config: DelayConfig,
    logger: logging.Logger,
) -> ig.Graph:
    """
    Build directed flight-connection graph.
    
    Nodes: flights
    Edges: flight u → flight v if delay in u can propagate to v
    Edge types:
      - 'pax': passenger connection (arrive at airport, depart within time window)
      - 'tail': same aircraft (sequential flights by same tail number)
    """
    logger.info("Building flight connection graph for delay propagation")
    # Keep a stable vertex id for graph mapping; avoid duplicating if already present
    if "vertex_id" in flights_df.columns:
        flights = flights_df
    else:
        flights = flights_df.with_row_index(name="vertex_id")
    
    # Check for required columns
    required_cols = ["dep_ts", "arr_ts", "origin", "dest"]
    for col in required_cols:
        if col not in flights.columns:
            raise ValueError(f"Missing required column: {col}. Available: {flights.columns}")
    
    n_flights = len(flights)
    logger.info(f"Processing {n_flights:,} flights")
    
    # Create graph
    g = ig.Graph(directed=True)
    g.add_vertices(n_flights)
    
    # Add flight metadata to vertices
    if "vertex_id" in flights.columns:
        g.vs["vertex_id"] = flights["vertex_id"].to_list()
    if "flight_id" in flights.columns:
        g.vs["flight_id"] = flights["flight_id"].to_list()
    if "carrier" in flights.columns:
        g.vs["carrier"] = flights["carrier"].to_list()
    if "origin" in flights.columns:
        g.vs["origin"] = flights["origin"].to_list()
    if "dest" in flights.columns:
        g.vs["dest"] = flights["dest"].to_list()
    if "dep_ts" in flights.columns:
        g.vs["dep_ts"] = flights["dep_ts"].to_list()
    if "arr_ts" in flights.columns:
        g.vs["arr_ts"] = flights["arr_ts"].to_list()
    
    edge_list = []
    edge_types = []
    
    # ========================================
    # Passenger Connection Edges (Time Window Join)
    # ========================================
    logger.info("Building passenger connection edges (arriving airport → departing flights)...")
    
    min_conn_dt = config.min_conn_minutes
    max_conn_dt = config.max_conn_minutes
    
    # Filter flights with valid timestamps
    flights_with_ts = flights.filter(
        pl.col("arr_ts").is_not_null() & pl.col("dep_ts").is_not_null()
    )
    
    # Convert to list for direct indexing (more efficient)
    origins = flights_with_ts["origin"].to_list()
    dests = flights_with_ts["dest"].to_list()
    dep_times = flights_with_ts["dep_ts"].to_list()
    arr_times = flights_with_ts["arr_ts"].to_list()
    vertex_ids = flights_with_ts["vertex_id"].to_list()
    
    # Build indices: airport -> arrivals/departures sorted by time for fast window queries
    arrivals_by_airport: Dict[str, List[Tuple[int, Any]]] = {}
    departures_by_airport: Dict[str, Tuple[List[Any], List[int]]] = {}
    
    for idx in range(len(flights_with_ts)):
        vid = vertex_ids[idx]
        arr_airport = dests[idx]
        dep_airport = origins[idx]
        
        arrivals_by_airport.setdefault(arr_airport, []).append((vid, arr_times[idx]))
        departures_by_airport.setdefault(dep_airport, ([], []))
        departures_by_airport[dep_airport][0].append(dep_times[idx])  # times
        departures_by_airport[dep_airport][1].append(vid)            # ids
    
    # Sort departures and arrivals by time for efficient window queries
    for airport, (times, ids) in departures_by_airport.items():
        sorted_pairs = sorted(zip(times, ids), key=lambda x: x[0])
        if sorted_pairs:
            dep_times_sorted, dep_ids_sorted = zip(*sorted_pairs)
            departures_by_airport[airport] = (list(dep_times_sorted), list(dep_ids_sorted))
        else:
            departures_by_airport[airport] = ([], [])
    for airport, arrs in arrivals_by_airport.items():
        arrivals_by_airport[airport] = sorted(arrs, key=lambda x: x[1])
    
    pax_edges = 0
    
    # For each airport, connect arriving to departing flights using binary search on departures
    for airport, arr_list in arrivals_by_airport.items():
        if airport not in departures_by_airport:
            continue
        
        dep_times_sorted, dep_ids_sorted = departures_by_airport[airport]
        if not dep_times_sorted:
            continue
        
        for arr_id, arr_time in arr_list:
            conn_min_time = arr_time + timedelta(minutes=min_conn_dt)
            conn_max_time = arr_time + timedelta(minutes=max_conn_dt)
            
            start_idx = bisect_left(dep_times_sorted, conn_min_time)
            for j in range(start_idx, len(dep_times_sorted)):
                dep_time = dep_times_sorted[j]
                if dep_time > conn_max_time:
                    break
                dep_id = dep_ids_sorted[j]
                if arr_id == dep_id:
                    continue
                edge_list.append((arr_id, dep_id))
                edge_types.append("pax")
                pax_edges += 1
    
    logger.info(f"Passenger connection edges: {pax_edges:,}")
    
    # ========================================
    # Tail Sequence Edges (Aircraft Rotations)
    # ========================================
    logger.info("Building tail sequence edges (same aircraft)...")
    
    tail_edges = 0
    if "tail" in flights.columns:
        flights_with_tail = flights_with_ts.filter(pl.col("tail").is_not_null()).sort("dep_ts")
        
        # Group by tail number
        for group in flights_with_tail.group_by("tail", maintain_order=True):
            # Polars may yield either a DataFrame or (key, DataFrame) tuple depending on version
            if isinstance(group, tuple) and len(group) == 2:
                _, tail_df = group
            else:
                tail_df = group
            tail_df = tail_df.sort("dep_ts")
            flight_ids_list = tail_df["vertex_id"].to_list()
            arr_times_list = tail_df["arr_ts"].to_list()
            dep_times_list = tail_df["dep_ts"].to_list()
            
            # Connect consecutive flights for same aircraft (ordered by dep_ts)
            for i in range(len(flight_ids_list) - 1):
                u_id = flight_ids_list[i]
                v_id = flight_ids_list[i + 1]
                u_arr = arr_times_list[i]
                v_dep = dep_times_list[i + 1]
                
                if u_arr is not None and v_dep is not None and u_arr < v_dep:
                    edge_list.append((u_id, v_id))
                    edge_types.append("tail")
                    tail_edges += 1
    else:
        logger.warning("No 'tail' column found; skipping tail sequence edges")
    
    logger.info(f"Tail sequence edges: {tail_edges:,}")
    
    # Add edges to graph
    if edge_list:
        g.add_edges(edge_list)
        g.es["edge_type"] = edge_types
    
    logger.info(f"Flight connection graph built: N={g.vcount():,}, E={g.ecount():,}")
    return g


# ============================================================================
# Delay Cascade Simulation
# ============================================================================

def _build_transition_lists(
    g: ig.Graph,
    p_pax: float,
    p_tail: float,
) -> List[List[Tuple[int, float]]]:
    """Precompute adjacency with transmission probabilities for faster cascades."""
    n = g.vcount()
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    has_edge_type = "edge_type" in g.es.attribute_names()
    edge_types = g.es["edge_type"] if has_edge_type else None

    for eid, (src, tgt) in enumerate(g.get_edgelist()):
        etype = edge_types[eid] if edge_types is not None else "pax"
        p = p_tail if etype == "tail" else p_pax
        adj[src].append((tgt, p))
    return adj


def simulate_delay_cascade(
    g: ig.Graph,
    seed_flights: Sequence[int],
    p_pax: float,
    p_tail: float,
    rng: np.random.Generator,
    logger: logging.Logger,
    transitions: Optional[List[List[Tuple[int, float]]]] = None,
) -> Tuple[Set[int], Dict[int, Optional[int]]]:
    """
    Simulate delay cascade using independent cascade model.
    
    A flight is "delayed" if:
    1. It's in the seed set (exogenous shock), OR
    2. One of its predecessors was delayed AND transmission succeeds
    
    Returns:
        delayed_flights: set of flight IDs that became delayed
        parent_map: mapping flight_id -> parent_flight_id (for tracing cascade)
    """
    delayed: Set[int] = set(seed_flights)
    parent: Dict[int, Optional[int]] = {int(f): None for f in seed_flights}

    if transitions is None:
        transitions = _build_transition_lists(g, p_pax, p_tail)

    queue: deque[int] = deque(int(f) for f in seed_flights)

    while queue:
        u = queue.popleft()
        for v, p in transitions[u]:
            if v in delayed:
                continue
            if rng.random() < p:
                delayed.add(v)
                parent[v] = u
                queue.append(v)

    return delayed, parent


def compute_cascade_statistics(
    cascade_sizes: List[int],
    n_flights: int,
) -> Dict[str, Any]:
    """Compute statistics on cascade size distribution."""
    cascade_array = np.array(cascade_sizes)
    
    return {
        "n_cascades": len(cascade_sizes),
        "mean_cascade_size": float(cascade_array.mean()),
        "median_cascade_size": float(np.median(cascade_array)),
        "std_cascade_size": float(cascade_array.std(ddof=1)) if len(cascade_sizes) > 1 else 0.0,
        "min_cascade_size": int(cascade_array.min()),
        "max_cascade_size": int(cascade_array.max()),
        "p50": float(np.percentile(cascade_array, 50)),
        "p90": float(np.percentile(cascade_array, 90)),
        "p99": float(np.percentile(cascade_array, 99)),
        "mean_fraction_delayed": float(cascade_array.mean() / n_flights),
        "p99_fraction_delayed": float(np.percentile(cascade_array, 99) / n_flights),
    }


def identify_super_spreaders(
    g: ig.Graph,
    n_simulations: int,
    p_pax: float,
    p_tail: float,
    rng: np.random.Generator,
    logger: logging.Logger,
    transitions: Optional[List[List[Tuple[int, float]]]] = None,
) -> Dict[int, float]:
    """
    Identify 'super-spreader' flights: those that cause largest cascades when seeded.
    
    For each flight, simulate seeding it with delay and measure cascade size.
    """
    logger.info(f"Computing super-spreader scores for {g.vcount():,} flights (this may take a while)...")
    
    n_flights = g.vcount()
    influence_scores: Dict[int, float] = {f: 0.0 for f in range(n_flights)}
    if transitions is None:
        transitions = _build_transition_lists(g, p_pax, p_tail)
    
    # Sample flights to compute influence (too many to do all)
    sample_size = min(n_flights, max(1000, int(0.01 * n_flights)))
    sampled_flights = list(rng.choice(n_flights, size=sample_size, replace=False))
    
    logger.info(f"Sampling {len(sampled_flights):,} flights for influence computation")
    
    for idx, flight_id in enumerate(sampled_flights):
        if (idx + 1) % max(1, len(sampled_flights) // 20) == 0:
            logger.info(f"Super-spreader computation: {idx + 1}/{len(sampled_flights)}")
        
        # Run multiple cascades with this flight as seed
        cascade_sizes = []
        for _ in range(n_simulations):
            delayed, _ = simulate_delay_cascade(
                g, [flight_id], p_pax, p_tail, rng, logger, transitions=transitions
            )
            cascade_sizes.append(len(delayed))
        
        # Influence = mean cascade size
        influence_scores[flight_id] = float(np.mean(cascade_sizes))
    
    return influence_scores


# ============================================================================
# Scenario Analysis
# ============================================================================

def seed_flights_from_airport_at_time(
    flights_df: pl.DataFrame,
    airport_code: str,
    hour: int,
    minute: int = 0,
) -> List[int]:
    """
    Get flight IDs departing from an airport at a specific time.
    
    Returns: list of flight indices (0-based)
    """
    matching = flights_df.filter(
        (pl.col("origin") == airport_code) &
        (pl.col("dep_ts").dt.hour() == hour) &
        (pl.col("dep_ts").dt.minute() == minute)
    )
    
    if "vertex_id" in matching.columns:
        return matching["vertex_id"].to_list()
    if "flight_id" in matching.columns:
        return matching["flight_id"].to_list()
    # Fallback to positional indices within filtered set (rare)
    return list(range(len(matching)))


def seed_all_flights_from_airport(
    flights_df: pl.DataFrame,
    airport_code: str,
) -> List[int]:
    """Get ALL flight IDs departing from an airport."""
    matching = flights_df.filter(pl.col("origin") == airport_code)
    
    if "vertex_id" in matching.columns:
        return matching["vertex_id"].to_list()
    if "flight_id" in matching.columns:
        return matching["flight_id"].to_list()
    return list(range(len(matching)))


# ============================================================================
# Main Analysis
# ============================================================================

def run_delay_propagation(
    flights_df: pl.DataFrame,
    g: ig.Graph,
    cfg: DelayConfig,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Run full delay propagation analysis."""
    logger.info(f"\n{'='*80}")
    logger.info("Delay Propagation Analysis")
    logger.info(f"{'='*80}")
    
    n_flights = g.vcount()
    logger.info(f"Flight network: N={n_flights:,} flights, E={g.ecount():,} connections")
    logger.info(f"Configuration: p_pax={cfg.p_pax}, p_tail={cfg.p_tail}, n_runs={cfg.n_runs}")
    
    rng = np.random.default_rng(cfg.rng_seed)
    transitions = _build_transition_lists(g, cfg.p_pax, cfg.p_tail)
    results = {}
    
    # ========================================
    # Baseline Random Shocks
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("Baseline: Random Initial Shocks")
    logger.info("=" * 80)
    
    cascade_sizes = []
    for run_id in range(cfg.n_runs):
        # Random seed: 1% of flights initially delayed
        k_seeds = max(1, int(0.01 * n_flights))
        seeds = list(rng.choice(n_flights, size=k_seeds, replace=False))
        
        delayed, _ = simulate_delay_cascade(
            g, seeds, cfg.p_pax, cfg.p_tail, rng, logger, transitions=transitions
        )
        cascade_sizes.append(len(delayed))
        
        if (run_id + 1) % max(1, cfg.n_runs // 10) == 0:
            logger.info(f"Random shocks: {run_id + 1}/{cfg.n_runs} runs")
    
    baseline_stats = compute_cascade_statistics(cascade_sizes, n_flights)
    results["baseline_random_shocks"] = {
        "description": "1% of flights randomly delayed",
        "statistics": baseline_stats,
        "cascade_sizes": cascade_sizes,
    }
    
    logger.info(f"Baseline cascade size: mean={baseline_stats['mean_cascade_size']:.0f}, "
                f"p99={baseline_stats['p99']:.0f} ({baseline_stats['p99_fraction_delayed']*100:.1f}% of network)")
    
    # ========================================
    # Super-Spreader Analysis
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("Super-Spreader Analysis")
    logger.info("=" * 80)
    
    # Only compute for smaller graphs or sample
    if n_flights <= 100_000:
        n_super_sims = 5
        logger.info(f"Computing super-spreader scores ({n_super_sims} simulations per flight)")
        influence = identify_super_spreaders(
            g, n_super_sims, cfg.p_pax, cfg.p_tail, rng, logger, transitions=transitions
        )
        
        # Top 20 super-spreaders
        top_spreaders = sorted(influence.items(), key=lambda x: x[1], reverse=True)[:20]
        logger.info("Top 20 super-spreader flights:")
        for rank, (flight_id, score) in enumerate(top_spreaders, 1):
            vertex_data = g.vs[flight_id]
            info = f"Flight {flight_id}"
            if "carrier" in vertex_data.attributes():
                info += f" ({vertex_data['carrier']})"
            if "origin" in vertex_data.attributes() and "dest" in vertex_data.attributes():
                info += f": {vertex_data['origin']} → {vertex_data['dest']}"
            logger.info(f"  {rank}. {info}: influence={score:.0f}")
        
        results["super_spreaders"] = {
            "top_20": [
                {
                    "rank": rank,
                    "flight_id": int(fid),
                    "influence_score": float(score)
                }
                for rank, (fid, score) in enumerate(top_spreaders, 1)
            ]
        }
    else:
        logger.warning(f"Graph too large ({n_flights:,}); skipping super-spreader analysis")
    
    # ========================================
    # Scenario: Major Hub Disruption
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("Scenario: Major Hub Disruption (Atlanta - ATL at 06:00)")
    logger.info("=" * 80)
    
    atl_flights = seed_flights_from_airport_at_time(flights_df, "ATL", hour=6, minute=0)
    
    if atl_flights:
        logger.info(f"Found {len(atl_flights)} flights departing ATL at 06:00")
        
        scenario_cascades = []
        for run_id in range(cfg.n_runs):
            delayed, _ = simulate_delay_cascade(
                g, atl_flights, cfg.p_pax, cfg.p_tail, rng, logger, transitions=transitions
            )
            scenario_cascades.append(len(delayed))
        
        scenario_stats = compute_cascade_statistics(scenario_cascades, n_flights)
        results["scenario_atl_hub_disruption"] = {
            "description": f"All {len(atl_flights)} flights departing ATL at 06:00 start delayed",
            "initial_delayed": len(atl_flights),
            "statistics": scenario_stats,
        }
        
        logger.info(f"ATL 06:00 scenario: mean cascade={scenario_stats['mean_cascade_size']:.0f}, "
                    f"p99={scenario_stats['p99']:.0f}")
    else:
        logger.warning("No flights found departing ATL at 06:00; skipping scenario")
    
    return results


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution."""
    root = get_project_root()
    log_dir = root / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(log_dir / "07_run_delay_propagation.log")
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("WS3 Script 07: Delay Propagation & Cascade Analysis")
    logger.info("=" * 80)
    
    # Load config
    config_path = root / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded config from {config_path}")
    set_global_seed(config.get("seed", 42))
    
    # Create delay config
    delay_cfg = DelayConfig(
        p_pax=config.get("analysis", {}).get("delay", {}).get("p_pax", 0.30),
        p_tail=config.get("analysis", {}).get("delay", {}).get("p_tail", 0.60),
        min_conn_minutes=config.get("analysis", {}).get("delay", {}).get("min_conn_minutes", 30),
        max_conn_minutes=config.get("analysis", {}).get("delay", {}).get("max_conn_minutes", 240),
        n_runs=config.get("analysis", {}).get("delay", {}).get("n_runs", 500),
        rng_seed=config.get("seed", 42),
        delay_threshold_minutes=config.get("analysis", {}).get("delay", {}).get("delay_threshold", 1.0),
        save_dir=str(root / "results" / "delay"),
    )
    
    # Create output directory
    Path(delay_cfg.save_dir).mkdir(parents=True, exist_ok=True)
    
    # Load flight data
    networks_dir = root / "results" / "networks"
    flight_nodes_path = networks_dir / "flight_nodes.parquet"
    flight_edges_path = networks_dir / "flight_edges.parquet"
    
    if not flight_nodes_path.exists() or not flight_edges_path.exists():
        logger.error(f"Flight network files not found at {networks_dir}")
        logger.error("Run WS1 scripts first (01_build_airport_network, 02_build_flight_network)")
        sys.exit(1)
    
    logger.info("Loading flight nodes and edges...")
    base_flights_df = pl.read_parquet(flight_nodes_path)
    flights_df = base_flights_df if "vertex_id" in base_flights_df.columns else base_flights_df.with_row_index(name="vertex_id")
    
    # Load flight graph
    edges_df = pl.read_parquet(flight_edges_path)
    n_flights = len(flights_df)
    
    logger.info(f"Loaded {n_flights:,} flights and {len(edges_df):,} edges")
    
    # Build connection graph from flight data
    logger.info("Building flight connection graph for delay propagation...")
    g = build_flight_connection_graph(flights_df, delay_cfg, logger)
    
    # Run analysis
    summary = {
        "script": "07_run_delay_propagation.py",
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "config": asdict(delay_cfg),
        "graph_info": {
            "n_flights": g.vcount(),
            "n_connections": g.ecount(),
        },
    }
    
    analysis_results = run_delay_propagation(flights_df, g, delay_cfg, logger)
    summary["results"] = analysis_results
    
    # Save results
    output_path = Path(delay_cfg.save_dir) / "delay_propagation_summary.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Saved results to {output_path}")
    
    # Write manifest
    manifest_path = log_dir / "07_run_delay_propagation_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "script": "07_run_delay_propagation.py",
            "timestamp": datetime.now().isoformat(),
            "results_dir": delay_cfg.save_dir,
            "summary_file": str(output_path),
        }, f, indent=2)
    logger.info(f"Wrote manifest: {manifest_path}")
    
    logger.info("=" * 80)
    logger.info("WS3 Script 07: Complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

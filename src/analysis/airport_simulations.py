"""
Robustness + delay contagion simulations for airport networks.

Inputs:
  - Airport network G_airport: nodes=airports, edges=routes (directed or undirected)
    Optional node attributes: "passengers" (annual pax), "traffic" (any numeric)
  - Airline subnetworks: dict[str, nx.Graph] where each graph is already filtered
  - Flights table (pandas DataFrame): one row per flight with at least:
      flight_id (unique), carrier, origin, dest, dep_time, arr_time
    Optional:
      tail_num, LateAircraftDelay (minutes or boolean), origin_lat, origin_lon, etc.

Outputs:
  - Robustness curves + metrics by graph
  - Cascade size distributions + scenario simulations

Author: (you)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

try:
    import pandas as pd  # optional but recommended for flight graph construction
except ImportError:
    pd = None


# -----------------------------
# Logging
# -----------------------------

def setup_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("airport_simulations")
    if logger.handlers:
        return logger  # already configured
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# -----------------------------
# Robustness Testing
# -----------------------------

@dataclass(frozen=True)
class RobustnessConfig:
    n_runs_random: int = 200
    random_seed: int = 42
    mode: str = "weak"  # "weak" for directed graphs, "strong", or "undirected"
    targeted_strategies: Tuple[str, ...] = ("degree", "traffic")  # "degree", "traffic"
    traffic_attr: str = "passengers"  # fallback handled
    k_values: Tuple[int, ...] = (1, 5, 10, 20, 50, 100)
    save_dir: str = "outputs/robustness"


def _largest_component_size(G: nx.Graph, mode: str = "weak") -> int:
    """Largest connected component size for undirected/directed graphs."""
    if G.number_of_nodes() == 0:
        return 0

    if G.is_directed():
        if mode == "strong":
            comps = nx.strongly_connected_components(G)
        else:
            # default: weak connectivity makes more sense for route reachability ignoring direction
            comps = nx.weakly_connected_components(G)
    else:
        comps = nx.connected_components(G)

    return max((len(c) for c in comps), default=0)


def _lcc_fraction(G: nx.Graph, n0: int, mode: str = "weak") -> float:
    """LCC size normalized by original node count n0 (stable x-axis comparison)."""
    if n0 <= 0:
        return 0.0
    return _largest_component_size(G, mode=mode) / float(n0)


def _rank_nodes_targeted(
    G: nx.Graph,
    strategy: str,
    traffic_attr: str = "passengers",
) -> List[Any]:
    """Return nodes sorted from most-to-least important for removal."""
    if strategy == "degree":
        deg = dict(G.degree())
        return sorted(G.nodes(), key=lambda n: deg.get(n, 0), reverse=True)

    if strategy == "traffic":
        # Accept multiple possible names; fallback to degree if missing.
        candidates = [traffic_attr, "traffic", "pax", "passenger_traffic"]
        def traffic(n):
            for a in candidates:
                if a in G.nodes[n]:
                    v = G.nodes[n].get(a)
                    try:
                        return float(v)
                    except Exception:
                        pass
            return float(G.degree(n))  # fallback
        return sorted(G.nodes(), key=traffic, reverse=True)

    raise ValueError(f"Unknown targeted strategy: {strategy}")


def simulate_random_removal(
    G: nx.Graph,
    n_runs: int,
    seed: int,
    mode: str,
    logger: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Random node removal: multiple runs.
    Returns:
      x_removed (0..n), mean_lcc_frac, std_lcc_frac  (normalized by original n)
    """
    n0 = G.number_of_nodes()
    if n0 == 0:
        return np.array([0]), np.array([0.0]), np.array([0.0])

    rng = np.random.default_rng(seed)
    curves = np.zeros((n_runs, n0 + 1), dtype=float)

    nodes = list(G.nodes())

    for r in range(n_runs):
        order = nodes.copy()
        rng.shuffle(order)

        H = G.copy()
        curves[r, 0] = _lcc_fraction(H, n0=n0, mode=mode)

        for i, node in enumerate(order, start=1):
            if H.has_node(node):
                H.remove_node(node)
            curves[r, i] = _lcc_fraction(H, n0=n0, mode=mode)

        if (r + 1) % max(1, n_runs // 10) == 0:
            logger.info(f"Random removal progress: {r+1}/{n_runs} runs")

    x = np.arange(n0 + 1)
    mean = curves.mean(axis=0)
    std = curves.std(axis=0, ddof=1) if n_runs > 1 else np.zeros_like(mean)
    return x, mean, std


def simulate_targeted_removal(
    G: nx.Graph,
    removal_order: Sequence[Any],
    mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Targeted node removal: deterministic order.
    Returns:
      x_removed (0..n), lcc_frac (normalized by original n)
    """
    n0 = G.number_of_nodes()
    if n0 == 0:
        return np.array([0]), np.array([0.0])

    H = G.copy()
    lcc = np.zeros(n0 + 1, dtype=float)
    lcc[0] = _lcc_fraction(H, n0=n0, mode=mode)

    for i, node in enumerate(removal_order, start=1):
        if H.has_node(node):
            H.remove_node(node)
        lcc[i] = _lcc_fraction(H, n0=n0, mode=mode)

    return np.arange(n0 + 1), lcc


def robustness_metrics_after_topk(
    G: nx.Graph,
    removal_order: Sequence[Any],
    k_values: Sequence[int],
    mode: str,
) -> Dict[int, Dict[str, float]]:
    """
    Metrics for "% disconnected after removing top-k hubs".
    Two normalizations:
      - disconnected_of_original = 1 - (LCC / N0)
      - disconnected_of_remaining = 1 - (LCC / (N0-k))  (if N0-k>0)
    """
    n0 = G.number_of_nodes()
    H = G.copy()
    out: Dict[int, Dict[str, float]] = {}

    removed = 0
    for k in sorted(set(k_values)):
        while removed < min(k, n0):
            node = removal_order[removed]
            if H.has_node(node):
                H.remove_node(node)
            removed += 1

        lcc = _largest_component_size(H, mode=mode)
        frac_orig = lcc / float(n0) if n0 else 0.0
        remaining = max(n0 - removed, 0)
        frac_rem = (lcc / float(remaining)) if remaining > 0 else 0.0

        out[k] = {
            "lcc_frac_of_original": frac_orig,
            "disconnected_frac_of_original": 1.0 - frac_orig,
            "lcc_frac_of_remaining": frac_rem,
            "disconnected_frac_of_remaining": 1.0 - frac_rem,
            "removed": float(removed),
            "remaining": float(remaining),
        }

    return out


def plot_robustness_curve_mean_std(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    title: str,
    save_path: Optional[str] = None,
):
    plt.figure()
    plt.plot(x, mean, label="mean LCC fraction")
    if std is not None and np.any(std > 0):
        plt.fill_between(x, mean - std, mean + std, alpha=0.2, label="±1 std")
    plt.xlabel("# nodes removed")
    plt.ylabel("Largest CC / original N")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()


def plot_robustness_curve_single(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    save_path: Optional[str] = None,
):
    plt.figure()
    plt.plot(x, y, label="LCC fraction")
    plt.xlabel("# nodes removed")
    plt.ylabel("Largest CC / original N")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()


def run_robustness_suite(
    graphs: Dict[str, nx.Graph],
    cfg: RobustnessConfig,
    logger: logging.Logger,
) -> Dict[str, Any]:
    os.makedirs(cfg.save_dir, exist_ok=True)

    summary: Dict[str, Any] = {"config": asdict(cfg), "results": {}}

    for name, G in graphs.items():
        logger.info(f"=== Robustness for graph: {name} (N={G.number_of_nodes()}, E={G.number_of_edges()}) ===")
        g_res: Dict[str, Any] = {}

        # Random removal
        x, mean, std = simulate_random_removal(
            G, n_runs=cfg.n_runs_random, seed=cfg.random_seed, mode=cfg.mode, logger=logger
        )
        g_res["random"] = {
            "x_removed": x.tolist(),
            "mean_lcc_frac": mean.tolist(),
            "std_lcc_frac": std.tolist(),
        }
        plot_robustness_curve_mean_std(
            x, mean, std,
            title=f"{name} - Random removal (n_runs={cfg.n_runs_random})",
            save_path=os.path.join(cfg.save_dir, f"{name}__random.png"),
        )

        # Targeted removals
        g_res["targeted"] = {}
        for strat in cfg.targeted_strategies:
            order = _rank_nodes_targeted(G, strat, traffic_attr=cfg.traffic_attr)
            xt, yt = simulate_targeted_removal(G, order, mode=cfg.mode)
            g_res["targeted"][strat] = {
                "x_removed": xt.tolist(),
                "lcc_frac": yt.tolist(),
                "topk_metrics": robustness_metrics_after_topk(G, order, cfg.k_values, mode=cfg.mode),
            }
            plot_robustness_curve_single(
                xt, yt,
                title=f"{name} - Targeted removal ({strat})",
                save_path=os.path.join(cfg.save_dir, f"{name}__targeted_{strat}.png"),
            )

        summary["results"][name] = g_res

    # Save JSON summary for reproducibility
    with open(os.path.join(cfg.save_dir, "robustness_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved robustness outputs to: {cfg.save_dir}")
    return summary


# -----------------------------
# Delay Propagation (Flights-as-nodes)
# -----------------------------

@dataclass(frozen=True)
class DelayConfig:
    p_pax: float = 0.30          # transmission probability for passenger/airport connection edges
    p_tail: float = 0.60         # transmission probability for same-aircraft edges (often higher)
    min_conn_minutes: int = 30   # min feasible connection time
    max_conn_minutes: int = 240  # max feasible connection time
    n_runs: int = 500
    rng_seed: int = 123
    save_dir: str = "outputs/delay"
    # For “baseline match”: what counts as "delayed" in data, if LateAircraftDelay is minutes
    late_aircraft_delay_threshold: float = 1.0


def _ensure_pandas():
    if pd is None:
        raise ImportError("pandas is required for flight-connection graph construction. pip install pandas")


def build_flight_connection_graph(
    flights_df: "pd.DataFrame",
    cfg: DelayConfig,
    logger: logging.Logger,
) -> nx.DiGraph:
    """
    Build a directed graph where:
      - node = flight_id
      - edge u->v if u can propagate delay to v
    Edge attribute: type in {"pax", "tail"} and p (transmission prob)
    """
    _ensure_pandas()

    required = {"flight_id", "origin", "dest", "dep_time", "arr_time"}
    missing = required - set(flights_df.columns)
    if missing:
        raise ValueError(f"Missing required flight columns: {missing}")

    # Ensure datetime
    flights = flights_df.copy()
    flights["dep_time"] = pd.to_datetime(flights["dep_time"])
    flights["arr_time"] = pd.to_datetime(flights["arr_time"])

    G = nx.DiGraph()

    # Add nodes with attributes (keep what you need)
    keep_cols = [c for c in flights.columns if c not in []]
    for _, row in flights[keep_cols].iterrows():
        fid = row["flight_id"]
        attrs = row.to_dict()
        G.add_node(fid, **attrs)

    # Passenger / airport connections: arrive at airport X then depart from X in [min,max]
    min_dt = pd.to_timedelta(cfg.min_conn_minutes, unit="m")
    max_dt = pd.to_timedelta(cfg.max_conn_minutes, unit="m")

    # Group by connection airport (= destination of inbound = origin of outbound)
    # We'll build edges by scanning departures for each airport
    logger.info("Building passenger-connection edges (time-window join by airport)...")
    by_airport_arr = flights.sort_values("arr_time").groupby("dest")
    by_airport_dep = flights.sort_values("dep_time").groupby("origin")

    airports = sorted(set(flights["origin"]).union(set(flights["dest"])))
    edge_count = 0

    for ap in airports:
        if ap not in by_airport_arr.groups or ap not in by_airport_dep.groups:
            continue
        arr = flights.loc[by_airport_arr.groups[ap], ["flight_id", "arr_time"]].sort_values("arr_time")
        dep = flights.loc[by_airport_dep.groups[ap], ["flight_id", "dep_time"]].sort_values("dep_time")

        # Two-pointer scan
        dep_times = dep["dep_time"].to_numpy()
        dep_ids = dep["flight_id"].to_numpy()

        j_start = 0
        for fid_u, t_arr in zip(arr["flight_id"].to_numpy(), arr["arr_time"].to_numpy()):
            lo = t_arr + min_dt
            hi = t_arr + max_dt

            # advance j_start to first dep_time >= lo
            while j_start < len(dep_times) and dep_times[j_start] < lo:
                j_start += 1

            j = j_start
            while j < len(dep_times) and dep_times[j] <= hi:
                fid_v = dep_ids[j]
                if fid_u != fid_v:
                    G.add_edge(fid_u, fid_v, type="pax", p=cfg.p_pax, airport=ap)
                    edge_count += 1
                j += 1

    logger.info(f"Passenger-connection edges added: {edge_count}")

    # Tail-number chaining: sequential flights by same aircraft
    if "tail_num" in flights.columns:
        logger.info("Building tail-number edges (aircraft rotations)...")
        tail_edges = 0
        f2 = flights.dropna(subset=["tail_num"]).sort_values("dep_time")
        for tail, grp in f2.groupby("tail_num"):
            grp = grp.sort_values("dep_time")
            ids = grp["flight_id"].to_numpy()
            # Link consecutive flights (or you can link all feasible sequences)
            for u, v in zip(ids[:-1], ids[1:]):
                if u != v:
                    G.add_edge(u, v, type="tail", p=cfg.p_tail, tail_num=tail)
                    tail_edges += 1
        logger.info(f"Tail-number edges added: {tail_edges}")
    else:
        logger.warning("No tail_num column found; skipping tail-number edges.")

    logger.info(f"Flight-connection graph built: N={G.number_of_nodes()}, E={G.number_of_edges()}")
    return G


def simulate_delay_cascade(
    G: nx.DiGraph,
    seed_flights: Sequence[Any],
    rng: np.random.Generator,
) -> Tuple[set, Dict[Any, Any]]:
    """
    Simple independent cascade model:
      - Start with seed_flights "delayed"
      - For each delayed u, attempt to delay each neighbor v with probability edge['p']
    Returns:
      infected_set, parent_map (for reconstructing example cascades)
    """
    infected = set(seed_flights)
    parent: Dict[Any, Any] = {s: None for s in seed_flights}
    queue = list(seed_flights)

    while queue:
        u = queue.pop(0)
        for v in G.successors(u):
            if v in infected:
                continue
            p = G[u][v].get("p", 0.0)
            if rng.random() < p:
                infected.add(v)
                parent[v] = u
                queue.append(v)

    return infected, parent


def run_delay_monte_carlo(
    G: nx.DiGraph,
    seed_selector,
    cfg: DelayConfig,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    seed_selector: callable (G) -> list of seed flight_ids
    """
    os.makedirs(cfg.save_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.rng_seed)

    sizes = []
    frac = []

    logger.info(f"Running delay Monte Carlo: n_runs={cfg.n_runs}")
    for r in range(cfg.n_runs):
        seeds = seed_selector(G)
        infected, _ = simulate_delay_cascade(G, seeds, rng)
        sizes.append(len(infected))
        frac.append(len(infected) / max(1, G.number_of_nodes()))

        if (r + 1) % max(1, cfg.n_runs // 10) == 0:
            logger.info(f"Delay sim progress: {r+1}/{cfg.n_runs}")

    sizes = np.array(sizes)
    frac = np.array(frac)

    # Plot cascade size distribution
    plt.figure()
    plt.hist(sizes, bins=50)
    plt.xlabel("Cascade size (# flights delayed)")
    plt.ylabel("Count")
    plt.title("Cascade size distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_dir, "cascade_size_hist.png"), dpi=200)
    plt.close()

    # Save summary
    out = {
        "config": asdict(cfg),
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "cascade_size": {
            "mean": float(sizes.mean()),
            "std": float(sizes.std(ddof=1)) if len(sizes) > 1 else 0.0,
            "p50": float(np.percentile(sizes, 50)),
            "p90": float(np.percentile(sizes, 90)),
            "p99": float(np.percentile(sizes, 99)),
        },
        "delayed_fraction": {
            "mean": float(frac.mean()),
            "std": float(frac.std(ddof=1)) if len(frac) > 1 else 0.0,
        },
    }
    with open(os.path.join(cfg.save_dir, "delay_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    logger.info(f"Saved delay outputs to: {cfg.save_dir}")
    return out


def calibrate_p_to_late_aircraft_delay(
    flights_df: "pd.DataFrame",
    base_cfg: DelayConfig,
    logger: logging.Logger,
    p_grid: Sequence[float] = tuple(np.linspace(0.05, 0.9, 18)),
) -> Tuple[DelayConfig, Dict[str, Any]]:
    """
    Coarse grid calibration:
      - target = observed fraction where LateAircraftDelay > threshold
      - try p_pax grid; keep ratio p_tail ~ 2*p_pax capped at 0.95
    """
    _ensure_pandas()
    if "LateAircraftDelay" not in flights_df.columns:
        raise ValueError("LateAircraftDelay column not found for calibration.")

    lad = flights_df["LateAircraftDelay"]
    # Accept boolean or minutes
    if lad.dtype == bool:
        target = float(lad.mean())
    else:
        target = float((pd.to_numeric(lad, errors="coerce").fillna(0.0) > base_cfg.late_aircraft_delay_threshold).mean())

    logger.info(f"Calibration target delayed fraction (LateAircraftDelay proxy): {target:.4f}")

    best = None
    best_err = float("inf")
    diagnostics = []

    for p in p_grid:
        cfg = DelayConfig(
            p_pax=float(p),
            p_tail=float(min(0.95, 2.0 * p)),
            min_conn_minutes=base_cfg.min_conn_minutes,
            max_conn_minutes=base_cfg.max_conn_minutes,
            n_runs=min(base_cfg.n_runs, 200),  # keep calibration cheaper
            rng_seed=base_cfg.rng_seed,
            save_dir=base_cfg.save_dir,
            late_aircraft_delay_threshold=base_cfg.late_aircraft_delay_threshold,
        )
        G = build_flight_connection_graph(flights_df, cfg, logger)

        # Seed: “exogenous” delayed flights — minimal assumption: small random fraction
        def seed_selector(_G):
            rng = np.random.default_rng(cfg.rng_seed)
            n = _G.number_of_nodes()
            k = max(1, int(0.01 * n))  # 1% initial shocks; tune if needed
            return rng.choice(list(_G.nodes()), size=k, replace=False).tolist()

        summary = run_delay_monte_carlo(G, seed_selector, cfg, logger)
        sim_frac = summary["delayed_fraction"]["mean"]
        err = abs(sim_frac - target)

        diagnostics.append({"p_pax": cfg.p_pax, "p_tail": cfg.p_tail, "sim": sim_frac, "target": target, "err": err})
        if err < best_err:
            best_err = err
            best = (cfg, summary)

    assert best is not None
    best_cfg, best_summary = best
    with open(os.path.join(base_cfg.save_dir, "calibration_grid.json"), "w", encoding="utf-8") as f:
        json.dump({"target": target, "grid": diagnostics, "best": asdict(best_cfg)}, f, indent=2)

    logger.info(f"Best calibration: p_pax={best_cfg.p_pax:.3f}, p_tail={best_cfg.p_tail:.3f}, err={best_err:.4f}")
    return best_cfg, best_summary


# -----------------------------
# Scenario helpers (e.g., JFK @ 6AM)
# -----------------------------

def seed_all_flights_from_airport_at_time(
    airport_code: str,
    hhmm: str,
    timezone_aware: bool = False,
):
    """
    Returns a seed_selector(G) that picks flights departing from `airport_code`
    with dep_time at the given HH:MM (local in your timestamps).
    """
    hh, mm = map(int, hhmm.split(":"))

    def selector(G: nx.DiGraph) -> List[Any]:
        seeds = []
        for n, data in G.nodes(data=True):
            dep = data.get("dep_time")
            origin = data.get("origin")
            if origin != airport_code or dep is None:
                continue
            try:
                # dep may be Timestamp/datetime
                if dep.hour == hh and dep.minute == mm:
                    seeds.append(n)
            except Exception:
                continue
        return seeds

    return selector


def save_example_cascade_subgraph(
    G: nx.DiGraph,
    seeds: Sequence[Any],
    cfg: DelayConfig,
    logger: logging.Logger,
    filename: str = "example_cascade_edgelist.json",
):
    """
    Runs one cascade and stores the induced subgraph edges + node attrs for inspection.
    """
    rng = np.random.default_rng(cfg.rng_seed + 999)
    infected, parent = simulate_delay_cascade(G, seeds, rng)

    H = G.subgraph(infected).copy()
    payload = {
        "seeds": list(seeds),
        "n_infected": len(infected),
        "nodes": [{"id": n, **H.nodes[n]} for n in H.nodes()],
        "edges": [{"u": u, "v": v, **H[u][v]} for u, v in H.edges()],
        "parent": {str(k): (None if v is None else str(v)) for k, v in parent.items()},
    }
    out_path = os.path.join(cfg.save_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info(f"Saved example cascade to {out_path}")


# -----------------------------
# CLI entrypoint (optional)
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loglevel", default="INFO")
    parser.add_argument("--demo", action="store_true", help="Run small synthetic demo graphs.")
    args = parser.parse_args()

    logger = setup_logger(args.loglevel)

    if args.demo:
        # Synthetic demo to verify stability/edge cases quickly
        logger.info("Running DEMO with synthetic graphs...")

        # Airport-like: hub-and-spoke vs more uniform
        G_hub = nx.barabasi_albert_graph(300, 2)  # scale-free-ish
        G_uni = nx.erdos_renyi_graph(300, 0.02)

        graphs = {"hub_like": G_hub, "uniform_like": G_uni}
        rob_cfg = RobustnessConfig(n_runs_random=100, save_dir="outputs_demo/robustness")
        run_robustness_suite(graphs, rob_cfg, logger)

        logger.info("DEMO done. (For real data, import these functions in your notebook/script.)")


if __name__ == "__main__":
    main()

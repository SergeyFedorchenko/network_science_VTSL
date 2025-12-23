# AI Implementation Instructions (Claude Sonnet 4.5 in Cursor / GitHub Copilot)
## Stack: polars + python-igraph (primary), leidenalg (recommended), gensim + scikit-learn (embeddings/link prediction)

This file is the single source of truth for implementing the entire Network Science project. All teammates must follow it to keep schemas, outputs, and results consistent and reproducible.

Project facts:
- Data already prepared (cleaned dataset is ready). We still implement validation checks and reproducibility packaging.
- Analyze one year: 2025.
- Build and compare:
  1) Baseline airport network (nodes = airports)
  2) Flight-centric network (nodes = flights)
  3) Advanced representation: multilayer network with airline layers
- Advanced methods required:
  - Community detection beyond modularity (Leiden with CPM objective as in-scope and scalable; optional DC-SBM using an extra library if approved)
  - Robustness/percolation analysis
  - Delay propagation modeling (contagion on flight graph)
  - Graph embeddings + link prediction

Non-negotiables:
- Deterministic final runs (fixed seeds, run manifests).
- Config-driven pipeline, no manual steps.
- Idempotent scripts that only write under results/.
- No O(n^2) edge creation for flight graphs.

---

## 1 Repository layout (must follow)

/
  README.md
  environment.yml
  config/
    config.yaml
  .github/
    copilot-instructions.md  (this file or link to it)
  .cursor/
    rules                    (optional mirror of this file)
  data/
    README.md
    cleaned/
      flights_2025.parquet   (preferred)
  src/
    io/
      load_data.py
      validate_data.py
      time_features.py
    networks/
      airport_network.py
      flight_network.py
      multilayer_network.py
      igraph_helpers.py
    analysis/
      centrality.py
      community.py
      robustness.py
      delay_propagation.py
      embeddings.py
      link_prediction.py
    business/
      airline_metrics.py
      hub_strategy.py
      disruption_cost_proxy.py
    viz/
      plotting.py
    utils/
      seeds.py
      logging.py
      manifests.py
      paths.py
  scripts/
    00_validate_inputs.py
    01_build_airport_network.py
    02_build_flight_network.py
    03_build_multilayer.py
    04_run_centrality.py
    05_run_communities.py
    06_run_robustness.py
    07_run_delay_propagation.py
    08_run_embeddings_linkpred.py
    09_run_business_module.py
    10_make_all_figures.py
  results/
    networks/
    analysis/
    business/
    figures/
    tables/
    logs/
  tests/
    fixtures/
      toy_flights.parquet
    test_validate_data.py
    test_time_features.py
    test_network_construction_small.py
    test_seed_determinism.py

---

## 2 Engineering conventions (required)

### 2.1 Determinism
- Implement `src/utils/seeds.py`:
  - Seed Python `random`, NumPy, and any other stochastic libs used.
  - Every script must call `set_global_seed(config.seed)` first.
- Every script writes a run manifest JSON to `results/logs/` containing:
  - git commit hash (if available)
  - timestamp
  - config snapshot (resolved)
  - input file fingerprints (row count, schema hash)
  - output file list

### 2.2 Performance and memory
- Use polars LazyFrame for all large operations:
  - `pl.scan_parquet()` or `pl.scan_csv()`
  - Avoid `.collect()` until the final aggregated tables are small enough.
- Use categorical encoding for high-cardinality strings where it helps (carrier, origin, dest).
- Avoid building the full flight graph if it becomes too large:
  - Use a config-driven scope strategy (top airports, sampling, monthly slice) for flight-centric analyses that are computationally heavy (communities, embeddings).
  - Keep the airport network full-year.

### 2.3 Code quality
- Type hints for public functions.
- Logging for major steps (row counts, unique counts, edge counts).
- Tests must cover validation, time parsing, toy graph correctness, and seed determinism.

---

## 3 Clean dataset contract (fixed schema)

The cleaned dataset has these columns:

- YEAR: int
- MONTH: int
- FL_DATE: date (YYYY-MM-DD)
- OP_UNIQUE_CARRIER: string
- TAIL_NUM: string (nullable)
- OP_CARRIER_FL_NUM: int
- ORIGIN_AIRPORT_ID: int
- ORIGIN: string (IATA)
- ORIGIN_CITY_NAME: string
- ORIGIN_STATE_NM: string
- DEST: string (IATA)
- DEST_CITY_NAME: string
- DEST_STATE_NM: string
- DEP_TIME: float/int (HHMM, nullable)
- DEP_DELAY: float (minutes, negative allowed)
- ARR_TIME: float/int (HHMM, nullable; may roll past midnight)
- ARR_DELAY: float (minutes, negative allowed)
- CANCELLED: float/int (0/1)
- AIR_TIME: float (minutes, nullable if cancelled)
- FLIGHTS: float/int (usually 1)
- DISTANCE: float (miles)

If additional columns exist, preserve them, but do not rely on them unless explicitly documented and validated.

Internal naming policy:
- Use the dataset column names directly (no remapping) unless absolutely necessary.
- If a mapping is created, store it in `results/logs/schema_mapping.json`.

---
 
## 4 config/config.yaml (required keys)

Minimum config keys:

seed: 42

data:
  cleaned_path: "data/cleaned/flights_2025.parquet"
  format: "parquet"

filters:
  year: 2025
  include_cancelled: false
  carriers: "ALL"              # or list like ["AA","DL"]

time_features:
  hhmm_columns: ["DEP_TIME","ARR_TIME"]
  midnight_roll_rule: "arr_lt_dep_add_1day_if_air_time_positive"

airport_network:
  directed: true
  edge_weight: "flight_count"
  edge_metrics:
    - "mean_dep_delay"
    - "mean_arr_delay"
    - "cancel_rate"
    - "mean_distance"

flight_graph:
  scope:
    mode: "top_airports"        # "full" | "top_airports" | "sample"
    top_airports_k: 50
    sample_frac: 0.10
  edges:
    include_tail_sequence: true
    include_same_route_knn: true
    route_knn_k: 3
    include_airport_transfer_knn: true
    airport_transfer_knn_k: 2
    transfer_same_carrier_only: true
    transfer_window_minutes: [30, 240]

multilayer:
  layer_key: "OP_UNIQUE_CARRIER"
  include_interlayer_transfer_edges: false

analysis:
  centrality:
    measures: ["degree","betweenness","pagerank"]
    betweenness_approx_cutoff: 20000   # if graph large
  communities:
    method: "leiden_cpm"        # primary
    leiden:
      objective: "CPM"
      resolution: 0.01
      n_runs: 10
    sbm_optional:
      enabled: false            # set true only if approved and dependency added
  robustness:
    strategies: ["random","highest_degree","highest_betweenness"]
    random_trials: 30
    recompute_betweenness_every: 10
  delay_propagation:
    model: "SIR"
    beta: 0.25
    gamma: 0.50
    delay_threshold_minutes: 15
    initial_seed_strategy: "top_k_outdegree"
    top_k: 50
    monte_carlo_runs: 200
    use_empirical_beta: true
  embeddings:
    method: "node2vec"
    dimensions: 128
    walk_length: 80
    num_walks: 10
    window_size: 10
    p: 1.0
    q: 1.0
  link_prediction:
    time_split:
      train_months: [1,2,3,4,5,6,7,8,9]
      test_months: [10,11,12]
    negative_ratio: 5
    classifier: "logreg"

outputs:
  overwrite: false

---

## 5 Data validation and time features (polars)

### 5.1 Validation (scripts/00_validate_inputs.py)
Implement `src/io/validate_data.py`:

Checks:
- Required columns exist and types are compatible.
- YEAR matches config year, MONTH in 1..12.
- ORIGIN and DEST are non-empty strings for non-cancelled flights.
- CANCELLED is 0/1.
- For non-cancelled flights: DEP_TIME and ARR_TIME must be present most of the time; quantify missingness.
- AIR_TIME should be null when cancelled and positive when not cancelled (report exceptions).
- DELAY columns numeric; report extreme values.

Outputs:
- results/tables/data_validation_summary.csv
- results/logs/data_fingerprint.json (row count, schema hash, min/max FL_DATE, null rates)

### 5.2 Time feature engineering (src/io/time_features.py)
We only have actual times (DEP_TIME, ARR_TIME) in HHMM.
Create:
- dep_minutes: minutes since midnight
- arr_minutes: minutes since midnight
- dep_ts: FL_DATE + dep_minutes
- arr_ts: FL_DATE + arr_minutes, with midnight roll handling:
  - If ARR_TIME < DEP_TIME and AIR_TIME > 0, then arr_ts += 1 day
- If DEP_TIME or ARR_TIME null, ts is null.

Implement using polars expressions, not Python loops, for scalability.

Write a unit test for HHMM conversion and midnight roll logic.

---

## 6 Network construction (polars -> igraph)

All graphs must have stable vertex IDs. Use integer vertex indices internally in igraph and store mapping tables.

### 6.1 Airport network (nodes = airports)
File: src/networks/airport_network.py
Script: scripts/01_build_airport_network.py

Nodes:
- Unique airport IATA codes from ORIGIN and DEST.

Edges:
- Directed edge ORIGIN -> DEST.

Edge weight:
- flight_count = sum(FLIGHTS) or count rows.

Edge metrics (aggregations):
- mean_dep_delay, mean_arr_delay
- cancel_rate (if include_cancelled true, else report as 0)
- mean_distance

Implementation approach:
- Polars LazyFrame:
  - filter year, optional filter CANCELLED
  - groupby(["ORIGIN","DEST"]) aggregate metrics
- Build igraph Graph from edge list:
  - Map airport codes to integer vertex IDs
  - Graph(directed=True).add_vertices(n).add_edges(list_of_pairs)
  - Set vertex attribute "code"
  - Set edge attributes for weight and metrics

Outputs:
- results/networks/airport_nodes.parquet (vertex_id, code, city/state if desired)
- results/networks/airport_edges.parquet (src_id, dst_id, metrics...)
- results/networks/airport_graph.graphml (optional, for inspection)
- results/logs/airport_network_summary.json (N, E, LCC size)

### 6.2 Flight graph (nodes = flights)
File: src/networks/flight_network.py
Script: scripts/02_build_flight_network.py

Node definition:
- One row = one flight instance.
- Create node_id as a stable hash or stable composite key stored explicitly:
  - Recommended composite key:
    FL_DATE, OP_UNIQUE_CARRIER, OP_CARRIER_FL_NUM, ORIGIN, DEST, DEP_TIME
  - Store a string key `flight_key` and map it to integer `flight_id`.

Because the dataset is huge, flight graph construction must be scope-limited via config:
- mode=top_airports: select flights where ORIGIN or DEST is in the top K airports by flight volume.
- mode=sample: stratified sampling by carrier and month (implement deterministic sample using hash mod).
- mode=full: only if it fits.

Edge types (must stay scalable):
A. Tail sequence edges (recommended, high value for delay propagation)
- For each TAIL_NUM (non-null), sort by dep_ts, connect consecutive legs:
  src = flight i, dst = flight i+1
- Edge attributes:
  - edge_type="tail_next_leg"
  - ground_time_minutes = next.dep_ts - current.arr_ts (if available)
  - same_carrier = (current carrier == next carrier)

Implement with polars:
- filter TAIL_NUM not null, dep_ts not null
- sort by [TAIL_NUM, dep_ts]
- create shifted columns per TAIL_NUM using `.over("TAIL_NUM")` and `shift(-1)`
- filter where next_flight_id not null and next.dep_ts > dep_ts

B. Same-route kNN edges (avoid cliques)
- For each route (ORIGIN, DEST), sort by dep_ts, connect each flight to next k flights.
- edge_type="route_knn"
- attribute: delta_dep_minutes

Implement with polars window:
- sort by [ORIGIN, DEST, dep_ts]
- create lead columns for i+1..i+k with `shift(-j).over(["ORIGIN","DEST"])`
- unpivot to edges (src_id, dst_id) for j in 1..k
- filter dst not null

C. Airport transfer kNN edges (approximate feasible connections without exploding)
Goal: represent potential passenger/crew/bank connections at airports without a full range join.
- For each airport and carrier (default same-carrier only), consider inbound arrivals and connect to the next k departures at that airport for that carrier.
- edge_type="airport_transfer_knn"
- attribute: transfer_slack_minutes = dep_ts_out - arr_ts_in
- Apply window constraints:
  - transfer_slack within [min,max] minutes from config
- Implementation (scalable approximation):
  1) Create arrivals table keyed by (airport=DEST, carrier, arr_ts, flight_id)
  2) Create departures table keyed by (airport=ORIGIN, carrier, dep_ts, flight_id)
  3) For each (airport, carrier), connect each arrival to the next k departures after it.
     - Implement by sorting departures and using as-of-like logic plus k-step shifts:
       - For each airport-carrier, maintain departures list and for arrivals find insertion index.
     - Practical approach:
       - If full implementation is too heavy, implement a simplified proxy:
         connect each flight to the next k flights departing from its DEST for the same carrier, ordered by dep_ts, using precomputed departure ranking and a join on rank ranges.
  4) Filter by slack window.

Important:
- If C is too complex under time pressure, prioritize A (tail edges) + B (route kNN). C can be marked optional in config and documented.

Graph build:
- Use igraph with vertices = number of flights in scoped dataset.
- Store node attributes:
  - carrier, origin, dest, dep_ts, arr_ts, dep_delay, arr_delay, cancelled, distance, month
- Store edge attributes:
  - edge_type, slack or delta, same_carrier

Outputs:
- results/networks/flight_nodes.parquet
- results/networks/flight_edges.parquet

### 6.3 Multilayer representation (airline layers)
File: src/networks/multilayer_network.py
Script: scripts/03_build_multilayer.py

Representation:
- Do not build separate igraph objects per layer unless needed.
- Store a single edge table with layer columns:
  - src_id, dst_id, src_layer, dst_layer, edge_type, attributes
- Layers are OP_UNIQUE_CARRIER.
- Within-layer edges are those where src_layer == dst_layer.
- Optional inter-layer edges can be enabled later, but default false for scale.

Outputs:
- results/networks/multilayer_edges.parquet
- results/networks/layer_summary.parquet (nodes/edges by airline)

---

## 7 Analyses (igraph-first, polars for tabulation)

### 7.1 Centrality (airport network primary)
File: src/analysis/centrality.py
Script: scripts/04_run_centrality.py

Compute on airport igraph Graph:
- degree (in/out), strength (weighted in/out)
- betweenness (use approximation if N large)
- PageRank (use weights optional)

Write results:
- results/analysis/airport_centrality.parquet

Also compute:
- components (weakly/strongly as appropriate)
- LCC size
- degree distribution table

### 7.2 Community detection beyond modularity
File: src/analysis/community.py
Script: scripts/05_run_communities.py

Primary method: Leiden with CPM objective (scalable, beyond modularity)
- Use python-igraph + leidenalg.
- Run multiple times with different seeds (config communities.leiden.n_runs).
- Select best partition by objective score.
- Apply to:
  - Airport network (full-year)
  - Flight graph (scoped subset only)

Outputs:
- results/analysis/airport_leiden_membership.parquet
- results/analysis/flight_leiden_membership.parquet
- results/analysis/community_summary_tables.parquet:
  - community size, dominant airline (for flight), dominant state/region (if using city/state fields)

Optional method: Degree-corrected SBM
- python-igraph does not provide full DC-SBM inference comparable to graph-tool.
- If you need SBM inference:
  - Add an approved library (for example graspologic) and set `analysis.communities.sbm_optional.enabled=true`.
  - Convert igraph adjacency to sparse matrix for SBM inference.
  - Store results alongside Leiden and clearly label as optional/approved.
- If not approved, Leiden CPM is the advanced community method for grading.

### 7.3 Robustness and percolation (airport network and airline subgraphs)
File: src/analysis/robustness.py
Script: scripts/06_run_robustness.py

Network:
- Airport graph.
- Airline-specific airport subgraphs:
  - Build per airline edge weights by filtering flights then aggregating routes for that airline.
  - Construct airline graphs (smaller) for comparative robustness.

Strategies:
- random removal: repeat random_trials, average LCC fraction
- targeted removal by current highest degree
- targeted removal by current highest betweenness:
  - recompute every N steps to reduce cost (config recompute_betweenness_every)

At each removal fraction:
- compute LCC size fraction
- optionally compute remaining total edge weight fraction

Outputs:
- results/analysis/robustness_curves.parquet
- results/tables/robustness_critical_nodes.csv
- figure-ready table: strategy curves for plotting

### 7.4 Delay propagation (contagion on flight graph)
File: src/analysis/delay_propagation.py
Script: scripts/07_run_delay_propagation.py

Goal:
- Model delay cascades on the flight graph using edges that represent plausible propagation paths.
- With this clean schema, the strongest empirical mechanism is aircraft rotation, proxied by tail sequence edges.

Define "delayed":
- infected if ARR_DELAY >= delay_threshold_minutes (default 15)
- cancelled flights excluded by default (config)

Graph edges used:
- tail_next_leg edges always included
- optionally route_knn and airport_transfer_knn edges if enabled

Empirical beta estimation (recommended):
- Compute empirical conditional probability on tail edges:
  P(next_arr_delay>=T | current_arr_delay>=T)
- Use this as beta for tail edges (or as baseline).
- If using multiple edge types, allow edge-type-specific betas:
  - beta_tail, beta_route, beta_transfer

Simulation:
- SIR:
  - infected nodes attempt to infect outgoing neighbors with probability beta
  - infected recover with probability gamma per step
- Run Monte Carlo with monte_carlo_runs.
- Seed selection:
  - top-k outdegree nodes in flight graph, or top-k by observed ARR_DELAY, configurable
- Outputs:
  - cascade sizes (per run, per seed strategy)
  - influence score per flight: mean cascade size when seeded at that flight (top candidates only)

Write:
- results/analysis/delay_cascades.parquet
- results/tables/delay_superspreaders.csv

---

## 8 Embeddings and link prediction (airport network)

### 8.1 node2vec embeddings (igraph-based random walks)
File: src/analysis/embeddings.py
Script: scripts/08_run_embeddings_linkpred.py

Implement node2vec on airport graph:
- Use igraph adjacency for random walks.
- Generate sequences of node IDs.
- Train skip-gram with gensim Word2Vec.
- Store embeddings.

Outputs:
- results/analysis/airport_embeddings.parquet

### 8.2 Link prediction with time split
File: src/analysis/link_prediction.py (called from script 08)

Task:
- Predict new airport routes in later months using earlier months.
- Build train graph from months in train_months.
- Build test edges from months in test_months not present in train graph.

Features:
- Embedding-based:
  - concatenate(u,v), hadamard(u,v), abs diff, dot product
- Heuristic baselines (igraph):
  - common neighbors, Jaccard, Adamic-Adar (implement via adjacency sets)
Classifier:
- logistic regression (scikit-learn)
Metrics:
- AUC, average precision
Outputs:
- results/analysis/linkpred_metrics.json
- results/tables/linkpred_top_predictions.csv

Data leakage rule:
- Embeddings must be trained on train graph only.

---

## 9 Business module (polars-heavy, consumes analysis outputs)

File: src/business/*
Script: scripts/09_run_business_module.py

Compute per airline (OP_UNIQUE_CARRIER):
Operational metrics:
- mean DEP_DELAY, mean ARR_DELAY (exclude cancelled by default)
- cancellation rate (mean(CANCELLED))
- mean DISTANCE
- volume: total flights

Network strategy metrics:
- hub concentration:
  - compute airport volumes for each airline: count flights touching airport as ORIGIN or DEST
  - compute share at top-1 and top-3 airports
- robustness proxy:
  - join airline robustness curves and report vulnerability score:
    - LCC fraction after removing top-1 hub
    - area under robustness curve (AURC) per airline

Disruption cost proxy:
- delay_cost = sum(max(ARR_DELAY,0)) * cost_per_minute
- cancellation_cost = count_cancelled * cost_per_cancellation
Both costs are parameters in config.

Outputs:
- results/business/airline_summary_metrics.parquet
- results/business/hub_concentration.parquet
- results/business/disruption_cost_proxy.parquet

---

## 10 Figures and tables (report-ready, no recomputation)

Script: scripts/10_make_all_figures.py
All plots must read from results/analysis and results/business only.

Required figures:
- fig01_airport_degree_distribution.png
- fig02_airport_centrality_rankings.png
- fig03_leiden_community_sizes_airport.png
- fig04_robustness_curves.png
- fig05_delay_cascade_distribution.png
- fig06_hub_dependence_by_airline.png
- fig07_connectivity_vs_delay_scatter.png
- fig08_link_prediction_performance.png (if link prediction included)

Required tables:
- tbl01_top_airports_by_centrality.csv
- tbl02_airport_communities_summary.csv
- tbl03_robustness_critical_nodes.csv
- tbl04_delay_superspreaders.csv
- tbl05_airline_business_metrics.csv
- tbl06_linkpred_top_predictions.csv

---

## 11 Pipeline execution order

1) scripts/00_validate_inputs.py
2) scripts/01_build_airport_network.py
3) scripts/02_build_flight_network.py
4) scripts/03_build_multilayer.py
5) scripts/04_run_centrality.py
6) scripts/05_run_communities.py
7) scripts/06_run_robustness.py
8) scripts/07_run_delay_propagation.py
9) scripts/08_run_embeddings_linkpred.py
10) scripts/09_run_business_module.py
11) scripts/10_make_all_figures.py

Add a Makefile:
- make validate
- make networks
- make analysis
- make business
- make figures
- make all

---

## 12 Dependency set (environment.yml guidance)

Include at minimum:
- python=3.11
- polars
- pyarrow
- numpy
- scipy
- pandas (optional, avoid in pipeline; allowed for small exports)
- python-igraph
- leidenalg
- matplotlib
- seaborn (optional)
- scikit-learn
- gensim
- tqdm
- pyyaml
- pytest

If DC-SBM optional is enabled:
- graspologic (or another approved SBM library)

---

## 13 Collaboration protocol (4 workstreams)

WS1 Data validation + network construction:
- Own: src/io, src/networks, scripts 00-03
- Contract: writes results/networks/*.parquet with stable schemas

WS2 Centrality + communities:
- Own: src/analysis/centrality.py, community.py, scripts 04-05
- Contract: consumes results/networks; writes results/analysis centrality and memberships

WS3 Robustness + delay propagation:
- Own: src/analysis/robustness.py, delay_propagation.py, scripts 06-07

WS4 Embeddings + link prediction + business + figures:
- Own: embeddings.py, link_prediction.py, business/*, scripts 08-10

Interfaces:
- No workstream edits another workstream’s output schema without explicit agreement and version bump.

Definition of done for any module:
- Config-driven script exists
- Writes outputs to results/
- Logs manifest
- Has at least one test or toy smoke test
- Documented run command in README

---

## 14 Prompt template for Claude Sonnet 4.5 (Cursor/Copilot)

Use this exact template when requesting code:

“Implement <module or script> according to copilot-instructions.md sections <X-Y>.
Use polars LazyFrame for large data.
Use python-igraph for graphs, leidenalg for Leiden.
Read config/config.yaml only.
Write outputs under results/<...>.
Write a run manifest JSON.
Add minimal unit tests under tests/.
Do not change existing schemas unless you also update all downstream consumers and tests.”

End of instructions.

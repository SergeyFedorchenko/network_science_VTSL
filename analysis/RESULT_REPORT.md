# Copilot / Claude Opus 4.5 Analysis Guide

## Research-Grade Notebook Program for Interpreting Network Science Pipeline Outputs

### Purpose and usage

This guide instructs Claude Opus 4.5 running inside GitHub Copilot (VS Code) to generate a set of research-grade Jupyter Notebook analysis documents that interpret the full set of pipeline outputs for this Network Science project.

Opus should use this guide as the controlling specification for notebook structure, evidence standards, output paths, and narrative style. The notebooks must analyze **only artifacts already produced** under `results/` (with manifests under `results/logs/`) and must produce report-ready tables and figures into the canonical report folders for downstream inclusion in `RESULTS_REPORT.md`\-style reporting.

* * *

## Operating rules

### Evidence-first interpretation

Opus should:

-   Treat every scientific claim as evidence-backed:
    
    -   Every finding must point to at least one concrete artifact generated in the notebooks, typically a table in `results/tables/report/` and/or a figure in `results/figures/report/`.
        
    -   If evidence is unavailable because an input artifact is missing or unreadable, Opus must explicitly label the claim as **Not available** and explain what is missing and where it was expected.
        
-   Prefer quantified statements over qualitative ones:
    
    -   Use ranks, shares, slopes, correlations, AUC/AP metrics, or distributional descriptors computed from outputs.
        
-   Keep interpretation grounded in network science mechanisms:
    
    -   For example: bridging structure for betweenness, hub dominance for heavy-tailed strength, fragmentation dynamics for percolation, cascade tails for contagion-like processes.
        

### Reproducibility and traceability requirements

Opus should:

-   Use manifests under `results/logs/` as the primary run ledger:
    
    -   Reconcile manifest-listed outputs against actual files on disk.
        
    -   Record which manifests were used, their timestamps, and any git hash or run identifiers present.
        
-   Record all notebook assumptions in a visible “Reproducibility” section:
    
    -   The exact file paths consumed (relative to repo root).
        
    -   Any sampling choices and seeds used in notebook computations.
        
    -   Any manual selection decisions (for example, “selected centrality table X among candidates Y and Z, because…”).
        
-   Maintain stable, deterministic ordering:
    
    -   Rankings must be stable, with deterministic tie-breaking (for example: sort by metric descending, then by label ascending).
        
    -   If sampling is required, it must be deterministic, and the seed must be stated.
        

### Robustness to missing artifacts

Opus should:

-   Never hard-fail the notebook program because an artifact is missing.
    
-   Continue analysis sections with:
    
    -   A clearly logged warning.
        
    -   “Not available” labels in narrative summaries.
        
    -   A concrete suggestion for which pipeline step likely needs rerun.
        
-   Maintain a consolidated “gaps” table:
    
    -   One row per missing or unreadable artifact.
        
    -   Include: expected location, detection method, impact on interpretation, likely pipeline step to rerun.
        

### No-code-in-this-guide constraint

-   This guide must not include code examples.
    
-   The notebooks Opus generates may include code as needed, but this guide describes notebook cells only in plain language.
    

* * *

## Project output map

### Expected output folders and typical file types

| Folder (relative to repo root) | Role in interpretation | Typical file types | Notes |
| --- | --- | --- | --- |
| `results/logs/` | Run manifests and step provenance | `.json` | Primary traceability source |
| `results/networks/` | Constructed network node/edge artifacts | `.parquet`, `.csv`, `.tsv` | Airport, flight, multilayer representations |
| `results/analysis/` | Analysis outputs | `.parquet`, `.csv`, `.tsv`, `.json` | Centrality, communities, robustness, delay cascades, embeddings, link prediction |
| `results/business/` | Airline-level and business-facing metrics | `.parquet`, `.csv`, `.json` | Hub concentration, disruption cost proxies, KPI summaries |
| `results/tables/` | Pipeline-emitted tables | `.csv`, `.tsv` | Report-ready tables may be here |
| `results/figures/` | Pipeline-emitted figures | `.png` | Final and intermediate figures |
| `results/tables/report/` | Notebook-generated report evidence tables | `.csv` | Canonical location for derived tables |
| `results/figures/report/` | Notebook-generated report evidence figures | `.png` | Canonical location for derived figures |
| `results/tables/report/_warnings.log` | Consolidated warnings | text | Must be appended to, not overwritten |

### Minimum run-completeness checklist

Opus should verify, at minimum:

**A. Manifests and provenance**

-   At least one manifest exists in `results/logs/`.
    
-   A manifest-to-disk reconciliation table can be produced.
    
-   A run index table (step, timestamp, git hash if available) can be produced into `results/tables/report/`.
    

**B. Network construction evidence**

-   At least one node table and one edge table exist under `results/networks/`.
    
-   Edge tables have identifiable endpoint columns (source/target or origin/dest) and a plausible weight/count column when expected.
    

**C. Core network science modules**

-   Centrality outputs: at least one table with node identifiers and numeric metric columns.
    
-   Communities outputs: at least one table with node identifiers and a community assignment column.
    
-   Robustness outputs: at least one table suitable for a curve (fraction removed vs largest component or equivalent).
    
-   Delay propagation outputs: at least one table enabling a cascade size distribution and/or superspreader ranking.
    

**D. ML and applied modules (when present)**

-   Embeddings and/or link prediction: at least one metrics artifact (often JSON or a metrics table) and, if available, top predictions table.
    
-   Business module: at least one airline-level summary or concentration/cost proxy output under `results/business/`.
    

**E. Report evidence writeability**

-   Opus can write at least one derived figure into `results/figures/report/` and one derived table into `results/tables/report/`.
    
-   Warnings can be appended to `results/tables/report/_warnings.log`.
    

* * *

## Notebook plan

Opus should produce multiple notebooks for readability and maintainability. The recommended set below is designed to be modular and align with the canonical reporting structure.

### Naming and placement conventions

-   Suggested location: `analysis/notebooks/`
    
-   Suggested naming: numeric prefix for stable ordering, descriptive slug, and scope tag.
    
    -   Example pattern: `01_run_inventory__provenance.ipynb`
        
-   Each notebook must:
    
    -   Declare its scope and which folders it reads.
        
    -   Write its report outputs to `results/tables/report/` and `results/figures/report/` using a notebook-specific prefix (for example, `nb03_community__...`).
        
    -   Append warnings to `results/tables/report/_warnings.log`.
        

* * *

### Notebook 01: Run inventory and manifest reconciliation

**Suggested filename**

-   `01_run_inventory__manifest_reconciliation.ipynb`
    

**Research goal / questions answered**

-   What pipeline steps ran, when, and with what run identity signals?
    
-   Which outputs are present, missing, duplicated, or inconsistent with manifests?
    
-   Is the run complete enough to support scientific interpretation?
    

**Inputs (artifact types and likely patterns)**

-   `results/logs/*.json` manifests
    
-   Any config snapshot artifacts if present (for example, `config/config.yaml`)
    

**Analyses to perform (cell-by-cell, plain English)**

1.  Markdown cell: state notebook scope, evidence rules, and output locations.
    
2.  Python cell: discover all manifests; summarize counts by step name and timestamp.
    
3.  Python cell: build a “manifest index” table:
    
    -   step name
        
    -   timestamp
        
    -   git hash if present
        
    -   number of outputs listed
        
    -   manifest path
        
4.  Python cell: reconcile manifest-listed outputs to disk:
    
    -   mark present vs missing
        
    -   compute missing-rate per step
        
5.  Python cell: generate a gaps table for missing/unreadable artifacts:
    
    -   include a “likely pipeline step to rerun” column.
        
6.  Markdown cell: interpret run completeness:
    
    -   explicitly list critical missing pieces that block downstream notebooks.
        
7.  Write outputs:
    
    -   run index table to `results/tables/report/`
        
    -   manifest reconciliation table to `results/tables/report/`
        
    -   gaps table to `results/tables/report/`
        

**Interpretation prompts**

-   Which steps are absent entirely?
    
-   Which steps list outputs that are not present on disk?
    
-   Are there multiple runs, and if so, which run is “canonical” for interpretation?
    

**Outputs to write**

-   `results/tables/report/nb01_run_index.csv`
    
-   `results/tables/report/nb01_manifest_reconciliation.csv`
    
-   `results/tables/report/nb01_missing_artifacts.csv`
    

**Quality checks**

-   Stable sorting by step name then timestamp.
    
-   Explicit statement of which run window is analyzed (date range).
    
-   If multiple manifest sets exist, Opus must justify which set is used for downstream analysis.
    

* * *

### Notebook 02: Network construction summaries (airport, flight, multilayer)

**Suggested filename**

-   `02_network_construction__structure_and_sanity.ipynb`
    

**Research goal / questions answered**

-   What networks were constructed and what is their structural signature?
    
-   Do node and edge tables look internally consistent and plausible for US flights?
    
-   Are hub-and-spoke and corridor patterns visible at the raw network level?
    

**Inputs**

-   `results/networks/` tables:
    
    -   node tables (airport identifiers, attributes if available)
        
    -   edge tables (endpoints and weights)
        
-   Optional: any pre-rendered network figures in `results/figures/`
    

**Analyses (cell-by-cell)**

1.  Markdown: define which network layers are expected and how they are detected.
    
2.  Python: inventory all network artifacts and summarize schema, row counts, and key columns.
    
3.  Python: detect edge tables and compute:
    
    -   top 20 origin-destination routes by weight/count
        
    -   degree or strength proxy distributions
        
4.  Python: compute basic integrity checks:
    
    -   missing endpoint rates
        
    -   self-loop rate
        
    -   duplicate edge rate
        
    -   weight non-negativity and heavy-tail checks
        
5.  Markdown: interpret network structure:
    
    -   identify trunk corridors and plausible mega-hub dominance
        
    -   discuss implications for downstream robustness and delay propagation
        

**Interpretation prompts**

-   Do top routes correspond to known high-traffic corridors?
    
-   Is the degree/strength proxy heavy-tailed, indicating hub dominance?
    
-   Are there anomalies (negative weights, extreme duplication, missing endpoints) that would bias centrality and robustness?
    

**Outputs**

-   `results/tables/report/nb02_network_inventory.csv`
    
-   `results/tables/report/nb02_top_routes_top20.csv`
    
-   `results/figures/report/nb02_top_routes_top20.png`
    
-   `results/figures/report/nb02_degree_strength_proxy_distribution.png`
    

**Quality checks**

-   Airport labeling quality check:
    
    -   resolve human-readable airport labels wherever possible.
        
    -   compute and report label resolution rate for the top 20 airports and warn if low.
        
-   Stable tie-breaking in top route rankings.
    

* * *

### Notebook 03: Centrality analysis (rankings, distributions, connector interpretation)

**Suggested filename**

-   `03_centrality__rankings_and_mechanisms.ipynb`
    

**Research goal / questions answered**

-   Which airports are structurally central under multiple centrality notions?
    
-   Do centrality distributions indicate extreme concentration or broad participation?
    
-   Which airports are plausible “connectors” (bridges) versus “volume hubs”?
    

**Inputs**

-   `results/analysis/` centrality-like tables:
    
    -   heuristic discovery for filenames containing centrality/rank/pagerank/betweenness/closeness/eigen
        
-   Optional: network node lookup for labels
    

**Analyses (cell-by-cell)**

1.  Markdown: state the centrality metrics expected and the meaning caveats for directed vs undirected and weighted vs unweighted graphs.
    
2.  Python: discover candidate centrality artifacts; choose the primary one explicitly if multiple exist.
    
3.  Python: identify node identifier column and numeric metric columns.
    
4.  Python: compute top-k rankings for each metric (k = 20 recommended):
    
    -   enforce stable tie-breaking
        
    -   attach `airport_label`
        
5.  Python: plot:
    
    -   bar charts for top-k per metric
        
    -   distribution plots for key metrics (log-scale when appropriate)
        
6.  Markdown: interpret findings:
    
    -   connector airports (high betweenness)
        
    -   globally influential hubs (high PageRank/eigenvector)
        
    -   cross-metric consistency and disagreements
        

**Interpretation prompts**

-   Are the same airports consistently top-ranked across metrics?
    
-   Which airports are high-betweenness but not top-degree, suggesting bridging roles?
    
-   How sensitive are conclusions to metric choice and graph definition?
    

**Outputs**

-   `results/tables/report/nb03_centrality_top20_by_metric.csv` (or one file per metric)
    
-   `results/figures/report/nb03_centrality_top20__<metric>.png`
    
-   `results/figures/report/nb03_centrality_distribution__<metric>.png`
    

**Quality checks**

-   Label resolution rate for top-k tables.
    
-   Explicit caveat section: metric semantics depend on graph construction choices.
    
-   Stable ordering and deterministic handling of missing values.
    

* * *

### Notebook 04: Community structure (size distribution and dominant airline/geography)

**Suggested filename**

-   `04_communities__structure_and_attributes.ipynb`
    

**Research goal / questions answered**

-   Does the network exhibit meaningful modular structure?
    
-   Are communities aligned with airline ecosystems, geography, or both?
    
-   Are there many tiny communities suggesting fragmentation or resolution issues?
    

**Inputs**

-   `results/analysis/` community-like tables:
    
    -   heuristic discovery for filenames containing community/leiden/partition/membership/cluster
        
-   Optional: airport attributes table if present (state/region/lat-lon)
    
-   Optional: airline identifiers if present in community tables
    

**Analyses (cell-by-cell)**

1.  Markdown: define the expected fields (node id, community id, optional airline).
    
2.  Python: discover community artifacts; select primary table explicitly.
    
3.  Python: compute community size distribution:
    
    -   stable sorting
        
    -   cap plots for readability (for example, top 30 communities)
        
4.  Python: if airline columns exist:
    
    -   compute dominant airline per community and its share
        
    -   quantify “purity” measures (best-effort, must be clearly defined)
        
5.  Python: if geography attributes exist:
    
    -   summarize community geographic concentration (best-effort)
        
6.  Markdown: interpret:
    
    -   airline-dominant communities as carrier subnetworks
        
    -   mixed communities around mega-hubs
        
    -   caveats about resolution parameters and weight definitions
        

**Interpretation prompts**

-   Do large communities correspond to major regions or airline networks?
    
-   Are dominant-airline communities evidence of hub-and-spoke strategies?
    
-   Are connectors between communities the same airports flagged in centrality?
    

**Outputs**

-   `results/tables/report/nb04_community_sizes.csv`
    
-   `results/figures/report/nb04_community_sizes_top30.png`
    
-   `results/tables/report/nb04_community_dominant_airline.csv` (if applicable)
    

**Quality checks**

-   Stable tie-breaking for community rankings.
    
-   Explicit note when airline/geography attributes are unavailable.
    
-   If community count is extremely high, Opus must justify any summarization strategy.
    

* * *

### Notebook 05: Robustness and percolation (random vs targeted attacks, critical hubs)

**Suggested filename**

-   `05_robustness__percolation_and_hub_dependence.ipynb`
    

**Research goal / questions answered**

-   How vulnerable is the network to random failures versus targeted hub removals?
    
-   What is the evidence for hub dependence and critical infrastructure?
    
-   Which nodes are consistently critical across scenarios?
    

**Inputs**

-   `results/analysis/` robustness-like tables:
    
    -   heuristic discovery for filenames containing robust/percol/attack/targeted/random/giant/lcc
        
-   Optional: any robustness figures already in `results/figures/`
    

**Analyses (cell-by-cell)**

1.  Markdown: define robustness curve semantics and key caveats (attack definition, dynamic vs static recomputation).
    
2.  Python: discover robustness artifacts; identify x-axis (fraction removed) and y-axis (largest component or equivalent).
    
3.  Python: plot robustness curves by scenario if scenario column exists:
    
    -   ensure consistent axes across scenarios
        
4.  Python: derive summary metrics:
    
    -   area under curve (AUC-like) where meaningful
        
    -   critical drop points (best-effort, clearly defined)
        
5.  Markdown: interpret:
    
    -   targeted vs random divergence as evidence of hub dependence
        
    -   operational meaning of “critical hubs”
        

**Interpretation prompts**

-   Does targeted removal produce a steep early collapse relative to random failure?
    
-   Are robustness results consistent with centrality and network concentration evidence?
    
-   What are plausible confounders (weight choices, directedness, airline-layer aggregation)?
    

**Outputs**

-   `results/figures/report/nb05_robustness_curves.png` (or per artifact)
    
-   `results/tables/report/nb05_robustness_summary_metrics.csv`
    

**Quality checks**

-   Scenario labeling clarity (random vs targeted must be explicit if present).
    
-   Stable sorting of scenario legends and deterministic plotting order.
    

* * *

### Notebook 06: Delay propagation (cascade distribution, superspreaders, overlap with centrality)

**Suggested filename**

-   `06_delay_propagation__cascades_and_superspreaders.ipynb`
    

**Research goal / questions answered**

-   Are delay cascades heavy-tailed, indicating rare but severe disruption episodes?
    
-   Which nodes act as delay superspreaders?
    
-   Do superspreaders overlap with structurally central connectors?
    

**Inputs**

-   `results/analysis/` delay propagation artifacts:
    
    -   heuristic discovery for filenames containing delay/propagation/cascade/spread/superspreader
        
-   Centrality outputs (from Notebook 03) for synthesis overlap
    

**Analyses (cell-by-cell)**

1.  Markdown: define cascade and impact metrics expected and how to interpret them.
    
2.  Python: discover delay artifacts; identify cascade size column and/or delay impact column.
    
3.  Python: compute cascade size distribution:
    
    -   frequency table
        
    -   distribution plot (log-scale when appropriate)
        
4.  Python: compute superspreader ranking:
    
    -   aggregate impact per node where meaningful
        
    -   stable tie-breaking
        
    -   attach `airport_label`
        
5.  Python: overlap analysis with centrality:
    
    -   join on node id where possible
        
    -   compute overlap rates between top-k centrality and top-k delay impact
        
6.  Markdown: interpret:
    
    -   heavy tails and operational risk
        
    -   connectors as amplifiers
        
    -   outliers: high delay impact but moderate structural centrality
        

**Interpretation prompts**

-   Are cascades dominated by small events with a long tail of rare large events?
    
-   Which nodes are risk concentrators, and do they match betweenness connectors?
    
-   What alternative explanations exist (weather exposure, capacity constraints, schedule banking)?
    

**Outputs**

-   `results/tables/report/nb06_cascade_size_distribution.csv`
    
-   `results/figures/report/nb06_cascade_size_distribution.png`
    
-   `results/tables/report/nb06_delay_superspreaders_top20.csv`
    
-   `results/figures/report/nb06_delay_superspreaders_top20.png`
    
-   `results/tables/report/nb06_centrality_delay_overlap.csv`
    

**Quality checks**

-   Label resolution rate for top superspreaders.
    
-   Explicit statement of whether delay propagation was simulated or empirically derived, based on artifact metadata.
    

* * *

### Notebook 07: Embeddings and link prediction (metrics, plausibility, evaluation caveats)

**Suggested filename**

-   `07_embeddings_linkpred__evaluation_and_plausibility.ipynb`
    

**Research goal / questions answered**

-   How well do embedding and link prediction models perform under reported metrics?
    
-   Are predicted “new routes” plausible and interpretable in network terms?
    
-   What are evaluation caveats (time split, leakage risk, class imbalance)?
    

**Inputs**

-   `results/analysis/` embedding and link prediction artifacts:
    
    -   heuristic discovery for filenames containing embed/embedding/node2vec/linkpred/auc/ap/mrr/hits
        
-   `results/tables/` or `results/analysis/` top predictions tables if present
    

**Analyses (cell-by-cell)**

1.  Markdown: define evaluation metrics and their limitations for imbalanced link prediction.
    
2.  Python: extract and normalize metrics:
    
    -   from JSON if present
        
    -   from tables if present
        
3.  Python: if embeddings exist as vectors:
    
    -   perform basic sanity checks (dimensions, missingness, norm distributions on a sample)
        
4.  Python: if top predictions exist:
    
    -   attach labels for endpoints
        
    -   compute plausibility descriptors:
        
        -   geographic plausibility if coordinates exist
            
        -   network plausibility (shared neighbors, existing carrier presence if attributes exist)
            
5.  Markdown: interpret:
    
    -   performance relative to baselines if present
        
    -   risk of spurious predictions
        
    -   how predicted links could reduce hub load or add redundancy (only if evidence supports)
        

**Interpretation prompts**

-   Are metrics consistent across methods or unstable?
    
-   Do top predictions cluster around mega-hubs (likely trivial) or fill structural gaps (more interesting)?
    
-   What aspects of the data split limit the external validity of predictions?
    

**Outputs**

-   `results/tables/report/nb07_linkpred_metrics_flat.csv`
    
-   `results/figures/report/nb07_embedding_norms_distribution.png` (if applicable)
    
-   `results/tables/report/nb07_top_predictions_annotated.csv` (if applicable)
    

**Quality checks**

-   Explicitly document which artifacts are used for metrics and which for predictions.
    
-   If no metrics are available, mark the section Not available and identify missing files.
    

* * *

### Notebook 08: Business module interpretation (hub dependence, disruption cost proxy, airline comparisons)

**Suggested filename**

-   `08_business__hub_strategy_and_resilience.ipynb`
    

**Research goal / questions answered**

-   How do airlines compare on hub concentration and operational performance signals?
    
-   Is there evidence of a trade-off between hub concentration and disruption vulnerability?
    
-   What is the business-facing interpretation of network science results?
    

**Inputs**

-   `results/business/` artifacts:
    
    -   heuristic discovery for airline/carrier/strategy/hub/concentration/cost/revenue/performance
        
-   Optional: robustness summaries (Notebook 05), delays (Notebook 06), centrality (Notebook 03)
    

**Analyses (cell-by-cell)**

1.  Markdown: define “network strategy signals” vs “performance signals” and the risk of ecological fallacies.
    
2.  Python: inventory all business artifacts; identify airline identifier column(s).
    
3.  Python: compute airline-level aggregates:
    
    -   sums for volume-like metrics
        
    -   means for rate-like metrics (must be explicitly justified per metric)
        
4.  Python: produce:
    
    -   top-15 airline comparisons for selected KPIs
        
    -   hub concentration exhibit (top-1 vs top-3 shares) if available
        
    -   disruption cost proxy exhibit if available
        
5.  Markdown: interpret:
    
    -   identify airlines with high concentration and high disruption costs as fragile strategies
        
    -   identify distributed strategies and discuss resilience implications
        
    -   caveat: cost proxies depend on parameter assumptions
        

**Interpretation prompts**

-   Which airlines appear most hub-concentrated, and does that align with robustness vulnerability?
    
-   Are delays and cancellations higher for more centralized carriers?
    
-   What assumptions underlie cost proxies, and how sensitive are conclusions?
    

**Outputs**

-   `results/tables/report/nb08_airline_kpi_summary.csv`
    
-   `results/figures/report/nb08_airline_kpi_top15__<metric>.png`
    
-   `results/figures/report/nb08_hub_concentration.png` (if applicable)
    
-   `results/figures/report/nb08_disruption_cost_proxy.png` (if applicable)
    

**Quality checks**

-   Clear separation of descriptive findings vs causal claims.
    
-   Explicit documentation of aggregation semantics for each KPI.
    

* * *

### Notebook 09: Cross-cutting synthesis and final research narrative

**Suggested filename**

-   `09_synthesis__integrated_findings.ipynb`
    

**Research goal / questions answered**

-   What is the integrated story across structure, dynamics, robustness, prediction, and business framing?
    
-   Which conclusions are strongly supported by multiple evidence streams?
    
-   Where do findings conflict, and what are plausible explanations?
    

**Inputs**

-   Derived report artifacts produced by Notebooks 02–08 from `results/tables/report/` and `results/figures/report/`
    
-   Optional: attempt direct best-effort joins across raw outputs when appropriate
    

**Analyses (cell-by-cell)**

1.  Markdown: define synthesis questions and strict evidence standard.
    
2.  Python: build a “master evidence index” table:
    
    -   list every report table and figure created, with a one-line description
        
3.  Python: centrality ↔ delay synthesis:
    
    -   attempt join on node id
        
    -   produce scatter and save joined sample table when feasible
        
4.  Python: communities ↔ airline/geography synthesis:
    
    -   quantify community dominance patterns where attributes exist
        
    -   relate community bridging nodes to centrality and delay superspreaders
        
5.  Python: robustness ↔ hub dependence synthesis:
    
    -   compare robustness vulnerability metrics to hub concentration proxies
        
6.  Python: link prediction ↔ plausible new routes synthesis:
    
    -   interpret whether predictions suggest redundancy additions or hub reinforcement
        
7.  Markdown: produce the final narrative using the Interpretation Framework template (below), including explicit caveats and Not available labels.
    

**Interpretation prompts**

-   Do structurally central airports also amplify delay cascades?
    
-   Are airline-dominant communities aligned with concentrated hub strategies?
    
-   Does the robustness curve evidence match hub concentration evidence?
    
-   Do predicted new routes suggest strategies for resilience improvement?
    

**Outputs**

-   `results/tables/report/nb09_evidence_index.csv`
    
-   `results/tables/report/nb09_synth_centrality_vs_delay_joined.csv` (if feasible)
    
-   `results/figures/report/nb09_synth_centrality_vs_delay_scatter.png` (if feasible)
    
-   `results/tables/report/nb09_synthesis_findings.csv` (structured bullet findings with evidence pointers)
    

**Quality checks**

-   Every synthesis statement must cite a table/figure path.
    
-   Any join performed must report join coverage (matched rows vs candidates).
    

* * *

### Notebook 10: Appendix (assumptions, limitations, reproducibility notes)

**Suggested filename**

-   `10_appendix__assumptions_limitations_reproducibility.ipynb`
    

**Research goal / questions answered**

-   What assumptions were required to interpret outputs?
    
-   What limitations constrain generalization and causal inference?
    
-   What is required to reproduce the run and all report evidence?
    

**Inputs**

-   `results/logs/` manifests
    
-   `results/tables/report/` evidence index
    
-   Optional: environment/config snapshots if present
    

**Analyses (cell-by-cell)**

1.  Markdown: list all assumptions made across notebooks (explicit and implicit).
    
2.  Markdown: enumerate limitations:
    
    -   data coverage, missing months, cancellation handling
        
    -   metric semantics dependence on network definition
        
    -   simulation realism and parameter sensitivity
        
    -   link prediction evaluation constraints
        
3.  Markdown: reproducibility checklist:
    
    -   artifacts needed to rerun analysis without recomputing pipeline
        
    -   artifacts needed to rerun pipeline, mapped to steps
        
4.  Python: generate an appendix table summarizing:
    
    -   each notebook’s inputs and outputs
        
    -   hash or file modification time for key report artifacts (best-effort)
        

**Outputs**

-   `results/tables/report/nb10_notebook_io_index.csv`
    
-   `results/tables/report/nb10_limitations_and_assumptions.csv` (structured bullets in table form)
    

**Quality checks**

-   “Not available” usage must be consistent and explicit.
    
-   No hidden assumptions. Every assumption must be enumerated.
    

* * *

## Interpretation framework (use consistently in every notebook)

Opus should use the following narrative template in each major notebook section.

### Template: interpretation block

**Key findings (evidence-grounded bullets)**

-   Bullet statements that include a quantitative anchor (rank, share, slope, delta) whenever possible.
    
-   If evidence is missing: start with **Not available** and specify the missing artifact.
    

**Evidence links**

-   List the exact paths to the relevant evidence artifacts, for example:
    
    -   Table: `results/tables/report/<name>.csv`
        
    -   Figure: `results/figures/report/<name>.png`
        

**Mechanistic explanation (network science reasoning)**

-   Explain _why_ the pattern is expected given known network mechanisms:
    
    -   hub dominance, brokerage/bridging, modularity structure, percolation thresholds, contagion-like cascades, embedding geometry.
        

**Alternative explanations and confounders**

-   At least 2 plausible confounders per major claim, such as:
    
    -   network definition choices (directed vs undirected, weight meaning)
        
    -   data coverage artifacts (missing routes, seasonal variation)
        
    -   aggregation artifacts (airline-level ecological fallacy)
        
    -   simulation parameterization effects
        

**Sensitivity / robustness notes**

-   Describe what would change the conclusion:
    
    -   alternative metric choice
        
    -   alternative time window
        
    -   alternative attack strategy
        
    -   alternative cascade metric
        
-   If sensitivity outputs do not exist, label as Not available and state what would be needed.
    

**Implications**

-   Operational implications (for reliability and resilience).
    
-   Research implications (for network science framing, future work, and validation).
    

* * *

## Deliverables

### Final checklist of what Opus must produce

**Notebook deliverables**

-   5 to 10 notebooks (recommended set: 10 notebooks above) placed under `analysis/notebooks/`.
    
-   Each notebook:
    
    -   runs end-to-end without requiring pipeline recomputation
        
    -   reads only from `results/` plus optional config snapshots
        
    -   writes report evidence into:
        
        -   `results/tables/report/`
            
        -   `results/figures/report/`
            
    -   appends warnings to `results/tables/report/_warnings.log`
        

**Evidence deliverables**

-   A figure index table: `results/tables/report/figure_index.csv`
    
-   A table index table: `results/tables/report/table_index.csv`
    
-   A master evidence index: `results/tables/report/evidence_index.csv` (may be produced in Notebook 09)
    

**Summary deliverables**

-   A short executive summary (1 to 2 pages equivalent) saved as:
    
    -   `results/tables/report/executive_summary.md` (if your repo allows markdown report outputs)
        
    -   If markdown outputs under report folders are discouraged in this repo, instead save:
        
        -   `results/tables/report/executive_summary.txt`
            
-   A synthesis findings table:
    
    -   `results/tables/report/synthesis_key_findings.csv`
        

### Guidance to keep outputs navigable

Opus should:

-   Use numbered section headings and a Table of Contents in each notebook.
    
-   Limit figures per notebook:
    
    -   Prefer a small number of high-value plots plus an appendix subsection for additional diagnostics.
        
-   Enforce consistent naming:
    
    -   Prefix artifacts with notebook id, for example `nb06_...`.
        
-   Avoid duplicative plots:
    
    -   If a figure already exists in `results/figures/` and is report-ready, embed it and only regenerate if necessary for consistency.
        
-   Use appendices rather than expanding the main narrative:
    
    -   Diagnostics, long-tailed tables, and sensitivity checks should go to a clearly labeled appendix section.
        

* * *

## If something is missing: troubleshooting guide (tied to pipeline steps and manifests)

Opus should treat this section as the first-response protocol when key artifacts are absent.

### Step-to-artifact mapping (best-effort)

-   `00_validate_inputs`
    
    -   Expect: validation manifests and any input sanity outputs
        
    -   If missing: interpretation can proceed, but flag increased risk of upstream data issues.
        
-   `01_build_airport_network`
    
    -   Expect: airport node/edge tables under `results/networks/`
        
    -   If missing: centrality, robustness, embeddings, and business modules may be blocked.
        
-   `02_build_flight_network`
    
    -   Expect: flight-centric network artifacts under `results/networks/`
        
    -   If missing: delay propagation and flight-level interpretations may be blocked.
        
-   `03_build_multilayer`
    
    -   Expect: multilayer artifacts under `results/networks/` or `results/analysis/`
        
    -   If missing: multilayer interpretations should be marked Not available.
        
-   `04_run_centrality`
    
    -   Expect: centrality tables under `results/analysis/`
        
    -   If missing: connector and hub inference should be qualified and rely on network proxies only.
        
-   `05_run_communities`
    
    -   Expect: membership or partition tables under `results/analysis/`
        
    -   If missing: modular structure claims must be Not available.
        
-   `06_run_robustness`
    
    -   Expect: robustness curve tables under `results/analysis/`
        
    -   If missing: hub dependence claims must rely on indirect evidence and be clearly qualified.
        
-   `07_run_delay_propagation`
    
    -   Expect: cascade distributions and superspreader tables under `results/analysis/` and/or `results/tables/`
        
    -   If missing: delay contagion narrative must be Not available.
        
-   `08_run_embeddings_linkpred`
    
    -   Expect: embedding artifacts and link prediction metrics under `results/analysis/`, plus prediction tables under `results/tables/`
        
    -   If missing: ML sections must be Not available and excluded from synthesis claims.
        
-   `09_run_business_module`
    
    -   Expect: airline summary, hub concentration, and disruption cost proxy outputs under `results/business/`
        
    -   If missing: business interpretation must be Not available or limited to non-business network findings.
        
-   `10_make_all_figures`
    
    -   Expect: report-ready figures under `results/figures/`
        
    -   If missing: notebooks should regenerate only the minimal figures needed into `results/figures/report/`.
        

### Concrete troubleshooting steps (what Opus should do)

-   First, use `results/logs/` manifests to determine whether the step ran:
    
    -   If the step is missing from manifests, suggest rerunning that step.
        
-   If the step exists in manifests but outputs are missing on disk:
    
    -   Treat as a failed or incomplete run.
        
    -   Record the discrepancy in `results/tables/report/nb01_missing_artifacts.csv`.
        
-   If outputs exist but are unreadable (schema issues, corruption):
    
    -   Record the file path, size, and last modified time.
        
    -   Attempt alternate readers only when format ambiguity exists (for example CSV vs TSV).
        
    -   Continue with warnings and mark affected analyses Not available.
        
-   If airport labels cannot be resolved:
    
    -   Treat it as a quality warning.
        
    -   Continue with raw IDs, but quantify label resolution rate and log it.
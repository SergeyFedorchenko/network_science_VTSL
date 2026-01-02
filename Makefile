# Makefile for Network Science Project (WS1)
# Usage: make <target>
# Note: Assumes PowerShell is available (SHELL configured below)

SHELL := pwsh
.SHELLFLAGS := -NoLogo -NoProfile -Command

.PHONY: help setup toy-data validate airport flight multilayer networks centrality communities robustness delay superspreaders embeddings business figures analysis pipeline all tests test-time test-validate test-network test-determinism test-toy test-small clean clean-logs clean-networks clean-analysis clean-results check-env list-results

help:
	@echo "Network Science Project Targets"
	@echo ""
	@echo "Setup:"
	@echo "  make setup            - Create conda environment"
	@echo "  make toy-data         - Generate toy dataset for testing"
	@echo ""
	@echo "Pipeline:"
	@echo "  make validate         - Run data validation (script 00)"
	@echo "  make airport          - Build airport network (01)"
	@echo "  make flight           - Build flight network (02)"
	@echo "  make multilayer       - Build multilayer network (03)"
	@echo "  make networks         - Run 01-03 network builds"
	@echo "  make centrality       - Run centrality metrics (04)"
	@echo "  make communities      - Run community detection (05)"
	@echo "  make robustness       - Run robustness analysis (06)"
	@echo "  make delay            - Run delay propagation (07)"
	@echo "  make superspreaders   - Compute airport superspreaders (07b)"
	@echo "  make embeddings       - Run embeddings & link prediction (08)"
	@echo "  make business         - Run business module (09)"
	@echo "  make figures          - Generate figures (10)"
	@echo "  make analysis         - Run analysis suite (04-09)"
	@echo "  make pipeline         - Run full pipeline (00-10)"
	@echo "  make all              - Alias for pipeline"
	@echo ""
	@echo "Testing:"
	@echo "  make tests            - Run all tests"
	@echo "  make test-time        - Time feature tests"
	@echo "  make test-validate    - Data validation tests"
	@echo "  make test-network     - Network construction tests"
	@echo "  make test-determinism - Seed determinism tests"
	@echo "  make test-toy         - Toy fixtures unit tests"
	@echo "  make test-small       - Small integration tests"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean-results    - Remove all results"
	@echo "  make clean-logs       - Remove logs only"
	@echo "  make clean-networks   - Remove network outputs only"
	@echo "  make clean-analysis   - Remove analysis outputs only"
	@echo "  make check-env        - Verify core deps are installed"
	@echo "  make list-results     - List results directory contents"
	@echo ""

# Setup environment
setup:
	conda env create -f environment.yml
	@echo "Activate with: conda activate network_science"

# Generate toy dataset
toy-data:
	python tests/fixtures/generate_toy_data.py

# Pipeline steps
validate:
	python scripts/00_validate_inputs.py

airport:
	python scripts/01_build_airport_network.py

flight:
	python scripts/02_build_flight_network.py

multilayer:
	python scripts/03_build_multilayer.py

networks: airport flight multilayer

centrality:
	python scripts/04_run_centrality.py

communities:
	python scripts/05_run_communities.py

robustness:
	python scripts/06_run_robustness.py

delay:
	python scripts/07_run_delay_propagation.py

superspreaders:
	python scripts/07b_compute_airport_superspreaders.py

embeddings:
	python scripts/08_run_embeddings_linkpred.py

business:
	python scripts/09_run_business_module.py

figures:
	python scripts/10_make_all_figures.py

analysis: centrality communities robustness delay superspreaders embeddings business

pipeline: validate networks analysis figures

all: pipeline

# Testing
tests:
	pytest tests/ -v

test-time:
	pytest tests/test_time_features.py -v

test-validate:
	pytest tests/test_validate_data.py -v

test-network:
	pytest tests/test_network_construction_small.py -v

test-determinism:
	pytest tests/test_seed_determinism.py -v

test-toy:
	pytest tests/test_*_toy.py -v

test-small:
	pytest tests/test_*_small.py -v

# Cleanup
clean-results:
	if (Test-Path results) { Remove-Item -Recurse -Force results }

clean-logs:
	if (Test-Path results/logs) { Remove-Item -Recurse -Force results/logs }

clean-networks:
	if (Test-Path results/networks) { Remove-Item -Recurse -Force results/networks }

clean-analysis:
	if (Test-Path results/analysis) { Remove-Item -Recurse -Force results/analysis }

# Development
check-env:
	python -c "import polars, igraph, leidenalg; print('All dependencies installed')"

list-results:
	@echo "Results directory contents:"
	@if (Test-Path results) { Get-ChildItem -Recurse results }

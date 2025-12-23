# Makefile for Network Science Project (WS1)
# Usage: make <target>
# Note: Requires PowerShell on Windows

.PHONY: help setup validate networks tests clean all

help:
	@echo "Network Science Project - WS1 Targets"
	@echo ""
	@echo "Setup:"
	@echo "  make setup      - Create conda environment"
	@echo "  make toy-data   - Generate toy dataset for testing"
	@echo ""
	@echo "Pipeline:"
	@echo "  make validate   - Run data validation (script 00)"
	@echo "  make networks   - Build both networks (scripts 01-02)"
	@echo "  make airport    - Build airport network only (script 01)"
	@echo "  make flight     - Build flight network only (script 02)"
	@echo "  make all        - Run complete WS1 pipeline (validate + networks)"
	@echo ""
	@echo "Testing:"
	@echo "  make tests      - Run all tests with pytest"
	@echo "  make test-time  - Run time feature tests only"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean      - Remove results/ directory"
	@echo "  make clean-logs - Remove logs only"
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

networks: airport flight

all: validate networks

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

# Cleanup
clean:
	rm -rf results/

clean-logs:
	rm -rf results/logs/*.log

clean-networks:
	rm -rf results/networks/*

# Development
check-env:
	python -c "import polars, igraph, leidenalg; print('✓ All dependencies installed')"

list-results:
	@echo "Results directory contents:"
	@ls -R results/

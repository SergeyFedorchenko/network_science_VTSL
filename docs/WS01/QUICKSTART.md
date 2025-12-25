# Quick Start Guide - WS1

## Prerequisites

1. **Python 3.11** installed via Conda
2. **Git** (optional, for tracking)
3. **Flight data** in parquet format

## Step-by-Step Setup

### 1. Create Environment

```powershell
# Navigate to project directory
cd c:\Users\aster\projects-source\network_science_VTSL

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate network_science

# Verify installation
python -c "import polars, igraph, leidenalg; print('✓ Ready!')"
```

### 2. Prepare Data

Place your cleaned dataset here:
```
data/cleaned/flights_2024.parquet
```

If you don't have data yet, use the toy dataset for testing:
```powershell
python tests/fixtures/generate_toy_data.py
# This creates tests/fixtures/toy_flights.parquet
```

### 3. Configure

Edit `config/config.yaml` if needed:
- Set correct data path
- Adjust scoping for flight graph (default: top 50 airports)
- Set include_cancelled flag

### 4. Run Pipeline

```powershell
# Step 1: Validate data
python scripts/00_validate_inputs.py

# Step 2: Build airport network
python scripts/01_build_airport_network.py

# Step 3: Build flight network
python scripts/02_build_flight_network.py
```

### 5. Check Outputs

```powershell
# View results
ls results/networks/
ls results/logs/

# Check logs
cat results/logs/00_validate_inputs.log
cat results/logs/01_build_airport_network_manifest.json
```

## Testing on Toy Dataset

```powershell
# Generate toy data
python tests/fixtures/generate_toy_data.py

# Update config to use toy data
# Edit config/config.yaml:
#   data:
#     cleaned_path: "tests/fixtures/toy_flights.parquet"

# Run pipeline
python scripts/00_validate_inputs.py
python scripts/01_build_airport_network.py
python scripts/02_build_flight_network.py

# Run tests
pytest tests/ -v
```

## Expected Results

**After script 00:**
- ✓ Validation summary CSV
- ✓ Data fingerprint JSON
- ✓ Log file

**After script 01:**
- ✓ airport_nodes.parquet (airports with IDs)
- ✓ airport_edges.parquet (routes with metrics)
- ✓ airport_graph.graphml (for inspection)
- ✓ Summary JSON with network stats

**After script 02:**
- ✓ flight_nodes.parquet (flights with timestamps)
- ✓ flight_edges.parquet (tail + route kNN edges)
- ✓ Summary JSON with edge type counts

## Common Issues

### Issue: "ModuleNotFoundError: No module named 'src'"

**Solution:** Always run from project root
```powershell
cd c:\Users\aster\projects-source\network_science_VTSL
python scripts/00_validate_inputs.py
```

### Issue: "File not found: data/cleaned/flights_2025.parquet"

**Solution:** Check data path in config.yaml or use absolute path
```yaml
data:
  cleaned_path: "C:/full/path/to/flights_2025.parquet"
```

### Issue: Scripts say "Output already exists"

**Solution:** Either delete results or set overwrite flag
```yaml
outputs:
  overwrite: true
```

Or delete specific outputs:
```powershell
rm results/networks/airport_nodes.parquet
```

## Next Steps

1. ✅ Validate outputs exist
2. ✅ Check log files for any warnings
3. ✅ Inspect network summaries (JSON files)
4. ✅ Run tests to verify correctness
5. ➡️ Hand off to WS2-4 for analysis

## Performance Tips

**For large datasets:**
- Reduce `flight_graph.scope.top_airports_k` from 50 to 20
- Use `mode: "sample"` with `sample_frac: 0.1`
- Monitor memory with Task Manager

**Expected runtimes (full dataset):**
- Validation: 30-60 seconds
- Airport network: 1-2 minutes
- Flight network: 2-5 minutes

## Getting Help

1. Check log files in `results/logs/`
2. Review [README.md](README.md) for detailed docs
3. See [WS1_SUMMARY.md](WS1_SUMMARY.md) for implementation details
4. Run tests to isolate issues: `pytest tests/ -v`

## Verification Checklist

- [ ] Environment activated
- [ ] Data file exists
- [ ] Config.yaml updated
- [ ] Script 00 passed (no validation errors)
- [ ] Script 01 produced outputs
- [ ] Script 02 produced outputs
- [ ] All tests pass
- [ ] Log files reviewed

**If all checked: WS1 complete! ✅**

---

Last updated: December 2025

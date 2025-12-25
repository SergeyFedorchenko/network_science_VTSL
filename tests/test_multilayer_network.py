"""
Test multilayer network construction.

Tests the airline-layered multilayer network building from airport network data.
"""
import pytest
import polars as pl
import sys
from pathlib import Path
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.networks.multilayer_network import (
    build_multilayer_airport_edges,
    build_interlayer_transfer_edges,
    compute_layer_summary,
    build_multilayer_network,
)


@pytest.fixture
def toy_flights() -> pl.LazyFrame:
    """Load toy flight data for testing."""
    project_root = Path(__file__).parent.parent
    data_path = project_root / "tests" / "fixtures" / "toy_flights.parquet"
    
    # Add FLIGHTS column if not present (for aggregation)
    lf = pl.scan_parquet(data_path)
    if "FLIGHTS" not in lf.collect_schema().names():
        lf = lf.with_columns(pl.lit(1).alias("FLIGHTS"))
    
    return lf


@pytest.fixture
def toy_airport_nodes() -> pl.DataFrame:
    """Create toy airport nodes DataFrame."""
    return pl.DataFrame({
        "node_id": [0, 1, 2],
        "code": ["JFK", "LAX", "ORD"],
        "city": ["New York, NY", "Los Angeles, CA", "Chicago, IL"],
        "state": ["New York", "California", "Illinois"],
    })


@pytest.fixture
def default_config() -> dict:
    """Create default multilayer config."""
    return {
        "layer_key": "OP_UNIQUE_CARRIER",
        "include_interlayer_transfer_edges": False,
    }


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestBuildMultilayerAirportEdges:
    """Tests for build_multilayer_airport_edges function."""
    
    def test_creates_edges_per_carrier(self, toy_flights, toy_airport_nodes, default_config):
        """Test that edges are created per carrier layer."""
        edges_df = build_multilayer_airport_edges(toy_flights, toy_airport_nodes, default_config)
        
        # Should have edges
        assert len(edges_df) > 0
        
        # Should have required columns
        required_cols = ["src_layer", "dst_layer", "src_id", "dst_id", "flight_count"]
        for col in required_cols:
            assert col in edges_df.columns, f"Missing column: {col}"
        
        # Should have edges for carriers in toy data
        carriers = edges_df["src_layer"].unique().to_list()
        assert len(carriers) >= 1
    
    def test_edge_weights_are_positive(self, toy_flights, toy_airport_nodes, default_config):
        """Test that edge weights (flight counts) are all positive."""
        edges_df = build_multilayer_airport_edges(toy_flights, toy_airport_nodes, default_config)
        
        assert all(edges_df["flight_count"] > 0)
    
    def test_handles_empty_flights(self, toy_airport_nodes, default_config):
        """Test handling of empty flight data."""
        empty_lf = pl.LazyFrame({
            "OP_UNIQUE_CARRIER": [],
            "ORIGIN": [],
            "DEST": [],
            "FLIGHTS": [],
            "DEP_DELAY": [],
            "ARR_DELAY": [],
            "DISTANCE": [],
        }).cast({
            "OP_UNIQUE_CARRIER": pl.Utf8, 
            "ORIGIN": pl.Utf8, 
            "DEST": pl.Utf8,
            "FLIGHTS": pl.Int64,
            "DEP_DELAY": pl.Float64,
            "ARR_DELAY": pl.Float64,
            "DISTANCE": pl.Float64,
        })
        
        edges_df = build_multilayer_airport_edges(empty_lf, toy_airport_nodes, default_config)
        assert len(edges_df) == 0


class TestBuildInterlayerTransferEdges:
    """Tests for build_interlayer_transfer_edges function."""
    
    def test_disabled_by_default(self, toy_flights, toy_airport_nodes, default_config):
        """Test that interlayer transfer edges are disabled by default."""
        interlayer_df = build_interlayer_transfer_edges(toy_flights, toy_airport_nodes, default_config)
        
        # Should be empty when disabled
        assert len(interlayer_df) == 0
    
    def test_creates_interlayer_edges_when_enabled(self, toy_flights, toy_airport_nodes):
        """Test that interlayer transfer edges are created when enabled."""
        config = {
            "layer_key": "OP_UNIQUE_CARRIER",
            "include_interlayer_transfer_edges": True,
        }
        
        interlayer_df = build_interlayer_transfer_edges(toy_flights, toy_airport_nodes, config)
        
        # Should have required columns
        required_cols = ["src_layer", "dst_layer", "src_id", "dst_id", "edge_type"]
        for col in required_cols:
            assert col in interlayer_df.columns, f"Missing column: {col}"


class TestComputeLayerSummary:
    """Tests for compute_layer_summary function."""
    
    def test_summary_columns(self, toy_flights, toy_airport_nodes, default_config):
        """Test that layer summary has expected columns."""
        layer_edges = build_multilayer_airport_edges(toy_flights, toy_airport_nodes, default_config)
        summary_df = compute_layer_summary(layer_edges)
        
        # Should have summary per carrier (column is "layer" not "src_layer")
        required_cols = ["layer", "edge_count", "total_flights"]
        for col in required_cols:
            assert col in summary_df.columns, f"Missing column: {col}"
    
    def test_summary_counts_correct(self, toy_flights, toy_airport_nodes, default_config):
        """Test that summary counts are consistent."""
        layer_edges = build_multilayer_airport_edges(toy_flights, toy_airport_nodes, default_config)
        summary_df = compute_layer_summary(layer_edges)
        
        # Total flights in summary should match sum of edge weights
        total_from_edges = layer_edges["flight_count"].sum()
        total_from_summary = summary_df["total_flights"].sum()
        
        assert total_from_edges == total_from_summary


class TestBuildMultilayerNetwork:
    """Tests for the main build_multilayer_network function."""
    
    def test_returns_summary_stats(self, toy_flights, toy_airport_nodes, default_config, temp_output_dir):
        """Test that the main function returns expected summary statistics."""
        result = build_multilayer_network(
            toy_flights, toy_airport_nodes, default_config, temp_output_dir
        )
        
        # Returns summary dict, not DataFrames
        expected_keys = ["n_layers", "n_edges", "n_interlayer_edges", "top_layers"]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
        
        # n_layers should be positive
        assert result["n_layers"] > 0
        assert result["n_edges"] > 0
    
    def test_creates_output_files(self, toy_flights, toy_airport_nodes, default_config, temp_output_dir):
        """Test that output parquet files are created."""
        build_multilayer_network(
            toy_flights, toy_airport_nodes, default_config, temp_output_dir
        )
        
        # Check output files exist
        assert (temp_output_dir / "multilayer_edges.parquet").exists()
        assert (temp_output_dir / "layer_summary.parquet").exists()
    
    def test_deterministic_output(self, toy_flights, toy_airport_nodes, default_config, temp_output_dir):
        """Test that output is deterministic."""
        result1 = build_multilayer_network(
            toy_flights, toy_airport_nodes, default_config, temp_output_dir
        )
        result2 = build_multilayer_network(
            toy_flights, toy_airport_nodes, default_config, temp_output_dir
        )
        
        # Summary stats should be identical
        assert result1["n_layers"] == result2["n_layers"]
        assert result1["n_edges"] == result2["n_edges"]

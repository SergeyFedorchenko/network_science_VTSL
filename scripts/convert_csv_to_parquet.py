"""
Convert CSV to Parquet format for efficient processing.
"""
import polars as pl
from pathlib import Path

# Input and output paths
csv_path = Path("data/united_flights/united_flights.csv")
parquet_path = Path("data/cleaned/flights_2024.parquet")

print(f"Reading CSV from: {csv_path}")
df = pl.read_csv(csv_path)

print(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
print(f"Columns: {df.columns}")

# Create output directory if needed
parquet_path.parent.mkdir(parents=True, exist_ok=True)

# Write to parquet
print(f"\nWriting Parquet to: {parquet_path}")
df.write_parquet(parquet_path)

print(f"✓ Conversion complete!")
print(f"  Original CSV size: {csv_path.stat().st_size / (1024*1024):.2f} MB")
print(f"  Parquet size: {parquet_path.stat().st_size / (1024*1024):.2f} MB")
print(f"  Compression ratio: {csv_path.stat().st_size / parquet_path.stat().st_size:.2f}x")

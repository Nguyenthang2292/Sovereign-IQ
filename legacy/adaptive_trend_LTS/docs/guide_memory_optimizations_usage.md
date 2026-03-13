# Memory Optimizations Usage Guide

This guide explains how to use the memory optimization features in the Adaptive Trend LTS module. These optimizations are designed to handle large datasets (backtesting) and reduce storage footprint (caching).

## Features

1.  **Memory-Mapped Arrays**: Process large datasets (e.g., years of tick data) without loading everything into RAM.
2.  **Data Compression**: Compress cached calculation results using `blosc` to save disk space (5-10x reduction).

## Configuration

You can enable these features in your `ATCConfig` or via the `config.yaml` file.

```python
from modules.adaptive_trend_LTS.utils.config import ATCConfig

config = ATCConfig(
    # ... other params ...
    
    # Enable Memory Mapping for Backtesting
    use_memory_mapped=True,
    
    # Enable Cache Compression
    use_compression=True,
    compression_level=5,      # 0-9 (higher = smaller size, slower)
    compression_algorithm="blosclz" # "blosclz", "lz4", "zlib", "zstd"
)
```

## 1. Memory-Mapped Arrays

**When to use:**
- You are backtesting on a very large CSV/Parquet file (> 1GB).
- You are running out of RAM (OOM errors).

**How it works:**
The system converts your source data into a binary memory-mapped file on disk. Python treats this file as an array in memory, but the OS handles paging data in/out of RAM as needed.

**Performance:**
- **RAM Usage**: ~0 MB (negligible)
- **Speed**: Slightly slower than pure RAM (due to disk I/O), but allows processing datasets larger than physical RAM.

**Example:**

```python
from modules.adaptive_trend_LTS.core.backtesting.dask_backtest import backtest_with_dask

# Run backtest with memory mapping
results = backtest_with_dask(
    historical_data_path="huge_data.csv",
    atc_config=config,
    use_memory_mapped=True
)
```

## 2. Data Compression

**When to use:**
- Your cache directory (`.cache/atc`) is taking up too much disk space.
- You want to reduce I/O load when saving/loading cache.

**How it works:**
Cache entries (intermediate calculation results like Moving Averages, Equity Curves) are compressed using high-performance `blosc` compression before being saved to disk.

**Performance:**
- **Storage**: 5x - 10x smaller files.
- **Speed**: Decompression is extremely fast (often faster than reading uncompressed data from slow disks). Compression adds a small CPU overhead.

**Example:**

```python
from modules.adaptive_trend_LTS.utils.cache_manager import get_cache_manager

# Initialize manager (happens automatically if configured in ATCConfig)
cache = get_cache_manager()

# Use normally - data will be compressed on disk
# The files will have .blosc extension
```

## Troubleshooting

- **Missing `blosc`**: Ensure you installed dependencies: `pip install blosc`.
- **Permission Errors**: On Windows, memory-mapped files might not be deleted immediately if they are still open. The system attempts to clean up old files automatically.
- **Performance is slow**: If using `use_memory_mapped` on a slow HDD, performance will degrade. Use an NVMe SSD for best results.


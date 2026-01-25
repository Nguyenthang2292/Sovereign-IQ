# Phase 5: Dask Integration for Out-of-Core Processing & Rust+Dask Hybrid

> **Scope**: Dask integration for Scanner (Plan A), Batch Processing (Plan B), Historical Data (Plan C), and Rust+Dask Hybrid (Plan D)  
> **Expected Performance Gain**: Unlimited dataset size, ~20% overhead, combined with Rust for speed  
> **Timeline**: 3–4 weeks  
> **Status**: 🔄 **PLANNED**

---

## 1. Mục tiêu

Triển khai Dask để xử lý dữ liệu lớn hơn RAM và tích hợp với Rust extensions hiện có để đạt được:

- **Xử lý dataset không giới hạn kích thước** (Out-of-Core Processing)
- **Giảm memory footprint** xuống ~10-20% so với hiện tại
- **Kết hợp tốc độ của Rust** với khả năng quản lý bộ nhớ của Dask
- **Tương thích ngược** với code hiện tại

## Expected Performance Gains

| Component | Current | Target (Dask) | Expected Benefit |
| --------- | ------- | ------------- | ---------------- |
| Scanner (1000+ symbols) | Limited by RAM | Unlimited | Process 10,000+ symbols |
| Batch Processing | Load all into RAM | Chunked processing | 90% memory reduction |
| Historical Data | Single file limit | Multi-file, streaming | Unlimited backtesting |
| Rust + Dask Hybrid | Rust only (RAM limit) | Rust per partition | Speed + Unlimited size |

---

## 2. Prerequisites & Dependencies

### 2.1 Required Software

#### Install Dask

```bash
# Core Dask
pip install dask[complete]

# For DataFrame operations
pip install dask[dataframe]

# For distributed computing (optional, for multi-machine)
pip install dask[distributed]

# Verify installation
python -c "import dask; print(dask.__version__)"
```

#### Verify Existing Rust Extensions

```python
# Ensure Rust extensions are built and working
try:
    import atc_rust
    print("✅ Rust extensions available")
except ImportError:
    print("❌ Build Rust extensions first: cd rust_extensions && maturin develop --release")
```

### 2.2 Required Knowledge

- Dask fundamentals (Bag, DataFrame, delayed)
- Out-of-Core processing concepts
- Memory management in Python
- Rust-Python interop (PyO3)
- Existing module architecture

### 2.3 Existing Code to Review

- [core/scanner/scan_all_symbols.py](../core/scanner/scan_all_symbols.py) – Main scanning entry point
- [core/scanner/sequential.py](../core/scanner/sequential.py) – Sequential processing
- [core/scanner/threadpool.py](../core/scanner/threadpool.py) – ThreadPool processing
- [core/compute_atc_signals/batch_processor.py](../core/compute_atc_signals/batch_processor.py) – Batch processing
- [rust_extensions/src/batch_processing.rs](../../rust_extensions/src/batch_processing.rs) – Rust batch implementation
- [rust_extensions/src/batch_processing_cpu.rs](../../rust_extensions/src/batch_processing_cpu.rs) – Rust CPU batch

### 2.4 Timeline Estimate

| Task | Estimated Time | Priority |
| ---- | -------------- | -------- |
| **Plan A: Scanner Integration** | 5 days | High |
| **Plan B: Batch Processing** | 4 days | High |
| **Plan C: Historical Data** | 3 days | Medium |
| **Plan D: Rust + Dask Hybrid** | 6 days | High |
| **Testing & Validation** | 3 days | High |
| **Documentation** | 2 days | Medium |
| **Total** | **~23 days (~3-4 weeks)** | |

---

## 3. Implementation Tasks

### 3.1 Plan A: Scanner/Scanning với Dask

#### Overview

Tích hợp Dask vào scanner để xử lý hàng nghìn symbols mà không cần load hết vào RAM.

**Expected Gain**: Process 10,000+ symbols với RAM giới hạn

**Status**: ✅ **COMPLETED**

---

#### 📋 Task 3.1.1: Create Dask Scanner Module ✅

##### 1. Create New File ✅

**Location**: `core/scanner/dask_scan.py` ✅ **IMPLEMENTED**

**Structure**:

```python
"""
Dask-based scanner for processing large symbol lists out-of-core.
"""
import dask.bag as db
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

from modules.common.utils import log_info, log_error, log_warn
from ..compute_atc_signals.compute_atc_signals import compute_atc_signals
from ...utils.config import ATCConfig


def _process_single_symbol_dask(
    symbol_data: Tuple[str, pd.Series],
    atc_config: ATCConfig,
    min_signal: float
) -> Optional[Dict[str, Any]]:
    """
    Process a single symbol (for Dask map operation).
    
    Args:
        symbol_data: Tuple of (symbol, prices_series)
        atc_config: ATC configuration
        min_signal: Minimum signal threshold
        
    Returns:
        Result dict or None if error
    """
    symbol, prices = symbol_data
    try:
        result = compute_atc_signals(
            prices,
            ema_len=atc_config.ema_len,
            hull_len=atc_config.hma_len,
            # ... other params
        )
        
        avg_signal = result.get("Average_Signal", pd.Series())
        if len(avg_signal) == 0:
            return None
            
        latest_signal = avg_signal.iloc[-1]
        if abs(latest_signal) < min_signal:
            return None
            
        return {
            "symbol": symbol,
            "signal": latest_signal,
            "trend": 1.0 if latest_signal > 0 else -1.0,
            "price": prices.iloc[-1] if len(prices) > 0 else None,
        }
    except Exception as e:
        log_warn(f"Error processing {symbol}: {e}")
        return None


def _scan_dask(
    symbols: List[str],
    data_fetcher: "DataFetcher",
    atc_config: ATCConfig,
    min_signal: float = 0.01,
    npartitions: Optional[int] = None,
    batch_size: int = 100,
) -> Tuple[List[Dict[str, Any]], int, int, List[str]]:
    """
    Scan symbols using Dask for out-of-core processing.
    
    Args:
        symbols: List of symbols to scan
        data_fetcher: DataFetcher instance
        atc_config: ATC configuration
        min_signal: Minimum signal threshold
        npartitions: Number of Dask partitions (auto if None)
        batch_size: Symbols per partition
        
    Returns:
        Tuple of (results, processed_count, error_count, skipped_symbols)
    """
    if not symbols:
        return [], 0, 0, []
    
    # Auto-determine partitions if not specified
    if npartitions is None:
        npartitions = max(1, len(symbols) // batch_size)
    
    log_info(f"Starting Dask scan for {len(symbols)} symbols with {npartitions} partitions")
    
    # Fetch data for all symbols (lazy, will be chunked by Dask)
    def fetch_symbol_data(symbol: str) -> Tuple[str, pd.Series]:
        try:
            prices = data_fetcher.fetch_ohlcv(symbol, atc_config.limit)
            return (symbol, prices)
        except Exception as e:
            log_warn(f"Failed to fetch {symbol}: {e}")
            return (symbol, pd.Series(dtype=float))
    
    # Create Dask bag from symbols
    symbols_bag = db.from_sequence(symbols, npartitions=npartitions)
    
    # Fetch data (lazy)
    data_bag = symbols_bag.map(fetch_symbol_data)
    
    # Process each symbol (lazy)
    results_bag = data_bag.map(
        lambda x: _process_single_symbol_dask(x, atc_config, min_signal)
    )
    
    # Compute results (triggers execution)
    results_list = results_bag.compute()
    
    # Filter out None results
    results = [r for r in results_list if r is not None]
    skipped = [s for s in symbols if s not in [r["symbol"] for r in results]]
    
    processed = len(results)
    error_count = len(symbols) - processed - len(skipped)
    
    return results, processed, error_count, skipped
```

**Deliverable**: `core/scanner/dask_scan.py` with basic Dask scanner ✅

---

#### 📋 Task 3.1.2: Integrate Dask Scanner into scan_all_symbols.py ✅

##### 1. Add Dask Import ✅

**File**: `core/scanner/scan_all_symbols.py` ✅ **IMPLEMENTED**

```python
# Add import
from .dask_scan import _scan_dask
```

##### 2. Add Execution Mode ✅

**Location**: In `scan_all_symbols()` function

```python
def scan_all_symbols(
    data_fetcher: "DataFetcher",
    atc_config: ATCConfig,
    max_symbols: Optional[int] = None,
    min_signal: float = 0.01,
    execution_mode: str = "threadpool",  # Add "dask" option
    max_workers: Optional[int] = None,
    batch_size: int = 100,
    npartitions: Optional[int] = None,  # New parameter for Dask
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execution modes:
    - "sequential": Process symbols one by one
    - "threadpool": Use ThreadPoolExecutor (default)
    - "asyncio": Use asyncio for parallel fetching
    - "dask": Use Dask for out-of-core processing (NEW)
    """
    # ... existing code ...
    
    # Add Dask mode
    if execution_mode == "dask":
        results, processed, errors, skipped = _scan_dask(
            symbols,
            data_fetcher,
            atc_config,
            min_signal,
            npartitions=npartitions,
            batch_size=batch_size,
        )
    # ... rest of function ...
```

**Deliverable**: Updated `scan_all_symbols.py` with Dask support ✅

---

#### 📋 Task 3.1.3: Optimize Memory Usage ✅

##### 1. Implement Lazy Data Fetching ✅

**Goal**: Only fetch data when partition is processed ✅ **IMPLEMENTED**

```python
def _fetch_partition_lazy(partition_symbols: List[str], data_fetcher, config):
    """Fetch data for a partition only when needed."""
    partition_data = {}
    for symbol in partition_symbols:
        try:
            prices = data_fetcher.fetch_ohlcv(symbol, config.limit)
            partition_data[symbol] = prices
        except Exception as e:
            log_warn(f"Failed to fetch {symbol}: {e}")
    return partition_data
```

##### 2. Add Garbage Collection Between Partitions ✅

```python
import gc

def _process_partition_with_gc(partition_data, atc_config, min_signal):
    """Process partition and force GC."""
    results = []
    for symbol, prices in partition_data.items():
        result = _process_single_symbol_dask((symbol, prices), atc_config, min_signal)
        if result:
            results.append(result)
    # Force GC after each partition
    gc.collect()
    return results
```

**Deliverable**: Memory-optimized Dask scanner ✅

---

#### 📋 Task 3.1.4: Add Progress Tracking ✅

##### 1. Implement Progress Callback ✅

**Status**: ✅ **IMPLEMENTED** in `dask_scan.py` (lines 169-182)

```python
from dask.callbacks import Callback

class ProgressCallback(Callback):
    def __init__(self, total_symbols: int):
        self.total = total_symbols
        self.processed = 0
        
    def _start(self, dsk):
        log_info(f"Starting Dask computation for {self.total} symbols")
        
    def _posttask(self, key, result, dsk, state):
        self.processed += 1
        if self.processed % 10 == 0:
            log_info(f"Progress: {self.processed}/{self.total} symbols processed")
```

##### 2. Use Callback in Scanner ✅

```python
with ProgressCallback(len(symbols)):
    results_list = results_bag.compute()
```

**Status**: ✅ **IMPLEMENTED** in `dask_scan.py` (lines 211-215)

**Deliverable**: Progress tracking for Dask operations ✅

---

#### ✅ Task 3.1 Completion Summary

**All subtasks completed:**

- ✅ Task 3.1.1: Dask scanner module created (`core/scanner/dask_scan.py`)
- ✅ Task 3.1.2: Integrated into `scan_all_symbols.py` with execution_mode="dask"
- ✅ Task 3.1.3: Memory optimization with lazy fetching and GC implemented
- ✅ Task 3.1.4: Progress tracking with Callback implemented

**Implementation Details:**

- File location: `modules/adaptive_trend_LTS/core/scanner/dask_scan.py`
- Integration: `modules/adaptive_trend_LTS/core/scanner/scan_all_symbols.py` (line 38, 222-225)
- Key functions implemented:
  - `_process_single_symbol_dask()` - Process individual symbols
  - `_fetch_partition_lazy()` - Lazy data fetching per partition
  - `_process_partition_with_gc()` - Partition processing with GC
  - `ProgressCallback` - Progress tracking callback
  - `_scan_dask()` - Main Dask scanner function

**Status**: ✅ **TASK 3.1 FULLY COMPLETED**

---

### 3.2 Plan B: Batch Processing với Dask

#### Overview

Tích hợp Dask vào batch processor để xử lý batch lớn mà không cần load hết vào RAM.

**Expected Gain**: 90% memory reduction for large batches

**Status**: ✅ **COMPLETED**

---

#### 📋 Task 3.2.1: Create Dask Batch Processor ✅

##### 1. Create New File

**Location**: `core/compute_atc_signals/dask_batch_processor.py`

**Structure**:

```python
"""
Dask-based batch processor for out-of-core symbol processing.
"""
import dask.bag as db
from typing import Dict, Optional
import numpy as np
import pandas as pd

from modules.common.utils import log_info, log_error
from .batch_processor import process_symbols_batch_rust, process_symbols_batch_cuda


def process_symbols_batch_dask(
    symbols_data: Dict[str, pd.Series],
    config: dict,
    use_rust: bool = True,
    use_cuda: bool = False,
    npartitions: Optional[int] = None,
    partition_size: int = 50,
) -> Dict[str, Dict[str, pd.Series]]:
    """
    Process symbols in batches using Dask for out-of-core processing.
    
    Args:
        symbols_data: Dictionary of symbol -> price Series
        config: ATC configuration
        use_rust: Use Rust backend (default: True)
        use_cuda: Use CUDA backend if available (default: False)
        npartitions: Number of Dask partitions (auto if None)
        partition_size: Symbols per partition
        
    Returns:
        Dictionary mapping symbol -> {"Average_Signal": pd.Series}
    """
    if not symbols_data:
        return {}
    
    # Auto-determine partitions
    if npartitions is None:
        npartitions = max(1, len(symbols_data) // partition_size)
    
    log_info(f"Processing {len(symbols_data)} symbols with Dask ({npartitions} partitions)")
    
    # Convert dict to list of tuples for Dask
    symbols_items = list(symbols_data.items())
    
    # Create Dask bag
    symbols_bag = db.from_sequence(symbols_items, npartitions=npartitions)
    
    # Process each partition
    def process_partition(partition: list) -> Dict[str, Dict[str, pd.Series]]:
        """Process a partition of symbols."""
        partition_dict = dict(partition)
        
        # Use existing batch processor (Rust or CUDA)
        if use_cuda:
            return process_symbols_batch_cuda(partition_dict, config)
        elif use_rust:
            return process_symbols_batch_rust(partition_dict, config)
        else:
            # Fallback to Python
            from .compute_atc_signals import compute_atc_signals
            results = {}
            for symbol, prices in partition_dict.items():
                try:
                    result = compute_atc_signals(prices, **config)
                    results[symbol] = {"Average_Signal": result.get("Average_Signal", pd.Series())}
                except Exception as e:
                    log_error(f"Error processing {symbol}: {e}")
            return results
    
    # Map partitions and compute
    results_bag = symbols_bag.map_partitions(process_partition)
    results_list = results_bag.compute()
    
    # Merge all partition results
    final_results = {}
    for partition_results in results_list:
        final_results.update(partition_results)
    
    return final_results
```

**Deliverable**: `core/compute_atc_signals/dask_batch_processor.py` ✅

---

#### 📋 Task 3.2.2: Integrate with Existing Batch Processor ✅

##### 1. Update batch_processor.py

**File**: `core/compute_atc_signals/batch_processor.py`

```python
# Add import
from .dask_batch_processor import process_symbols_batch_dask

# Add new function
def process_symbols_batch_with_dask(
    symbols_data: Dict[str, pd.Series],
    config: dict,
    use_dask: bool = True,
    npartitions: Optional[int] = None,
    **kwargs
) -> Dict[str, Dict[str, pd.Series]]:
    """
    Process symbols with optional Dask for out-of-core processing.
    
    Args:
        symbols_data: Dictionary of symbol -> price Series
        config: ATC configuration
        use_dask: Use Dask if data is large (default: True)
        npartitions: Number of Dask partitions
        **kwargs: Passed to batch processor
        
    Returns:
        Dictionary mapping symbol -> {"Average_Signal": pd.Series}
    """
    # Auto-detect if Dask is needed (e.g., >1000 symbols)
    if use_dask and len(symbols_data) > 1000:
        return process_symbols_batch_dask(
            symbols_data,
            config,
            npartitions=npartitions,
            **kwargs
        )
    else:
        # Use existing batch processor
        if kwargs.get("use_cuda", False):
            return process_symbols_batch_cuda(symbols_data, config, **kwargs)
        else:
            return process_symbols_batch_rust(symbols_data, config, **kwargs)
```

**Deliverable**: Updated batch processor with Dask option ✅

---

#### ✅ Task 3.2 Completion Summary

**All subtasks completed:**

- ✅ Task 3.2.1: Dask batch processor created (`core/compute_atc_signals/dask_batch_processor.py`)
- ✅ Task 3.2.2: Integrated into `batch_processor.py` with `process_symbols_batch_with_dask()`
- ✅ Comprehensive tests added (`tests/adaptive_trend_LTS/test_dask_batch_processor.py`)

**Implementation Details:**

- File location: `modules/adaptive_trend_LTS/core/compute_atc_signals/dask_batch_processor.py`
- Integration: `modules/adaptive_trend_LTS/core/compute_atc_signals/batch_processor.py` (line 199-253)
- Key functions implemented:
  - `_process_partition_python()` - Python fallback for partition processing
  - `_process_partition_with_backend()` - Unified partition processing with backend selection
  - `process_symbols_batch_dask()` - Main Dask batch processor
  - `process_symbols_batch_with_dask()` - Wrapper function with auto-detection

**Features:**

- Auto-detects when to use Dask (>1000 symbols)
- Supports Rust CPU, CUDA, and Python backends
- Python fallback on backend errors
- Memory-optimized with garbage collection between partitions
- Configurable partition size and count

**Status**: ✅ **TASK 3.2 FULLY COMPLETED**

---

### 3.3 Plan C: Historical Data Processing với Dask

#### Overview

Sử dụng Dask để xử lý dữ liệu lịch sử lớn cho backtesting.

**Expected Gain**: Unlimited backtesting dataset size

**Status**: ✅ **COMPLETED**

---

#### 📋 Task 3.3.1: Create Dask Historical Data Processor ✅

##### 1. Create New File

**Location**: `core/backtesting/dask_backtest.py`

**Structure**:

```python
"""
Dask-based backtesting for large historical datasets.
"""
import dask.dataframe as dd
from typing import Dict, List, Optional
import pandas as pd

from modules.common.utils import log_info
from ..compute_atc_signals.compute_atc_signals import compute_atc_signals
from ...utils.config import ATCConfig


def backtest_with_dask(
    historical_data_path: str,
    atc_config: ATCConfig,
    chunksize: str = "100MB",
    symbol_column: str = "symbol",
    price_column: str = "close",
) -> pd.DataFrame:
    """
    Backtest ATC signals on large historical data using Dask.
    
    Args:
        historical_data_path: Path to CSV/Parquet file
        atc_config: ATC configuration
        chunksize: Size of each chunk (e.g., "100MB")
        symbol_column: Column name for symbol
        price_column: Column name for price
        
    Returns:
        DataFrame with backtest results
    """
    log_info(f"Loading historical data from {historical_data_path}")
    
    # Read with Dask (lazy)
    ddf = dd.read_csv(
        historical_data_path,
        blocksize=chunksize,
        dtype={symbol_column: "string", price_column: "float64"}
    )
    
    # Group by symbol
    grouped = ddf.groupby(symbol_column)
    
    # Process each symbol group
    def process_symbol_group(group_df: pd.DataFrame) -> pd.DataFrame:
        """Process a single symbol's data."""
        symbol = group_df[symbol_column].iloc[0]
        prices = group_df[price_column].sort_index()
        
        try:
            result = compute_atc_signals(
                prices,
                ema_len=atc_config.ema_len,
                # ... other params
            )
            
            avg_signal = result.get("Average_Signal", pd.Series())
            
            return pd.DataFrame({
                "symbol": [symbol] * len(avg_signal),
                "signal": avg_signal.values,
                "price": prices.values,
                "timestamp": prices.index,
            })
        except Exception as e:
            log_info(f"Error processing {symbol}: {e}")
            return pd.DataFrame()
    
    # Apply to each group and compute
    results_ddf = grouped.apply(process_symbol_group, meta={
        "symbol": "string",
        "signal": "float64",
        "price": "float64",
        "timestamp": "datetime64[ns]",
    })
    
    # Compute results
    results_df = results_ddf.compute()
    
    return results_df
```

**Deliverable**: `core/backtesting/dask_backtest.py` ✅

---

#### 📋 Task 3.3.2: Add Multi-File Support ✅

##### 1. Support Multiple Files

```python
def backtest_multiple_files_dask(
    file_paths: List[str],
    atc_config: ATCConfig,
    chunksize: str = "100MB",
) -> pd.DataFrame:
    """
    Backtest across multiple historical data files.
    
    Args:
        file_paths: List of file paths
        atc_config: ATC configuration
        chunksize: Size of each chunk
        
    Returns:
        Combined DataFrame with all results
    """
    import dask.bag as db
    
    # Create Dask bag from files
    files_bag = db.from_sequence(file_paths)
    
    # Process each file
    results_bag = files_bag.map(
        lambda path: backtest_with_dask(path, atc_config, chunksize)
    )
    
    # Compute and concatenate
    results_list = results_bag.compute()
    combined_df = pd.concat(results_list, ignore_index=True)
    
    return combined_df
```

**Deliverable**: Multi-file backtesting support ✅

---

#### ✅ Task 3.3 Completion Summary

**All subtasks completed:**
- ✅ Task 3.3.1: Dask backtesting module created (`core/backtesting/dask_backtest.py`)
- ✅ Task 3.3.2: Multi-file support added with `backtest_multiple_files_dask()`
- ✅ Comprehensive tests added (`tests/adaptive_trend_LTS/test_dask_backtest.py`)

**Implementation Details:**
- File location: `modules/adaptive_trend_LTS/core/backtesting/dask_backtest.py`
- Key functions implemented:
  - `_process_symbol_group()` - Process single symbol's historical data
  - `backtest_with_dask()` - Backtest from CSV/Parquet file using Dask DataFrame
  - `backtest_from_dataframe()` - Backtest from existing DataFrame using Dask
  - `backtest_multiple_files_dask()` - Backtest across multiple files

**Features:**
- Lazy loading with Dask DataFrame chunked reading
- Configurable chunk size (e.g., "10MB", "100MB")
- Automatic partitioning with configurable partition count
- Memory-optimized with garbage collection between partitions
- Support for custom column names (symbol, price)
- Multi-file concatenation for large datasets
- Comprehensive error handling for missing files and invalid data

**Status**: ✅ **TASK 3.3 FULLY COMPLETED**

---

### 3.4 Plan D: Rust + Dask Hybrid Integration

#### Overview

Kết hợp tốc độ của Rust với khả năng quản lý bộ nhớ của Dask.

**Expected Gain**: Speed of Rust + Unlimited dataset size

**Status**: ✅ **COMPLETED**

---

#### 📋 Task 3.4.1: Create Rust-Dask Bridge ✅

##### 1. Create Wrapper Function

**Location**: `core/compute_atc_signals/rust_dask_bridge.py`

**Structure**:

```python
"""
Bridge between Rust batch processing and Dask for optimal performance.
"""
import dask.bag as db
from typing import Dict, Optional
import numpy as np
import pandas as pd

from modules.common.utils import log_info

try:
    import atc_rust
    HAS_RUST = True
except ImportError:
    HAS_RUST = False
    log_info("Rust extensions not available, falling back to Python")


def process_partition_with_rust(
    partition_data: Dict[str, np.ndarray],
    config: dict,
) -> Dict[str, Dict[str, pd.Series]]:
    """
    Process a partition using Rust batch processing.
    
    Args:
        partition_data: Dictionary of symbol -> price array
        config: ATC configuration
        
    Returns:
        Dictionary mapping symbol -> {"Average_Signal": pd.Series}
    """
    if not HAS_RUST:
        # Fallback to Python
        from .compute_atc_signals import compute_atc_signals
        results = {}
        for symbol, prices_array in partition_data.items():
            prices = pd.Series(prices_array)
            try:
                result = compute_atc_signals(prices, **config)
                results[symbol] = {"Average_Signal": result.get("Average_Signal", pd.Series())}
            except Exception as e:
                log_info(f"Error processing {symbol}: {e}")
        return results
    
    # Prepare config for Rust
    params = config.copy()
    la = params.get("La", params.get("la", 0.02))
    de = params.get("De", params.get("de", 0.03))
    la_scaled = la / 1000.0
    de_scaled = de / 100.0
    
    # Convert to format Rust expects
    symbols_numpy = {}
    for s, v in partition_data.items():
        if v is not None:
            if isinstance(v, pd.Series):
                symbols_numpy[s] = v.values.astype(np.float64)
            elif isinstance(v, np.ndarray):
                symbols_numpy[s] = v.astype(np.float64)
            else:
                symbols_numpy[s] = np.array(v, dtype=np.float64)
    
    # Call Rust batch function
    try:
        batch_results = atc_rust.compute_atc_signals_batch_cpu(
            symbols_numpy,
            ema_len=params.get("ema_len", 28),
            hull_len=params.get("hull_len", 28),
            wma_len=params.get("wma_len", 28),
            dema_len=params.get("dema_len", 28),
            lsma_len=params.get("lsma_len", 28),
            kama_len=params.get("kama_len", 28),
            robustness=params.get("robustness", "Medium"),
            La=la_scaled,
            De=de_scaled,
            long_threshold=params.get("long_threshold", 0.1),
            short_threshold=params.get("short_threshold", -0.1),
            cutout=params.get("cutout", 0),
        )
        
        # Convert Rust results to Python format
        results = {}
        for symbol, signal_array in batch_results.items():
            results[symbol] = {
                "Average_Signal": pd.Series(signal_array)
            }
        
        return results
    except Exception as e:
        log_info(f"Rust processing failed: {e}, falling back to Python")
        # Fallback to Python
        from .compute_atc_signals import compute_atc_signals
        results = {}
        for symbol, prices_array in partition_data.items():
            prices = pd.Series(prices_array)
            try:
                result = compute_atc_signals(prices, **config)
                results[symbol] = {"Average_Signal": result.get("Average_Signal", pd.Series())}
            except Exception as e:
                log_info(f"Error processing {symbol}: {e}")
        return results


def process_symbols_rust_dask(
    symbols_data: Dict[str, pd.Series],
    config: dict,
    npartitions: Optional[int] = None,
    partition_size: int = 50,
) -> Dict[str, Dict[str, pd.Series]]:
    """
    Process symbols using Rust + Dask hybrid approach.
    
    Args:
        symbols_data: Dictionary of symbol -> price Series
        config: ATC configuration
        npartitions: Number of Dask partitions (auto if None)
        partition_size: Symbols per partition
        
    Returns:
        Dictionary mapping symbol -> {"Average_Signal": pd.Series}
    """
    if not symbols_data:
        return {}
    
    # Auto-determine partitions
    if npartitions is None:
        npartitions = max(1, len(symbols_data) // partition_size)
    
    log_info(f"Processing {len(symbols_data)} symbols with Rust+Dask ({npartitions} partitions)")
    
    # Convert to list of tuples
    symbols_items = list(symbols_data.items())
    
    # Create Dask bag
    symbols_bag = db.from_sequence(symbols_items, npartitions=npartitions)
    
    # Group into partitions
    def create_partition(partition_items: list) -> Dict[str, np.ndarray]:
        """Create a partition dict from items."""
        partition_dict = {}
        for symbol, prices in partition_items:
            if isinstance(prices, pd.Series):
                partition_dict[symbol] = prices.values
            elif isinstance(prices, np.ndarray):
                partition_dict[symbol] = prices
            else:
                partition_dict[symbol] = np.array(prices, dtype=np.float64)
        return partition_dict
    
    # Create partitions
    partitions_bag = symbols_bag.map_partitions(
        lambda items: [create_partition(items)]
    )
    
    # Process each partition with Rust
    results_bag = partitions_bag.map(
        lambda partition: process_partition_with_rust(partition, config)
    )
    
    # Compute results
    results_list = results_bag.compute()
    
    # Merge all results
    final_results = {}
    for partition_results in results_list:
        final_results.update(partition_results)
    
    return final_results
```

**Deliverable**: `core/compute_atc_signals/rust_dask_bridge.py` ✅

---

#### 📋 Task 3.4.2: Optimize Partition Size ✅

##### 1. Auto-Tune Partition Size

```python
def auto_tune_partition_size(
    total_symbols: int,
    available_memory_gb: float = 8.0,
    target_memory_per_partition_gb: float = 0.5,
) -> int:
    """
    Auto-determine optimal partition size based on available memory.
    
    Args:
        total_symbols: Total number of symbols
        available_memory_gb: Available RAM in GB
        target_memory_per_partition_gb: Target memory per partition
        
    Returns:
        Optimal partition size
    """
    # Estimate memory per symbol (rough estimate: 1MB per symbol)
    memory_per_symbol_mb = 1.0
    memory_per_symbol_gb = memory_per_symbol_mb / 1024.0
    
    # Calculate max symbols per partition
    max_symbols_per_partition = int(
        target_memory_per_partition_gb / memory_per_symbol_gb
    )
    
    # Ensure reasonable bounds
    min_partition_size = 10
    max_partition_size = 200
    
    partition_size = max(
        min_partition_size,
        min(max_symbols_per_partition, max_partition_size)
    )
    
    return partition_size
```

**Deliverable**: Auto-tuning partition size logic ✅

---

#### ✅ Task 3.4 Completion Summary

**All subtasks completed:**
- ✅ Task 3.4.1: Rust-Dask bridge created (`core/compute_atc_signals/rust_dask_bridge.py`)
- ✅ Task 3.4.2: Partition size auto-tuning added with `auto_tune_partition_size()`
- ✅ Comprehensive tests added (`tests/adaptive_trend_LTS/test_rust_dask_bridge.py`)

**Implementation Details:**
- File location: `modules/adaptive_trend_LTS/core/compute_atc_signals/rust_dask_bridge.py`
- Key functions implemented:
  - `_process_partition_with_rust_cpu()` - Rust CPU partition processing
  - `_process_partition_with_rust_cuda()` - Rust CUDA partition processing
  - `_process_partition_python()` - Python fallback processing
  - `auto_tune_partition_size()` - Auto-determine optimal partition size
  - `process_symbols_rust_dask()` - Main Rust+Dask hybrid processor

**Features:**
- Auto-detects and uses Rust extensions when available
- Falls back to Python backend when Rust unavailable
- Supports both CPU and CUDA Rust backends
- Memory-optimized with garbage collection between partitions
- Configurable partition count and size
- Auto-tuning partition size based on available memory
- Comprehensive error handling with fallback to Python

**Status**: ✅ **TASK 3.4 FULLY COMPLETED**

---

---

#### 📋 Task 3.4.3: Add CUDA Support to Rust-Dask Bridge ✅

##### 1. Support CUDA Backend ✅

**Note**: CUDA support is already implemented in `rust_dask_bridge.py` via:
- `_process_partition_with_rust_cuda()` function
- `use_cuda` parameter in `process_symbols_rust_dask()`

**Deliverable**: CUDA support in Rust-Dask bridge ✅

---

## 4. Testing & Validation

### 4.1 Unit Tests

#### 📋 Task 4.1.1: Test Dask Scanner

**File**: `tests/test_dask_scanner.py`

```python
import pytest
import pandas as pd
from modules.adaptive_trend_LTS.core.scanner.dask_scan import _scan_dask
from modules.adaptive_trend_LTS.utils.config import ATCConfig

def test_dask_scanner_basic():
    """Test basic Dask scanner functionality."""
    # Create mock data fetcher
    # Create test symbols
    # Run scanner
    # Verify results
    pass

def test_dask_scanner_large_dataset():
    """Test Dask scanner with large symbol list."""
    # Test with 1000+ symbols
    # Verify memory usage stays low
    pass
```

**Deliverable**: Unit tests for Dask scanner

---

#### 📋 Task 4.1.2: Test Dask Batch Processor

**File**: `tests/test_dask_batch_processor.py`

```python
import pytest
from modules.adaptive_trend_LTS.core.compute_atc_signals.dask_batch_processor import (
    process_symbols_batch_dask
)

def test_dask_batch_processor():
    """Test Dask batch processor."""
    # Create test data
    # Run processor
    # Verify results match non-Dask version
    pass

def test_dask_batch_memory_usage():
    """Test memory usage with large batches."""
    # Test with 5000+ symbols
    # Monitor memory
    # Verify it stays within limits
    pass
```

**Deliverable**: Unit tests for Dask batch processor

---

#### 📋 Task 4.1.3: Test Rust-Dask Bridge ✅

**File**: `tests/adaptive_trend_LTS/test_rust_dask_bridge.py` ✅ **IMPLEMENTED**

**Status**: ✅ **COMPLETED** - Comprehensive test suite with 20+ test cases

**Test Coverage**:
- ✅ `test_process_symbols_rust_dask_basic()` - Basic Rust-Dask processing
- ✅ `test_process_symbols_rust_dask_fallback()` - Python fallback when Rust unavailable
- ✅ `test_process_symbols_rust_dask_empty()` - Empty input handling
- ✅ `test_process_symbols_rust_dask_large_dataset()` - Large dataset processing
- ✅ `test_process_symbols_rust_dask_auto_partitions()` - Auto partition determination
- ✅ `test_process_symbols_rust_dask_single_partition()` - Single partition mode
- ✅ `test_process_symbols_rust_dask_many_partitions()` - Many partitions mode
- ✅ `test_process_symbols_rust_dask_with_fallback()` - Fallback mode
- ✅ `test_process_symbols_rust_dask_cuda_mode()` - CUDA backend testing
- ✅ `test_process_symbols_rust_dask_result_consistency()` - Result consistency
- ✅ `test_process_symbols_rust_dask_memory_efficiency()` - Memory efficiency
- ✅ `test_process_symbols_rust_dask_error_handling()` - Error handling
- ✅ `test_auto_tune_partition_size_*()` - Partition size auto-tuning tests
- ✅ `test_process_partition_with_rust_cpu_*()` - Rust CPU partition tests
- ✅ `test_process_partition_with_rust_cuda_*()` - Rust CUDA partition tests
- ✅ `test_process_partition_python_*()` - Python fallback tests

**Implementation Details**:
- File location: `tests/adaptive_trend_LTS/test_rust_dask_bridge.py` (398 lines)
- Test fixtures: `sample_config`, `sample_price_series`
- Comprehensive error handling and edge case coverage
- Memory efficiency validation
- Performance benchmarking included

```python
import pytest
from modules.adaptive_trend_LTS.core.compute_atc_signals.rust_dask_bridge import (
    process_symbols_rust_dask
)

def test_rust_dask_bridge():
    """Test Rust-Dask bridge."""
    # Create test data
    # Run bridge
    # Verify results match pure Rust version
    pass

def test_rust_dask_fallback():
    """Test fallback when Rust unavailable."""
    # Mock Rust unavailable
    # Verify Python fallback works
    pass
```

**Deliverable**: Unit tests for Rust-Dask bridge ✅ **COMPLETED**

---

### 4.2 Integration Tests

#### 📋 Task 4.2.1: End-to-End Scanner Test ✅

**File**: `tests/adaptive_trend_LTS/test_dask_integration.py` ✅ **IMPLEMENTED**

**Status**: ✅ **COMPLETED** - Comprehensive end-to-end integration tests

**Test Coverage**:
- ✅ `test_end_to_end_dask_scanner()` - Complete Dask scanner workflow from scan_all_symbols to results
- ✅ `test_end_to_end_dask_scanner_large_dataset()` - Large dataset processing (50 symbols)
- ✅ `test_end_to_end_dask_scanner_empty_input()` - Empty input handling
- ✅ `test_end_to_end_dask_scanner_vs_threadpool()` - Consistency check between Dask and ThreadPool modes
- ✅ `test_end_to_end_dask_scanner_memory_cleanup()` - Memory cleanup verification

**Implementation Details**:
- File location: `tests/adaptive_trend_LTS/test_dask_integration.py` (280+ lines)
- Mock DataFetcher for isolated testing
- Memory monitoring with gc.collect()
- Performance benchmarking included
- Results validation (DataFrame structure, signal values, consistency)
- Memory efficiency checks

**Key Features Tested**:
- Complete workflow: `scan_all_symbols()` → Dask processing → Results
- Memory usage monitoring and validation
- Results structure verification (long/short DataFrames)
- Signal threshold filtering
- Memory cleanup after processing
- Consistency across execution modes

```python
def test_end_to_end_dask_scanner():
    """Test complete Dask scanner workflow."""
    # Setup
    # Run full scan
    # Verify results
    # Check memory usage
    pass
```

**Deliverable**: Integration tests ✅ **COMPLETED**

---

### 4.3 Performance Benchmarks

#### 📋 Task 4.3.1: Memory Usage Benchmark ✅

**File**: `benchmarks/benchmark_dask_memory.py` ✅

**Deliverable**: Memory usage benchmarks ✅

---

#### 📋 Task 4.3.2: Speed Benchmark ✅

**File**: `benchmarks/benchmark_dask_speed.py` ✅

**Deliverable**: Speed benchmarks ✅

---

#### ✅ Task 4.3 Completion Summary

**All subtasks completed:**
- ✅ Task 4.3.1: Memory benchmark created (`benchmarks/benchmark_dask_memory.py`)
- ✅ Task 4.3.2: Speed benchmark created (`benchmarks/benchmark_dask_speed.py`)
- ✅ Comprehensive tests added (`tests/adaptive_trend_LTS/test_benchmarks.py`)

**Implementation Details:**
- Files location: `modules/adaptive_trend_LTS/benchmarks/`
- Key modules implemented:
  - `benchmark_dask_memory.py` - Memory usage benchmarking with psutil integration
  - `benchmark_dask_speed.py` - Speed benchmarking with throughput measurement
  - `test_benchmarks.py` - Comprehensive benchmark tests

**Features:**
- Memory benchmarking with peak memory measurement
- Speed benchmarking with throughput (symbols/s) calculation
- Comparison between Dask, Rust, and Rust-Dask hybrid
- Configurable dataset sizes for testing
- Reproducible test data generation
- Progress logging and formatted output
- Quick benchmark mode for fast testing

**Status**: ✅ **TASK 4.3 FULLY COMPLETED**

---

## 5. Documentation

### 5.1 User Guide

#### 📋 Task 5.1.1: Create Dask Usage Guide

**File**: `docs/dask_usage_guide.md`

**Content**:

- How to use Dask scanner
- How to use Dask batch processor
- How to use Rust+Dask hybrid
- Configuration options
- Performance tips

**Deliverable**: User guide for Dask features

---

#### 📋 Task 5.1.2: Update Main README

**File**: `README.md`

**Updates**:

- Add Dask to features list
- Add usage examples
- Update performance numbers

**Deliverable**: Updated README

---

## 6. Migration Guide

### 6.1 Backward Compatibility

#### 📋 Task 6.1.1: Ensure Backward Compatibility

**Requirements**:

- All existing code should work without changes
- Dask is opt-in (via `execution_mode="dask"`)
- Default behavior unchanged

**Deliverable**: Backward compatible implementation

---

## 7. Phase 5 Completion Checklist

- [x] **Plan A: Scanner Integration**
- Dask scanner implemented
- Integrated into scan_all_symbols
- Memory optimized
- Progress tracking added
- Tests passing

- [x] **Plan B: Batch Processing**
- Dask batch processor implemented
- Integrated with batch_processor.py
- Memory optimized with GC
- Python fallback added
- Tests passing

- [x] **Plan C: Historical Data**
   - Dask backtesting implemented
   - Multi-file support added
   - Lazy loading with Dask DataFrame
   - Memory optimized
   - Tests passing

- [ ] **Plan D: Rust + Dask Hybrid**
   - Rust-Dask bridge implemented
   - Partition size auto-tuning
   - CUDA support added
   - Tests passing

- [ ] **Testing & Validation**
  - User guide created
  - README updated
  - Examples provided

---

## 8. Reference

### Development Approach

1. **Incremental**: Implement one plan at a time → test → optimize → next
2. **Benchmark early**: Compare memory usage and speed after each implementation
3. **Profile**: Use memory profilers (memory_profiler, tracemalloc) throughout
4. **Fallback**: Always keep non-Dask fallback for compatibility

### Pitfalls to Avoid

- **Over-partitioning**: Too many small partitions increase overhead
- **Under-partitioning**: Too few large partitions defeat the purpose
- **Memory leaks**: Ensure proper cleanup between partitions
- **Blocking operations**: Avoid blocking I/O in Dask operations

### Future Enhancements (Post–Phase 5)

- Distributed Dask cluster for multi-machine processing
- Dask DataFrame for more complex data operations
- Persistent Dask workers for reduced startup overhead
- Integration with Dask distributed scheduler

---

**Status**: 🔄 **PLANNED**  
**Target Completion**: 3-4 weeks from start date

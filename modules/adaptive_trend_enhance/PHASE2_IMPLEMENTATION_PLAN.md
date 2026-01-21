# Implementation Plan: Adaptive Trend Enhanced - Phase 2 Optimizations

## 🎯 Executive Summary

This plan outlines the technical approach for Phase 2 memory and performance optimizations to the `adaptive_trend_enhance` module. Building on the **5.71x speedup** achieved in Phase 1, Phase 2 targets:

- **50% memory reduction** in Scanner through batch processing
- **2-3x faster** equity calculations via vectorization
- **Overall 1.5-2x additional speedup** (total: 8.5-11x vs baseline)

---

## 📐 Architecture Overview

### Current Architecture (Phase 1)

```
Scanner → [All Symbols] → Process All → Accumulate Results → Filter & Sort
  └─ Memory Issue: All results held in memory simultaneously

Equity Calculation → 60 calls (54 Layer1 + 6 Layer2) → Numba JIT loops
  └─ Performance Issue: Sequential processing, no caching

Signal Processing → Many intermediate Series → Keep all in memory
  └─ Memory Issue: No cleanup of temporary data
```

### Target Architecture (Phase 2)

```
Scanner → [Batches of Symbols] → Process Batch → Yield Results → GC
  └─ Memory Optimization: Batch processing with forced cleanup

Equity Calculation → Vectorized/Cached/Parallel → Fast processing
  └─ Performance Optimization: NumPy vectorization + LRU cache + parallel

Signal Processing → Context managers → Automatic cleanup
  └─ Memory Optimization: Explicit cleanup of intermediate data
```

---

## 🔧 Detailed Implementation

### 1. Scanner Batch Processing

#### 1.1 Current Implementation Analysis

**File:** `modules/adaptive_trend_enhance/core/scanner.py`

Current problematic pattern:

```python
# scanner.py (lines 49-87)
def _scan_sequential(symbols, data_fetcher, atc_config, min_signal):
    results = []  # ❌ Accumulates ALL results in memory
    for symbol in symbols:
        result = _process_symbol(symbol, ...)
        if result is not None:
            results.append(result)  # ❌ Memory grows linearly
    return results
```

**Memory Profile (estimated):**

- 1000 symbols × ~5KB/result = **5MB**
- Peak memory includes: data fetches, calculations, intermediate series
- Total peak: **~100-200MB** for 1000 symbols

---

#### 1.2 Proposed Solution: Generator with Batch Processing

**New Architecture:**

```python
# scanner.py - NEW batched implementation

def _process_symbols_batched(
    symbols: list,
    data_fetcher: "DataFetcher",
    atc_config: ATCConfig,
    min_signal: float,
    batch_size: int = 100,
) -> Generator[Dict[str, Any], None, None]:
    """
    Process symbols in batches with forced garbage collection.

    Args:
        symbols: List of symbols to process
        data_fetcher: DataFetcher instance
        atc_config: ATC configuration
        min_signal: Minimum signal threshold
        batch_size: Number of symbols per batch (default: 100)

    Yields:
        Dictionary with symbol data for each valid signal

    Benefits:
        - Memory usage: O(batch_size) instead of O(len(symbols))
        - Forced GC between batches prevents memory creep
        - Early yielding allows incremental result processing
    """
    import gc

    total_batches = (len(symbols) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(symbols))
        batch = symbols[start_idx:end_idx]

        # Process batch
        for symbol in batch:
            result = _process_symbol(symbol, data_fetcher, atc_config, min_signal)
            if result is not None:
                yield result

        # Force garbage collection after each batch
        gc.collect()

        # Optional: Memory status logging
        if batch_idx % 10 == 0:
            mem_manager = get_memory_manager()
            status, snapshot = mem_manager.check_memory_status()
            if status != 'ok':
                log_warn(f"Memory {status}: {snapshot.ram_percent:.1f}% used")
```

**Integration with existing execution modes:**

```python
def _scan_sequential(symbols, data_fetcher, atc_config, min_signal, batch_size=100):
    """Sequential scan with batched processing."""
    results = []
    skipped_count = 0
    error_count = 0
    skipped_symbols = []

    # Use generator for memory efficiency
    result_generator = _process_symbols_batched(
        symbols, data_fetcher, atc_config, min_signal, batch_size
    )

    for idx, result in enumerate(result_generator, 1):
        results.append(result)

        # Progress update
        if idx % 10 == 0:
            log_progress(f"Scanned {idx}/{len(symbols)} symbols...")

    return results, skipped_count, error_count, skipped_symbols
```

**ThreadPool version:**

```python
def _scan_threadpool(
    symbols, data_fetcher, atc_config, min_signal, max_workers, batch_size=100
):
    """
    Threadpool scan with batched memory management.

    Strategy: Process in batches to limit concurrent memory usage
    """
    results = []
    total_batches = (len(symbols) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(symbols))
        batch = symbols[start_idx:end_idx]

        # Process batch in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process_symbol, sym, data_fetcher, atc_config, min_signal): sym
                for sym in batch
            }

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)

        # Cleanup after batch
        gc.collect()

    return results, 0, 0, []
```

---

#### 1.3 Configuration & CLI Integration

**Add to ATCConfig:**

```python
# modules/adaptive_trend_enhance/utils/config.py

@dataclass
class ATCConfig:
    # ... existing fields ...

    # Memory optimization settings
    batch_size: int = 100  # Symbols per batch for scanning
    enable_batch_gc: bool = True  # Force GC between batches
    max_memory_percent: float = 85.0  # Pause if memory exceeds this
```

**Add CLI argument:**

```python
# modules/adaptive_trend_enhance/cli/argument_parser.py

parser.add_argument(
    "--batch-size",
    type=int,
    default=100,
    help="Number of symbols to process per batch (default: 100). "
         "Smaller batches use less memory but may be slower."
)
```

---

### 2. Equity Curve Vectorization

#### 2.1 Current Implementation Analysis

**File:** `modules/adaptive_trend_enhance/core/compute_equity.py`

Current Numba implementation:

```python
@njit(cache=True)
def _calculate_equity_core(
    initial_weight: float,
    signal_arr: np.ndarray,
    roc_arr: np.ndarray,
    cutout: int,
    De: float,
) -> np.ndarray:
    """
    Current: Loop-based equity calculation
    Called 60 times per symbol (54 Layer1 + 6 Layer2)
    """
    n = len(signal_arr)
    equity = np.empty(n, dtype=np.float64)

    for i in range(n):  # ❌ Sequential loop
        if i < cutout:
            equity[i] = initial_weight
        else:
            prev_equity = equity[i - 1] if i > 0 else initial_weight
            change = signal_arr[i - 1] * roc_arr[i] if i > 0 else 0
            equity[i] = prev_equity * (1 + change - De)

    return equity
```

**Performance analysis:**

- Numba JIT is fast, but still sequential
- 60 calls means 60 separate Numba invocations
- No opportunity for batching or vectorization across signals

---

#### 2.2 Proposed Vectorized Implementation

**Strategy 1: Pure NumPy Vectorization (for single equity)**

```python
def _calculate_equity_vectorized(
    initial_weight: float,
    signal_series: pd.Series,
    roc_series: pd.Series,
    cutout: int,
    decay: float,
) -> pd.Series:
    """
    Vectorized equity calculation using NumPy.

    Formula: equity[i] = equity[i-1] * (1 + signal[i-1] * roc[i] - decay)

    Vectorization approach:
    1. Pre-compute all change factors: (1 + signal[i-1] * roc[i] - decay)
    2. Use np.cumprod() for cumulative product
    3. Apply initial_weight scaling

    Performance: ~2-3x faster than Numba loop for n > 500
    """
    n = len(signal_series)
    signal_arr = signal_series.values
    roc_arr = roc_series.values

    # Pre-allocate result
    equity = np.empty(n, dtype=np.float64)

    # Handle cutout period
    if cutout > 0:
        equity[:cutout] = initial_weight

    # Calculate change factors: (1 + signal[i-1] * roc[i] - decay)
    # Need to shift signal by 1 (signal[i-1])
    signal_shifted = np.roll(signal_arr, 1)
    signal_shifted[0] = 0  # No signal before first bar

    change_factors = 1.0 + signal_shifted * roc_arr - decay

    # For cutout period, change_factor should be 1 (no change)
    if cutout > 0:
        change_factors[:cutout] = 1.0

    # Cumulative product gives us equity curve
    # equity[i] = initial_weight * prod(change_factors[0:i+1])
    equity = initial_weight * np.cumprod(change_factors)

    return pd.Series(equity, index=signal_series.index)
```

**Strategy 2: Batch Vectorization (for multiple equities)**

```python
def _calculate_equities_batch_vectorized(
    initial_weights: np.ndarray,  # shape: (n_signals,)
    signals_matrix: np.ndarray,   # shape: (n_signals, n_bars)
    roc_arr: np.ndarray,          # shape: (n_bars,)
    cutout: int,
    decay: float,
) -> np.ndarray:
    """
    Vectorized batch equity calculation for multiple signals.

    Args:
        initial_weights: Initial weight for each signal
        signals_matrix: Matrix of signals (rows=signals, cols=bars)
        roc_arr: Rate of change array (shared across all signals)
        cutout: Cutout period
        decay: Decay rate

    Returns:
        Matrix of equity curves (shape: n_signals × n_bars)

    Performance: Process 6 signals simultaneously, ~4-5x faster than 6 sequential calls
    """
    n_signals, n_bars = signals_matrix.shape

    # Initialize equity matrix
    equity_matrix = np.empty((n_signals, n_bars), dtype=np.float64)

    # Shift signals by 1 column (signal[i-1])
    signals_shifted = np.roll(signals_matrix, 1, axis=1)
    signals_shifted[:, 0] = 0

    # Broadcast ROC to match signal matrix shape
    roc_broadcasted = np.broadcast_to(roc_arr, (n_signals, n_bars))

    # Calculate change factors for all signals at once
    change_factors = 1.0 + signals_shifted * roc_broadcasted - decay

    # Handle cutout
    if cutout > 0:
        change_factors[:, :cutout] = 1.0

    # Cumulative product along time axis (axis=1)
    # Broadcast initial weights
    initial_weights_col = initial_weights.reshape(-1, 1)
    equity_matrix = initial_weights_col * np.cumprod(change_factors, axis=1)

    return equity_matrix
```

**Integration example:**

```python
# In calculate_layer2_equities()
def calculate_layer2_equities(
    layer1_signals: Dict[str, pd.Series],
    ma_configs: list,
    R: pd.Series,
    L: float,
    De: float,
    cutout: int,
) -> Dict[str, pd.Series]:
    """Calculate Layer 2 equities using vectorized batch processing."""

    # Prepare batch data
    ma_types = [cfg[0] for cfg in ma_configs]
    initial_weights = np.array([cfg[2] for cfg in ma_configs])

    # Stack signals into matrix
    signals_matrix = np.vstack([
        layer1_signals[ma_type].values for ma_type in ma_types
    ])

    # Batch calculate all equities at once
    equity_matrix = _calculate_equities_batch_vectorized(
        initial_weights=initial_weights,
        signals_matrix=signals_matrix,
        roc_arr=R.values,
        cutout=cutout,
        decay=De,
    )

    # Convert back to dictionary of Series
    index = R.index
    layer2_equities = {
        ma_type: pd.Series(equity_matrix[i], index=index)
        for i, ma_type in enumerate(ma_types)
    }

    return layer2_equities
```

---

#### 2.3 Equity Caching Strategy

**Cache Key Design:**

```python
def _generate_equity_cache_key(
    signal: pd.Series,
    roc: pd.Series,
    initial_weight: float,
    decay: float,
    cutout: int,
) -> str:
    """
    Generate cache key for equity calculation.

    Key components:
    - Signal hash (first 16 chars of SHA256)
    - ROC hash (first 16 chars of SHA256)
    - Parameters: initial_weight, decay, cutout

    Note: Only cache if signal/ROC series are long enough (>100 bars)
          to justify hashing overhead
    """
    if len(signal) < 100:
        return None  # Don't cache short series

    import hashlib

    signal_hash = hashlib.sha256(signal.values.tobytes()).hexdigest()[:16]
    roc_hash = hashlib.sha256(roc.values.tobytes()).hexdigest()[:16]

    key = f"equity_{signal_hash}_{roc_hash}_{initial_weight}_{decay}_{cutout}"
    return key
```

**Cache Implementation:**

```python
# Extend CacheManager for equity curves
class EquityCacheManager:
    """Specialized cache for equity curves."""

    def __init__(self, max_entries: int = 200, max_size_mb: float = 100.0):
        self._cache = {}
        self.max_entries = max_entries
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self._hits = 0
        self._misses = 0

    def get_or_calculate(
        self,
        signal: pd.Series,
        roc: pd.Series,
        initial_weight: float,
        decay: float,
        cutout: int,
        calculator: Callable,
    ) -> pd.Series:
        """Get cached equity or calculate and cache."""

        key = _generate_equity_cache_key(signal, roc, initial_weight, decay, cutout)

        if key is None:
            # Don't cache short series
            return calculator()

        # Check cache
        if key in self._cache:
            self._hits += 1
            return self._cache[key].copy()

        # Calculate
        self._misses += 1
        result = calculator()

        # Store in cache (with LRU eviction if needed)
        self._cache[key] = result

        return result

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0

        return {
            'entries': len(self._cache),
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate_percent': hit_rate,
        }

# Global instance
_equity_cache = EquityCacheManager()

def get_equity_cache() -> EquityCacheManager:
    """Get global equity cache instance."""
    return _equity_cache
```

---

#### 2.4 Parallel Equity Processing

```python
def _calculate_equities_parallel(
    signals_dict: Dict[str, pd.Series],
    initial_weights_dict: Dict[str, float],
    roc: pd.Series,
    decay: float,
    cutout: int,
    max_workers: int = None,
) -> Dict[str, pd.Series]:
    """
    Calculate multiple equity curves in parallel.

    Args:
        signals_dict: Dictionary of MA signals
        initial_weights_dict: Initial weight for each MA
        roc: Rate of change series (shared)
        decay: Decay rate
        cutout: Cutout period
        max_workers: Number of parallel workers

    Returns:
        Dictionary of equity curves

    Usage: For Layer 1 (6 MAs) or Layer 2 (6 weights)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if max_workers is None:
        hw_mgr = get_hardware_manager()
        config = hw_mgr.get_optimal_workload_config(len(signals_dict))
        max_workers = config.num_threads

    equity_cache = get_equity_cache()
    results = {}

    def calculate_one_equity(ma_type, signal):
        """Worker function for one equity calculation."""
        initial_weight = initial_weights_dict[ma_type]

        # Try cache first
        result = equity_cache.get_or_calculate(
            signal=signal,
            roc=roc,
            initial_weight=initial_weight,
            decay=decay,
            cutout=cutout,
            calculator=lambda: _calculate_equity_vectorized(
                initial_weight, signal, roc, cutout, decay
            ),
        )

        return ma_type, result

    # Process in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(calculate_one_equity, ma_type, signal): ma_type
            for ma_type, signal in signals_dict.items()
        }

        for future in as_completed(futures):
            ma_type, equity = future.result()
            results[ma_type] = equity

    return results
```

---

### 3. Intermediate Series Cleanup

#### 3.1 Context Manager for Temporary Series

```python
# modules/adaptive_trend_enhance/utils/memory_utils.py

import gc
from contextlib import contextmanager
from typing import Union, Tuple, Any
import pandas as pd

@contextmanager
def temp_series(*series: pd.Series, cleanup_threshold_mb: float = 10.0):
    """
    Context manager for temporary Series with automatic cleanup.

    Args:
        *series: One or more pandas Series to manage
        cleanup_threshold_mb: Force GC if total size exceeds this (MB)

    Usage:
        with temp_series(ma1, ma2, ma3, cleanup_threshold_mb=50.0) as (m1, m2, m3):
            result = compute_something(m1, m2, m3)
        # ma1, ma2, ma3 are explicitly deleted and GC is triggered

    Benefits:
        - Explicit cleanup scope
        - Automatic GC for large Series
        - Clear code intent
    """
    try:
        # Check total size
        total_size_mb = sum(
            s.memory_usage(deep=True) / (1024**2) for s in series if isinstance(s, pd.Series)
        )

        yield series

    finally:
        # Delete all Series
        for s in series:
            del s

        # Force GC if size exceeded threshold
        if total_size_mb > cleanup_threshold_mb:
            gc.collect()


@contextmanager
def temp_dataframe(*dfs: pd.DataFrame, cleanup_threshold_mb: float = 50.0):
    """Similar to temp_series but for DataFrames."""
    try:
        total_size_mb = sum(
            df.memory_usage(deep=True).sum() / (1024**2)
            for df in dfs if isinstance(df, pd.DataFrame)
        )

        yield dfs

    finally:
        for df in dfs:
            del df

        if total_size_mb > cleanup_threshold_mb:
            gc.collect()


def cleanup_series(*series: pd.Series, force_gc: bool = True) -> None:
    """
    Utility function to cleanup series without context manager.

    Usage:
        ma1 = calculate_ema(...)
        ma2 = calculate_sma(...)
        result = ma1 + ma2
        cleanup_series(ma1, ma2)  # Cleanup intermediates
    """
    for s in series:
        try:
            del s
        except:
            pass

    if force_gc:
        gc.collect()
```

---

#### 3.2 Application in MA Calculations

```python
# modules/adaptive_trend_enhance/core/compute_moving_averages.py

def set_of_moving_averages_enhanced(
    length: int,
    source: pd.Series,
    ma_type: str,
    robustness: str = "Medium",
    use_cache: bool = True,
    use_parallel: bool = True,
    prefer_gpu: bool = True,
) -> Optional[Tuple[pd.Series, ...]]:
    """Generate 9 MAs with automatic cleanup of intermediates."""

    from modules.adaptive_trend_enhance.utils.memory_utils import temp_series

    with track_memory("set_of_moving_averages"):
        # Calculate length offsets
        L1, L2, L3, L4, L_1, L_2, L_3, L_4 = diflen(length, robustness=robustness)
        ma_lengths = [length, L1, L2, L3, L4, L_1, L_2, L_3, L_4]

        # Calculate MAs (these are temporary during calculation)
        if use_parallel:
            # ... parallel calculation ...
            mas = [f.result() for f in futures]
        else:
            mas = [
                ma_calculation_enhanced(source, ma_len, ma_type, use_cache, prefer_gpu)
                for ma_len in ma_lengths
            ]

        # Unpack results
        MA, MA1, MA2, MA3, MA4, MA_1, MA_2, MA_3, MA_4 = mas

        # The intermediate 'mas' list will be cleaned up when function exits
        # No need for explicit cleanup here since we're returning the Series

        return MA, MA1, MA2, MA3, MA4, MA_1, MA_2, MA_3, MA_4
```

---

#### 3.3 Application in Signal Processing

```python
# modules/adaptive_trend_enhance/core/compute_atc_signals.py

def compute_atc_signals(...) -> Dict[str, pd.Series]:
    """Compute ATC signals with automatic cleanup."""

    from modules.adaptive_trend_enhance.utils.memory_utils import temp_series, cleanup_series

    # ... initialization ...

    # Step 1: Calculate MAs
    with track_memory("Computing Moving Averages"):
        ema_set = set_of_moving_averages(length=ema_len, source=prices, ma_type="EMA", ...)
        hma_set = set_of_moving_averages(length=hull_len, source=prices, ma_type="HMA", ...)
        # ... other MAs ...

    # Step 2: Calculate signals (cleanup MAs after use)
    with track_memory("Computing Layer 1 signals"):
        # Use temp_series for intermediate MA sets
        with temp_series(*ema_set, *hma_set, cleanup_threshold_mb=100) as all_mas:
            ema_signal = _layer1_signal_for_ma("EMA", ema_set, ...)
            hma_signal = _layer1_signal_for_ma("HMA", hma_set, ...)
            # ... other signals ...
        # MAs are cleaned up here

    # Step 3: Layer 2 calculations
    layer1_signals = {
        "EMA": ema_signal,
        "HMA": hma_signal,
        # ...
    }

    with track_memory("Computing Layer 2 equities"):
        layer2_equities = calculate_layer2_equities(layer1_signals, ...)

    # Step 4: Final aggregation
    with track_memory("Computing Average_Signal"):
        signals = [layer1_signals[ma] for ma in MA_TYPES]
        weights = [layer2_equities[f"{ma}_S"] for ma in MA_TYPES]

        average_signal = weighted_signal(signals, weights)

        # Cleanup intermediate signals and weights
        cleanup_series(*signals, *weights, force_gc=True)

    # Return only final results
    return {
        "Average_Signal": average_signal,
        # ... other final results ...
    }
```

---

### 4. Additional Optimizations

#### 4.1 `weighted_signal()` with `einsum`

```python
# Alternative implementation using np.einsum

def weighted_signal_einsum(
    signals: Iterable[pd.Series],
    weights: Iterable[pd.Series],
) -> pd.Series:
    """
    Weighted signal calculation using Einstein summation.

    Performance: Potentially faster for large number of signals (>10)
    """
    signals = list(signals)
    weights = list(weights)

    # Stack into 2D arrays
    # signals_matrix: shape (n_signals, n_bars)
    # weights_matrix: shape (n_signals, n_bars)
    signals_matrix = np.vstack([s.values for s in signals])
    weights_matrix = np.vstack([w.values for w in weights])

    # Einstein summation: sum over signals axis (axis 0)
    # 'ij,ij->j' means: element-wise multiply, then sum over i (signals)
    numerator = np.einsum('ij,ij->j', signals_matrix, weights_matrix)
    denominator = np.einsum('ij->j', weights_matrix)

    # Division with NaN handling
    with np.errstate(divide='ignore', invalid='ignore'):
        result_arr = np.divide(numerator, denominator)
        result_arr = np.where(np.isfinite(result_arr), result_arr, np.nan)

    return pd.Series(result_arr, index=signals[0].index).round(2)
```

#### 4.2 DataFrame Optimization

```python
# In scanner.py

def scan_all_symbols(...) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Optimized DataFrame creation."""

    # Instead of:
    # results_df = pd.DataFrame(results)  # ❌ Slow for large lists

    # Use:
    results_df = pd.DataFrame.from_records(
        results,
        columns=['symbol', 'signal', 'trend', 'price', 'exchange']
    )

    # Pre-allocate with known structure (if possible)
    # Or use copy=False when safe
```

---

## 📊 Performance Targets & Metrics

### Target Improvements

| Metric                    | Phase 1  | Phase 2 Target | Total vs Baseline |
| ------------------------- | -------- | -------------- | ----------------- |
| **Speedup**               | 5.71x    | +1.5-2x        | **8.5-11x**       |
| **Memory (1000 symbols)** | ~200MB   | <100MB         | **50% reduction** |
| **Equity calc time**      | Baseline | 2-3x faster    | **Vectorization** |
| **Cache hit rate**        | N/A      | >60%           | **Equity reuse**  |

### Benchmarking Plan

```python
# tests/adaptive_trend_enhance/test_phase2_performance.py

def test_scanner_memory_batch_vs_nobatch():
    """Compare memory usage: batch vs non-batch processing."""

    # Test with 1000 symbols
    symbols = get_test_symbols(1000)

    # Baseline (no batching)
    mem_before = get_memory_usage()
    results_nobatch = scan_all_symbols(symbols, batch_size=None)
    mem_peak_nobatch = get_peak_memory()
    mem_after = get_memory_usage()

    # With batching
    mem_before = get_memory_usage()
    results_batch = scan_all_symbols(symbols, batch_size=100)
    mem_peak_batch = get_peak_memory()
    mem_after = get_memory_usage()

    # Verify memory reduction
    memory_reduction = (mem_peak_nobatch - mem_peak_batch) / mem_peak_nobatch
    assert memory_reduction >= 0.40  # At least 40% reduction

    # Verify results are identical
    assert_results_equal(results_nobatch, results_batch)


def test_equity_vectorized_vs_numba():
    """Compare vectorized vs Numba equity calculation."""

    # Setup test data
    signal, roc = generate_test_data(n_bars=2000)

    # Numba version
    time_numba = timeit(lambda: equity_series_numba(...), number=100)

    # Vectorized version
    time_vectorized = timeit(lambda: _calculate_equity_vectorized(...), number=100)

    speedup = time_numba / time_vectorized
    assert speedup >= 2.0  # At least 2x faster

    # Verify correctness
    result_numba = equity_series_numba(...)
    result_vectorized = _calculate_equity_vectorized(...)
    np.testing.assert_allclose(result_numba, result_vectorized, rtol=1e-10)


def test_equity_cache_hit_rate():
    """Test equity cache effectiveness."""

    cache = get_equity_cache()
    cache.clear()

    # Simulate typical workload: same signals repeated
    for _ in range(10):  # 10 iterations
        for ma_type in ['EMA', 'HMA', 'WMA', 'DEMA', 'LSMA', 'KAMA']:
            calculate_equity_with_cache(ma_type, ...)

    stats = cache.get_stats()
    hit_rate = stats['hit_rate_percent']

    # Should have >60% hit rate (54 out of 60 calls are duplicates)
    assert hit_rate >= 60.0
```

---

## 🔬 Testing Strategy

### Unit Tests

- [ ] Test `_process_symbols_batched()` generator
- [ ] Test `_calculate_equity_vectorized()` correctness
- [ ] Test `_calculate_equities_batch_vectorized()` correctness
- [ ] Test equity cache key generation
- [ ] Test `temp_series` context manager cleanup

### Integration Tests

- [ ] Test batched scanner with all execution modes
- [ ] Test vectorized equity in full ATC pipeline
- [ ] Test cleanup effectiveness in real workflows

### Performance Tests

- [ ] Benchmark batch sizes (50, 100, 200, 500)
- [ ] Benchmark vectorized vs Numba for different n_bars
- [ ] Measure cache hit rates in realistic scenarios
- [ ] Profile memory usage under various loads

### Stress Tests

- [ ] Test with 5000 symbols (if data available)
- [ ] Test with limited memory (2GB constraint)
- [ ] Test parallel equity with high worker counts
- [ ] Test cache eviction under pressure

---

## 🚀 Deployment Checklist

### Pre-deployment

- [ ] All tests pass (unit + integration + performance)
- [ ] Benchmarks meet targets (memory, speed, cache hit rate)
- [ ] Code review completed
- [ ] Documentation updated

### Backward Compatibility

- [ ] Existing tests still pass
- [ ] Default behavior unchanged (batch_size=100 by default)
- [ ] CLI maintains existing interface
- [ ] No breaking changes to public APIs

### Monitoring & Rollback

- [ ] Add performance metrics logging
- [ ] Add memory usage tracking
- [ ] Create rollback plan (feature flags)
- [ ] Document known limitations

---

## 📝 Implementation Risks & Mitigations

### Risk 1: Batch Processing Complexity

**Risk:** Batch processing adds complexity to scanner logic  
**Mitigation:**

- Thorough unit testing of generator
- Integration tests with all execution modes
- Careful progress tracking implementation

### Risk 2: Vectorization Correctness

**Risk:** Vectorized equity may have numerical differences vs Numba  
**Mitigation:**

- Extensive correctness tests (`np.testing.assert_allclose`)
- Parallel Numba/vectorized validation during transition
- Gradual rollout with feature flag

### Risk 3: Memory Management Overhead

**Risk:** Context managers and cleanup may hurt performance  
**Mitigation:**

- Benchmark cleanup overhead (should be <1%)
- Use cleanup only for large Series (>threshold)
- Profile real-world impact before full rollout

### Risk 4: Cache Hit Rate Variability

**Risk:** Cache hit rate may vary with usage patterns  
**Mitigation:**

- Adaptive cache sizing based on workload
- LRU eviction to keep most useful entries
- Monitoring and tuning in production

---

## ✅ Success Criteria

### Functional Requirements

- ✅ All existing tests pass
- ✅ Batch processing produces identical results
- ✅ Vectorized equity matches Numba output (within tolerance)
- ✅ Memory cleanup doesn't affect final results

### Performance Requirements

- ✅ Memory reduction ≥50% for 1000 symbols
- ✅ Equity calculation 2-3x faster
- ✅ Overall speedup 1.5-2x (total: 8.5-11x)
- ✅ Cache hit rate >60% for typical workloads

### Quality Requirements

- ✅ Code coverage ≥90% for new code
- ✅ No pylint/flake8 errors
- ✅ Documentation complete and accurate
- ✅ Performance regressions: 0

---

## 📅 Timeline

**Week 1:** Scanner batch processing + memory profiling  
**Week 2:** Equity vectorization + caching  
**Week 3:** Parallel equity + cleanup utilities  
**Week 4:** Testing, optimization, documentation

**Total Duration:** 3-4 weeks  
**Estimated Effort:** 80-100 hours

---

This implementation plan provides a comprehensive roadmap for Phase 2 enhancements while maintaining code quality, test coverage, and backward compatibility.

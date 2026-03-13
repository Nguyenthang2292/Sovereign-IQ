"""
Test to verify if CUDA+Dask actually uses GPU or silently falls back to CPU.
This will add detailed logging and monitoring to detect the actual execution path.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.common.utils import log_error, log_info, log_success, log_warn

print("=" * 80)
print("CUDA+DASK GPU USAGE VERIFICATION TEST")
print("=" * 80)

# Generate test data
np.random.seed(42)
n_symbols = 10
n_bars = 100

symbols_data = {}
for i in range(n_symbols):
    symbol = f"TEST{i:02d}"
    prices = pd.Series(
        100 + np.cumsum(np.random.randn(n_bars) * 0.5), index=pd.date_range("2024-01-01", periods=n_bars, freq="1h")
    )
    symbols_data[symbol] = prices

log_info(f"Generated {n_symbols} symbols with {n_bars} bars each")

# Config
config = {
    "ema_len": 28,
    "hma_len": 28,
    "wma_len": 28,
    "dema_len": 28,
    "lsma_len": 28,
    "kama_len": 28,
    "robustness": "Medium",
    "La": 0.02,
    "De": 0.03,
    "long_threshold": 0.1,
    "short_threshold": -0.1,
}

print("\n" + "=" * 80)
print("TEST 1: Import and Check Module Availability")
print("=" * 80)

try:
    import atc_rust

    log_success("✅ atc_rust module imported successfully")

    # Check for CUDA batch function
    has_cuda_batch = hasattr(atc_rust, "compute_atc_signals_batch")
    log_info(f"   Has compute_atc_signals_batch: {has_cuda_batch}")

except ImportError as e:
    log_error(f"❌ Failed to import atc_rust: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("TEST 2: Direct CUDA Batch Call (Standalone)")
print("=" * 80)

try:
    # Monkey-patch to detect if function is called
    original_cuda_batch = atc_rust.compute_atc_signals_batch
    cuda_batch_called = {"count": 0}

    def traced_cuda_batch(*args, **kwargs):
        cuda_batch_called["count"] += 1
        log_info(f"   🎯 compute_atc_signals_batch CALLED (call #{cuda_batch_called['count']})")
        log_info(f"      Args: {len(args)} positional")
        log_info(f"      Kwargs: {list(kwargs.keys())}")
        return original_cuda_batch(*args, **kwargs)

    atc_rust.compute_atc_signals_batch = traced_cuda_batch

    # Import batch processor
    from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.batch_processor import process_symbols_batch_cuda

    log_info("Testing standalone CUDA batch processing...")
    start_time = time.time()

    cuda_results = process_symbols_batch_cuda(symbols_data, config, num_threads=2)

    elapsed = time.time() - start_time

    log_success(f"✅ Standalone CUDA completed in {elapsed:.3f}s")
    log_info(f"   Results: {len(cuda_results)} symbols")
    log_info(f"   CUDA function calls: {cuda_batch_called['count']}")

    if cuda_batch_called["count"] > 0:
        log_success("   ✅ CONFIRMED: CUDA function WAS called")
    else:
        log_warn("   ⚠️ WARNING: CUDA function NOT called - using fallback?")

except Exception as e:
    log_error(f"❌ Standalone CUDA failed: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 80)
print("TEST 3: CUDA+Dask Execution")
print("=" * 80)

# Reset counter
cuda_batch_called["count"] = 0

try:
    from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.rust_dask_bridge import process_symbols_rust_dask

    log_info("Testing CUDA+Dask processing...")
    start_time = time.time()

    dask_results = process_symbols_rust_dask(
        symbols_data,
        config,
        use_cuda=True,  # ← Explicitly request CUDA
        partition_size=5,  # Small partitions for testing
    )

    elapsed = time.time() - start_time

    log_success(f"✅ CUDA+Dask completed in {elapsed:.3f}s")
    log_info(f"   Results: {len(dask_results)} symbols")
    log_info(f"   CUDA function calls: {cuda_batch_called['count']}")

    if cuda_batch_called["count"] > 0:
        log_success(f"   ✅ CONFIRMED: CUDA function called {cuda_batch_called['count']} times")
        log_info(f"   Expected ~{n_symbols // 5} calls for partition_size=5")
    else:
        log_error("   ❌ CRITICAL: CUDA function NOT called!")
        log_warn("   CUDA+Dask is NOT using GPU - using Python fallback!")

except Exception as e:
    log_error(f"❌ CUDA+Dask failed: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 80)
print("TEST 4: Compare Results (If Both Succeeded)")
print("=" * 80)

if "cuda_results" in locals() and "dask_results" in locals():
    # Compare first symbol
    test_symbol = list(symbols_data.keys())[0]

    if test_symbol in cuda_results and test_symbol in dask_results:
        cuda_signal = cuda_results[test_symbol]["Average_Signal"]
        dask_signal = dask_results[test_symbol]["Average_Signal"]

        # Align indices
        cuda_aligned = cuda_signal.reindex(symbols_data[test_symbol].index)
        dask_aligned = dask_signal.reindex(symbols_data[test_symbol].index)

        diff = np.abs(cuda_aligned - dask_aligned)
        max_diff = diff.max()
        match_pct = (diff < 1e-10).sum() / len(diff) * 100

        log_info(f"   Symbol: {test_symbol}")
        log_info(f"   Max difference: {max_diff:.6e}")
        log_info(f"   Match rate: {match_pct:.1f}%")

        if max_diff < 1e-10:
            log_success("   ✅ Perfect match - both using same code path")
        elif max_diff < 1e-3:
            log_warn("   ⚠️ Small differences - might be numerical precision")
        else:
            log_error("   ❌ Large differences - using different implementations!")
    else:
        log_warn(f"   Symbol {test_symbol} missing in results")
else:
    log_warn("   Cannot compare - one or both tests failed")

print("\n" + "=" * 80)
print("VERIFICATION SUMMARY")
print("=" * 80)

print("""
🎯 CONCLUSIONS:

1. If standalone CUDA calls the function:
   → CUDA kernels are accessible and working
   
2. If CUDA+Dask does NOT call the function:
   → CUDA+Dask is silently falling back to Python!
   → This explains why it shows 100% match (using Python, not CUDA)
   → The benchmark result is misleading!

3. If both call the function but give different results:
   → They're using different code paths or configurations
   → Need to investigate parameter passing

4. If CUDA+Dask calls the function AND matches standalone CUDA:
   → Both are actually using CUDA
   → But standalone CUDA still has numerical drift bug
   → Need to investigate why Dask version doesn't drift
""")

# Restore original function
atc_rust.compute_atc_signals_batch = original_cuda_batch

print("=" * 80)

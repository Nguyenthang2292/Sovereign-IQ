import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is in path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from modules.adaptive_trend_LTS.core.compute_atc_signals import compute_atc_signals
from modules.adaptive_trend_LTS.core.compute_atc_signals.batch_processor import process_symbols_batch_cuda


def generate_dummy_data(rows=1000):
    prices = 100 + np.random.randn(rows).cumsum()
    s = pd.Series(prices, name="close")
    # Make sure it has an index
    s.index = pd.date_range(start="2021-01-01", periods=rows, freq="h")
    return s


def test_concurrency_correctness():
    """
    Verify that concurrent batch processing produces identical results to sequential processing.
    This ensures that threading doesn't introduce race conditions in the Rust CUDA cache.
    """
    print("\n=== Integration Test: Concurrent CUDA Correctness ===")

    num_symbols = 50
    print(f"Generating dummy data for {num_symbols} symbols...")
    symbols_data = {f"SYM_{i}": generate_dummy_data() for i in range(num_symbols)}

    # Minimal config for ATC
    config = {}

    # 1. Sequential Processing
    print("Running Sequential Processing (Baseline)...")
    t0 = time.time()
    results_seq = {}
    for sym, prices in symbols_data.items():
        # Ensure use_cuda=True explicitly
        results_seq[sym] = compute_atc_signals(prices, **config, use_cuda=True)
    t_seq = time.time() - t0
    print(f"Sequential Time: {t_seq:.4f}s")

    # 2. Concurrent Processing
    print("Running Concurrent Batch Processing (4 threads)...")
    t0 = time.time()
    results_conc = process_symbols_batch_cuda(symbols_data, config, num_threads=4)
    t_conc = time.time() - t0
    print(f"Concurrent Time: {t_conc:.4f}s")

    # 3. Verification
    print("Verifying results...")
    match_count = 0
    mismatch_details = []

    for sym in symbols_data:
        # Check if key exists
        if sym not in results_seq or sym not in results_conc:
            print(f"Missing result for {sym}")
            continue

        res_seq = results_seq[sym].get("Average_Signal")
        res_conc = results_conc[sym].get("Average_Signal")

        if res_seq is None or res_conc is None:
            print(f"Missing Average_Signal for {sym}")
            continue

        # Compare values (handle NaNs correctly)
        # Using a small epsilon for float comparison, though exact match expected for same CUDA kernel
        if np.allclose(res_seq, res_conc, equal_nan=True, atol=1e-8):
            match_count += 1
        else:
            diff = np.abs(res_seq - res_conc).max()
            mismatch_details.append((sym, diff))

    success_rate = (match_count / num_symbols) * 100
    print(f"\nResults: {match_count}/{num_symbols} matched ({success_rate:.2f}%)")

    if len(mismatch_details) > 0:
        print("Mismatches found:")
        for sym, diff in mismatch_details[:5]:
            print(f"  {sym}: Max Diff = {diff}")

    if success_rate == 100:
        print("\n✅ PASSED: Concurrent execution is strictly consistent with sequential execution.")
    else:
        print("\n❌ FAILED: Inconsistencies detected.")
        sys.exit(1)

    # Speedup check
    # Note: For small 50 symbols, speedup might be negligible due to thread overhead,
    # but we print it anyway.
    if t_conc < t_seq:
        print(f"🚀 Speedup achieved: {t_seq / t_conc:.2f}x")
    else:
        print(f"⚠️ No speedup (likely dataset too small or overhead too high): {t_seq / t_conc:.2f}x")


if __name__ == "__main__":
    test_concurrency_correctness()

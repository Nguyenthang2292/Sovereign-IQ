"""Test individual MA kernels - Find the source of CUDA mismatch."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pandas_ta as ta

# Test data
np.random.seed(42)
prices = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.5))

print("=" * 60)
print("MA Kernel Accuracy Test")
print("=" * 60)

# Test each MA type
ma_types = ["EMA", "WMA", "DEMA", "LSMA", "HMA", "KAMA"]
length = 28

results = {}

for ma_type in ma_types:
    print(f"\nTesting {ma_type}...")

    # pandas_ta (reference)
    if ma_type == "EMA":
        ref = ta.ema(prices, length=length)
    elif ma_type == "WMA":
        ref = ta.wma(prices, length=length)
    elif ma_type == "DEMA":
        ref = ta.dema(prices, length=length)
    elif ma_type == "LSMA":
        ref = ta.linreg(prices, length=length)
    elif ma_type == "HMA":
        ref = ta.hma(prices, length=length)
    elif ma_type == "KAMA":
        from modules.adaptive_trend_LTS.core.compute_moving_averages.calculate_kama_atc import calculate_kama_atc

        ref = calculate_kama_atc(prices, length=length)

    # Rust CPU
    try:
        from modules.adaptive_trend_LTS.core.compute_moving_averages.calculate_kama_atc import (
            calculate_kama_atc as kama_rust,
        )
        from modules.adaptive_trend_LTS.core.rust_backend import (
            calculate_dema,
            calculate_ema,
            calculate_hma,
            calculate_lsma,
            calculate_wma,
        )

        if ma_type == "EMA":
            rust = pd.Series(calculate_ema(prices.values, length, use_rust=True, use_cuda=False), index=prices.index)
        elif ma_type == "WMA":
            rust = pd.Series(calculate_wma(prices.values, length, use_rust=True, use_cuda=False), index=prices.index)
        elif ma_type == "DEMA":
            rust = pd.Series(calculate_dema(prices.values, length, use_rust=True), index=prices.index)
        elif ma_type == "LSMA":
            rust = pd.Series(calculate_lsma(prices.values, length, use_rust=True), index=prices.index)
        elif ma_type == "HMA":
            rust = pd.Series(calculate_hma(prices.values, length, use_rust=True, use_cuda=False), index=prices.index)
        elif ma_type == "KAMA":
            rust = kama_rust(prices, length=length)

        # Compare
        diff = np.abs(ref.values - rust.values)
        max_diff = np.nanmax(diff)
        mean_diff = np.nanmean(diff)
        match_pct = np.sum(diff < 1e-6) / len(diff) * 100

        print(f"  pandas_ta vs Rust CPU:")
        print(f"    Max diff: {max_diff:.2e}")
        print(f"    Mean diff: {mean_diff:.2e}")
        print(f"    Match rate: {match_pct:.1f}%")

        results[ma_type] = {"ref": ref, "rust": rust, "max_diff": max_diff, "match_pct": match_pct}

    except Exception as e:
        print(f"  Rust test failed: {e}")

print(f"\n{'=' * 60}")
print("Summary")
print("=" * 60)

for ma_type, res in results.items():
    status = "✓" if res["match_pct"] > 99.9 else "✗"
    print(f"{ma_type:6s}: {status} {res['match_pct']:5.1f}% match, max diff = {res['max_diff']:.2e}")

print(f"\n{'=' * 60}")
print("Conclusion")
print("=" * 60)

all_match = all(r["match_pct"] > 99.9 for r in results.values())
if all_match:
    print("✓ All Rust MA kernels match pandas_ta (>99.9%)")
    print("\nIssue is NOT in MA calculations.")
    print("Next: Check signal persistence or Layer 1/2 logic.")
else:
    print("✗ Some Rust MA kernels have discrepancies")
    failing = [ma for ma, r in results.items() if r["match_pct"] <= 99.9]
    print(f"Failing MAs: {', '.join(failing)}")
    print("\nThese MA kernels need fixing before CUDA can work correctly.")

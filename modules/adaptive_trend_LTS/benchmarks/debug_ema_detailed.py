"""Detailed comparison of pandas_ta vs Rust EMA to find exact difference."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pandas_ta as ta

from modules.adaptive_trend_LTS.core.rust_backend import calculate_ema

# Test data
np.random.seed(42)
prices = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.5))
length = 28

print("=" * 60)
print("EMA Detailed Comparison")
print("=" * 60)

# pandas_ta
ema_pandas = ta.ema(prices, length=length)

# Rust
ema_rust = pd.Series(calculate_ema(prices.values, length, use_rust=True, use_cuda=False), index=prices.index)

print(f"\nFirst 35 bars comparison:")
print(f"{'Bar':<5} {'Price':<12} {'pandas_ta':<12} {'Rust':<12} {'Diff':<12} {'Match'}")
print("-" * 70)

for i in range(35):
    p = prices.iloc[i]
    pt = ema_pandas.iloc[i]
    rt = ema_rust.iloc[i]

    if pd.isna(pt) and pd.isna(rt):
        diff = 0.0
        match = "✓ (both NaN)"
    elif pd.isna(pt):
        diff = float("inf")
        match = "✗ (pandas NaN)"
    elif pd.isna(rt):
        diff = float("inf")
        match = "✗ (Rust NaN)"
    else:
        diff = abs(pt - rt)
        match = "✓" if diff < 1e-6 else f"✗ ({diff:.2e})"

    pt_str = f"{pt:.6f}" if not pd.isna(pt) else "NaN"
    rt_str = f"{rt:.6f}" if not pd.isna(rt) else "NaN"
    diff_str = f"{diff:.2e}" if diff != float("inf") else "inf"

    print(f"{i:<5} {p:<12.6f} {pt_str:<12} {rt_str:<12} {diff_str:<12} {match}")

print(f"\n{'=' * 60}")
print("Analysis")
print("=" * 60)

# Count NaN positions
pandas_nan_count = ema_pandas.isna().sum()
rust_nan_count = ema_rust.isna().sum()

print(f"pandas_ta NaN count: {pandas_nan_count}")
print(f"Rust NaN count: {rust_nan_count}")

# Find first valid index
pandas_first_valid = ema_pandas.first_valid_index()
rust_first_valid = ema_rust.first_valid_index()

print(f"\npandas_ta first valid index: {pandas_first_valid}")
print(f"Rust first valid index: {rust_first_valid}")

if pandas_first_valid != rust_first_valid:
    print(f"\n⚠️  ISSUE: First valid index mismatch!")
    print(f"   Expected (pandas_ta): {pandas_first_valid}")
    print(f"   Actual (Rust): {rust_first_valid}")
    print(f"   Offset: {rust_first_valid - pandas_first_valid if rust_first_valid and pandas_first_valid else 'N/A'}")
else:
    print(f"\n✓ First valid index matches: {pandas_first_valid}")

# Check values after first valid
if pandas_first_valid is not None and rust_first_valid is not None:
    start_idx = max(pandas_first_valid, rust_first_valid)
    valid_diff = np.abs(ema_pandas.iloc[start_idx:].values - ema_rust.iloc[start_idx:].values)
    max_valid_diff = np.nanmax(valid_diff)
    mean_valid_diff = np.nanmean(valid_diff)

    print(f"\nAfter first valid index:")
    print(f"   Max diff: {max_valid_diff:.2e}")
    print(f"   Mean diff: {mean_valid_diff:.2e}")

    if max_valid_diff < 1e-10:
        print(f"   ✓ Values match perfectly after warmup")
    else:
        print(f"   ✗ Values differ even after warmup")

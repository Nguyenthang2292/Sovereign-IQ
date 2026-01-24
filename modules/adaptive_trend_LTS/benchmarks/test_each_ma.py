"""Test each MA individually with detailed output."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pandas_ta as ta

from modules.adaptive_trend_LTS.core.compute_moving_averages.calculate_kama_atc import calculate_kama_atc
from modules.adaptive_trend_LTS.core.rust_backend import (
    calculate_dema,
    calculate_ema,
    calculate_hma,
    calculate_lsma,
    calculate_wma,
)

# Test data
np.random.seed(42)
prices = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.5))
length = 28


def test_ma(ma_type, ref_func, rust_func):
    print(f"\n{'=' * 60}")
    print(f"Testing {ma_type}")
    print("=" * 60)

    # Reference
    ref = ref_func()

    # Rust
    rust = rust_func()

    # Find first valid indices
    ref_first = ref.first_valid_index()
    rust_first = rust.first_valid_index()

    print(f"pandas_ta first valid: {ref_first}")
    print(f"Rust first valid: {rust_first}")

    if ref_first != rust_first:
        print(f"⚠️  MISMATCH: Offset = {rust_first - ref_first if rust_first and ref_first else 'N/A'}")
        return False

    # Compare values after first valid
    if ref_first is not None:
        start = ref_first
        ref_vals = ref.iloc[start:].values
        rust_vals = rust.iloc[start:].values

        diff = np.abs(ref_vals - rust_vals)
        max_diff = np.nanmax(diff)
        mean_diff = np.nanmean(diff)

        print(f"After warmup: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")

        if max_diff < 1e-10:
            print(f"✓ {ma_type} matches perfectly")
            return True
        else:
            print(f"✗ {ma_type} has numerical differences")
            # Show first few mismatches
            mismatch_idx = np.where(diff > 1e-10)[0]
            if len(mismatch_idx) > 0:
                print(f"  First mismatch at bar {start + mismatch_idx[0]}: diff={diff[mismatch_idx[0]]:.2e}")
            return False

    return True


# Test each MA
results = {}

results["EMA"] = test_ma(
    "EMA",
    lambda: ta.ema(prices, length=length),
    lambda: pd.Series(calculate_ema(prices.values, length, use_rust=True, use_cuda=False), index=prices.index),
)

results["WMA"] = test_ma(
    "WMA",
    lambda: ta.wma(prices, length=length),
    lambda: pd.Series(calculate_wma(prices.values, length, use_rust=True, use_cuda=False), index=prices.index),
)

results["DEMA"] = test_ma(
    "DEMA",
    lambda: ta.dema(prices, length=length),
    lambda: pd.Series(calculate_dema(prices.values, length, use_rust=True), index=prices.index),
)

results["LSMA"] = test_ma(
    "LSMA",
    lambda: ta.linreg(prices, length=length),
    lambda: pd.Series(calculate_lsma(prices.values, length, use_rust=True), index=prices.index),
)

results["HMA"] = test_ma(
    "HMA",
    lambda: ta.hma(prices, length=length),
    lambda: pd.Series(calculate_hma(prices.values, length, use_rust=True, use_cuda=False), index=prices.index),
)

results["KAMA"] = test_ma(
    "KAMA", lambda: calculate_kama_atc(prices, length=length), lambda: calculate_kama_atc(prices, length=length)
)

print(f"\n{'=' * 60}")
print("Summary")
print("=" * 60)

for ma, passed in results.items():
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"{ma:6s}: {status}")

all_pass = all(results.values())
if all_pass:
    print(f"\n✓ All MAs match! The issue must be elsewhere.")
else:
    failing = [ma for ma, passed in results.items() if not passed]
    print(f"\n✗ Failing MAs: {', '.join(failing)}")

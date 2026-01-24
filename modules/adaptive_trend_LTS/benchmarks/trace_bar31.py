"""Trace exact intermediate values at bar 31 to find CUDA bug."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from modules.adaptive_trend_LTS.core.compute_atc_signals import compute_atc_signals

# Test data
np.random.seed(42)
prices = pd.Series(100 + np.cumsum(np.random.randn(50) * 0.5))

config = {
    "ema_len": 28,
    "hull_len": 28,
    "wma_len": 28,
    "dema_len": 28,
    "lsma_len": 28,
    "kama_len": 28,
    "robustness": "Medium",
    "La": 0.02,
    "De": 0.03,
    "cutout": 0,
    "long_threshold": 0.1,
    "short_threshold": -0.1,
    "use_cuda": False,
}

# Get reference result
result = compute_atc_signals(prices=prices, **config)

print("=" * 70)
print("TRACING BAR 31 - EXPECTED VALUES")
print("=" * 70)

bar = 31
ma_types = ["EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"]

print(f"\n1. Layer 1 Signals (weighted average of 9 MA lengths):")
print("-" * 70)
for ma in ma_types:
    l1_sig = result[f"{ma}_Signal"].iloc[bar]
    print(f"  {ma}_Signal[{bar}] = {l1_sig:.10f}")

print(f"\n2. Layer 2 Equities:")
print("-" * 70)
for ma in ma_types:
    l2_eq = result[f"{ma}_S"].iloc[bar]
    print(f"  {ma}_S[{bar}] = {l2_eq:.10f}")

print(f"\n3. Final Average Signal Calculation:")
print("-" * 70)

long_th = 0.1
short_th = -0.1

nom = 0.0
den = 0.0

for ma in ma_types:
    s = result[f"{ma}_Signal"].iloc[bar]
    e = result[f"{ma}_S"].iloc[bar]

    if pd.notna(s) and pd.notna(e):
        # Discretize
        if s > long_th:
            c = 1.0
        elif s < short_th:
            c = -1.0
        else:
            c = 0.0

        nom += c * e
        den += e

        print(f"  {ma:6s}: s={s:10.6f}, e={e:10.6f}, c={c:4.1f} | nom+={c * e:10.6f}, den+={e:10.6f}")

final_avg = nom / den if den > 0 else 0.0

print(f"\n  Total: nom={nom:.10f}, den={den:.10f}")
print(f"  Final Average_Signal[{bar}] = {final_avg:.10f}")

# Now get CUDA result
print(f"\n{'=' * 70}")
print("CUDA BATCH RESULT")
print("=" * 70)

import atc_rust

symbols_data = {"TEST": prices.values}
cuda_config = {
    "ema_len": 28,
    "hull_len": 28,
    "wma_len": 28,
    "dema_len": 28,
    "lsma_len": 28,
    "kama_len": 28,
    "robustness": "Medium",
    "La": 0.02 / 1000.0,
    "De": 0.03 / 100.0,
    "long_threshold": 0.1,
    "short_threshold": -0.1,
}

cuda_result = atc_rust.compute_atc_signals_batch(symbols_data, **cuda_config)
cuda_sig = cuda_result["TEST"][bar]

print(f"\nCUDA Average_Signal[{bar}] = {cuda_sig:.10f}")
print(f"Expected: {final_avg:.10f}")
print(f"Difference: {abs(cuda_sig - final_avg):.10f}")

print(f"\n{'=' * 70}")
print("ANALYSIS")
print("=" * 70)

if abs(cuda_sig - final_avg) < 1e-6:
    print("✓ CUDA matches expected!")
else:
    print("✗ CUDA differs from expected")
    print("\nPossible causes:")
    print("1. Layer 1 signals are different in CUDA")
    print("2. Layer 2 equities are different in CUDA")
    print("3. Final weighted average calculation has a bug")
    print("4. Discretization (s > 0.1 → c=1.0) is different")

    # Reverse engineer what CUDA might be calculating
    print(f"\nReverse engineering CUDA result ({cuda_sig:.6f}):")

    # Try to find which combination gives cuda_sig
    # If all equities are same, and cuda_sig ≈ 0.159:
    # nom/den = 0.159
    # If den = sum of all equities, what should nom be?

    total_eq = den
    expected_nom = cuda_sig * total_eq

    print(f"  If den={total_eq:.6f}, then nom should be {expected_nom:.6f}")
    print(f"  Actual nom={nom:.6f}")
    print(f"  Difference in nom: {abs(expected_nom - nom):.6f}")

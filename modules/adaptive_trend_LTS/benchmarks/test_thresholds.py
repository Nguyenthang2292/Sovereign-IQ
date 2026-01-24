"""Test if CUDA is using wrong thresholds."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from modules.adaptive_trend_LTS.core.compute_atc_signals import compute_atc_signals

# Test data
np.random.seed(42)
prices_series = pd.Series(100 + np.cumsum(np.random.randn(50) * 0.5))

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

result = compute_atc_signals(prices=prices_series, **config)

bar = 31
ma_types = ["EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"]

# Get CUDA result
import atc_rust

symbols_data = {"TEST": prices_series.values}
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

print("=" * 70)
print("TESTING DIFFERENT THRESHOLD VALUES")
print("=" * 70)

# Try different thresholds
test_thresholds = [
    (0.0, 0.0),
    (0.1, -0.1),
    (0.5, -0.5),
    (1.0, -1.0),
]

for long_th, short_th in test_thresholds:
    nom = 0.0
    den = 0.0

    for ma in ma_types:
        s = result[f"{ma}_Signal"].iloc[bar]
        e = result[f"{ma}_S"].iloc[bar]

        if pd.notna(s) and pd.notna(e):
            if s > long_th:
                c = 1.0
            elif s < short_th:
                c = -1.0
            else:
                c = 0.0

            nom += c * e
            den += e

    test_result = nom / den if den > 0 else 0.0
    match = abs(test_result - cuda_sig) < 1e-6

    print(f"Thresholds ({long_th:4.1f}, {short_th:5.1f}): {test_result:.10f} {'✓ MATCH!' if match else ''}")

# Also test if CUDA is using unrounded Layer 1 signals
print(f"\n{'=' * 70}")
print("TESTING IF CUDA USES UNROUNDED LAYER 1 SIGNALS")
print("=" * 70)

# Layer 1 signals might be rounded in Python but not in CUDA
# Try with exact floating point values

print("\nThis requires access to CUDA intermediate values...")
print("Current CUDA result: {:.10f}".format(cuda_sig))
print("Expected result: {:.10f}".format(result["Average_Signal"].iloc[bar]))

# Calculate what the Layer 1 signals would need to be
print(f"\n{'=' * 70}")
print("WORKING BACKWARDS FROM CUDA RESULT")
print("=" * 70)

# If CUDA result = 0.159, and we know equities, what should signals be?
total_eq = sum(result[f"{ma}_S"].iloc[bar] for ma in ma_types)
cuda_nom = cuda_sig * total_eq

print(f"CUDA result: {cuda_sig:.10f}")
print(f"Total equity: {total_eq:.10f}")
print(f"Implied nom: {cuda_nom:.10f}")

# Try to find which combination of classified signals gives this nom
print("\nTrying all possible classifications (2^6 = 64 combinations)...")

from itertools import product

best_match = None
best_diff = float("inf")

for classification in product([-1.0, 0.0, 1.0], repeat=6):
    nom = sum(c * result[f"{ma}_S"].iloc[bar] for c, ma in zip(classification, ma_types))
    den = total_eq
    test_result = nom / den

    diff = abs(test_result - cuda_sig)
    if diff < best_diff:
        best_diff = diff
        best_match = classification

if best_diff < 1e-6:
    print(f"\n✓ FOUND MATCH!")
    print(f"  CUDA is using classifications: {best_match}")
    print(f"  Expected classifications:")
    for ma in ma_types:
        s = result[f"{ma}_Signal"].iloc[bar]
        if s > 0.1:
            c = 1.0
        elif s < -0.1:
            c = -1.0
        else:
            c = 0.0
        print(f"    {ma}: {c:.1f} (signal={s:.6f})")

    print(f"\n  CUDA classifications:")
    for ma, c in zip(ma_types, best_match):
        print(f"    {ma}: {c:.1f}")
else:
    print(f"\n✗ NO EXACT MATCH FOUND")
    print(f"  Best match has difference: {best_diff:.2e}")
    print(f"  This suggests CUDA is using different equity values too")

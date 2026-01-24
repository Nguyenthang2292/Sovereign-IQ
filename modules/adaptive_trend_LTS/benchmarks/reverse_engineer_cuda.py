"""Workaround: Extract intermediate values by calling single-MA CUDA functions."""

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

# Get reference (Python/Rust CPU)
print("=" * 70)
print("EXTRACTING REFERENCE VALUES (Python/Rust CPU)")
print("=" * 70)

result = compute_atc_signals(prices=prices_series, **config)

bar = 31
ma_types = ["EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"]

print(f"\nLayer 1 Signals at bar {bar}:")
for ma in ma_types:
    sig = result[f"{ma}_Signal"].iloc[bar]
    print(f"  {ma:6s}: {sig:.10f}")

print(f"\nLayer 2 Equities at bar {bar}:")
for ma in ma_types:
    eq = result[f"{ma}_S"].iloc[bar]
    print(f"  {ma:6s}: {eq:.10f}")

# Now get CUDA batch result
print(f"\n{'=' * 70}")
print("CUDA BATCH RESULT")
print("=" * 70)

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

print(f"\nCUDA Final Signal at bar {bar}: {cuda_sig:.10f}")
print(f"Expected: {result['Average_Signal'].iloc[bar]:.10f}")
print(f"Difference: {abs(cuda_sig - result['Average_Signal'].iloc[bar]):.10f}")

# Manual reverse engineering
print(f"\n{'=' * 70}")
print("REVERSE ENGINEERING CUDA VALUES")
print("=" * 70)

# If CUDA only processes EMA (i=0), what would the result be?
print("\nHypothesis 1: CUDA only processes EMA (loop exits after i=0)")

ema_sig = result["EMA_Signal"].iloc[bar]
ema_eq = result["EMA_S"].iloc[bar]

# Discretize EMA signal
if ema_sig > 0.1:
    ema_c = 1.0
elif ema_sig < -0.1:
    ema_c = -1.0
else:
    ema_c = 0.0

ema_only_result = (ema_c * ema_eq) / ema_eq if ema_eq > 0 else 0.0

print(f"  EMA signal: {ema_sig:.6f} → classified: {ema_c:.1f}")
print(f"  EMA equity: {ema_eq:.6f}")
print(f"  Result if only EMA: {ema_only_result:.10f}")
print(f"  CUDA actual: {cuda_sig:.10f}")
print(f"  Match: {'✓' if abs(ema_only_result - cuda_sig) < 1e-6 else '✗'}")

# Try all combinations
print("\nHypothesis 2: Testing which MAs are included")

for num_mas in range(1, 7):
    nom = 0.0
    den = 0.0

    for i, ma in enumerate(ma_types[:num_mas]):
        s = result[f"{ma}_Signal"].iloc[bar]
        e = result[f"{ma}_S"].iloc[bar]

        if pd.notna(s) and pd.notna(e):
            if s > 0.1:
                c = 1.0
            elif s < -0.1:
                c = -1.0
            else:
                c = 0.0

            nom += c * e
            den += e

    partial_result = nom / den if den > 0 else 0.0
    match = abs(partial_result - cuda_sig) < 1e-6

    mas_included = ", ".join(ma_types[:num_mas])
    print(f"  {num_mas} MAs ({mas_included}): {partial_result:.10f} {'✓ MATCH!' if match else ''}")

print(f"\n{'=' * 70}")
print("CONCLUSION")
print("=" * 70)

# Find which combination matches
for num_mas in range(1, 7):
    nom = 0.0
    den = 0.0

    for ma in ma_types[:num_mas]:
        s = result[f"{ma}_Signal"].iloc[bar]
        e = result[f"{ma}_S"].iloc[bar]

        if pd.notna(s) and pd.notna(e):
            if s > 0.1:
                c = 1.0
            elif s < -0.1:
                c = -1.0
            else:
                c = 0.0

            nom += c * e
            den += e

    partial_result = nom / den if den > 0 else 0.0

    if abs(partial_result - cuda_sig) < 1e-6:
        print(f"\n✓ CUDA is processing exactly {num_mas} MA type(s): {', '.join(ma_types[:num_mas])}")
        print(f"  Missing: {', '.join(ma_types[num_mas:])}")
        break

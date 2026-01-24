"""Simulate CUDA batch logic in Python to find the bug."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from modules.adaptive_trend_LTS.core.compute_atc_signals import compute_atc_signals

# Simple test data
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
    "use_cuda": False,  # Use Rust CPU for reference
}

print("=" * 60)
print("Simulating CUDA Batch Logic in Python")
print("=" * 60)

# Get reference result
result = compute_atc_signals(prices=prices, **config)

print(f"\nReference (Rust CPU) Average_Signal:")
print(f"  First 10: {result['Average_Signal'].iloc[:10].values}")
print(f"  Bars 27-37: {result['Average_Signal'].iloc[27:37].values}")

# Extract Layer 1 and Layer 2 for manual calculation
ma_types = ["EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"]

print(f"\n{'=' * 60}")
print("Layer 1 Signals (at bar 35):")
print("=" * 60)

for ma in ma_types:
    sig = result[f"{ma}_Signal"].iloc[35]
    print(f"  {ma}_Signal[35] = {sig:.6f}")

print(f"\n{'=' * 60}")
print("Layer 2 Equities (at bar 35):")
print("=" * 60)

for ma in ma_types:
    eq = result[f"{ma}_S"].iloc[35]
    print(f"  {ma}_S[35] = {eq:.6f}")

# Manually calculate final average signal for bar 35
print(f"\n{'=' * 60}")
print("Manual Final Average Calculation (bar 35):")
print("=" * 60)

long_threshold = 0.1
short_threshold = -0.1

nom = 0.0
den = 0.0

for ma in ma_types:
    s = result[f"{ma}_Signal"].iloc[35]
    e = result[f"{ma}_S"].iloc[35]

    # Discretize signal
    if pd.notna(s) and pd.notna(e):
        if s > long_threshold:
            c = 1.0
        elif s < short_threshold:
            c = -1.0
        else:
            c = 0.0

        nom += c * e
        den += e

        print(f"  {ma:6s}: s={s:8.4f}, e={e:8.4f}, c={c:4.1f}, nom+={c * e:8.4f}, den+={e:8.4f}")

final_avg = nom / den if den > 0 else 0.0

print(f"\n  Total: nom={nom:.6f}, den={den:.6f}")
print(f"  Final Average_Signal[35] = {final_avg:.6f}")
print(f"  Reference Average_Signal[35] = {result['Average_Signal'].iloc[35]:.6f}")
print(f"  Difference = {abs(final_avg - result['Average_Signal'].iloc[35]):.2e}")

if abs(final_avg - result["Average_Signal"].iloc[35]) < 1e-10:
    print(f"\n  ✓ Manual calculation matches reference!")
else:
    print(f"\n  ✗ Manual calculation differs from reference")

print(f"\n{'=' * 60}")
print("CUDA Kernel Logic Check")
print("=" * 60)

print("""
The CUDA kernel does:
1. For each MA type (6 total):
   - Read L1 signal from: all_l1_signals[i * total_bars + idx]
   - Read L2 equity from: all_l2_equities[i * total_bars + idx]
   - Discretize: c = 1.0 if s > 0.1, -1.0 if s < -0.1, else 0.0
   - Accumulate: nom += c * e, den += e
2. Calculate: result = nom / den

Potential issues:
- Index calculation: i * total_bars + idx
- NaN handling
- Threshold comparison precision
""")

# Check if there's an indexing issue
print(f"\nChecking index calculation:")
print(f"  Symbol 0, bar 35, total_bars=50")
for i, ma in enumerate(ma_types):
    cuda_idx = i * 50 + 35
    print(f"  {ma} (i={i}): index = {i} * 50 + 35 = {cuda_idx}")

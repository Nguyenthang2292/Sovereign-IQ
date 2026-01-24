"""Final investigation: Compare CUDA batch intermediate arrays."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

# Simple test
np.random.seed(42)
prices = 100 + np.cumsum(np.random.randn(50) * 0.5)

# Get single-symbol reference
from modules.adaptive_trend_LTS.core.compute_atc_signals import compute_atc_signals

prices_series = pd.Series(prices)
config = {"ema_len": 28, "robustness": "Medium", "La": 0.02, "De": 0.03, "use_cuda": False}

result = compute_atc_signals(prices=prices_series, **config)

print("Keys in result:", list(result.keys()))
print("\nEMA_Signal bars 27-37:", result["EMA_Signal"].iloc[27:38].values)
print("Average_Signal bars 27-37:", result["Average_Signal"].iloc[27:38].values)

# The issue: CUDA batch gives different results
# Let's check if it's a data layout problem

print("\n" + "=" * 60)
print("HYPOTHESIS: CUDA batch may be using wrong MA order or indexing")
print("=" * 60)

# Check all Layer 1 signals at bar 31 (first mismatch)
print("\nLayer 1 Signals at bar 31:")
for ma in ["EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"]:
    sig = result[f"{ma}_Signal"].iloc[31]
    print(f"  {ma}_Signal[31] = {sig:.6f}")

print("\nLayer 2 Equities at bar 31:")
for ma in ["EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"]:
    eq = result[f"{ma}_S"].iloc[31]
    print(f"  {ma}_S[31] = {eq:.6f}")

# Manual calculation of Average_Signal[31]
long_th = 0.1
short_th = -0.1

nom = 0.0
den = 0.0

print("\nManual calculation of Average_Signal[31]:")
for ma in ["EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"]:
    s = result[f"{ma}_Signal"].iloc[31]
    e = result[f"{ma}_S"].iloc[31]

    if pd.notna(s) and pd.notna(e):
        c = 1.0 if s > long_th else (-1.0 if s < short_th else 0.0)
        nom += c * e
        den += e
        print(f"  {ma}: s={s:.4f}, e={e:.4f}, c={c:.1f}")

avg = nom / den if den > 0 else 0.0
print(f"\nCalculated: {avg:.6f}")
print(f"Reference:  {result['Average_Signal'].iloc[31]:.6f}")
print(f"Match: {'✓' if abs(avg - result['Average_Signal'].iloc[31]) < 1e-10 else '✗'}")

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
print("""
If manual calculation matches reference, then:
- Python logic is correct
- CUDA batch must have a bug in:
  1. How it packs/unpacks data
  2. How it indexes into flattened arrays
  3. Signal persistence or equity calculation

CUDA batch gives 0.159 at bar 31, but expected is 0.327.
This suggests CUDA is using WRONG Layer 1 or Layer 2 values.
""")

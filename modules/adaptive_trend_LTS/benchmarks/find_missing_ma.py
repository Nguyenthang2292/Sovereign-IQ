"""Find which MA contribution is missing in CUDA."""

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

result = compute_atc_signals(prices=prices, **config)

bar = 31
ma_types = ["EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"]

print("=" * 70)
print(f"FINDING MISSING MA CONTRIBUTION AT BAR {bar}")
print("=" * 70)

# Calculate expected
long_th = 0.1
short_th = -0.1

contributions = {}
total_nom = 0.0
total_den = 0.0

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

        contrib_nom = c * e
        contrib_den = e

        contributions[ma] = {
            "signal": s,
            "equity": e,
            "classified": c,
            "nom_contrib": contrib_nom,
            "den_contrib": contrib_den,
        }

        total_nom += contrib_nom
        total_den += contrib_den

expected_avg = total_nom / total_den if total_den > 0 else 0.0

# Get CUDA result
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
cuda_avg = cuda_result["TEST"][bar]

# Reverse engineer CUDA
cuda_nom = cuda_avg * total_den

print(f"\nExpected (Python):")
print(f"  nom = {total_nom:.10f}")
print(f"  den = {total_den:.10f}")
print(f"  avg = {expected_avg:.10f}")

print(f"\nCUDA:")
print(f"  avg = {cuda_avg:.10f}")
print(f"  Reverse engineered nom = {cuda_nom:.10f} (assuming den is same)")

print(f"\nDifference:")
print(f"  nom_diff = {total_nom - cuda_nom:.10f}")

print(f"\n{'=' * 70}")
print("CONTRIBUTIONS BY MA TYPE:")
print("=" * 70)

for ma in ma_types:
    if ma in contributions:
        c = contributions[ma]
        print(f"\n{ma}:")
        print(f"  Signal: {c['signal']:.6f} → Classified: {c['classified']:.1f}")
        print(f"  Equity: {c['equity']:.6f}")
        print(f"  Nom contribution: {c['nom_contrib']:.6f}")
        print(f"  Den contribution: {c['den_contrib']:.6f}")

        # Check if this MA's contribution matches the missing amount
        if abs(c["nom_contrib"] - (total_nom - cuda_nom)) < 0.001:
            print(f"  ⚠️  THIS MA'S CONTRIBUTION MATCHES THE MISSING AMOUNT!")

print(f"\n{'=' * 70}")
print("HYPOTHESIS:")
print("=" * 70)

# Find which MA is missing
missing_nom = total_nom - cuda_nom

for ma in ma_types:
    if ma in contributions:
        c = contributions[ma]
        if abs(c["nom_contrib"] - missing_nom) < 0.001:
            print(f"\n✗ CUDA is MISSING contribution from {ma}!")
            print(f"   {ma} should contribute {c['nom_contrib']:.6f} to nom")
            print(f"   But CUDA nom is short by {missing_nom:.6f}")
            print(f"\n   Possible causes:")
            print(f"   1. {ma} Layer 1 signal is NaN in CUDA")
            print(f"   2. {ma} Layer 2 equity is NaN in CUDA")
            print(f"   3. {ma} is not included in the loop (i < 6)")
            print(f"   4. Indexing error for {ma} in flattened array")

"""Calculate expected EMA Layer 1 signal at bar 31."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pandas_ta as ta

# Test data
np.random.seed(42)
prices = pd.Series(100 + np.cumsum(np.random.randn(50) * 0.5))

# EMA parameters
ema_len = 28
robustness = "Medium"


# Get diflen for Medium robustness
def get_diflen(length, robustness):
    """Get different lengths based on robustness."""
    if robustness == "Low":
        return [length]
    elif robustness == "Medium":
        step = max(1, length // 10)
        return [
            max(2, length - 4 * step),
            max(2, length - 3 * step),
            max(2, length - 2 * step),
            max(2, length - step),
            length,
            length + step,
            length + 2 * step,
            length + 3 * step,
            length + 4 * step,
        ]
    else:  # High
        step = max(1, length // 20)
        return [
            max(2, length - 9 * step),
            max(2, length - 7 * step),
            max(2, length - 5 * step),
            max(2, length - 3 * step),
            max(2, length - step),
            length,
            length + step,
            length + 3 * step,
            length + 5 * step,
            length + 7 * step,
            length + 9 * step,
        ]


lengths = get_diflen(ema_len, robustness)

print("=" * 70)
print(f"EMA Layer 1 Signal Calculation (bar 31)")
print("=" * 70)

print(f"\nEMA base length: {ema_len}")
print(f"Robustness: {robustness}")
print(f"Different lengths: {lengths}")

# Calculate EMA for each length
bar = 31

print(f"\n{'Length':<8} {'EMA[31]':<12} {'Price[31]':<12} {'Signal':<10}")
print("-" * 50)

signals = []
equities = []

for length in lengths:
    ema = ta.ema(prices, length=length)
    ema_val = ema.iloc[bar]
    price_val = prices.iloc[bar]

    # Signal persistence (simplified - just check current crossover)
    # In reality, this is more complex with state tracking
    if pd.notna(ema_val):
        # Simplified: if price > ema, signal = 1, else if price < ema, signal = -1
        # This is NOT the full signal persistence logic, just for illustration
        if price_val > ema_val:
            sig = 1.0
        elif price_val < ema_val:
            sig = -1.0
        else:
            sig = 0.0
    else:
        sig = np.nan

    signals.append(sig)

    # Equity (simplified - assume 1.0 for now)
    equities.append(1.0)

    print(f"{length:<8} {ema_val:<12.6f} {price_val:<12.6f} {sig:<10.1f}")

# Calculate weighted average
print(f"\n{'=' * 70}")
print("Weighted Average Calculation")
print("=" * 70)

weighted_sum = 0.0
weight_sum = 0.0

for i, (sig, eq) in enumerate(zip(signals, equities)):
    if not np.isnan(sig):
        weighted_sum += sig * eq
        weight_sum += eq
        print(f"Length {lengths[i]:2d}: sig={sig:5.1f}, eq={eq:5.1f}, contrib={sig * eq:7.3f}")

if weight_sum > 0:
    avg_signal = weighted_sum / weight_sum
else:
    avg_signal = 0.0

print(f"\nWeighted sum: {weighted_sum:.6f}")
print(f"Weight sum: {weight_sum:.6f}")
print(f"Average signal: {avg_signal:.6f}")

# Compare with reference
from modules.adaptive_trend_LTS.core.compute_atc_signals import compute_atc_signals

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
ref_ema_signal = result["EMA_Signal"].iloc[bar]

print(f"\n{'=' * 70}")
print("Comparison")
print("=" * 70)

print(f"Calculated (simplified): {avg_signal:.10f}")
print(f"Reference (full logic): {ref_ema_signal:.10f}")
print(f"Difference: {abs(avg_signal - ref_ema_signal):.2e}")

print(f"\nNote: This is a SIMPLIFIED calculation.")
print(f"The actual Layer 1 signal uses full signal persistence logic")
print(f"with state tracking across all bars, not just bar 31.")

"""Debug ROC growth kernel - Compare CUDA vs Python."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from modules.adaptive_trend.utils.exp_growth import exp_growth

# Create synthetic test data
np.random.seed(42)
n_bars = 100
prices = pd.Series(100 + np.cumsum(np.random.randn(n_bars) * 0.5), index=pd.RangeIndex(n_bars))


# Python ROC calculation
def python_roc_with_growth(prices, La):
    """Python reference implementation."""
    # Calculate ROC
    roc = prices.pct_change().fillna(0.0)

    # Apply growth factor
    growth = exp_growth(L=La, index=prices.index, cutout=0)
    roc_with_growth = roc * growth

    return roc, growth, roc_with_growth


# Test with La = 0.02 (scaled from 20/1000)
La_scaled = 0.02 / 1000.0  # 2e-05

print("=" * 60)
print("ROC Growth Kernel Investigation")
print("=" * 60)

# Python calculation
roc_py, growth_py, roc_growth_py = python_roc_with_growth(prices, La_scaled)

print(f"\nPython Results (first 10 bars):")
print(f"Prices: {prices.iloc[:10].values}")
print(f"ROC: {roc_py.iloc[:10].values}")
print(f"Growth: {growth_py.iloc[:10].values}")
print(f"ROC*Growth: {roc_growth_py.iloc[:10].values}")

# Check growth formula
print(f"\nGrowth Formula Check:")
for i in range(5):
    expected = np.exp(La_scaled * i)
    actual = growth_py.iloc[i]
    print(
        f"  Bar {i}: exp({La_scaled} * {i}) = {expected:.10f}, actual = {actual:.10f}, diff = {abs(expected - actual):.2e}"
    )

# Now test CUDA kernel
print(f"\n{'=' * 60}")
print("Testing CUDA Kernel")
print("=" * 60)

try:
    import atc_rust

    # Prepare data for CUDA
    symbols_data = {"TEST": prices.values}

    # Call batch processing (which uses batch_roc_with_growth_kernel)
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
    }

    # We need to extract ROC from intermediate calculations
    # For now, let's just verify the final signal uses correct ROC

    print("\nNote: Direct ROC kernel testing requires exposing intermediate values.")
    print("Current CUDA batch doesn't expose ROC directly.")
    print("\nRecommendation: Add debug output to batch_roc_with_growth_kernel")

except Exception as e:
    print(f"CUDA test failed: {e}")
    import traceback

    traceback.print_exc()

# Manual verification of CUDA kernel logic
print(f"\n{'=' * 60}")
print("CUDA Kernel Logic Verification")
print("=" * 60)

print("""
CUDA kernel (batch_signal_kernels.cu lines 85-115):

for (int i = threadIdx.x; i < n; i += blockDim.x) {
    if (i == 0) {
        roc[i] = 0.0;
    } else {
        double p = prices[i];
        double p_prev = prices[i - 1];
        double r = (p_prev != 0.0 && !isnan(p) && !isnan(p_prev)) ? (p - p_prev) / p_prev : 0.0;
        
        // Python: bars = i == 0 ? 1 : i
        // Although i=0 is handled above, let's match the math strictly
        double growth = exp(La * (double)i);
        roc[i] = r * growth;
    }
}

Python equivalent:
    roc[0] = 0.0
    for i in range(1, n):
        r = (prices[i] - prices[i-1]) / prices[i-1]
        growth = exp(La * i)
        roc[i] = r * growth
""")

# Verify this matches
print("\nManual verification (first 10 bars):")
for i in range(10):
    if i == 0:
        r_manual = 0.0
        growth_manual = 1.0  # Not used since r=0
        roc_manual = 0.0
    else:
        r_manual = (prices.iloc[i] - prices.iloc[i - 1]) / prices.iloc[i - 1]
        growth_manual = np.exp(La_scaled * i)
        roc_manual = r_manual * growth_manual

    r_py = roc_py.iloc[i]
    g_py = growth_py.iloc[i]
    rg_py = roc_growth_py.iloc[i]

    match_r = "✓" if abs(r_manual - r_py) < 1e-10 else "✗"
    match_g = "✓" if abs(growth_manual - g_py) < 1e-10 else "✗"
    match_rg = "✓" if abs(roc_manual - rg_py) < 1e-10 else "✗"

    print(f"Bar {i}: r={match_r} g={match_g} rg={match_rg} | Manual: {roc_manual:.10f}, Python: {rg_py:.10f}")

print(f"\n{'=' * 60}")
print("Conclusion")
print("=" * 60)
print("""
If all bars show ✓, then CUDA kernel logic matches Python.
If there are ✗, the issue is in the CUDA implementation.

Next step: Add debug output to CUDA kernel to verify actual values.
""")

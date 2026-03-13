"""
Simulate CUDA HMA logic in Python and compare with pandas_ta.
This script replicates the exact logic found in Rust/CUDA implementation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta

# Disable pandas warnings
pd.options.mode.chained_assignment = None

print("=" * 80)
print("SIMULATING CUDA HMA LOGIC vs PANDAS_TA")
print("=" * 80)


def simulate_wma_cuda(prices: np.ndarray, length: int) -> np.ndarray:
    """Simulates batch_wma_kernel logic exactly."""
    n = len(prices)
    result = np.full(n, np.nan, dtype=np.float64)

    denom = length * (length + 1) / 2.0

    # Kernel loop
    for i in range(n):
        if i < length - 1:
            result[i] = np.nan
            continue

        # NaN guard
        is_valid = True
        for j in range(length):
            if np.isnan(prices[i - j]):
                is_valid = False
                break

        if not is_valid:
            result[i] = np.nan
            continue

        weighted_sum = 0.0
        for j in range(length):
            # i - j traverses backwards: i, i-1, i-2...
            # weight: length - j traverses: length, length-1, ... 1
            val = prices[i - j]
            weight = length - j
            weighted_sum += val * weight

        result[i] = weighted_sum / denom

    return result


def simulate_hma_cuda(prices: np.ndarray, length: int) -> np.ndarray:
    """Simulates Rust HMA orchestration logic found in batch_processing.rs."""

    # Rust logic:
    # let h = std::cmp::max(length / 2, 1);
    # let sq = std::cmp::max((length as f64).sqrt() as usize, 1);

    h = max(int(length / 2), 1)
    sq = max(int(np.sqrt(length)), 1)

    print(f"   HMA({length}) params: h={h}, sq={sq}")

    # 1. wh = WMA(prices, h)
    wh = simulate_wma_cuda(prices, h)

    # 2. wf = WMA(prices, length)
    wf = simulate_wma_cuda(prices, length)

    # 3. diff = 2*wh - wf
    # Note: batch_linear_combine_kernel handles NaNs by propagating them
    diff = np.full_like(prices, np.nan)
    mask = ~np.isnan(wh) & ~np.isnan(wf)
    diff[mask] = 2.0 * wh[mask] - 1.0 * wf[mask]

    # 4. res = WMA(diff, sq)
    res = simulate_wma_cuda(diff, sq)

    return res


# --- TESTING ---

# 1. Generate Data
np.random.seed(42)
n_bars = 100
# Use random walk to simulate price
price_values = 100 + np.cumsum(np.random.randn(n_bars))
prices_series = pd.Series(price_values)
prices_numpy = prices_series.values.astype(np.float64)

# 2. Run Simulations
length = 28

print("\nrunning pandas_ta.hma...")
pta_hma = ta.hma(prices_series, length=length)

print("running simulated cuda hma...")
sim_hma = simulate_hma_cuda(prices_numpy, length)

# 3. Compare Results
print("\n" + "=" * 80)
print("COMPARISON RESULTS")
print("=" * 80)

# Align series
df = pd.DataFrame({"Price": prices_series, "Pandas_TA": pta_hma, "Sim_CUDA": sim_hma})
df["Diff"] = (df["Pandas_TA"] - df["Sim_CUDA"]).abs()

# Analyze diff
first_valid_idx = max(length, int(length / 2), int(np.sqrt(length))) + length  # Rough estimate
filtered_diff = df["Diff"].iloc[length:]  # Skip warmup

print(f"Max Difference: {filtered_diff.max():.6e}")
print(f"Mean Difference: {filtered_diff.mean():.6e}")

if filtered_diff.max() > 1e-10:
    print("❌ MISMATCH DETECTED!")

    # Find first mismtach
    mismatch = df[df["Diff"] > 1e-10]
    if not mismatch.empty:
        print(f"\nFirst mismatch at index {mismatch.index[0]}:")
        print(mismatch.head(5))

        # Check intermediate values at divergence
        idx = mismatch.index[0]
        print(f"\nDebugging index {idx}:")

        # Manually verify WMA
        print("--- Intermediate Calculations check ---")
        h = max(int(length / 2), 1)
        wma_half_sim = simulate_wma_cuda(prices_numpy, h)[idx]
        wma_full_sim = simulate_wma_cuda(prices_numpy, length)[idx]

        wma_half_pta = ta.wma(prices_series, length=h).iloc[idx]
        wma_full_pta = ta.wma(prices_series, length=length).iloc[idx]

        print(
            f"WMA({h}): Sim={wma_half_sim:.6f}, Pandas={wma_half_pta:.6f}, Diff={abs(wma_half_sim - wma_half_pta):.6e}"
        )
        print(
            f"WMA({length}): Sim={wma_full_sim:.6f}, Pandas={wma_full_pta:.6f}, Diff={abs(wma_full_sim - wma_full_pta):.6e}"
        )

else:
    print("✅ PERFECT MATCH!")
    print("This means the algorithm logic is identical.")
    print("The bug MUST be in: 1. Initialization, 2. Floating point association, or 3. Data marshalling.")

print("=" * 80)

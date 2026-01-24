"""Simulate CUDA signal persistence kernel in Python to find the bug."""

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

# Get reference result
result = compute_atc_signals(prices=prices_series, **config)

# Get EMA for testing
ema = result["EMA"]
prices = prices_series.values
ema_vals = ema.values

print("=" * 60)
print("Simulating CUDA Signal Persistence Kernel")
print("=" * 60)


# Simulate CUDA kernel logic EXACTLY
def cuda_signal_persistence(prices, ma):
    """Exact simulation of batch_signal_persistence_kernel."""
    n = len(prices)
    signals = np.zeros(n)
    current_sig = 0.0

    for i in range(n):
        if i > 0:
            p_curr = prices[i]
            p_prev = prices[i - 1]
            m_curr = ma[i]
            m_prev = ma[i - 1]

            # Handle NaNs
            valid = not (np.isnan(p_curr) or np.isnan(p_prev) or np.isnan(m_curr) or np.isnan(m_prev))

            if valid:
                crossover = (p_prev <= m_prev) and (p_curr > m_curr)
                crossunder = (p_prev >= m_prev) and (p_curr < m_curr)

                # Debug print for bars 28-35
                if 28 <= i <= 35:
                    print(f"Bar {i}: p={p_curr:.4f} m={m_curr:.4f} | p_prev={p_prev:.4f} m_prev={m_prev:.4f}")
                    print(
                        f"        crossover={crossover} crossunder={crossunder} | sig_before={current_sig:.1f}", end=""
                    )

                if crossover:
                    current_sig = 1.0
                elif crossunder:
                    current_sig = -1.0

                if 28 <= i <= 35:
                    print(f" sig_after={current_sig:.1f}")
            else:
                if 28 <= i <= 35:
                    print(f"Bar {i}: INVALID (NaN)")

        signals[i] = current_sig

    return signals


# Run simulation
cuda_sim_signals = cuda_signal_persistence(prices, ema_vals)

# Compare with reference
ref_signals = result["EMA_Signal"].values

print(f"\n{'=' * 60}")
print("Comparison with Reference")
print("=" * 60)

print(f"\nBars 27-37:")
print(f"{'Bar':<5} {'CUDA Sim':<12} {'Reference':<12} {'Match'}")
print("-" * 45)

for i in range(27, 38):
    cuda_val = cuda_sim_signals[i]
    ref_val = ref_signals[i]
    match = "✓" if abs(cuda_val - ref_val) < 1e-10 else "✗"
    print(f"{i:<5} {cuda_val:<12.6f} {ref_val:<12.6f} {match}")

# Check if simulation matches reference
diff = np.abs(cuda_sim_signals - ref_signals)
max_diff = np.max(diff)
match_count = np.sum(diff < 1e-10)
match_pct = match_count / len(diff) * 100

print(f"\nOverall:")
print(f"  Max diff: {max_diff:.2e}")
print(f"  Match rate: {match_pct:.1f}% ({match_count}/{len(diff)})")

if match_pct == 100:
    print(f"\n✓ CUDA simulation matches reference perfectly!")
    print(f"  → Signal persistence kernel logic is CORRECT")
    print(f"  → Bug must be elsewhere (equity kernel or data layout)")
else:
    print(f"\n✗ CUDA simulation differs from reference")
    print(f"  → Signal persistence kernel has a bug")

    # Find first mismatch
    mismatch_idx = np.where(diff > 1e-10)[0]
    if len(mismatch_idx) > 0:
        idx = mismatch_idx[0]
        print(f"\n  First mismatch at bar {idx}:")
        print(f"    CUDA sim: {cuda_sim_signals[idx]:.6f}")
        print(f"    Reference: {ref_signals[idx]:.6f}")

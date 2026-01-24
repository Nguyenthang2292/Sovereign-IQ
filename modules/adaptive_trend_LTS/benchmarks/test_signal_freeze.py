"""Test signal persistence with REAL data to find freeze cause."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pandas_ta as ta

# REAL test data (same as used in CUDA batch)
np.random.seed(42)
prices = pd.Series(100 + np.cumsum(np.random.randn(50) * 0.5))

# Calculate EMA
ema = ta.ema(prices, length=28)

print("=" * 60)
print("Signal Persistence Test with REAL Data")
print("=" * 60)


# CUDA kernel logic (EXACT)
def cuda_signal_persistence_exact(prices, ma):
    """Exact CUDA kernel implementation."""
    n = len(prices)
    signals = []
    current_sig = 0.0

    for i in range(n):
        if i > 0:
            p_curr = prices.iloc[i]
            p_prev = prices.iloc[i - 1]
            m_curr = ma.iloc[i]
            m_prev = ma.iloc[i - 1]

            # NaN check
            valid = not (pd.isna(p_curr) or pd.isna(p_prev) or pd.isna(m_curr) or pd.isna(m_prev))

            if valid:
                # EXACT CUDA logic
                crossover = (p_prev <= m_prev) and (p_curr > m_curr)
                crossunder = (p_prev >= m_prev) and (p_curr < m_curr)

                if i >= 28 and i <= 35:
                    print(f"Bar {i}:")
                    print(f"  p_curr={p_curr:.4f}, m_curr={m_curr:.4f}")
                    print(f"  p_prev={p_prev:.4f}, m_prev={m_prev:.4f}")
                    print(f"  p_prev <= m_prev: {p_prev <= m_prev}")
                    print(f"  p_curr > m_curr: {p_curr > m_curr}")
                    print(f"  p_prev >= m_prev: {p_prev >= m_prev}")
                    print(f"  p_curr < m_curr: {p_curr < m_curr}")
                    print(f"  → crossover={crossover}, crossunder={crossunder}")
                    print(f"  → sig_before={current_sig:.1f}", end="")

                if crossover:
                    current_sig = 1.0
                elif crossunder:
                    current_sig = -1.0

                if i >= 28 and i <= 35:
                    print(f" → sig_after={current_sig:.1f}\n")

        signals.append(current_sig)

    return pd.Series(signals, index=prices.index)


# Run exact simulation
cuda_sim = cuda_signal_persistence_exact(prices, ema)

print("=" * 60)
print("Results")
print("=" * 60)

print(f"\nSignals bars 27-37:")
print(cuda_sim.iloc[27:38].values)

# Check if it freezes
diffs = cuda_sim.diff().iloc[28:38]
print(f"\nSignal changes (diff) bars 28-37:")
print(diffs.values)

freeze_count = (diffs == 0).sum()
print(f"\nBars where signal didn't change: {freeze_count}/10")

if freeze_count >= 5:
    print("\n❌ Signal IS freezing - crossover/crossunder not detecting changes")
else:
    print("\n✓ Signal is updating - freeze must be elsewhere")

"""
Deep diagnostic: Compare CUDA vs CPU calculations for a single symbol.
This will help identify exactly where CUDA diverges from CPU.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import atc_rust

from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import compute_atc_signals

print("=" * 80)
print("CUDA vs CPU DETAILED COMPARISON - Single Symbol Deep Dive")
print("=" * 80)

# Generate simple test data
np.random.seed(42)
n_bars = 100
prices = pd.Series(
    100 + np.cumsum(np.random.randn(n_bars) * 0.5), index=pd.date_range("2024-01-01", periods=n_bars, freq="1h")
)

print(f"\nTest Data:")
print(f"  Symbol: TEST")
print(f"  Bars: {len(prices)}")
print(f"  Price range: {prices.min():.2f} - {prices.max():.2f}")
print(f"  First 5: {list(prices.head())}")
print(f"  Last 5: {list(prices.tail())}")

# Config
config = {
    "ema_len": 28,
    "hma_len": 28,
    "wma_len": 28,
    "dema_len": 28,
    "lsma_len": 28,
    "kama_len": 28,
    "robustness": "Medium",
    "La": 0.02,
    "De": 0.03,
    "long_threshold": 0.1,
    "short_threshold": -0.1,
}

print("\n" + "=" * 80)
print("METHOD 1: Python CPU (Original)")
print("=" * 80)

try:
    cpu_result = compute_atc_signals(prices=prices, **config)
    cpu_signal = cpu_result["Average_Signal"]
    print(f"✅ SUCCESS")
    print(f"   Signal shape: {cpu_signal.shape}")
    print(f"   Signal range: {cpu_signal.min():.6f} to {cpu_signal.max():.6f}")
    print(f"   Last 10 signals: {list(cpu_signal.tail(10).values)}")
    print(f"   Unique values: {sorted(cpu_signal.unique())}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback

    traceback.print_exc()
    cpu_signal = None

print("\n" + "=" * 80)
print("METHOD 2: Rust CUDA Batch")
print("=" * 80)

try:
    test_data_cuda = {"TEST": prices.values.astype(np.float64)}

    cuda_result = atc_rust.compute_atc_signals_batch(
        test_data_cuda,
        ema_len=config["ema_len"],
        hull_len=config["hma_len"],  # Note: Rust uses hull_len
        wma_len=config["wma_len"],
        dema_len=config["dema_len"],
        lsma_len=config["lsma_len"],
        kama_len=config["kama_len"],
        robustness=config["robustness"],
        La=config["La"] / 1000.0,  # Scaled
        De=config["De"] / 100.0,  # Scaled
        long_threshold=config["long_threshold"],
        short_threshold=config["short_threshold"],
    )

    cuda_signal = pd.Series(cuda_result["TEST"], index=prices.index)
    print(f"✅ SUCCESS")
    print(f"   Signal shape: {cuda_signal.shape}")
    print(f"   Signal range: {cuda_signal.min():.6f} to {cuda_signal.max():.6f}")
    print(f"   Last 10 signals: {list(cuda_signal.tail(10).values)}")
    print(f"   Unique values: {sorted(cuda_signal.unique())}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback

    traceback.print_exc()
    cuda_signal = None

print("\n" + "=" * 80)
print("COMPARISON ANALYSIS")
print("=" * 80)

if cpu_signal is not None and cuda_signal is not None:
    # Align to same index
    cpu_aligned = cpu_signal.reindex(prices.index)
    cuda_aligned = cuda_signal.reindex(prices.index)

    # Calculate differences
    diff = cuda_aligned - cpu_aligned
    abs_diff = np.abs(diff)

    print(f"\n📊 Overall Statistics:")
    print(f"   Exact matches: {(diff == 0).sum()}/{len(diff)} ({(diff == 0).sum() / len(diff) * 100:.1f}%)")
    print(f"   Max difference: {abs_diff.max():.6e}")
    print(f"   Mean difference: {abs_diff.mean():.6e}")
    print(f"   Median difference: {abs_diff.median():.6e}")

    # Find divergence point
    print(f"\n🔍 Divergence Analysis:")
    if (diff != 0).any():
        first_diff_idx = diff[diff != 0].index[0]
        first_diff_pos = prices.index.get_loc(first_diff_idx)
        print(f"   First divergence at bar {first_diff_pos}: {first_diff_idx}")
        print(f"   CPU signal: {cpu_aligned.loc[first_diff_idx]:.6f}")
        print(f"   CUDA signal: {cuda_aligned.loc[first_diff_idx]:.6f}")
        print(f"   Difference: {diff.loc[first_diff_idx]:.6e}")

        # Show context around first divergence
        context_start = max(0, first_diff_pos - 5)
        context_end = min(len(prices), first_diff_pos + 5)

        print(f"\n   Context (bars {context_start} to {context_end}):")
        for i in range(context_start, context_end):
            idx = prices.index[i]
            marker = "👉" if i == first_diff_pos else "  "
            print(
                f"   {marker} Bar {i}: CPU={cpu_aligned.iloc[i]:.4f}, CUDA={cuda_aligned.iloc[i]:.4f}, diff={diff.iloc[i]:.6e}"
            )
    else:
        print(f"   ✅ Perfect match across all bars!")

    # Signal classification comparison
    print(f"\n📋 Signal Classification:")
    print(
        f"   CPU:  {(cpu_aligned > 0).sum()} long, {(cpu_aligned < 0).sum()} short, {(cpu_aligned == 0).sum()} neutral"
    )
    print(
        f"   CUDA: {(cuda_aligned > 0).sum()} long, {(cuda_aligned < 0).sum()} short, {(cuda_aligned == 0).sum()} neutral"
    )

    # Correlation
    if len(cpu_aligned) > 1:
        correlation = np.corrcoef(cpu_aligned, cuda_aligned)[0, 1]
        print(f"\n📈 Correlation: {correlation:.6f}")

    # Save detailed comparison
    comparison_df = pd.DataFrame(
        {
            "Price": prices,
            "CPU_Signal": cpu_aligned,
            "CUDA_Signal": cuda_aligned,
            "Difference": diff,
            "Abs_Diff": abs_diff,
        }
    )

    output_file = Path(__file__).parent / "cuda_vs_cpu_comparison.csv"
    comparison_df.to_csv(output_file)
    print(f"\n💾 Detailed comparison saved to: {output_file}")

else:
    print("❌ Cannot compare - one or both calculations failed")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)

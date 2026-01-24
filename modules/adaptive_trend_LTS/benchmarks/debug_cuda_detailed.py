"""Detailed comparison of CUDA vs Rust signals."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from modules.adaptive_trend_LTS.core.compute_atc_signals import compute_atc_signals
from modules.adaptive_trend_LTS.core.compute_atc_signals.batch_processor import process_symbols_batch_cuda

# Use pre-generated synthetic data for faster testing
np.random.seed(42)
prices_data = {
    "TEST1": pd.Series(100 + np.cumsum(np.random.randn(500) * 0.5), index=pd.RangeIndex(500)),
    "TEST2": pd.Series(200 + np.cumsum(np.random.randn(500) * 0.8), index=pd.RangeIndex(500)),
}

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
    "use_cuda": True,
}

print("Testing with synthetic data...")
print("=" * 60)

# CUDA batch
cuda_results = process_symbols_batch_cuda(prices_data, config)
sym = "TEST1"
cuda_sig = cuda_results[sym]["Average_Signal"]

# Rust single
rust_result = compute_atc_signals(prices=prices_data[sym], **config)
rust_sig = rust_result["Average_Signal"]

print(f"\nSymbol: {sym}")
print(f"CUDA signal range: [{cuda_sig.min():.3f}, {cuda_sig.max():.3f}]")
print(f"Rust signal range: [{rust_sig.min():.3f}, {rust_sig.max():.3f}]")

# Check first 50 bars (warmup period)
print(f"\nFirst 50 bars (warmup):")
print(f"CUDA: {cuda_sig.iloc[:50].values}")
print(f"Rust: {rust_sig.iloc[:50].values}")

# Check where differences occur
diff = np.abs(cuda_sig.values - rust_sig.values)
large_diff_idx = np.where(diff > 0.01)[0]
print(f"\nLarge differences (> 0.01): {len(large_diff_idx)} positions")
if len(large_diff_idx) > 0:
    print(f"First 10 positions: {large_diff_idx[:10]}")
    for idx in large_diff_idx[:5]:
        print(f"  Bar {idx}: CUDA={cuda_sig.iloc[idx]:.6f}, Rust={rust_sig.iloc[idx]:.6f}, diff={diff[idx]:.6f}")

# Check signal distribution
print(f"\nSignal distribution:")
print(f"CUDA: {np.bincount((cuda_sig.values + 1.5).astype(int), minlength=4)}")  # -1, 0, 1
print(f"Rust: {np.bincount((rust_sig.values + 1.5).astype(int), minlength=4)}")

# Check if it's a systematic offset or random noise
correlation = np.corrcoef(cuda_sig.values, rust_sig.values)[0, 1]
print(f"\nCorrelation: {correlation:.6f}")

# Check cutout application
print(f"\nCutout check (first 28 bars should be affected by warmup):")
print(f"CUDA first 28: {cuda_sig.iloc[:28].unique()}")
print(f"Rust first 28: {rust_sig.iloc[:28].unique()}")

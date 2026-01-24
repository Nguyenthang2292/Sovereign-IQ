"""Compare CUDA batch vs single-symbol processing step by step."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from modules.adaptive_trend_LTS.core.compute_atc_signals import compute_atc_signals
from modules.adaptive_trend_LTS.core.compute_atc_signals.batch_processor import process_symbols_batch_cuda

# Create simple test data
np.random.seed(42)
n_bars = 100
prices = pd.Series(100 + np.cumsum(np.random.randn(n_bars) * 0.5), index=pd.RangeIndex(n_bars))

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

print("=" * 60)
print("CUDA Batch vs Single-Symbol Comparison")
print("=" * 60)

# Single-symbol processing (Rust CPU)
print("\n1. Single-symbol processing (Rust CPU)...")
single_result = compute_atc_signals(prices=prices, **config)

print(f"   Keys: {list(single_result.keys())}")
print(f"   Average_Signal shape: {single_result['Average_Signal'].shape}")
print(f"   Average_Signal first 10: {single_result['Average_Signal'].iloc[:10].values}")
print(
    f"   Average_Signal stats: min={single_result['Average_Signal'].min():.3f}, max={single_result['Average_Signal'].max():.3f}"
)

# CUDA batch processing
print("\n2. CUDA batch processing...")
symbols_data = {"TEST": prices}

try:
    batch_result = process_symbols_batch_cuda(symbols_data, config)

    print(f"   Batch result keys: {list(batch_result.keys())}")

    if "TEST" in batch_result:
        cuda_sig = batch_result["TEST"]["Average_Signal"]
        print(f"   CUDA Average_Signal shape: {cuda_sig.shape}")
        print(
            f"   CUDA Average_Signal first 10: {cuda_sig.iloc[:10].values if isinstance(cuda_sig, pd.Series) else cuda_sig[:10]}"
        )

        if isinstance(cuda_sig, pd.Series):
            print(f"   CUDA Average_Signal stats: min={cuda_sig.min():.3f}, max={cuda_sig.max():.3f}")
        else:
            print(f"   CUDA Average_Signal stats: min={np.min(cuda_sig):.3f}, max={np.max(cuda_sig):.3f}")

        # Compare
        print(f"\n3. Comparison:")

        single_vals = single_result["Average_Signal"].values
        cuda_vals = cuda_sig.values if isinstance(cuda_sig, pd.Series) else cuda_sig

        if len(single_vals) == len(cuda_vals):
            diff = np.abs(single_vals - cuda_vals)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            match_count = np.sum(diff < 1e-6)
            match_pct = match_count / len(diff) * 100

            print(f"   Max diff: {max_diff:.6f}")
            print(f"   Mean diff: {mean_diff:.6f}")
            print(f"   Match rate: {match_pct:.1f}% ({match_count}/{len(diff)})")

            # Show first few mismatches
            mismatch_idx = np.where(diff > 1e-6)[0]
            if len(mismatch_idx) > 0:
                print(f"\n   First 5 mismatches:")
                for idx in mismatch_idx[:5]:
                    print(
                        f"      Bar {idx}: Single={single_vals[idx]:.6f}, CUDA={cuda_vals[idx]:.6f}, diff={diff[idx]:.6f}"
                    )
            else:
                print(f"\n   ✓ All values match!")
        else:
            print(f"   ✗ Length mismatch: Single={len(single_vals)}, CUDA={len(cuda_vals)}")
    else:
        print(f"   ✗ TEST symbol not in batch result")

except Exception as e:
    print(f"   ✗ CUDA batch failed: {e}")
    import traceback

    traceback.print_exc()

print(f"\n{'=' * 60}")
print("Diagnosis")
print("=" * 60)

print("""
If match rate is 0% or very low, the issue is in:
1. CUDA batch processing logic (batch_processing.rs)
2. CUDA kernel implementations (batch_signal_kernels.cu)
3. Data layout/indexing when passing to CUDA

If match rate is high (>90%), the issue was in the benchmark comparison logic.
""")

"""Extract and compare CUDA batch intermediate values."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import atc_rust
import numpy as np
import pandas as pd

# Simple test
np.random.seed(42)
prices_arr = 100 + np.cumsum(np.random.randn(50) * 0.5)

# Call CUDA batch to get classified signals
symbols_data = {"TEST": prices_arr}

config = {
    "ema_len": 28,
    "hull_len": 28,
    "wma_len": 28,
    "dema_len": 28,
    "lsma_len": 28,
    "kama_len": 28,
    "robustness": "Medium",
    "La": 0.02 / 1000.0,  # Already scaled
    "De": 0.03 / 100.0,  # Already scaled
    "long_threshold": 0.1,
    "short_threshold": -0.1,
}

print("=" * 60)
print("CUDA Batch Intermediate Values Extraction")
print("=" * 60)

try:
    # Call batch function
    batch_result = atc_rust.compute_atc_signals_batch(symbols_data, **config)

    print(f"\nCUDA batch result keys: {list(batch_result.keys())}")

    if "TEST" in batch_result:
        cuda_sig = batch_result["TEST"]
        print(f"CUDA result type: {type(cuda_sig)}")
        print(f"CUDA result shape: {cuda_sig.shape if hasattr(cuda_sig, 'shape') else len(cuda_sig)}")
        print(f"CUDA first 10: {cuda_sig[:10]}")
        print(f"CUDA bars 27-37: {cuda_sig[27:37]}")

        # Compare with single-symbol Rust
        from modules.adaptive_trend_LTS.core.compute_atc_signals import compute_atc_signals

        prices_series = pd.Series(prices_arr)
        single_config = {
            "ema_len": 28,
            "hull_len": 28,
            "wma_len": 28,
            "dema_len": 28,
            "lsma_len": 28,
            "kama_len": 28,
            "robustness": "Medium",
            "La": 0.02,  # Unscaled (will be scaled inside)
            "De": 0.03,  # Unscaled
            "long_threshold": 0.1,
            "short_threshold": -0.1,
            "use_cuda": False,
        }

        single_result = compute_atc_signals(prices=prices_series, **single_config)
        single_sig = single_result["Average_Signal"].values

        print(f"\nSingle (Rust CPU) first 10: {single_sig[:10]}")
        print(f"Single bars 27-37: {single_sig[27:37]}")

        # Compare
        diff = np.abs(cuda_sig - single_sig)
        print(f"\nComparison:")
        print(f"  Max diff: {np.max(diff):.6f}")
        print(f"  Mean diff: {np.mean(diff):.6f}")
        print(f"  First mismatch: bar {np.argmax(diff > 1e-6)}")

        # Show detailed comparison for first mismatch
        mismatch_idx = np.where(diff > 1e-6)[0]
        if len(mismatch_idx) > 0:
            print(f"\n  First 5 mismatches:")
            for idx in mismatch_idx[:5]:
                print(f"    Bar {idx}: CUDA={cuda_sig[idx]:.6f}, Single={single_sig[idx]:.6f}, diff={diff[idx]:.6f}")

                # Show Layer 1 and Layer 2 for this bar
                if idx >= 27:  # After warmup
                    print(f"      Layer 1 signals:")
                    for ma in ["EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"]:
                        l1 = single_result[f"{ma}_Signal"].iloc[idx]
                        print(f"        {ma}_Signal[{idx}] = {l1:.6f}")
                    print(f"      Layer 2 equities:")
                    for ma in ["EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"]:
                        l2 = single_result[f"{ma}_S"].iloc[idx]
                        print(f"        {ma}_S[{idx}] = {l2:.6f}")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()

print(f"\n{'=' * 60}")
print("Hypothesis")
print("=" * 60)
print("""
If CUDA batch differs from single-symbol at specific bars,
the issue could be:

1. Layer 1 signal calculation in CUDA differs
2. Layer 2 equity calculation in CUDA differs  
3. Final weighted average has a bug
4. Data packing/unpacking issue

Next step: Add printf to CUDA kernels to see intermediate values.
""")

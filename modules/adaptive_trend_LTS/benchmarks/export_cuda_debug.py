"""Export CUDA intermediate values for debugging."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import atc_rust
import numpy as np
import pandas as pd

# Test data
np.random.seed(42)
prices = 100 + np.cumsum(np.random.randn(50) * 0.5)

symbols_data = {"TEST": prices}

config = {
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

print("Calling CUDA batch with debug export...")

# Call the debug version that exports intermediate values
try:
    result = atc_rust.compute_atc_signals_batch_debug(symbols_data, **config)

    print("\nDebug result keys:", list(result.keys()))

    # Save to files
    output_dir = Path(__file__).parent / "cuda_debug_output"
    output_dir.mkdir(exist_ok=True)

    for key, value in result.items():
        if isinstance(value, dict):
            # Nested dict (e.g., Layer 1 signals by MA type)
            for subkey, subvalue in value.items():
                filename = output_dir / f"{key}_{subkey}.npy"
                np.save(filename, subvalue)
                print(f"Saved {filename}")
        else:
            # Direct array
            filename = output_dir / f"{key}.npy"
            np.save(filename, value)
            print(f"Saved {filename}")

    print(f"\nAll intermediate values exported to {output_dir}")

except AttributeError:
    print("\nDEBUG FUNCTION NOT AVAILABLE")
    print("Need to add compute_atc_signals_batch_debug to Rust code")
    print("\nFalling back to manual extraction...")

    # Fallback: Call regular batch and try to infer values
    result = atc_rust.compute_atc_signals_batch(symbols_data, **config)
    print(f"CUDA result at bar 31: {result['TEST'][31]:.6f}")

"""Simple test to capture CUDA printf output."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Force flush
import os

import atc_rust
import numpy as np

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Simple test
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

print("Calling CUDA batch...")
print("=" * 60)

result = atc_rust.compute_atc_signals_batch(symbols_data, **config)

print("=" * 60)
print("CUDA batch completed")
print(f"Result: {result['TEST'][:10]}")

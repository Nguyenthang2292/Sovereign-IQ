"""Test CUDA batch function signature and behavior."""

import sys
from pathlib import Path

import numpy as np

# Add project root
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import inspect

import atc_rust

print("=" * 70)
print("RUST FUNCTION SIGNATURE INVESTIGATION")
print("=" * 70)

# Check signature
sig = inspect.signature(atc_rust.compute_atc_signals_batch)
print("\nFunction: compute_atc_signals_batch")
print(f"Signature: {sig}")
print("\nParameters:")
for name, param in sig.parameters.items():
    print(f"  - {name}: {param}")

# Test with dummy data
print("\n" + "=" * 70)
print("TESTING WITH DUMMY DATA")
print("=" * 70)

test_data = {
    "TEST1": np.array([100.0, 101.0, 102.0, 103.0, 104.0] * 20, dtype=np.float64),
    "TEST2": np.array([50.0, 51.0, 50.5, 51.5, 50.0] * 20, dtype=np.float64),
}

print(f"\nTest data: {len(test_data)} symbols, {len(test_data['TEST1'])} bars each")

# Test with hull_len (current Rust signature)
print("\n--- Test 1: Using hull_len parameter ---")
try:
    result1 = atc_rust.compute_atc_signals_batch(
        test_data,
        ema_len=28,
        hull_len=28,  # <- Current signature
        wma_len=28,
        dema_len=28,
        lsma_len=28,
        kama_len=28,
        robustness="Medium",
        La=0.02 / 1000,
        De=0.03 / 100,
        long_threshold=0.1,
        short_threshold=-0.1,
    )
    print("✅ SUCCESS with hull_len")
    print(f"   Results: {len(result1)} symbols")
    for sym, arr in result1.items():
        print(f"   {sym}: shape={arr.shape}, last_value={arr[-1]}")
except Exception as e:
    print(f"❌ FAILED with hull_len: {e}")

# Test with hma_len (what I tried to fix to)
print("\n--- Test 2: Using hma_len parameter ---")
try:
    result2 = atc_rust.compute_atc_signals_batch(
        test_data,
        ema_len=28,
        hma_len=28,  # <- My attempted fix
        wma_len=28,
        dema_len=28,
        lsma_len=28,
        kama_len=28,
        robustness="Medium",
        La=0.02 / 1000,
        De=0.03 / 100,
        long_threshold=0.1,
        short_threshold=-0.1,
    )
    print("✅ SUCCESS with hma_len")
    print(f"   Results: {len(result2)} symbols")
except Exception as e:
    print(f"❌ FAILED with hma_len: {type(e).__name__}: {e}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("The Rust function signature ACTUALLY uses 'hull_len', not 'hma_len'")
print("My previous fix was WRONG - I should have left it as hull_len")
print("The real question: WHY does CUDA+Dask work with hma_len?")

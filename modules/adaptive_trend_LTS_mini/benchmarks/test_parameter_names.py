"""Deep investigation into Rust function parameter handling."""

import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import atc_rust

print("=" * 70)
print("TESTING RUST PARAMETER FLEXIBILITY")
print("=" * 70)

test_data = {
    "TEST": np.array([100.0, 101.0, 102.0, 103.0, 104.0] * 20, dtype=np.float64),
}

# Test 1: Using hull_len
print("\n--- Test 1: hull_len only ---")
try:
    result1 = atc_rust.compute_atc_signals_batch(
        test_data,
        ema_len=28,
        hull_len=28,
        wma_len=28,
        dema_len=28,
        lsma_len=28,
        kama_len=28,
        robustness="Medium",
        La=0.00002,
        De=0.0003,
        long_threshold=0.1,
        short_threshold=-0.1,
    )
    print(f"✅ SUCCESS: hull_len works")
    print(f"   Result: {result1['TEST'][-1]}")
except Exception as e:
    print(f"❌ FAIL: {e}")

# Test 2: Using hma_len
print("\n--- Test 2: hma_len only ---")
try:
    result2 = atc_rust.compute_atc_signals_batch(
        test_data,
        ema_len=28,
        hma_len=28,
        wma_len=28,
        dema_len=28,
        lsma_len=28,
        kama_len=28,
        robustness="Medium",
        La=0.00002,
        De=0.0003,
        long_threshold=0.1,
        short_threshold=-0.1,
    )
    print(f"✅ SUCCESS: hma_len works")
    print(f"   Result: {result2['TEST'][-1]}")
except Exception as e:
    print(f"❌ FAIL: {e}")

# Test 3: Using BOTH
print("\n--- Test 3: BOTH hull_len AND hma_len ---")
try:
    result3 = atc_rust.compute_atc_signals_batch(
        test_data,
        ema_len=28,
        hull_len=28,
        hma_len=30,  # Different value to see which one is used
        wma_len=28,
        dema_len=28,
        lsma_len=28,
        kama_len=28,
        robustness="Medium",
        La=0.00002,
        De=0.0003,
        long_threshold=0.1,
        short_threshold=-0.1,
    )
    print(f"✅ SUCCESS: both work")
    print(f"   Result: {result3['TEST'][-1]}")
except Exception as e:
    print(f"❌ FAIL: {e}")

# Test 4: Missing BOTH
print("\n--- Test 4: Missing BOTH (should use default) ---")
try:
    result4 = atc_rust.compute_atc_signals_batch(
        test_data,
        ema_len=28,
        # No hull_len or hma_len
        wma_len=28,
        dema_len=28,
        lsma_len=28,
        kama_len=28,
        robustness="Medium",
        La=0.00002,
        De=0.0003,
        long_threshold=0.1,
        short_threshold=-0.1,
    )
    print(f"✅ SUCCESS: defaults work")
    print(f"   Result: {result4['TEST'][-1]}")
except Exception as e:
    print(f"❌ FAIL: {e}")

print("\n" + "=" * 70)
print("COMPARING RESULTS")
print("=" * 70)
if "result1" in locals() and "result2" in locals():
    print(f"hull_len result: {result1['TEST'][-1]}")
    print(f"hma_len result:  {result2['TEST'][-1]}")
    print(f"Match: {np.array_equal(result1['TEST'], result2['TEST'])}")

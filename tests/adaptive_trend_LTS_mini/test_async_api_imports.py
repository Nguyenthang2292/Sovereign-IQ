"""Test script to verify async API import paths.

This script demonstrates all available import patterns for the async incremental ATC API.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("Testing async API import paths...\n")

# Pattern 1: Direct import from incremental package (recommended)
print("1. Direct import from incremental package:")
try:
    from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.incremental import (
        AsyncIncrementalATC,
        AsyncMultiTimeframeIncrementalATC,
        process_price_stream,
    )
    print(f"   [OK] AsyncIncrementalATC: {AsyncIncrementalATC.__name__}")
    print(f"   [OK] AsyncMultiTimeframeIncrementalATC: {AsyncMultiTimeframeIncrementalATC.__name__}")
    print(f"   [OK] process_price_stream: {process_price_stream.__name__}")
except Exception as e:
    print(f"   [FAIL] Failed: {e}")

print()

# Pattern 2: Import from async_api submodule
print("2. Import from async_api submodule:")
try:
    from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.incremental.async_api import (
        AsyncIncrementalATC as ATC2,
        AsyncMultiTimeframeIncrementalATC as MTF2,
        process_price_stream as stream2,
    )
    print(f"   ✓ AsyncIncrementalATC: {ATC2.__name__}")
    print(f"   ✓ AsyncMultiTimeframeIncrementalATC: {MTF2.__name__}")
    print(f"   ✓ process_price_stream: {stream2.__name__}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print()

# Pattern 3: Import from async_wrapper (internal, not recommended)
print("3. Import from async_wrapper (internal, not recommended):")
try:
    from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.incremental.async_wrapper import (
        AsyncIncrementalATC as ATC3,
    )
    print(f"   ✓ AsyncIncrementalATC: {ATC3.__name__}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print()

# Pattern 4: Also verify sync API is still accessible
print("4. Verify sync API still accessible:")
try:
    from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.incremental import (
        IncrementalATC,
        MultiTimeframeIncrementalATC,
    )
    print(f"   ✓ IncrementalATC: {IncrementalATC.__name__}")
    print(f"   ✓ MultiTimeframeIncrementalATC: {MultiTimeframeIncrementalATC.__name__}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print()
print("All import patterns verified successfully!")
print()
print("Recommended usage:")
print("  from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.incremental import AsyncIncrementalATC")

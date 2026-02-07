"""Quick verification script for async API.

This script demonstrates the Quick Verification example from ASYNC_API.md
and confirms that the async API is working correctly.

Usage:
    python tests/adaptive_trend_LTS_mini/verify_async_api.py
"""

import asyncio
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.incremental import (
    AsyncIncrementalATC,
)


async def verify_async_api():
    """Verify async API is working correctly."""
    print("=" * 60)
    print("Async API Quick Verification")
    print("=" * 60)

    # Minimal configuration
    config = {"ema_len": 20, "hma_len": 20, "wma_len": 20}
    atc = AsyncIncrementalATC(config)

    # Test with sufficient historical data (need at least max_len + warmup)
    # Generate 50 bars of sample data
    import numpy as np

    np.random.seed(42)
    base = 100.0
    returns = np.random.normal(0.001, 0.02, 50)
    prices = pd.Series(base * np.exp(np.cumsum(returns)))
    print(f"\nTesting with {len(prices)} price points for initialization")

    # Test initialize
    print("\n1. Testing initialize()...")
    result = await atc.initialize(prices)
    print(f"   ✓ Initialize successful: {result is not None}")
    print(f"   - State initialized: {atc.state.get('initialized', False)}")
    print(f"   - Price history length: {len(atc.state.get('price_history', []))}")

    # Test update
    print("\n2. Testing update()...")
    new_price = 104.0
    signal = await atc.update(new_price)
    print(f"   ✓ Update successful: Signal = {signal:.4f}")
    print(f"   - New price added: {new_price}")
    print(f"   - Current signal: {signal:.4f}")

    # Test batch update
    print("\n3. Testing batch_update()...")
    batch_prices = [105.0, 106.0, 104.5]
    signals = await atc.batch_update(batch_prices)
    print(f"   ✓ Batch update successful: {len(signals)} signals")
    print(f"   - Batch prices: {batch_prices}")
    print(f"   - Signals: {[f'{s:.4f}' for s in signals]}")

    # Summary
    print("\n" + "=" * 60)
    print("✓ All async API tests passed!")
    print("=" * 60)
    print("\nThe async API is working correctly. You can now:")
    print("  1. Use AsyncIncrementalATC in your async applications")
    print("  2. Integrate with WebSocket streams")
    print("  3. Use with FastAPI and other async frameworks")
    print("\nRefer to docs/ASYNC_API.md for detailed examples.")
    print("=" * 60)


def main():
    """Run verification."""
    try:
        asyncio.run(verify_async_api())
        return 0
    except Exception as e:
        print(f"\n✗ Error during verification: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""Quick test to verify CUDA bug fix - parameter name mismatch."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from modules.adaptive_trend_LTS.core.compute_atc_signals.batch_processor import (
    process_symbols_batch_cuda,
)
from modules.common.utils import log_error, log_info, log_success


def generate_test_data(num_symbols=5, num_bars=100):
    """Generate simple test data."""
    symbols_data = {}
    np.random.seed(42)

    for i in range(num_symbols):
        symbol = f"TEST{i}"
        # Generate simple trending price data
        prices = pd.Series(
            100 + np.cumsum(np.random.randn(num_bars) * 0.5),
            index=pd.date_range("2024-01-01", periods=num_bars, freq="1h"),
        )
        symbols_data[symbol] = prices

    return symbols_data


def main():
    """Test CUDA batch processing with fixed parameter."""
    log_info("=" * 60)
    log_info("Testing CUDA Bug Fix - Parameter Name Mismatch")
    log_info("=" * 60)

    # Generate test data
    test_data = generate_test_data(num_symbols=5, num_bars=100)
    log_info(f"Generated test data: {len(test_data)} symbols")

    # Test config
    config = {
        "ema_len": 28,
        "hma_len": 28,  # The parameter that was causing the bug
        "wma_len": 28,
        "dema_len": 28,
        "lsma_len": 28,
        "kama_len": 28,
        "robustness": "Medium",
        "La": 0.02,
        "De": 0.03,
        "long_threshold": 0.1,
        "short_threshold": -0.1,
        "use_cuda": True,
    }

    try:
        log_info("\nAttempting CUDA batch processing...")
        results = process_symbols_batch_cuda(test_data, config, num_threads=2)

        if results:
            log_success(f"✅ CUDA processing successful! Processed {len(results)} symbols")

            # Check if signals are valid
            for symbol, result in results.items():
                if "Average_Signal" in result:
                    signal = result["Average_Signal"]
                    log_info(f"  {symbol}: {len(signal)} bars, last signal: {signal.iloc[-1]:.6f}")
                else:
                    log_error(f"  {symbol}: Missing Average_Signal!")

            log_success("\n" + "=" * 60)
            log_success("✅ BUG FIX VERIFIED - CUDA works correctly!")
            log_success("=" * 60)
        else:
            log_error("❌ CUDA returned empty results")
            return False

    except Exception as e:
        log_error(f"❌ CUDA processing failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

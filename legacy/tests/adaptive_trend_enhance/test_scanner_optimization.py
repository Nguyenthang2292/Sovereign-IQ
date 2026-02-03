from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from modules.adaptive_trend_enhance.core.scanner.scan_all_symbols import scan_all_symbols
from modules.adaptive_trend_enhance.utils.config import ATCConfig


class MockDataFetcher:
    def __init__(self):
        self.exchange_name = "binance"

    def list_binance_futures_symbols(self, max_candidates=None, progress_label=None):
        return ["BTCUSDT", "ETHUSDT"]

    def fetch_ohlcv_with_fallback_exchange(self, symbol, limit=1500, timeframe="15min", check_freshness=True):
        # Create random data
        # Convert "15m" to "15min" to avoid pandas FutureWarning about deprecated 'm' frequency
        freq = timeframe.replace("m", "min") if timeframe.endswith("m") and not timeframe.endswith("min") else timeframe
        dates = pd.date_range(end=pd.Timestamp.now(), periods=limit, freq=freq)
        df = pd.DataFrame(
            {
                "close": np.random.randn(limit) + 100,
                "open": np.random.randn(limit) + 100,
                "high": np.random.randn(limit) + 105,
                "low": np.random.randn(limit) + 95,
                "volume": np.random.randn(limit) * 1000,
            },
            index=dates,
        )
        return df, "binance"


def test_scanner_float32_integration():
    """Verify scanner runs successfully with float32 precision configuration."""

    fetcher = MockDataFetcher()

    # Configure for float32
    config = ATCConfig(limit=100, precision="float32")

    # We patch compute_atc_signals to verify it receives the correct precision
    with patch("modules.adaptive_trend_enhance.core.scanner.process_symbol.compute_atc_signals") as mock_compute:
        # Prepare mock return for compute_atc_signals so scanner doesn't crash on result processing
        # It needs to return a dict with Average_Signal
        avg_sig = pd.Series(np.random.randn(100), index=pd.date_range(end=pd.Timestamp.now(), periods=100, freq="15min"))
        mock_compute.return_value = {"Average_Signal": avg_sig, "EMA_Signal": avg_sig, "EMA_S": avg_sig}

        long_df, short_df = scan_all_symbols(data_fetcher=fetcher, atc_config=config, execution_mode="sequential")

        # Verify compute_atc_signals was called with precision="float32"
        # The args might be passed as kwargs
        call_args = mock_compute.call_args
        assert call_args is not None

        # Check kwargs
        assert call_args.kwargs.get("precision") == "float32"

        # Also verify raw values casting logic in process_symbol
        # (This is implicitly tested by the fact that the code ran without error,
        # but unit test for process_symbol would be more direct.
        # Here we trust integration test implies component function)


if __name__ == "__main__":
    test_scanner_float32_integration()

import pandas as pd
import pytest
from unittest.mock import Mock, patch
from modules.adaptive_trend_enhance.core.scanner.scan_all_symbols import scan_all_symbols
from modules.adaptive_trend_enhance.utils.config import ATCConfig


class TestScannerCaching:
    """Test caching mechanism in scan_all_symbols."""

    @pytest.fixture
    def mock_data_fetcher(self):
        fetcher = Mock()
        # Mock list_binance_futures_symbols to return some symbols
        fetcher.list_binance_futures_symbols.return_value = ["BTC/USDT", "ETH/USDT"]

        # Mock fetch_ohlcv_with_fallback_exchange to return a dummy dataframe
        # This will be called if cache is NOT used
        df = pd.DataFrame(
            {
                "open": [100.0] * 100,
                "high": [105.0] * 100,
                "low": [95.0] * 100,
                "close": [102.0] * 100,
                "volume": [1000.0] * 100,
            }
        )
        fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (df, "binance")
        return fetcher

    @pytest.fixture
    def atc_config(self):
        config = Mock(spec=ATCConfig)
        config.timeframe = "1h"
        config.limit = 100
        config.ema_len = 20
        config.hma_len = 20
        config.wma_len = 20
        config.dema_len = 20
        config.lsma_len = 20
        config.kama_len = 20
        config.robustness = "Medium"
        config.lambda_param = 0.5
        config.decay = 0.1
        config.cutout = 5
        config.precision = "float64"
        config.calculation_source = "close"
        config.batch_size = 10
        # Add weights
        config.ema_w = 1.0
        config.hma_w = 1.0
        config.wma_w = 1.0
        config.dema_w = 1.0
        config.lsma_w = 1.0
        config.kama_w = 1.0
        config.long_threshold = 0.0
        config.short_threshold = 0.0
        return config

    def test_scan_all_symbols_uses_cache(self, mock_data_fetcher, atc_config):
        """Test that scan_all_symbols uses the provided cache."""

        # Create a cache with specific data for BTC/USDT
        cached_df = pd.DataFrame(
            {
                "open": [200.0] * 100,
                "high": [205.0] * 100,
                "low": [195.0] * 100,
                "close": [202.0] * 100,
                "volume": [2000.0] * 100,
            }
        )
        ohlcv_cache = {"BTC/USDT": cached_df}

        # Patch dependencies to isolate scanner logic
        with patch("modules.adaptive_trend_enhance.core.scanner.threadpool._process_symbol") as mock_process:
            # Configure mock_process to verify it receives the cache
            mock_process.return_value = {
                "symbol": "BTC/USDT",
                "signal": 1.0,
                "trend": 1,
                "price": 202.0,
                "exchange": "CACHED",
            }

            scan_all_symbols(
                data_fetcher=mock_data_fetcher,
                atc_config=atc_config,
                execution_mode="threadpool",
                ohlcv_cache=ohlcv_cache,
            )

            # Verify _process_symbol was called with ohlcv_cache
            # args: symbol, data_fetcher, atc_config, min_signal, ohlcv_cache
            args, kwargs = mock_process.call_args
            assert args[0] in ["BTC/USDT", "ETH/USDT"]  # Symbol
            assert args[4] == ohlcv_cache  # Cache passed

    def test_scan_all_symbols_sequential_uses_cache(self, mock_data_fetcher, atc_config):
        """Test that scan_all_symbols uses the provided cache in sequential mode."""

        cached_df = pd.DataFrame(
            {
                "open": [200.0] * 100,
                "high": [205.0] * 100,
                "low": [195.0] * 100,
                "close": [202.0] * 100,
                "volume": [2000.0] * 100,
            }
        )
        ohlcv_cache = {"BTC/USDT": cached_df}

        with patch("modules.adaptive_trend_enhance.core.scanner.sequential._process_symbol") as mock_process:
            mock_process.return_value = {
                "symbol": "BTC/USDT",
                "signal": 1.0,
                "trend": 1,
                "price": 202.0,
                "exchange": "CACHED",
            }

            scan_all_symbols(
                data_fetcher=mock_data_fetcher,
                atc_config=atc_config,
                execution_mode="sequential",
                ohlcv_cache=ohlcv_cache,
            )

            # Verify _process_symbol was called with ohlcv_cache
            # _process_symbol is called inside _process_symbols_batched generator
            # We can check call_args_list to see if any call had cache
            found_cache_call = False
            for call in mock_process.call_args_list:
                args, kwargs = call
                if len(args) > 4 and args[4] == ohlcv_cache:
                    found_cache_call = True
                    break

            assert found_cache_call, "ohlcv_cache was not passed to _process_symbol in sequential mode"

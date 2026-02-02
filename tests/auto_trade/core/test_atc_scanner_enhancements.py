from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from modules.auto_trade.core.atc_scanner import ATCScanner


@pytest.fixture
def mock_data_fetcher():
    return MagicMock()


@pytest.fixture
def mock_scan_all_symbols():
    with patch("modules.auto_trade.core.atc_scanner.scan_all_symbols") as mock:
        yield mock


class TestATCScannerEnhancements:
    """Tests for signal strength and parallel execution enhancements."""

    def test_init_extended_config(self, mock_data_fetcher):
        """Test initialization with new config parameters."""
        config = {
            "use_signal_strength": True,
        }
        scanner = ATCScanner(mock_data_fetcher, config=config)
        assert scanner.use_signal_strength is True
        # max_workers should be auto-detected (not None)
        assert scanner.max_workers is not None
        assert scanner.max_workers > 0

    def test_parallel_execution_auto_detect(self, mock_data_fetcher):
        """Test that max_workers is auto-detected and passed to ThreadPoolExecutor."""
        # No max_workers in config
        config = {"timeframes": ["1h", "15m", "5m"]}
        scanner = ATCScanner(mock_data_fetcher, config=config)

        with (
            patch("modules.auto_trade.core.atc_scanner.scan_all_symbols") as mock_scan,
            patch("modules.auto_trade.core.atc_scanner.ThreadPoolExecutor") as mock_executor_cls,
            patch("modules.auto_trade.core.atc_scanner.as_completed") as mock_as_completed,
        ):
            mock_scan.return_value = (pd.DataFrame(), pd.DataFrame())

            # Configure as_completed to yield futures immediately
            def side_effect(futures):
                return list(futures)

            mock_as_completed.side_effect = side_effect

            # Need to setup context manager mock
            mock_executor = MagicMock()
            mock_executor_cls.return_value.__enter__.return_value = mock_executor

            # Configure future.result() to return (longs, shorts)
            mock_future = MagicMock()
            mock_future.result.return_value = (pd.DataFrame(), pd.DataFrame())
            mock_executor.submit.return_value = mock_future

            scanner.scan_symbols(["BTCUSDT"])

            # Verify executor initialized with auto-detected max_workers
            # We don't know exact number (hardware dependent), but check it was called with scanner.max_workers
            mock_executor_cls.assert_called_with(max_workers=scanner.max_workers)

    def test_signal_strength_capture(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test that signal strengths are captured in SignalResult."""
        scanner = ATCScanner(mock_data_fetcher)

        def side_effect(data_fetcher, atc_config, symbols, **kwargs):
            if atc_config.timeframe == "1h":
                # Strong LONG
                return pd.DataFrame({"symbol": ["BTCUSDT"], "signal": [0.9]}), pd.DataFrame()
            elif atc_config.timeframe == "15m":
                # Weak LONG
                return pd.DataFrame({"symbol": ["BTCUSDT"], "signal": [0.2]}), pd.DataFrame()
            else:
                # Moderate SHORT
                return pd.DataFrame(), pd.DataFrame({"symbol": ["BTCUSDT"], "signal": [-0.5]})

        mock_scan_all_symbols.side_effect = side_effect

        results = scanner.scan_symbols(["BTCUSDT"])

        assert len(results) == 1
        btc = results[0]

        # Verify strengths dictionary
        assert btc.strengths["1h"] == 0.9
        assert btc.strengths["15m"] == 0.2
        assert btc.strengths["5m"] == -0.5

    def test_weighted_signal_strength_scoring(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test scoring using signal strength weights."""
        config = {
            "use_signal_strength": True,
            "weights": {"1h": 0.5, "15m": 0.3, "5m": 0.2},
            "threshold": 0.0,  # Lower threshold to ensure we get results
        }
        scanner = ATCScanner(mock_data_fetcher, config=config)

        def side_effect(data_fetcher, atc_config, symbols, **kwargs):
            if atc_config.timeframe == "1h":
                # 1h: Strong LONG (0.8) -> 0.5 * 0.8 = 0.4
                return pd.DataFrame({"symbol": ["BTCUSDT"], "signal": [0.8]}), pd.DataFrame()
            elif atc_config.timeframe == "15m":
                # 15m: Weak LONG (0.2) -> 0.3 * 0.2 = 0.06
                return pd.DataFrame({"symbol": ["BTCUSDT"], "signal": [0.2]}), pd.DataFrame()
            elif atc_config.timeframe == "5m":
                # 5m: Moderate SHORT (-0.5) -> 0.2 * -0.5 = -0.1
                return pd.DataFrame(), pd.DataFrame({"symbol": ["BTCUSDT"], "signal": [-0.5]})
            return pd.DataFrame(), pd.DataFrame()

        mock_scan_all_symbols.side_effect = side_effect

        results = scanner.scan_symbols(["BTCUSDT"])

        assert len(results) == 1
        btc = results[0]

        # Expected score: 0.4 + 0.06 - 0.1 = 0.36
        assert btc.score == 0.36

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from modules.auto_trade.core.atc_scanner import ATCScanner, SignalResult

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_data_fetcher():
    return MagicMock()


@pytest.fixture
def mock_scan_all_symbols():
    with patch("modules.auto_trade.core.atc_scanner.scan_all_symbols") as mock:
        yield mock


# ============================================================================
# Initialization Tests
# ============================================================================


class TestATCScannerInitialization:
    """Tests for ATCScanner initialization and validation."""

    def test_init_with_defaults(self, mock_data_fetcher):
        """Test initialization with default configuration."""
        scanner = ATCScanner(mock_data_fetcher)
        assert scanner.weights == {"1h": 0.5, "15m": 0.3, "5m": 0.2}
        assert scanner.threshold == 0.6
        assert scanner.timeframes == ["1h", "15m", "5m"]
        assert scanner.min_signal == 0.0

    def test_init_with_custom_config(self, mock_data_fetcher):
        """Test initialization with custom configuration."""
        config = {
            "weights": {"4h": 0.6, "1h": 0.4},
            "threshold": 0.7,
            "timeframes": ["4h", "1h"],
            "min_signal": 0.01,
        }
        scanner = ATCScanner(mock_data_fetcher, config=config)
        assert scanner.weights == {"4h": 0.6, "1h": 0.4}
        assert scanner.threshold == 0.7
        assert scanner.timeframes == ["4h", "1h"]
        assert scanner.min_signal == 0.01

    def test_init_with_negative_weights_raises_error(self, mock_data_fetcher):
        """Test that negative weights raise ValueError."""
        config = {"weights": {"1h": -0.5, "15m": 0.3, "5m": 0.2}}
        with pytest.raises(ValueError, match="non-negative"):
            ATCScanner(mock_data_fetcher, config=config)

    def test_init_with_zero_sum_weights_raises_error(self, mock_data_fetcher):
        """Test that weights summing to zero raise ValueError."""
        config = {"weights": {"1h": 0.0, "15m": 0.0, "5m": 0.0}}
        with pytest.raises(ValueError, match="sum to zero"):
            ATCScanner(mock_data_fetcher, config=config)

    def test_init_with_invalid_threshold_raises_error(self, mock_data_fetcher):
        """Test that invalid threshold raises ValueError."""
        # Threshold > 1.0
        with pytest.raises(ValueError, match="between 0 and 1"):
            ATCScanner(mock_data_fetcher, config={"threshold": 1.5})

        # Threshold < 0
        with pytest.raises(ValueError, match="between 0 and 1"):
            ATCScanner(mock_data_fetcher, config={"threshold": -0.1})

    def test_init_warns_on_non_normalized_weights(self, mock_data_fetcher, caplog):
        """Test that non-normalized weights generate a warning."""
        config = {"weights": {"1h": 0.5, "15m": 0.5, "5m": 0.5}}  # Sum = 1.5
        scanner = ATCScanner(mock_data_fetcher, config=config)
        # Scanner should still initialize, just warn
        assert scanner.weights == {"1h": 0.5, "15m": 0.5, "5m": 0.5}


# ============================================================================
# Scan Tests
# ============================================================================


class TestATCScannerScan:
    """Tests for ATCScanner scan functionality."""

    def test_scan_symbols_returns_signals(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test basic scan returns expected signals."""
        scanner = ATCScanner(mock_data_fetcher)
        symbols = ["BTCUSDT"]

        # All timeframes show LONG for BTC
        mock_scan_all_symbols.return_value = (
            pd.DataFrame({"symbol": ["BTCUSDT"]}),
            pd.DataFrame(),
        )

        results = scanner.scan_symbols(symbols)
        assert len(results) == 1
        assert results[0].symbol == "BTCUSDT"
        assert results[0].signal_type == "LONG"

    def test_scan_symbols_filters_by_threshold(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test that weak signals are filtered out."""
        scanner = ATCScanner(mock_data_fetcher, config={"threshold": 0.8})
        symbols = ["WEAK_SYMBOL"]

        # Weak signal: Only 1h Long (0.5 score < 0.8)
        def side_effect(data_fetcher, atc_config, symbols, **kwargs):
            if atc_config.timeframe == "1h":
                return pd.DataFrame({"symbol": ["WEAK_SYMBOL"]}), pd.DataFrame()
            return pd.DataFrame(), pd.DataFrame()

        mock_scan_all_symbols.side_effect = side_effect

        results = scanner.scan_symbols(symbols)
        assert len(results) == 0

    def test_scan_symbols_handles_empty_results(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test safe handling of empty scan results."""
        scanner = ATCScanner(mock_data_fetcher)
        mock_scan_all_symbols.return_value = (pd.DataFrame(), pd.DataFrame())

        results = scanner.scan_symbols(["BTCUSDT"])
        assert len(results) == 0

    def test_scan_symbols_handles_scan_errors(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test that scan errors are handled gracefully."""
        scanner = ATCScanner(mock_data_fetcher)
        mock_scan_all_symbols.side_effect = Exception("Network error")

        results = scanner.scan_symbols(["BTCUSDT"])
        # Should return empty list, not raise
        assert len(results) == 0


# ============================================================================
# Aggregation Tests
# ============================================================================


class TestATCScannerAggregation:
    """Tests for signal aggregation logic."""

    def test_score_calculation_long_signals(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test score calculation for LONG signals."""
        scanner = ATCScanner(mock_data_fetcher)
        symbols = ["BTCUSDT"]

        # All timeframes LONG
        mock_scan_all_symbols.return_value = (
            pd.DataFrame({"symbol": ["BTCUSDT"]}),
            pd.DataFrame(),
        )

        results = scanner.scan_symbols(symbols)
        btc = results[0]
        # 0.5 + 0.3 + 0.2 = 1.0
        assert btc.score == 1.0
        assert btc.signal_type == "LONG"

    def test_score_calculation_short_signals(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test score calculation for SHORT signals."""
        scanner = ATCScanner(mock_data_fetcher)
        symbols = ["BTCUSDT"]

        # All timeframes SHORT
        mock_scan_all_symbols.return_value = (
            pd.DataFrame(),
            pd.DataFrame({"symbol": ["BTCUSDT"]}),
        )

        results = scanner.scan_symbols(symbols)
        btc = results[0]
        # -0.5 - 0.3 - 0.2 = -1.0
        assert btc.score == -1.0
        assert btc.signal_type == "SHORT"

    def test_score_calculation_mixed_signals(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test score calculation for mixed signals."""
        scanner = ATCScanner(mock_data_fetcher)
        symbols = ["BTCUSDT", "ETHUSDT"]

        def side_effect(data_fetcher, atc_config, symbols, **kwargs):
            tf = atc_config.timeframe
            if tf == "1h":
                return (
                    pd.DataFrame({"symbol": ["BTCUSDT"]}),  # Longs
                    pd.DataFrame({"symbol": ["ETHUSDT"]}),  # Shorts
                )
            elif tf == "15m":
                return pd.DataFrame({"symbol": ["BTCUSDT"]}), pd.DataFrame()
            elif tf == "5m":
                return pd.DataFrame({"symbol": ["BTCUSDT"]}), pd.DataFrame({"symbol": ["ETHUSDT"]})
            return pd.DataFrame(), pd.DataFrame()

        mock_scan_all_symbols.side_effect = side_effect

        results = scanner.scan_symbols(symbols)

        # BTC: 1h(0.5) + 15m(0.3) + 5m(0.2) = 1.0 -> LONG
        # ETH: 1h(-0.5) + 15m(0.0) + 5m(-0.2) = -0.7 -> SHORT

        assert len(results) == 2

        btc_result = next(r for r in results if r.symbol == "BTCUSDT")
        assert btc_result.signal_type == "LONG"
        assert btc_result.score == 1.0
        assert btc_result.details["1h"] == "LONG"
        assert btc_result.details["15m"] == "LONG"
        assert btc_result.details["5m"] == "LONG"

        eth_result = next(r for r in results if r.symbol == "ETHUSDT")
        assert eth_result.signal_type == "SHORT"
        assert eth_result.score == -0.7
        assert eth_result.details["1h"] == "SHORT"
        assert eth_result.details["15m"] == "NEUTRAL"
        assert eth_result.details["5m"] == "SHORT"

    def test_threshold_application(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test threshold correctly filters results."""
        # Low threshold - should include more signals
        scanner_low = ATCScanner(mock_data_fetcher, config={"threshold": 0.3})
        # High threshold - should include fewer signals
        scanner_high = ATCScanner(mock_data_fetcher, config={"threshold": 0.8})

        def side_effect(data_fetcher, atc_config, symbols, **kwargs):
            if atc_config.timeframe == "1h":
                return pd.DataFrame({"symbol": ["BTCUSDT"]}), pd.DataFrame()
            return pd.DataFrame(), pd.DataFrame()

        mock_scan_all_symbols.side_effect = side_effect

        # With low threshold (0.3), 0.5 score should pass
        results_low = scanner_low.scan_symbols(["BTCUSDT"])
        assert len(results_low) == 1

        # With high threshold (0.8), 0.5 score should not pass
        results_high = scanner_high.scan_symbols(["BTCUSDT"])
        assert len(results_high) == 0


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestATCScannerEdgeCases:
    """Tests for edge cases."""

    def test_single_symbol(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test scanning a single symbol."""
        scanner = ATCScanner(mock_data_fetcher)
        mock_scan_all_symbols.return_value = (
            pd.DataFrame({"symbol": ["BTCUSDT"]}),
            pd.DataFrame(),
        )

        results = scanner.scan_symbols(["BTCUSDT"])
        assert len(results) == 1
        assert results[0].symbol == "BTCUSDT"

    def test_all_neutral_signals(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test when all signals are neutral."""
        scanner = ATCScanner(mock_data_fetcher)
        # No longs, no shorts = neutral for all timeframes
        mock_scan_all_symbols.return_value = (pd.DataFrame(), pd.DataFrame())

        results = scanner.scan_symbols(["BTCUSDT", "ETHUSDT"])
        assert len(results) == 0

    def test_conflicting_timeframe_signals(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test symbol with conflicting signals across timeframes."""
        scanner = ATCScanner(mock_data_fetcher, config={"threshold": 0.4})

        def side_effect(data_fetcher, atc_config, symbols, **kwargs):
            tf = atc_config.timeframe
            if tf == "1h":
                # 1h: LONG (weight 0.5)
                return pd.DataFrame({"symbol": ["BTCUSDT"]}), pd.DataFrame()
            elif tf == "15m":
                # 15m: SHORT (weight -0.3)
                return pd.DataFrame(), pd.DataFrame({"symbol": ["BTCUSDT"]})
            elif tf == "5m":
                # 5m: SHORT (weight -0.2)
                return pd.DataFrame(), pd.DataFrame({"symbol": ["BTCUSDT"]})
            return pd.DataFrame(), pd.DataFrame()

        mock_scan_all_symbols.side_effect = side_effect

        results = scanner.scan_symbols(["BTCUSDT"])
        # Score: 0.5 - 0.3 - 0.2 = 0.0 -> NEUTRAL (not > 0.4, not < -0.4)
        assert len(results) == 0

    def test_custom_timeframes(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test scanner with custom timeframes."""
        config = {
            "timeframes": ["4h", "1h"],
            "weights": {"4h": 0.7, "1h": 0.3},
        }
        scanner = ATCScanner(mock_data_fetcher, config=config)

        def side_effect(data_fetcher, atc_config, symbols, **kwargs):
            # Both timeframes LONG
            return pd.DataFrame({"symbol": ["BTCUSDT"]}), pd.DataFrame()

        mock_scan_all_symbols.side_effect = side_effect

        results = scanner.scan_symbols(["BTCUSDT"])
        assert len(results) == 1
        btc = results[0]
        assert btc.score == 1.0
        # Only 4h and 1h should be in details
        assert "4h" in btc.details
        assert "1h" in btc.details
        assert "15m" not in btc.details


# ============================================================================
# SignalResult Tests
# ============================================================================


class TestSignalResult:
    """Tests for SignalResult NamedTuple."""

    def test_signal_result_creation(self):
        """Test SignalResult can be created and accessed."""
        result = SignalResult(
            symbol="BTCUSDT",
            score=0.8,
            signal_type="LONG",
            details={"1h": "LONG", "15m": "LONG", "5m": "NEUTRAL"},
        )
        assert result.symbol == "BTCUSDT"
        assert result.score == 0.8
        assert result.signal_type == "LONG"
        assert result.details["1h"] == "LONG"

    def test_signal_result_is_immutable(self):
        """Test SignalResult is immutable (NamedTuple behavior)."""
        result = SignalResult(
            symbol="BTCUSDT",
            score=0.8,
            signal_type="LONG",
            details={"1h": "LONG"},
        )
        with pytest.raises(AttributeError):
            result.symbol = "ETHUSDT"


# ============================================================================
# Extended Scan Tests (from Review v2)
# ============================================================================


class TestATCScannerScanningExtended:
    """Extended tests for scanning functionality from review recommendations."""

    def _create_mock_scanner(self, mock_data_fetcher, config=None):
        """Helper to create scanner with mocked dependencies."""
        scanner = ATCScanner(mock_data_fetcher, config)
        return scanner

    def test_scan_symbols_all_long(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test scanning when all timeframes show LONG."""
        scanner = self._create_mock_scanner(mock_data_fetcher)

        # Mock all timeframes to return LONG for BTC/USDT
        mock_scan_all_symbols.return_value = (
            pd.DataFrame({"symbol": ["BTCUSDT"]}),
            pd.DataFrame(),
        )

        results = scanner.scan_symbols(["BTCUSDT"])

        assert len(results) == 1
        assert results[0].symbol == "BTCUSDT"
        assert results[0].signal_type == "LONG"
        assert results[0].score == 1.0
        assert results[0].details == {"1h": "LONG", "15m": "LONG", "5m": "LONG"}

    def test_scan_symbols_all_short(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test scanning when all timeframes show SHORT."""
        scanner = self._create_mock_scanner(mock_data_fetcher)

        # Mock all timeframes to return SHORT for ETH/USDT
        mock_scan_all_symbols.return_value = (
            pd.DataFrame(),
            pd.DataFrame({"symbol": ["ETHUSDT"]}),
        )

        results = scanner.scan_symbols(["ETHUSDT"])

        assert len(results) == 1
        assert results[0].symbol == "ETHUSDT"
        assert results[0].signal_type == "SHORT"
        assert results[0].score == -1.0
        assert results[0].details == {"1h": "SHORT", "15m": "SHORT", "5m": "SHORT"}

    def test_scan_symbols_mixed_below_threshold(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test scanning with mixed signals below threshold (NEUTRAL)."""
        scanner = self._create_mock_scanner(mock_data_fetcher)

        call_count = 0

        def side_effect(data_fetcher, atc_config, symbols, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # 1h LONG
                return pd.DataFrame({"symbol": ["BTCUSDT"]}), pd.DataFrame()
            else:  # 15m, 5m NEUTRAL
                return pd.DataFrame(), pd.DataFrame()

        mock_scan_all_symbols.side_effect = side_effect

        results = scanner.scan_symbols(["BTCUSDT"])

        # 1h LONG = 0.5, but threshold is 0.6 → NEUTRAL (filtered out)
        assert len(results) == 0

    def test_scan_symbols_just_above_threshold(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test scanning with score just above threshold."""
        scanner = self._create_mock_scanner(mock_data_fetcher)

        call_count = 0

        def side_effect(data_fetcher, atc_config, symbols, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # 1h and 15m LONG (0.5 + 0.3 = 0.8 > 0.6)
                return pd.DataFrame({"symbol": ["BTCUSDT"]}), pd.DataFrame()
            else:  # 5m NEUTRAL
                return pd.DataFrame(), pd.DataFrame()

        mock_scan_all_symbols.side_effect = side_effect

        results = scanner.scan_symbols(["BTCUSDT"])

        assert len(results) == 1
        assert results[0].signal_type == "LONG"
        assert results[0].score == 0.8

    def test_scan_symbols_conflicting_signals(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test scanning with conflicting signals (LONG vs SHORT)."""
        scanner = self._create_mock_scanner(mock_data_fetcher)

        call_count = 0

        def side_effect(data_fetcher, atc_config, symbols, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # 1h LONG
                return pd.DataFrame({"symbol": ["BTCUSDT"]}), pd.DataFrame()
            elif call_count == 2:  # 15m SHORT
                return pd.DataFrame(), pd.DataFrame({"symbol": ["BTCUSDT"]})
            else:  # 5m SHORT
                return pd.DataFrame(), pd.DataFrame({"symbol": ["BTCUSDT"]})

        mock_scan_all_symbols.side_effect = side_effect

        results = scanner.scan_symbols(["BTCUSDT"])

        # 1h LONG (0.5) - 15m SHORT (0.3) - 5m SHORT (0.2) = 0.0 → NEUTRAL
        assert len(results) == 0

    def test_scan_symbols_multiple_symbols(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test scanning multiple symbols."""
        scanner = self._create_mock_scanner(mock_data_fetcher)

        def side_effect(data_fetcher, atc_config, symbols, **kwargs):
            # BTC: all LONG, ETH: all SHORT, BNB: mixed (below threshold)
            if atc_config.timeframe == "1h":
                longs = pd.DataFrame({"symbol": ["BTCUSDT", "BNBUSDT"]})
                shorts = pd.DataFrame({"symbol": ["ETHUSDT"]})
            elif atc_config.timeframe == "15m":
                longs = pd.DataFrame({"symbol": ["BTCUSDT"]})
                shorts = pd.DataFrame({"symbol": ["ETHUSDT"]})
            else:  # 5m
                longs = pd.DataFrame({"symbol": ["BTCUSDT"]})
                shorts = pd.DataFrame({"symbol": ["ETHUSDT"]})
            return longs, shorts

        mock_scan_all_symbols.side_effect = side_effect

        results = scanner.scan_symbols(["BTCUSDT", "ETHUSDT", "BNBUSDT"])

        assert len(results) == 2  # BTC and ETH exceed threshold, BNB doesn't

        btc_result = [r for r in results if r.symbol == "BTCUSDT"][0]
        assert btc_result.signal_type == "LONG"
        assert btc_result.score == 1.0

        eth_result = [r for r in results if r.symbol == "ETHUSDT"][0]
        assert eth_result.signal_type == "SHORT"
        assert eth_result.score == -1.0

    def test_scan_symbols_empty_list(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test scanning with empty symbol list."""
        scanner = self._create_mock_scanner(mock_data_fetcher)
        mock_scan_all_symbols.return_value = (pd.DataFrame(), pd.DataFrame())

        results = scanner.scan_symbols([])

        assert results == []

    @patch("modules.auto_trade.core.atc_scanner.log_error")
    def test_scan_symbols_handles_scan_error(self, mock_log_error, mock_data_fetcher, mock_scan_all_symbols):
        """Test that scan errors are caught and logged."""
        scanner = self._create_mock_scanner(mock_data_fetcher)

        # First call succeeds, second call fails, third succeeds
        call_count = 0

        def side_effect(data_fetcher, atc_config, symbols, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("Network error")
            return pd.DataFrame({"symbol": ["BTCUSDT"]}), pd.DataFrame()

        mock_scan_all_symbols.side_effect = side_effect

        results = scanner.scan_symbols(["BTCUSDT"])

        # Error was logged
        mock_log_error.assert_called_once()
        assert "Network error" in str(mock_log_error.call_args[0][0])

        # Results still calculated with available timeframes
        # 1h LONG (0.5) + 5m LONG (0.2) = 0.7 > 0.6 → LONG
        assert len(results) == 1


# ============================================================================
# Run Single Scan Tests
# ============================================================================


class TestATCScannerRunSingleScan:
    """Test _run_single_scan helper method."""

    def test_run_single_scan_success(self, mock_data_fetcher):
        """Test successful single scan."""
        scanner = ATCScanner(mock_data_fetcher)

        with (
            patch("modules.auto_trade.core.atc_scanner.create_atc_config_from_dict") as mock_create_config,
            patch("modules.auto_trade.core.atc_scanner.scan_all_symbols") as mock_scan_all,
        ):
            expected_longs = pd.DataFrame({"symbol": ["BTCUSDT"]})
            expected_shorts = pd.DataFrame({"symbol": ["ETHUSDT"]})
            mock_scan_all.return_value = (expected_longs, expected_shorts)

            longs, shorts = scanner._run_single_scan(["BTCUSDT", "ETHUSDT"], "1h")

            assert longs.equals(expected_longs)
            assert shorts.equals(expected_shorts)
            mock_create_config.assert_called_once()

    def test_run_single_scan_filters_excluded_keys(self, mock_data_fetcher):
        """Test that scanner-specific keys are filtered from config."""
        config = {
            "weights": {"1h": 0.5},
            "threshold": 0.6,
            "timeframes": ["1h"],
            "min_signal": 0.5,
            "some_atc_param": "value",  # This should be passed through
        }
        scanner = ATCScanner(mock_data_fetcher, config)

        with (
            patch("modules.auto_trade.core.atc_scanner.create_atc_config_from_dict") as mock_create_config,
            patch(
                "modules.auto_trade.core.atc_scanner.scan_all_symbols", return_value=(pd.DataFrame(), pd.DataFrame())
            ) as mock_scan_all,
        ):
            scanner._run_single_scan(["BTCUSDT"], "1h")

            # Only non-excluded keys should be passed
            passed_config = mock_create_config.call_args[0][0]
            assert "some_atc_param" in passed_config
            assert "weights" not in passed_config
            assert "threshold" not in passed_config
            assert "timeframes" not in passed_config
            assert "min_signal" not in passed_config

    def test_run_single_scan_config_error_propagates(self, mock_data_fetcher):
        """Test that configuration errors propagate."""
        scanner = ATCScanner(mock_data_fetcher)

        with patch("modules.auto_trade.core.atc_scanner.create_atc_config_from_dict") as mock_create_config:
            mock_create_config.side_effect = ValueError("Invalid config")

            with pytest.raises(ValueError, match="Invalid ATC config for 1h"):
                scanner._run_single_scan(["BTCUSDT"], "1h")

    @patch("modules.auto_trade.core.atc_scanner.log_error")
    def test_run_single_scan_runtime_error_returns_empty(self, mock_log_error, mock_data_fetcher):
        """Test that runtime errors return empty DataFrames."""
        scanner = ATCScanner(mock_data_fetcher)

        with (
            patch("modules.auto_trade.core.atc_scanner.create_atc_config_from_dict"),
            patch("modules.auto_trade.core.atc_scanner.scan_all_symbols") as mock_scan_all,
        ):
            mock_scan_all.side_effect = Exception("Network timeout")

            longs, shorts = scanner._run_single_scan(["BTCUSDT"], "1h")

            assert longs.empty
            assert shorts.empty
            mock_log_error.assert_called_once()


# ============================================================================
# Additional Edge Cases from Review v2
# ============================================================================


class TestATCScannerExtendedEdgeCases:
    """Additional edge case tests from review v2."""

    def test_single_timeframe(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test with only one timeframe configured."""
        config = {
            "timeframes": ["1h"],
            "weights": {"1h": 1.0},
            "threshold": 0.9,
        }
        scanner = ATCScanner(mock_data_fetcher, config=config)

        mock_scan_all_symbols.return_value = (
            pd.DataFrame({"symbol": ["BTCUSDT"]}),
            pd.DataFrame(),
        )

        results = scanner.scan_symbols(["BTCUSDT"])

        assert len(results) == 1
        assert results[0].score == 1.0
        assert results[0].details == {"1h": "LONG"}

    def test_custom_timeframes_extended(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test with custom timeframes (not just 5m/15m/1h)."""
        config = {
            "timeframes": ["4h", "1d"],
            "weights": {"4h": 0.6, "1d": 0.4},
        }
        scanner = ATCScanner(mock_data_fetcher, config=config)

        mock_scan_all_symbols.return_value = (
            pd.DataFrame({"symbol": ["BTCUSDT"]}),
            pd.DataFrame(),
        )

        results = scanner.scan_symbols(["BTCUSDT"])

        assert len(results) == 1
        assert results[0].score == 1.0
        assert results[0].details == {"4h": "LONG", "1d": "LONG"}

    def test_threshold_zero(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test with threshold = 0 (all non-zero scores pass)."""
        config = {"threshold": 0.0}
        scanner = ATCScanner(mock_data_fetcher, config=config)

        # Only 5m shows LONG (0.2 score)
        call_count = 0

        def side_effect(data_fetcher, atc_config, symbols, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 3:  # 5m
                return pd.DataFrame({"symbol": ["BTCUSDT"]}), pd.DataFrame()
            return pd.DataFrame(), pd.DataFrame()

        mock_scan_all_symbols.side_effect = side_effect

        results = scanner.scan_symbols(["BTCUSDT"])

        # 0.2 > 0.0 → LONG
        assert len(results) == 1
        assert results[0].signal_type == "LONG"

    def test_threshold_one(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test with threshold = 1.0 (only perfect scores pass)."""
        config = {"threshold": 1.0}
        scanner = ATCScanner(mock_data_fetcher, config=config)

        # 1h and 15m LONG (0.8 total), but threshold is 1.0
        call_count = 0

        def side_effect(data_fetcher, atc_config, symbols, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return pd.DataFrame({"symbol": ["BTCUSDT"]}), pd.DataFrame()
            return pd.DataFrame(), pd.DataFrame()

        mock_scan_all_symbols.side_effect = side_effect

        results = scanner.scan_symbols(["BTCUSDT"])

        # 0.8 < 1.0 → NEUTRAL (filtered out)
        assert len(results) == 0

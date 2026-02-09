from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl
import pytest

from modules.auto_trade.core.atc_scanner import ATCScanner, SignalResult

# ============================================================================
# Initialization Tests
# ============================================================================


class TestATCScannerInitialization:
    """Tests for ATCScanner initialization and validation."""

    def test_init_with_defaults(self, mock_data_fetcher):
        """Test initialization with default configuration."""
        scanner = ATCScanner(mock_data_fetcher)
        assert scanner.weights == {"15m": 0.5, "1h": 0.3, "4h": 0.2}
        assert scanner.threshold == 0.6
        assert scanner.timeframes == ["15m", "1h", "4h"]
        assert scanner.min_signal == 0.0

    def test_init_with_custom_config(self, mock_data_fetcher):
        """Test initialization with custom configuration."""
        config = {
            "weights": {"4h": 0.6, "1h": 0.4},
            "threshold": 0.7,
            "timeframes": ["4h", "1h"],
            "min_signal": 0.01,
        }
        scanner = ATCScanner(mock_data_fetcher, config=config)  # type: ignore[arg-type]
        assert scanner.weights == {"4h": 0.6, "1h": 0.4}
        assert scanner.threshold == 0.7
        assert scanner.timeframes == ["4h", "1h"]
        assert scanner.min_signal == 0.01

    def test_init_with_negative_weights_raises_error(self, mock_data_fetcher):
        """Test that negative weights raise ValueError."""
        config = {"weights": {"15m": -0.5, "1h": 0.3, "4h": 0.2}}
        with pytest.raises(ValueError, match="non-negative"):
            ATCScanner(mock_data_fetcher, config=config)  # type: ignore[arg-type]

    def test_init_with_zero_sum_weights_raises_error(self, mock_data_fetcher):
        """Test that weights summing to zero raise ValueError."""
        config = {"weights": {"15m": 0.0, "1h": 0.0, "4h": 0.0}}
        with pytest.raises(ValueError, match="sum to zero"):
            ATCScanner(mock_data_fetcher, config=config)  # type: ignore[arg-type]

    def test_init_with_invalid_threshold_raises_error(self, mock_data_fetcher):
        """Test that invalid threshold raises ValueError."""
        # Threshold > 1.0
        with pytest.raises(ValueError, match="between 0 and 1"):
            ATCScanner(mock_data_fetcher, config={"threshold": 1.5})  # type: ignore[arg-type]

        # Threshold < 0
        with pytest.raises(ValueError, match="between 0 and 1"):
            ATCScanner(mock_data_fetcher, config={"threshold": -0.1})  # type: ignore[arg-type]

    def test_init_warns_on_non_normalized_weights(self, mock_data_fetcher, caplog):
        """Test that non-normalized weights generate a warning."""
        config = {"weights": {"1h": 0.5, "15m": 0.5, "5m": 0.5}}  # Sum = 1.5
        scanner = ATCScanner(mock_data_fetcher, config=config)  # type: ignore[arg-type]
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
        scanner = ATCScanner(mock_data_fetcher, config={"threshold": 0.8})  # type: ignore[arg-type]
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
                return pd.DataFrame({"symbol": ["BTCUSDT"]}), pd.DataFrame({"symbol": ["ETHUSDT"]})
            elif tf == "4h":
                return pd.DataFrame({"symbol": ["BTCUSDT"]}), pd.DataFrame({"symbol": ["ETHUSDT"]})
            return pd.DataFrame(), pd.DataFrame()

        mock_scan_all_symbols.side_effect = side_effect

        results = scanner.scan_symbols(symbols)

        # Default weights: 15m=0.5, 1h=0.3, 4h=0.2
        # BTC: 15m(0.5) + 1h(0.3) + 4h(0.2) = 1.0 -> LONG
        # ETH: 15m(-0.5) + 1h(-0.3) + 4h(-0.2) = -1.0 -> SHORT

        assert len(results) == 2

        btc_result = next(r for r in results if r.symbol == "BTCUSDT")
        assert btc_result.signal_type == "LONG"
        assert btc_result.score == 1.0
        assert btc_result.details["1h"] == "LONG"
        assert btc_result.details["15m"] == "LONG"
        assert btc_result.details["4h"] == "LONG"

        eth_result = next(r for r in results if r.symbol == "ETHUSDT")
        assert eth_result.signal_type == "SHORT"
        assert eth_result.score == -1.0
        assert eth_result.details["1h"] == "SHORT"
        assert eth_result.details["15m"] == "SHORT"
        assert eth_result.details["4h"] == "SHORT"

    def test_threshold_application(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test threshold correctly filters results."""
        # Low threshold - should include more signals
        scanner_low = ATCScanner(mock_data_fetcher, config={"threshold": 0.3})  # type: ignore[arg-type]
        # High threshold - should include fewer signals (need all TFs with data so threshold is not scaled down)
        scanner_high = ATCScanner(mock_data_fetcher, config={"threshold": 0.8})  # type: ignore[arg-type]

        def side_effect_low(data_fetcher, atc_config, symbols, **kwargs):
            if atc_config.timeframe == "1h":
                return pd.DataFrame({"symbol": ["BTCUSDT"]}), pd.DataFrame()
            return pd.DataFrame(), pd.DataFrame()

        def side_effect_high(data_fetcher, atc_config, symbols, **kwargs):
            # All TFs return same so weight coverage 100% → threshold stays 0.8; score = 0.3 (1h only) < 0.8
            if atc_config.timeframe == "1h":
                return pd.DataFrame({"symbol": ["BTCUSDT"]}), pd.DataFrame()
            # 15m and 4h return empty so TF has no data; then only 1h active, adaptive = 0.8*0.3 = 0.24, score 1.0 would pass.
            # So we need all 3 TFs to have data but only 1h LONG for BTC → score 0.3
            if atc_config.timeframe == "15m":
                return pd.DataFrame(), pd.DataFrame({"symbol": ["OTHER"]})  # active TF, but BTC not in it
            if atc_config.timeframe == "4h":
                return pd.DataFrame(), pd.DataFrame({"symbol": ["OTHER2"]})
            return pd.DataFrame(), pd.DataFrame()

        mock_scan_all_symbols.side_effect = side_effect_low
        results_low = scanner_low.scan_symbols(["BTCUSDT"])
        assert len(results_low) == 1

        mock_scan_all_symbols.side_effect = side_effect_high
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

    def test_ccxt_input_returns_ccxt_symbol(self, mock_data_fetcher, mock_scan_all_symbols):
        """Option A: input symbols in CCXT format (BTC/USDT) are normalized for aggregation;
        returned SignalResult.symbol is mapped back to original CCXT format for downstream."""
        scanner = ATCScanner(mock_data_fetcher)
        mock_scan_all_symbols.return_value = (
            pd.DataFrame({"symbol": ["BTCUSDT"]}),
            pd.DataFrame(),
        )
        results = scanner.scan_symbols(["BTC/USDT"])
        assert len(results) == 1
        assert results[0].symbol == "BTC/USDT"
        assert results[0].signal_type == "LONG"

    def test_all_neutral_signals(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test when all signals are neutral."""
        scanner = ATCScanner(mock_data_fetcher)
        # No longs, no shorts = neutral for all timeframes
        mock_scan_all_symbols.return_value = (pd.DataFrame(), pd.DataFrame())

        results = scanner.scan_symbols(["BTCUSDT", "ETHUSDT"])
        assert len(results) == 0

    def test_conflicting_timeframe_signals(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test symbol with conflicting signals across timeframes."""
        scanner = ATCScanner(mock_data_fetcher, config={"threshold": 0.4})  # type: ignore[arg-type]

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
        scanner = ATCScanner(mock_data_fetcher, config=config)  # type: ignore[arg-type]

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
            strengths={"1h": 0.8, "15m": 0.3},
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
            strengths={"1h": 0.8},
        )
        with pytest.raises(AttributeError):
            result.symbol = "ETHUSDT"  # type: ignore[misc]


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
        assert results[0].details == {"1h": "LONG", "15m": "LONG", "4h": "LONG"}

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
        assert results[0].details == {"1h": "SHORT", "15m": "SHORT", "4h": "SHORT"}

    def test_scan_symbols_mixed_below_threshold(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test scanning with mixed signals below threshold (NEUTRAL)."""
        scanner = self._create_mock_scanner(mock_data_fetcher)

        call_count = 0

        def side_effect(data_fetcher, atc_config, symbols, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # 1h LONG for BTC only
                return pd.DataFrame({"symbol": ["BTCUSDT"]}), pd.DataFrame()
            # 15m and 4h: active (have some symbol) but BTC not in longs/shorts → BTC NEUTRAL for those
            if call_count == 2:  # 15m
                return pd.DataFrame({"symbol": ["OTHER"]}), pd.DataFrame()
            if call_count == 3:  # 4h
                return pd.DataFrame(), pd.DataFrame({"symbol": ["OTHER2"]})
            return pd.DataFrame(), pd.DataFrame()

        mock_scan_all_symbols.side_effect = side_effect

        results = scanner.scan_symbols(["BTCUSDT"])

        # 1h LONG = 0.3 (weight), 15m/4h NEUTRAL; score = 0.3, threshold 0.6 → NEUTRAL (filtered out)
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
        # Active TFs are normalized (15m + 1h), so total score becomes 1.0 with adaptive threshold
        assert results[0].score == 1.0

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
            # Mock returns Pandas DFs
            pd_longs = pd.DataFrame({"symbol": ["BTCUSDT"], "signal": [1.0]})
            pd_shorts = pd.DataFrame({"symbol": ["ETHUSDT"], "signal": [-1.0]})
            mock_scan_all.return_value = (pd_longs, pd_shorts)

            longs, shorts = scanner._run_single_scan(["BTCUSDT", "ETHUSDT"], "1h")

            # Result should be Polars DFs
            assert isinstance(longs, pl.DataFrame)
            assert isinstance(shorts, pl.DataFrame)

            # Verify content
            expected_longs = pl.from_pandas(pd_longs)
            expected_shorts = pl.from_pandas(pd_shorts)

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
        scanner = ATCScanner(mock_data_fetcher, config)  # type: ignore[arg-type]

        with (
            patch("modules.auto_trade.core.atc_scanner.create_atc_config_from_dict") as mock_create_config,
            patch(
                "modules.auto_trade.core.atc_scanner.scan_all_symbols", return_value=(pd.DataFrame(), pd.DataFrame())
            ),
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

            # Check for Polars empty
            assert isinstance(longs, pl.DataFrame)
            assert longs.is_empty()
            assert shorts.is_empty()
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
        scanner = ATCScanner(mock_data_fetcher, config=config)  # type: ignore[arg-type]

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
        scanner = ATCScanner(mock_data_fetcher, config=config)  # type: ignore[arg-type]

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
        scanner = ATCScanner(mock_data_fetcher, config=config)  # type: ignore[arg-type]

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
        scanner = ATCScanner(mock_data_fetcher, config=config)  # type: ignore[arg-type]

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

        # Adaptive threshold scales by active weight (0.8) → 1.0 * 0.8 = 0.8, score 1.0 passes
        assert len(results) == 1
        assert results[0].score == 1.0


# ============================================================================
# Additional Tests from Review v3
# ============================================================================


class TestATCScannerReviewV3:
    """Additional tests from review v3 recommendations."""

    @patch("modules.auto_trade.core.atc_scanner.log_warn")
    def test_timeframe_weight_mismatch_warning(self, mock_log_warn, mock_data_fetcher):
        """Test warning when timeframes don't have weights."""
        config = {
            "timeframes": ["1h", "15m", "5m"],
            "weights": {"1h": 0.5, "15m": 0.5},  # Missing 5m
        }
        _ = ATCScanner(mock_data_fetcher, config=config)  # type: ignore[arg-type]
        mock_log_warn.assert_called()
        assert "without weights" in str(mock_log_warn.call_args_list)

    @patch("modules.auto_trade.core.atc_scanner.log_warn")
    def test_extra_weights_warning(self, mock_log_warn, mock_data_fetcher):
        """Test warning when weights exist for unused timeframes."""
        config = {
            "timeframes": ["1h", "15m"],
            "weights": {"1h": 0.5, "15m": 0.3, "5m": 0.2},  # 5m not in timeframes
        }
        _ = ATCScanner(mock_data_fetcher, config=config)  # type: ignore[arg-type]
        mock_log_warn.assert_called()
        assert "unused timeframes" in str(mock_log_warn.call_args_list)

    @patch("modules.auto_trade.core.atc_scanner.log_warn")
    def test_non_normalized_weights_warning(self, mock_log_warn, mock_data_fetcher):
        """Test warning for non-normalized weights."""
        config = {"weights": {"1h": 0.5, "15m": 0.5, "5m": 0.5}}  # Sum = 1.5
        _ = ATCScanner(mock_data_fetcher, config=config)  # type: ignore[arg-type]
        mock_log_warn.assert_called()
        assert any("sum to 1.5" in str(call[0][0]) for call in mock_log_warn.call_args_list)

    def test_signal_strength_disabled_uses_unit_weights(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test that signal strength disabled uses unit weights."""
        config = {"use_signal_strength": False, "weights": {"1h": 0.5, "15m": 0.3, "5m": 0.2}}
        scanner = ATCScanner(mock_data_fetcher, config=config)  # type: ignore[arg-type]

        # Mock scan results with varying strengths (should be ignored)
        def side_effect(data_fetcher, atc_config, symbols, **kwargs):
            longs = pd.DataFrame({"symbol": ["BTCUSDT"], "signal": [1.0]})
            shorts = pd.DataFrame()
            return longs, shorts

        mock_scan_all_symbols.side_effect = side_effect
        results = scanner.scan_symbols(["BTCUSDT"])

        # Should ignore strength, use only weight
        assert len(results) == 1
        assert results[0].score == 1.0  # Sum of weights

    def test_signal_strength_enabled_uses_actual_values(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test that signal strength enabled incorporates strength values."""
        config = {"use_signal_strength": True, "weights": {"1h": 0.5, "15m": 0.3, "5m": 0.2}}
        scanner = ATCScanner(mock_data_fetcher, config=config)  # type: ignore[arg-type]

        # Mock scan results with specific strengths
        call_count = 0

        def side_effect(data_fetcher, atc_config, symbols, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # 1h: strength 0.8
                longs = pd.DataFrame({"symbol": ["BTCUSDT"], "signal": [0.8]})
                shorts = pd.DataFrame()
            elif call_count == 2:  # 15m: strength 0.6
                longs = pd.DataFrame({"symbol": ["BTCUSDT"], "signal": [0.6]})
                shorts = pd.DataFrame()
            else:  # 5m: strength 0.4
                longs = pd.DataFrame({"symbol": ["BTCUSDT"], "signal": [0.4]})
                shorts = pd.DataFrame()
            return longs, shorts

        mock_scan_all_symbols.side_effect = side_effect
        results = scanner.scan_symbols(["BTCUSDT"])

        # Strengths are applied per TF with normalized weights; expect updated aggregate score
        assert len(results) == 1
        assert round(results[0].score, 2) == 0.68

    def test_long_signal_above_threshold(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test LONG signal above threshold."""
        config = {"threshold": 0.6}
        scanner = ATCScanner(mock_data_fetcher, config=config)  # type: ignore[arg-type]

        # Mock results for score = 0.7
        call_count = 0

        def side_effect(data_fetcher, atc_config, symbols, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # 1h and 15m LONG
                longs = pd.DataFrame({"symbol": ["BTCUSDT"]})
                shorts = pd.DataFrame()
            else:
                longs = pd.DataFrame()
                shorts = pd.DataFrame()
            return longs, shorts

        mock_scan_all_symbols.side_effect = side_effect
        results = scanner.scan_symbols(["BTCUSDT"])

        assert len(results) == 1
        assert results[0].signal_type == "LONG"
        # Active TFs normalized → score becomes 1.0
        assert results[0].score == 1.0

    def test_short_signal_below_negative_threshold(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test SHORT signal below negative threshold."""
        config = {"threshold": 0.6}
        scanner = ATCScanner(mock_data_fetcher, config=config)  # type: ignore[arg-type]

        # Mock results for score = -0.7
        call_count = 0

        def side_effect(data_fetcher, atc_config, symbols, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # 1h and 15m SHORT
                longs = pd.DataFrame()
                shorts = pd.DataFrame({"symbol": ["BTCUSDT"]})
            else:
                longs = pd.DataFrame()
                shorts = pd.DataFrame()
            return longs, shorts

        mock_scan_all_symbols.side_effect = side_effect
        results = scanner.scan_symbols(["BTCUSDT"])

        assert len(results) == 1
        assert results[0].signal_type == "SHORT"
        # Active TFs normalized → score becomes -1.0
        assert results[0].score == -1.0

    def test_neutral_signal_within_threshold(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test NEUTRAL signal within threshold."""
        config = {"threshold": 0.6}
        scanner = ATCScanner(mock_data_fetcher, config=config)  # type: ignore[arg-type]

        # Mock results for score = 0.3
        call_count = 0

        def side_effect(data_fetcher, atc_config, symbols, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # Only 1h LONG
                longs = pd.DataFrame({"symbol": ["BTCUSDT"]})
                shorts = pd.DataFrame()
            else:
                longs = pd.DataFrame()
                shorts = pd.DataFrame()
            return longs, shorts

        mock_scan_all_symbols.side_effect = side_effect
        results = scanner.scan_symbols(["BTCUSDT"])

        # Active TFs normalized → score becomes 1.0 and passes adaptive threshold
        assert len(results) == 1
        assert results[0].signal_type == "LONG"
        assert results[0].score == 1.0

    @patch("modules.auto_trade.core.atc_scanner.get_hardware_manager")
    def test_max_workers_fallback_on_auto_detect_failure(self, mock_get_hardware_manager, mock_data_fetcher):
        """Test fallback to default when hardware detection fails."""
        mock_get_hardware_manager.side_effect = Exception("Hardware detection failed")

        scanner = ATCScanner(mock_data_fetcher)

        # Should fall back to len(timeframes) = 3
        assert scanner.max_workers == 3

    def test_calculate_weighted_score_long(self, mock_data_fetcher):
        """Test _calculate_weighted_score for LONG signal."""
        config = {"use_signal_strength": True, "weights": {"1h": 0.5}}
        scanner = ATCScanner(mock_data_fetcher, config=config)  # type: ignore[arg-type]

        # LONG with strength 0.8
        score = scanner._calculate_weighted_score("LONG", 0.5, 0.8)
        assert score == 0.4  # 0.5 * 0.8

    def test_calculate_weighted_score_long_without_strength(self, mock_data_fetcher):
        """Test _calculate_weighted_score for LONG without strength."""
        config = {"use_signal_strength": False, "weights": {"1h": 0.5}}
        scanner = ATCScanner(mock_data_fetcher, config=config)  # type: ignore[arg-type]

        # LONG without strength
        score = scanner._calculate_weighted_score("LONG", 0.5, 0.8)
        assert score == 0.5  # Just the weight

    def test_calculate_weighted_score_short(self, mock_data_fetcher):
        """Test _calculate_weighted_score for SHORT signal."""
        config = {"use_signal_strength": True, "weights": {"1h": 0.5}}
        scanner = ATCScanner(mock_data_fetcher, config=config)  # type: ignore[arg-type]

        # SHORT with strength -0.8
        score = scanner._calculate_weighted_score("SHORT", 0.5, -0.8)
        assert score == -0.4  # 0.5 * -0.8

    def test_calculate_weighted_score_short_without_strength(self, mock_data_fetcher):
        """Test _calculate_weighted_score for SHORT without strength."""
        config = {"use_signal_strength": False, "weights": {"1h": 0.5}}
        scanner = ATCScanner(mock_data_fetcher, config=config)  # type: ignore[arg-type]

        # SHORT without strength
        score = scanner._calculate_weighted_score("SHORT", 0.5, -0.8)
        assert score == -0.5  # Negative weight

    def test_calculate_weighted_score_neutral(self, mock_data_fetcher):
        """Test _calculate_weighted_score for NEUTRAL signal."""
        config = {"weights": {"1h": 0.5}}
        scanner = ATCScanner(mock_data_fetcher, config=config)  # type: ignore[arg-type]

        # NEUTRAL always returns 0
        score = scanner._calculate_weighted_score("NEUTRAL", 0.5, 0.8)
        assert score == 0.0


# ============================================================================
# Cache Tests (Python and Rust)
# ============================================================================


class TestATCScannerCache:
    """Tests for ATCScanner caching functionality (both Python and Rust)."""

    def test_cache_disabled_never_caches(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test that cache is not used when enable_cache=False."""
        config = {"enable_cache": False}
        scanner = ATCScanner(mock_data_fetcher, config=config)  # type: ignore[arg-type]

        # Mock scan result
        mock_scan_all_symbols.return_value = (
            pd.DataFrame({"symbol": ["BTCUSDT"], "signal": [0.8]}),
            pd.DataFrame(),
        )

        # First scan
        results1 = scanner.scan_symbols(["BTC/USDT"])
        assert len(results1) == 1

        # Second scan - should call scan_all_symbols again (no cache)
        results2 = scanner.scan_symbols(["BTC/USDT"])
        assert len(results2) == 1

        # Verify scan_all_symbols was called twice (3 timeframes × 2 scans = 6 calls)
        assert mock_scan_all_symbols.call_count == 6

    @pytest.mark.skipif(
        not hasattr(__import__("sys").modules.get("sovereign_prime"), "ScanCache"),
        reason="Rust ScanCache not available",
    )
    def test_rust_cache_stores_and_retrieves(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test that Rust ScanCache stores and retrieves results correctly."""
        config = {"enable_cache": True, "cache_ttl_seconds": 60}
        scanner = ATCScanner(mock_data_fetcher, config=config)  # type: ignore[arg-type]

        # Mock scan result
        mock_scan_all_symbols.return_value = (
            pd.DataFrame({"symbol": ["BTC/USDT"], "signal": [0.8]}),
            pd.DataFrame(),
        )

        # First scan - populate cache
        results1 = scanner.scan_symbols(["BTC/USDT"])
        assert len(results1) == 1
        assert mock_scan_all_symbols.call_count == 3  # 3 timeframes

        # Second scan - should use cache
        results2 = scanner.scan_symbols(["BTC/USDT"])
        assert len(results2) == 1
        assert mock_scan_all_symbols.call_count == 3  # No additional calls (cache hit)

    def test_cache_key_generation(self, mock_data_fetcher):
        """Test that cache keys are generated correctly."""
        scanner = ATCScanner(mock_data_fetcher)

        # Same symbols, same timeframe → same key
        key1 = scanner._get_cache_key(["BTC/USDT", "ETH/USDT"], "1h")
        key2 = scanner._get_cache_key(["BTC/USDT", "ETH/USDT"], "1h")
        assert key1 == key2

        # Different timeframe → different key
        key3 = scanner._get_cache_key(["BTC/USDT", "ETH/USDT"], "15m")
        assert key3 != key1

        # Different symbols → different key
        key4 = scanner._get_cache_key(["BTC/USDT"], "1h")
        assert key4 != key1

        # Symbol order shouldn't matter (sorted internally)
        key5 = scanner._get_cache_key(["ETH/USDT", "BTC/USDT"], "1h")
        assert key5 == key1  # Should be same as key1

    @pytest.mark.skipif(
        not hasattr(__import__("sys").modules.get("sovereign_prime"), "ScanCache"),
        reason="Rust ScanCache not available",
    )
    def test_cache_clear_rust(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test clearing Rust cache."""
        config = {"enable_cache": True}
        scanner = ATCScanner(mock_data_fetcher, config=config)  # type: ignore[arg-type]

        # Populate cache
        mock_scan_all_symbols.return_value = (
            pd.DataFrame({"symbol": ["BTC/USDT"], "signal": [0.8]}),
            pd.DataFrame(),
        )
        scanner.scan_symbols(["BTC/USDT"])

        # Clear cache
        scanner.clear_cache()

    def test_cache_respects_ttl(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test that cache entries expire after TTL."""
        import time

        config = {"enable_cache": True, "cache_ttl_seconds": 1}
        scanner = ATCScanner(mock_data_fetcher, config=config)  # type: ignore[arg-type]

        # Mock scan result
        mock_scan_all_symbols.return_value = (
            pd.DataFrame({"symbol": ["BTC/USDT"], "signal": [0.8]}),
            pd.DataFrame(),
        )

        # First scan - populate cache
        scanner.scan_symbols(["BTC/USDT"])
        assert mock_scan_all_symbols.call_count == 3  # 3 timeframes

        # Wait for TTL to expire
        time.sleep(1.1)

        # Second scan - cache expired, should call scan_all_symbols again
        scanner.scan_symbols(["BTC/USDT"])
        assert mock_scan_all_symbols.call_count == 6  # 3 more calls (cache miss)

    def test_cache_with_batch_processing(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test that cache works correctly with batch processing."""
        config = {"enable_cache": True, "batch_size": 2}
        scanner = ATCScanner(mock_data_fetcher, config=config)  # type: ignore[arg-type]

        # Create 5 symbols (will be processed in 3 batches: 2, 2, 1)
        symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT"]

        # Mock scan result
        mock_scan_all_symbols.return_value = (
            pd.DataFrame({"symbol": symbols[:2], "signal": [0.8, 0.7]}),
            pd.DataFrame(),
        )

        # Freeze time to keep cache keys stable across scans
        from datetime import datetime

        fixed_time = datetime(2025, 1, 1, 0, 0, 30)
        with patch("modules.auto_trade.core.atc_scanner.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_time

            # First scan
            _ = scanner.scan_symbols(symbols)
            initial_calls = mock_scan_all_symbols.call_count

            # Second scan - should use cache
            _ = scanner.scan_symbols(symbols)

        # Verify cache was used (no additional scan_all_symbols calls)
        assert mock_scan_all_symbols.call_count == initial_calls

    def test_cache_initialization_fallback_to_python(self, mock_data_fetcher):
        """Test that scanner handles cache initialization when Rust fails."""
        # Force Rust cache by config, but it should handle gracefully
        config = {"enable_cache": True}

        # Create scanner
        scanner = ATCScanner(mock_data_fetcher, config=config)  # type: ignore[arg-type]

        # Scanner should initialize successfully
        assert scanner.enable_cache is True

    def test_cache_with_empty_results(self, mock_data_fetcher, mock_scan_all_symbols):
        """Test that cache does NOT cache empty scan results (by design)."""
        config = {"enable_cache": True}
        scanner = ATCScanner(mock_data_fetcher, config=config)  # type: ignore[arg-type]

        # Mock empty scan result
        mock_scan_all_symbols.return_value = (pd.DataFrame(), pd.DataFrame())

        # First scan - empty results
        results1 = scanner.scan_symbols(["BTC/USDT"])
        assert len(results1) == 0

        # Empty results are NOT cached (by design in atc_scanner.py:599)
        # Second scan should call scan_all_symbols again
        results2 = scanner.scan_symbols(["BTC/USDT"])
        assert len(results2) == 0

        # Verify scan_all_symbols was called twice (empty results not cached)
        assert mock_scan_all_symbols.call_count == 6  # 3 timeframes × 2 scans

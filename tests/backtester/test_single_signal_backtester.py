"""
Tests for single signal (highest confidence) backtester.

This module tests the new single signal mode where signals are selected
based on highest confidence rather than majority vote.
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from modules.backtester import FullBacktester
from modules.position_sizing.core.hybrid_signal_calculator import HybridSignalCalculator
from modules.position_sizing.core.position_sizer import PositionSizer


@pytest.fixture
def mock_data_fetcher():
    """Create a mock DataFetcher."""
    fetcher = Mock()
    return fetcher


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    dates = pd.date_range("2024-01-01", periods=100, freq="1h")
    return pd.DataFrame(
        {
            "open": np.random.uniform(40000, 50000, 100),
            "high": np.random.uniform(40000, 51000, 100),
            "low": np.random.uniform(39000, 50000, 100),
            "close": np.random.uniform(40000, 50000, 100),
            "volume": np.random.uniform(1000, 10000, 100),
        },
        index=dates,
    )


@pytest.fixture
def hybrid_calculator(mock_data_fetcher):
    """Create a HybridSignalCalculator instance."""
    return HybridSignalCalculator(
        data_fetcher=mock_data_fetcher,
        enabled_indicators=["range_oscillator", "spc", "xgboost"],
        use_confidence_weighting=True,
        min_indicators_agreement=3,
    )


class TestCalculateSingleSignalHighestConfidence:
    """Test calculate_single_signal_highest_confidence method."""

    def test_no_signals_returns_zero(self, hybrid_calculator, sample_df):
        """Test that no signals returns (0, 0.0)."""
        with (
            patch.object(hybrid_calculator, "_calculate_indicators_sequential", return_value=[]) as mock_seq,
            patch.object(hybrid_calculator, "_calculate_indicators_parallel", return_value=[]) as mock_par,
        ):
            result = hybrid_calculator.calculate_single_signal_highest_confidence(
                df=sample_df,
                symbol="BTC/USDT",
                timeframe="1h",
                period_index=50,
            )
            assert result == (0, 0.0)
            # Verify which implementation was invoked (parallel is default when ENABLE_MULTITHREADING=True)
            assert mock_par.called is True
            assert mock_seq.called is False

    def test_only_neutral_signals_returns_zero(self, hybrid_calculator, sample_df):
        """Test that only neutral signals (0) returns (0, 0.0)."""
        indicator_signals = [
            {"indicator": "range_oscillator", "signal": 0, "confidence": 0.5},
            {"indicator": "spc", "signal": 0, "confidence": 0.6},
        ]
        with (
            patch.object(
                hybrid_calculator, "_calculate_indicators_sequential", return_value=indicator_signals
            ) as mock_seq,
            patch.object(
                hybrid_calculator, "_calculate_indicators_parallel", return_value=indicator_signals
            ) as mock_par,
        ):
            result = hybrid_calculator.calculate_single_signal_highest_confidence(
                df=sample_df,
                symbol="BTC/USDT",
                timeframe="1h",
                period_index=50,
            )
            assert result == (0, 0.0)
            # Verify which implementation was invoked (parallel is default when ENABLE_MULTITHREADING=True)
            assert mock_par.called is True
            assert mock_seq.called is False

    def test_selects_highest_confidence_long(self, hybrid_calculator, sample_df):
        """Test that method selects signal with highest confidence (LONG)."""
        indicator_signals = [
            {"indicator": "range_oscillator", "signal": 1, "confidence": 0.5},
            {"indicator": "spc", "signal": 1, "confidence": 0.8},  # Highest
            {"indicator": "xgboost", "signal": 1, "confidence": 0.6},
        ]
        with (
            patch.object(
                hybrid_calculator, "_calculate_indicators_sequential", return_value=indicator_signals
            ) as mock_seq,
            patch.object(
                hybrid_calculator, "_calculate_indicators_parallel", return_value=indicator_signals
            ) as mock_par,
        ):
            result = hybrid_calculator.calculate_single_signal_highest_confidence(
                df=sample_df,
                symbol="BTC/USDT",
                timeframe="1h",
                period_index=50,
            )
            assert result == (1, 0.8)
            # Verify which implementation was invoked (parallel is default when ENABLE_MULTITHREADING=True)
            assert mock_par.called is True
            assert mock_seq.called is False

    def test_selects_highest_confidence_short(self, hybrid_calculator, sample_df):
        """Test that method selects signal with highest confidence (SHORT)."""
        indicator_signals = [
            {"indicator": "range_oscillator", "signal": -1, "confidence": 0.5},
            {"indicator": "spc", "signal": -1, "confidence": 0.9},  # Highest
            {"indicator": "xgboost", "signal": -1, "confidence": 0.6},
        ]
        with (
            patch.object(
                hybrid_calculator, "_calculate_indicators_sequential", return_value=indicator_signals
            ) as mock_seq,
            patch.object(
                hybrid_calculator, "_calculate_indicators_parallel", return_value=indicator_signals
            ) as mock_par,
        ):
            result = hybrid_calculator.calculate_single_signal_highest_confidence(
                df=sample_df,
                symbol="BTC/USDT",
                timeframe="1h",
                period_index=50,
            )
            assert result == (-1, 0.9)
            # Verify which implementation was invoked (parallel is default when ENABLE_MULTITHREADING=True)
            assert mock_par.called is True
            assert mock_seq.called is False

    def test_tie_confidence_prefers_long(self, hybrid_calculator, sample_df):
        """Test that when confidence is tied, LONG is preferred over SHORT."""
        indicator_signals = [
            {"indicator": "range_oscillator", "signal": -1, "confidence": 0.7},
            {"indicator": "spc", "signal": 1, "confidence": 0.7},  # Same confidence, but LONG
        ]
        with (
            patch.object(
                hybrid_calculator, "_calculate_indicators_sequential", return_value=indicator_signals
            ) as mock_seq,
            patch.object(
                hybrid_calculator, "_calculate_indicators_parallel", return_value=indicator_signals
            ) as mock_par,
        ):
            result = hybrid_calculator.calculate_single_signal_highest_confidence(
                df=sample_df,
                symbol="BTC/USDT",
                timeframe="1h",
                period_index=50,
            )
            assert result == (1, 0.7)  # LONG preferred
            # Verify which implementation was invoked (parallel is default when ENABLE_MULTITHREADING=True)
            assert mock_par.called is True
            assert mock_seq.called is False

    def test_mixed_signals_selects_highest(self, hybrid_calculator, sample_df):
        """Test that method selects highest confidence from mixed signals."""
        indicator_signals = [
            {"indicator": "range_oscillator", "signal": 1, "confidence": 0.5},
            {"indicator": "spc", "signal": -1, "confidence": 0.9},  # Highest
            {"indicator": "xgboost", "signal": 1, "confidence": 0.6},
        ]
        with (
            patch.object(
                hybrid_calculator, "_calculate_indicators_sequential", return_value=indicator_signals
            ) as mock_seq,
            patch.object(
                hybrid_calculator, "_calculate_indicators_parallel", return_value=indicator_signals
            ) as mock_par,
        ):
            result = hybrid_calculator.calculate_single_signal_highest_confidence(
                df=sample_df,
                symbol="BTC/USDT",
                timeframe="1h",
                period_index=50,
            )
            assert result == (-1, 0.9)  # SHORT has highest confidence
            # Verify which implementation was invoked (parallel is default when ENABLE_MULTITHREADING=True)
            assert mock_par.called is True
            assert mock_seq.called is False


class TestCalculateSingleSignalFromPrecomputed:
    """Test calculate_single_signal_from_precomputed method."""

    def test_no_signals_returns_zero(self, hybrid_calculator):
        """Test that no signals returns (0, 0.0)."""
        precomputed_indicators = {}
        result = hybrid_calculator.calculate_single_signal_from_precomputed(
            precomputed_indicators=precomputed_indicators,
            period_index=50,
        )
        assert result == (0, 0.0)

    def test_selects_highest_confidence_from_precomputed(self, hybrid_calculator, sample_df):
        """Test that method selects highest confidence from precomputed indicators."""
        precomputed_indicators = {
            "range_oscillator": pd.DataFrame(
                {
                    "signal": [0] * 50 + [1] + [0] * 49,
                    "confidence": [0.0] * 50 + [0.5] + [0.0] * 49,
                },
                index=sample_df.index,
            ),
            "spc": pd.DataFrame(
                {
                    "signal": [0] * 50 + [1] + [0] * 49,
                    "confidence": [0.0] * 50 + [0.8] + [0.0] * 49,  # Highest
                },
                index=sample_df.index,
            ),
            "xgboost": pd.DataFrame(
                {
                    "signal": [0] * 50 + [1] + [0] * 49,
                    "confidence": [0.0] * 50 + [0.6] + [0.0] * 49,
                },
                index=sample_df.index,
            ),
        }
        result = hybrid_calculator.calculate_single_signal_from_precomputed(
            precomputed_indicators=precomputed_indicators,
            period_index=50,
        )
        assert result == (1, 0.8)


class TestFullBacktesterSingleSignalMode:
    """Test FullBacktester with single_signal mode."""

    def test_backtester_initializes_with_single_signal_mode(self, mock_data_fetcher):
        """Test that FullBacktester can be initialized with single_signal mode."""
        backtester = FullBacktester(
            data_fetcher=mock_data_fetcher,
            signal_mode="single_signal",
        )
        assert backtester.signal_mode == "single_signal"

    def test_backtester_defaults_to_majority_vote(self, mock_data_fetcher):
        """Test that FullBacktester defaults to majority_vote mode."""
        backtester = FullBacktester(data_fetcher=mock_data_fetcher)
        assert backtester.signal_mode == "majority_vote"

    def test_backtester_raises_error_invalid_mode(self, mock_data_fetcher):
        """Test that FullBacktester raises error for invalid signal_mode."""
        with pytest.raises(ValueError, match="Invalid signal_mode"):
            FullBacktester(
                data_fetcher=mock_data_fetcher,
                signal_mode="invalid_mode",
            )

    @patch("modules.backtester.core.backtester.calculate_single_signals")
    def test_backtest_uses_single_signal_calculator(self, mock_calc, mock_data_fetcher, sample_df):
        """Test that backtest() uses calculate_single_signals when signal_mode is single_signal."""
        mock_calc.return_value = pd.Series([0] * len(sample_df), index=sample_df.index)

        backtester = FullBacktester(
            data_fetcher=mock_data_fetcher,
            signal_mode="single_signal",
        )

        # Mock the data fetcher
        mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (sample_df, None)

        backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=100,
            signal_type="LONG",
            df=sample_df,
        )

        # Verify that calculate_single_signals was called (not calculate_signals)
        assert mock_calc.called


class TestPositionSizingWithCumulativePerformance:
    """Test PositionSizer with cumulative performance adjustment."""

    @patch("modules.position_sizing.core.position_sizer.FullBacktester")
    def test_position_sizing_uses_cumulative_performance(self, mock_backtester_class, mock_data_fetcher):
        """Test that position sizing uses cumulative performance from equity curve."""
        # Setup mock backtester
        mock_backtester = Mock()
        mock_backtester_class.return_value = mock_backtester

        # Mock backtest result with equity curve
        initial_capital = 10000.0
        final_equity = 11000.0  # 10% gain
        equity_curve = pd.Series([initial_capital] * 50 + [final_equity] * 50)

        mock_backtest_result = {
            "metrics": {
                "win_rate": 0.6,
                "total_return": 0.1,
                "sharpe_ratio": 1.5,
            },
            "equity_curve": equity_curve,
        }
        mock_backtester.backtest.return_value = mock_backtest_result

        # Create PositionSizer
        position_sizer = PositionSizer(data_fetcher=mock_data_fetcher)

        # Mock regime detector
        with patch.object(position_sizer.regime_detector, "detect_regime", return_value="trending"):
            # Mock data fetcher
            mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (
                pd.DataFrame(
                    {
                        "open": [40000] * 100,
                        "high": [41000] * 100,
                        "low": [39000] * 100,
                        "close": [40000] * 100,
                        "volume": [1000] * 100,
                    }
                ),
                None,
            )

            result = position_sizer.calculate_position_size(
                symbol="BTC/USDT",
                account_balance=10000.0,
                signal_type="LONG",
            )

            # Verify that cumulative_performance_multiplier is in result
            assert "cumulative_performance_multiplier" in result
            # With 10% gain, multiplier should be > 1.0
            assert result["cumulative_performance_multiplier"] > 1.0

    @patch("modules.position_sizing.core.position_sizer.FullBacktester")
    def test_position_sizing_handles_negative_performance(self, mock_backtester_class, mock_data_fetcher):
        """Test that position sizing handles negative cumulative performance."""
        # Setup mock backtester
        mock_backtester = Mock()
        mock_backtester_class.return_value = mock_backtester

        # Mock backtest result with negative equity curve
        initial_capital = 10000.0
        final_equity = 9000.0  # 10% loss
        equity_curve = pd.Series([initial_capital] * 50 + [final_equity] * 50)

        mock_backtest_result = {
            "metrics": {
                "win_rate": 0.4,
                "total_return": -0.1,
                "sharpe_ratio": -0.5,
            },
            "equity_curve": equity_curve,
        }
        mock_backtester.backtest.return_value = mock_backtest_result

        # Create PositionSizer
        position_sizer = PositionSizer(data_fetcher=mock_data_fetcher)

        # Mock regime detector
        with patch.object(position_sizer.regime_detector, "detect_regime", return_value="trending"):
            # Mock data fetcher
            mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (
                pd.DataFrame(
                    {
                        "open": [40000] * 100,
                        "high": [41000] * 100,
                        "low": [39000] * 100,
                        "close": [40000] * 100,
                        "volume": [1000] * 100,
                    }
                ),
                None,
            )

            result = position_sizer.calculate_position_size(
                symbol="BTC/USDT",
                account_balance=10000.0,
                signal_type="LONG",
            )

            # Verify that cumulative_performance_multiplier is in result
            assert "cumulative_performance_multiplier" in result
            # With 10% loss, multiplier should be < 1.0
            assert result["cumulative_performance_multiplier"] < 1.0

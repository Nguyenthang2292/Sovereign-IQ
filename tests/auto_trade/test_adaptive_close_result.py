"""
tests/auto_trade/test_adaptive_close_result.py
==============================================
Tests for AdaptiveCloseResult dataclass and compute_adaptive_deadline_with_meta().
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from modules.auto_trade.execution.adaptive_close_calculator import (
    AdaptiveCloseCalculator,
    AdaptiveCloseResult,
    DEFAULT_FALLBACK_DURATION_HOURS,
)


class MockSettingsManager:
    """Mock settings manager for testing."""

    def __init__(self, settings=None):
        self.settings = settings or {}

    def get(self, key, default=None):
        keys = key.split(".")
        value = self.settings
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value


class TestAdaptiveCloseResult:
    """Tests for AdaptiveCloseResult dataclass."""

    def test_adaptive_result_creation(self):
        """Test creating AdaptiveCloseResult with all fields."""
        deadline = datetime.now(timezone.utc)
        result = AdaptiveCloseResult(
            deadline_utc=deadline,
            source="adaptive",
            duration_hours=5.5,
            pelt_hours=4.2,
            hmm_hours=6.1,
        )

        assert result.deadline_utc == deadline
        assert result.source == "adaptive"
        assert result.duration_hours == 5.5
        assert result.pelt_hours == 4.2
        assert result.hmm_hours == 6.1

    def test_adaptive_result_with_none_fields(self):
        """Test creating AdaptiveCloseResult with optional fields as None."""
        deadline = datetime.now(timezone.utc)
        result = AdaptiveCloseResult(
            deadline_utc=deadline,
            source="static",
            duration_hours=4.0,
            pelt_hours=None,
            hmm_hours=None,
        )

        assert result.deadline_utc == deadline
        assert result.source == "static"
        assert result.duration_hours == 4.0
        assert result.pelt_hours is None
        assert result.hmm_hours is None


class TestComputeAdaptiveDeadlineWithMeta:
    """Tests for compute_adaptive_deadline_with_meta() method."""

    @staticmethod
    def _make_ohlcv_df(periods: int = 120) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "open": [50000] * periods,
                "high": [50100] * periods,
                "low": [49900] * periods,
                "close": [50100] * periods,
                "volume": [1000] * periods,
            },
            index=pd.date_range(end=datetime.now(), periods=periods, freq="15min", tz="UTC"),
        )

    def test_disabled_returns_static_source(self):
        """Test that disabled adaptive close returns source='static'."""
        settings = MockSettingsManager({
            "auto_close": {
                "adaptive": {"enabled": False},
                "max_duration_hours": 4.0,
            }
        })
        calculator = AdaptiveCloseCalculator(settings)

        opened_at = datetime.now(timezone.utc)
        result = calculator.compute_adaptive_deadline_with_meta(
            symbol="BTC/USDT",
            opened_at=opened_at,
        )

        assert isinstance(result, AdaptiveCloseResult)
        assert result.source == "static"
        assert result.deadline_utc is not None
        assert result.duration_hours == 4.0
        assert result.pelt_hours is None
        assert result.hmm_hours is None

    @patch("modules.auto_trade.execution.adaptive_close_calculator.RegimeDurationAnalyzer")
    def test_adaptive_source_when_analysis_valid(self, mock_analyzer_class):
        """Test source='adaptive' when analysis is valid."""
        mock_analyzer = MagicMock()
        mock_result = MagicMock()
        mock_result.is_valid = True
        mock_result.recommended_duration_hours = 5.5
        mock_result.pelt_avg_duration_hours = 4.2
        mock_result.hmm_next_state_duration_hours = 6.1
        mock_result.error = None
        mock_analyzer.analyze.return_value = mock_result
        mock_analyzer_class.return_value = mock_analyzer

        settings = MockSettingsManager({
            "auto_close": {
                "adaptive": {
                    "enabled": True,
                    "min_duration_hours": 1.0,
                    "max_duration_hours": 12.0,
                    "lookback_days": 60,
                    "timeframe": "15m",
                },
                "max_duration_hours": 4.0,
            }
        })
        calculator = AdaptiveCloseCalculator(settings)

        df = self._make_ohlcv_df(periods=120)
        opened_at = datetime.now(timezone.utc)
        result = calculator.compute_adaptive_deadline_with_meta(
            symbol="BTC/USDT",
            opened_at=opened_at,
            ohlcv_df=df,
        )

        assert isinstance(result, AdaptiveCloseResult)
        assert result.source == "adaptive"
        assert result.duration_hours == 5.5
        assert result.pelt_hours == 4.2
        assert result.hmm_hours == 6.1
        assert result.deadline_utc is not None

    @patch("modules.auto_trade.execution.adaptive_close_calculator.RegimeDurationAnalyzer")
    def test_adaptive_fallback_source_when_analysis_invalid(self, mock_analyzer_class):
        """Test source='adaptive_fallback' when analysis is invalid."""
        mock_analyzer = MagicMock()
        mock_result = MagicMock()
        mock_result.is_valid = False
        mock_result.recommended_duration_hours = None
        mock_result.error = "Analysis failed"
        mock_result.pelt_avg_duration_hours = 3.5
        mock_result.hmm_next_state_duration_hours = None
        mock_analyzer.analyze.return_value = mock_result
        mock_analyzer_class.return_value = mock_analyzer

        settings = MockSettingsManager({
            "auto_close": {
                "adaptive": {
                    "enabled": True,
                    "min_duration_hours": 1.0,
                    "max_duration_hours": 12.0,
                },
                "max_duration_hours": 4.0,
            }
        })
        calculator = AdaptiveCloseCalculator(settings)

        df = self._make_ohlcv_df(periods=120)
        opened_at = datetime.now(timezone.utc)
        result = calculator.compute_adaptive_deadline_with_meta(
            symbol="BTC/USDT",
            opened_at=opened_at,
            ohlcv_df=df,
        )

        assert isinstance(result, AdaptiveCloseResult)
        assert result.source == "adaptive_fallback"
        assert result.duration_hours == 4.0  # fallback duration
        assert result.pelt_hours == 3.5
        assert result.hmm_hours is None

    @patch("modules.auto_trade.execution.adaptive_close_calculator.RegimeDurationAnalyzer")
    def test_pelt_and_hmm_hours_assigned_correctly(self, mock_analyzer_class):
        """Test that pelt_hours and hmm_hours are correctly extracted from analysis."""
        mock_analyzer = MagicMock()
        mock_result = MagicMock()
        mock_result.is_valid = True
        mock_result.recommended_duration_hours = 6.0
        mock_result.pelt_avg_duration_hours = 5.0
        mock_result.hmm_next_state_duration_hours = 7.0
        mock_result.error = None
        mock_analyzer.analyze.return_value = mock_result
        mock_analyzer_class.return_value = mock_analyzer

        settings = MockSettingsManager({
            "auto_close": {
                "adaptive": {
                    "enabled": True,
                    "min_duration_hours": 1.0,
                    "max_duration_hours": 12.0,
                },
                "max_duration_hours": 4.0,
            }
        })
        calculator = AdaptiveCloseCalculator(settings)

        df = self._make_ohlcv_df(periods=120)
        opened_at = datetime.now(timezone.utc)
        result = calculator.compute_adaptive_deadline_with_meta(
            symbol="BTC/USDT",
            opened_at=opened_at,
            ohlcv_df=df,
        )

        assert result.pelt_hours == 5.0
        assert result.hmm_hours == 7.0

    def test_adaptive_fallback_source_with_insufficient_data(self):
        """Test source='adaptive_fallback' when OHLCV data is insufficient."""
        settings = MockSettingsManager({
            "auto_close": {
                "adaptive": {
                    "enabled": True,
                    "min_duration_hours": 1.0,
                    "max_duration_hours": 12.0,
                },
                "max_duration_hours": 4.0,
            }
        })
        calculator = AdaptiveCloseCalculator(settings)

        # Only 50 candles (below 100 threshold)
        df = pd.DataFrame({
            "open": [50000] * 50,
            "high": [50100] * 50,
            "low": [49900] * 50,
            "close": [50100] * 50,
            "volume": [1000] * 50,
        }, index=pd.date_range(end=datetime.now(), periods=50, freq="15min", tz="UTC"))

        opened_at = datetime.now(timezone.utc)
        result = calculator.compute_adaptive_deadline_with_meta(
            symbol="BTC/USDT",
            opened_at=opened_at,
            ohlcv_df=df,
        )

        assert isinstance(result, AdaptiveCloseResult)
        assert result.source == "adaptive_fallback"
        assert result.duration_hours == 4.0
        assert result.pelt_hours is None
        assert result.hmm_hours is None

    @patch("modules.auto_trade.execution.adaptive_close_calculator.RegimeDurationAnalyzer")
    def test_adaptive_fallback_on_exception(self, mock_analyzer_class):
        """Test source='adaptive_fallback' when exception occurs."""
        mock_analyzer_class.side_effect = Exception("Analyzer error")

        settings = MockSettingsManager({
            "auto_close": {
                "adaptive": {
                    "enabled": True,
                    "min_duration_hours": 1.0,
                    "max_duration_hours": 12.0,
                },
                "max_duration_hours": 4.0,
            }
        })
        calculator = AdaptiveCloseCalculator(settings)

        df = self._make_ohlcv_df(periods=120)
        opened_at = datetime.now(timezone.utc)
        result = calculator.compute_adaptive_deadline_with_meta(
            symbol="BTC/USDT",
            opened_at=opened_at,
            ohlcv_df=df,
        )

        assert isinstance(result, AdaptiveCloseResult)
        assert result.source == "adaptive_fallback"
        assert result.duration_hours == 4.0
        assert result.pelt_hours is None
        assert result.hmm_hours is None

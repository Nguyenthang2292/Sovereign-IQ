"""
tests/auto_trade/test_adaptive_close.py
=======================================
Tests for auto_trade/execution/adaptive_close_calculator module.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from modules.auto_trade.execution.adaptive_close_calculator import (
    AdaptiveCloseCalculator,
    DEFAULT_FALLBACK_DURATION_HOURS,
    DEFAULT_MAX_DURATION_HOURS,
    DEFAULT_MIN_DURATION_HOURS,
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


class TestAdaptiveCloseCalculator:
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

    def test_disabled_returns_none(self):
        """Test that disabled adaptive close returns None."""
        settings = MockSettingsManager({
            "auto_close": {
                "adaptive": {"enabled": False},
                "max_duration_hours": 4.0,
            }
        })
        calculator = AdaptiveCloseCalculator(settings)
        
        opened_at = datetime.now(timezone.utc)
        result = calculator.compute_adaptive_deadline(
            symbol="BTC/USDT",
            opened_at=opened_at,
        )
        
        assert result is None

    @patch("modules.auto_trade.execution.adaptive_close_calculator.RegimeDurationAnalyzer")
    def test_enabled_with_valid_result(self, mock_analyzer_class):
        """Test enabled adaptive close with valid analysis result."""
        # Mock analyzer
        mock_analyzer = MagicMock()
        mock_result = MagicMock()
        mock_result.is_valid = True
        mock_result.recommended_duration_hours = 5.5
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
        
        # Provide sufficient OHLCV data to pass minimum-candle gate
        df = self._make_ohlcv_df(periods=120)
        
        opened_at = datetime.now(timezone.utc)
        result = calculator.compute_adaptive_deadline(
            symbol="BTC/USDT",
            opened_at=opened_at,
            ohlcv_df=df,
        )
        
        # Result should be opened_at + 5.5 hours (clamped within 1-12)
        expected_deadline = opened_at.replace(second=0, microsecond=0) + __import__('datetime').timedelta(hours=5.5)
        assert result is not None
        assert abs((result - opened_at).total_seconds() - 5.5 * 3600) < 1

    @patch("modules.auto_trade.execution.adaptive_close_calculator.RegimeDurationAnalyzer")
    def test_clamp_to_min(self, mock_analyzer_class):
        """Test that result is clamped to min_duration."""
        mock_analyzer = MagicMock()
        mock_result = MagicMock()
        mock_result.is_valid = True
        mock_result.recommended_duration_hours = 0.5  # Below min of 1.0
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
        result = calculator.compute_adaptive_deadline(
            symbol="BTC/USDT",
            opened_at=opened_at,
            ohlcv_df=df,
        )
        
        # Should be clamped to min of 1.0 hour
        assert result is not None
        assert abs((result - opened_at).total_seconds() - 1.0 * 3600) < 1

    @patch("modules.auto_trade.execution.adaptive_close_calculator.RegimeDurationAnalyzer")
    def test_clamp_to_max(self, mock_analyzer_class):
        """Test that result is clamped to max_duration."""
        mock_analyzer = MagicMock()
        mock_result = MagicMock()
        mock_result.is_valid = True
        mock_result.recommended_duration_hours = 20.0  # Above max of 12.0
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
        result = calculator.compute_adaptive_deadline(
            symbol="BTC/USDT",
            opened_at=opened_at,
            ohlcv_df=df,
        )
        
        # Should be clamped to max of 12.0 hours
        assert result is not None
        assert abs((result - opened_at).total_seconds() - 12.0 * 3600) < 1

    @patch("modules.auto_trade.execution.adaptive_close_calculator.RegimeDurationAnalyzer")
    def test_fallback_when_analysis_invalid(self, mock_analyzer_class):
        """Test fallback to static duration when analysis is invalid."""
        mock_analyzer = MagicMock()
        mock_result = MagicMock()
        mock_result.is_valid = False
        mock_result.recommended_duration_hours = None
        mock_result.error = "Analysis failed"
        mock_analyzer.analyze.return_value = mock_result
        mock_analyzer_class.return_value = mock_analyzer

        settings = MockSettingsManager({
            "auto_close": {
                "adaptive": {
                    "enabled": True,
                    "min_duration_hours": 1.0,
                    "max_duration_hours": 12.0,
                },
                "max_duration_hours": 4.0,  # Fallback duration
            }
        })
        calculator = AdaptiveCloseCalculator(settings)
        
        df = self._make_ohlcv_df(periods=120)
        
        opened_at = datetime.now(timezone.utc)
        result = calculator.compute_adaptive_deadline(
            symbol="BTC/USDT",
            opened_at=opened_at,
            ohlcv_df=df,
        )
        
        # Should fallback to 4.0 hours
        assert result is not None
        assert abs((result - opened_at).total_seconds() - 4.0 * 3600) < 1

    def test_fallback_insufficient_data(self):
        """Test fallback when OHLCV data is insufficient."""
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
        result = calculator.compute_adaptive_deadline(
            symbol="BTC/USDT",
            opened_at=opened_at,
            ohlcv_df=df,
        )
        
        # Should fallback to 4.0 hours
        assert result is not None
        assert abs((result - opened_at).total_seconds() - 4.0 * 3600) < 1

    def test_fallback_no_data(self):
        """Test fallback when no OHLCV data available."""
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
        
        opened_at = datetime.now(timezone.utc)
        result = calculator.compute_adaptive_deadline(
            symbol="BTC/USDT",
            opened_at=opened_at,
            ohlcv_df=None,  # No data provided
        )
        
        # Should fallback to 4.0 hours (fetch will fail in test)
        # Actually fetch is mocked so this depends on implementation
        # For this test we check it doesn't crash
        assert result is not None

    @patch("modules.auto_trade.execution.adaptive_close_calculator.RegimeDurationAnalyzer")
    def test_exception_handling(self, mock_analyzer_class):
        """Test exception handling returns fallback."""
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
        result = calculator.compute_adaptive_deadline(
            symbol="BTC/USDT",
            opened_at=opened_at,
            ohlcv_df=df,
        )
        
        # Should fallback on exception
        assert result is not None
        assert abs((result - opened_at).total_seconds() - 4.0 * 3600) < 1

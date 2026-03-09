"""
tests/detect_regime_change/test_analyzer.py
===========================================
Tests for detect_regime_change.regime_duration_analyzer module.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from modules.detect_regime_change.models import RegimeDurationResult
from modules.detect_regime_change.regime_duration_analyzer import RegimeDurationAnalyzer


class TestRegimeDurationAnalyzer:
    def create_ohlcv_df(
        self,
        interval_minutes: int = 15,
        days: int = 60,
        base_price: float = 50000,
    ) -> pd.DataFrame:
        """Create a sample OHLCV DataFrame."""
        periods = int(days * 24 * 60 / interval_minutes)
        index = pd.date_range(
            end=datetime.now(),
            periods=periods,
            freq=f"{interval_minutes}min",
            tz="UTC",
        )
        
        np.random.seed(42)
        
        data = []
        current_price = base_price
        for i in range(periods):
            change = np.random.randn() * (base_price * 0.002)
            current_price += change
            data.append({
                "open": current_price - abs(np.random.randn() * 10),
                "high": current_price + abs(np.random.randn() * 20),
                "low": current_price - abs(np.random.randn() * 20),
                "close": current_price,
                "volume": np.random.randint(1000, 10000),
            })
        
        df = pd.DataFrame(data, index=index)
        return df

    @patch("modules.detect_regime_change.regime_duration_analyzer.detect_change_points_pelt")
    @patch("modules.detect_regime_change.regime_duration_analyzer.estimate_hmm_regime_duration")
    def test_analyze_with_both_pelt_and_hmm_high_conf(
        self, mock_hmm, mock_pelt
    ):
        """Test analyzer with both PELT and HMM results (high confidence)."""
        # Mock PELT results
        from modules.detect_regime_change.models import ChangePoint, RegimeSegment
        mock_pelt.return_value = (
            [ChangePoint(index=100, timestamp=None)],
            [
                RegimeSegment(
                    start_index=0, end_index=100,
                    duration_seconds=3600*3, duration_hours=3.0,
                    mean_return=0.01, volatility=0.02,
                ),
                RegimeSegment(
                    start_index=100, end_index=200,
                    duration_seconds=3600*4, duration_hours=4.0,
                    mean_return=0.02, volatility=0.03,
                ),
            ],
        )
        
        # Mock HMM results with high confidence
        mock_hmm.return_value = (5.0, 1, 0.85)  # duration=5h, state=BULLISH, prob=0.85
        
        analyzer = RegimeDurationAnalyzer()
        df = self.create_ohlcv_df()
        
        result = analyzer.analyze(df, symbol="BTC/USDT", timeframe="15m")
        
        assert result.symbol == "BTC/USDT"
        assert result.timeframe == "15m"
        assert result.pelt_avg_duration_hours == 3.5  # (3+4)/2
        assert result.hmm_next_state_duration_hours == 5.0
        assert result.hmm_state == 1
        assert result.hmm_state_probability == 0.85
        
        # High confidence: 0.4 * 3.5 + 0.6 * 5.0 = 1.4 + 3.0 = 4.4
        assert result.recommended_duration_hours == pytest.approx(4.4)
        assert result.is_valid is True

    @patch("modules.detect_regime_change.regime_duration_analyzer.detect_change_points_pelt")
    @patch("modules.detect_regime_change.regime_duration_analyzer.estimate_hmm_regime_duration")
    def test_analyze_with_both_pelt_and_hmm_low_conf(
        self, mock_hmm, mock_pelt
    ):
        """Test analyzer with both PELT and HMM results (low confidence)."""
        from modules.detect_regime_change.models import ChangePoint, RegimeSegment
        mock_pelt.return_value = (
            [ChangePoint(index=100, timestamp=None)],
            [
                RegimeSegment(
                    start_index=0, end_index=100,
                    duration_seconds=3600*3, duration_hours=3.0,
                    mean_return=0.01, volatility=0.02,
                ),
            ],
        )
        
        # Mock HMM results with low confidence
        mock_hmm.return_value = (5.0, 1, 0.5)  # duration=5h, prob=0.5 (< 0.7 threshold)
        
        analyzer = RegimeDurationAnalyzer()
        df = self.create_ohlcv_df()
        
        result = analyzer.analyze(df, symbol="BTC/USDT", timeframe="15m")
        
        # Low confidence: 0.7 * 3.0 + 0.3 * 5.0 = 2.1 + 1.5 = 3.6
        assert result.recommended_duration_hours == pytest.approx(3.6)
        assert result.is_valid is True

    @patch("modules.detect_regime_change.regime_duration_analyzer.detect_change_points_pelt")
    @patch("modules.detect_regime_change.regime_duration_analyzer.estimate_hmm_regime_duration")
    def test_analyze_only_pelt(self, mock_hmm, mock_pelt):
        """Test analyzer when only PELT succeeds."""
        from modules.detect_regime_change.models import ChangePoint, RegimeSegment
        mock_pelt.return_value = (
            [ChangePoint(index=100, timestamp=None)],
            [
                RegimeSegment(
                    start_index=0, end_index=100,
                    duration_seconds=3600*4, duration_hours=4.0,
                    mean_return=0.01, volatility=0.02,
                ),
            ],
        )
        
        # HMM fails
        mock_hmm.return_value = (None, None, None)
        
        analyzer = RegimeDurationAnalyzer()
        df = self.create_ohlcv_df()
        
        result = analyzer.analyze(df, symbol="BTC/USDT", timeframe="15m")
        
        # Only PELT: use pelt_avg directly
        assert result.recommended_duration_hours == 4.0
        assert result.is_valid is True

    @patch("modules.detect_regime_change.regime_duration_analyzer.detect_change_points_pelt")
    @patch("modules.detect_regime_change.regime_duration_analyzer.estimate_hmm_regime_duration")
    def test_analyze_only_hmm(self, mock_hmm, mock_pelt):
        """Test analyzer when only HMM succeeds."""
        # PELT fails (no segments)
        mock_pelt.return_value = ([], [])
        
        # HMM succeeds
        mock_hmm.return_value = (3.5, 0, 0.8)
        
        analyzer = RegimeDurationAnalyzer()
        df = self.create_ohlcv_df()
        
        result = analyzer.analyze(df, symbol="BTC/USDT", timeframe="15m")
        
        # Only HMM: use hmm_duration directly
        assert result.recommended_duration_hours == 3.5
        assert result.is_valid is True

    @patch("modules.detect_regime_change.regime_duration_analyzer.detect_change_points_pelt")
    @patch("modules.detect_regime_change.regime_duration_analyzer.estimate_hmm_regime_duration")
    def test_analyze_both_fail(self, mock_hmm, mock_pelt):
        """Test analyzer when both PELT and HMM fail."""
        # PELT fails
        mock_pelt.return_value = ([], [])
        
        # HMM fails
        mock_hmm.return_value = (None, None, None)
        
        analyzer = RegimeDurationAnalyzer()
        df = self.create_ohlcv_df()
        
        result = analyzer.analyze(df, symbol="BTC/USDT", timeframe="15m")
        
        # Both fail: no recommendation
        assert result.recommended_duration_hours is None
        assert result.is_valid is False

    @patch("modules.detect_regime_change.regime_duration_analyzer.detect_change_points_pelt")
    @patch("modules.detect_regime_change.regime_duration_analyzer.estimate_hmm_regime_duration")
    def test_analyze_pelt_exception(self, mock_hmm, mock_pelt):
        """Test analyzer when PELT raises exception."""
        mock_pelt.side_effect = Exception("PELT error")
        
        # HMM still works
        mock_hmm.return_value = (4.0, 1, 0.75)
        
        analyzer = RegimeDurationAnalyzer()
        df = self.create_ohlcv_df()
        
        result = analyzer.analyze(df, symbol="BTC/USDT", timeframe="15m")
        
        # PELT fails but HMM succeeds → use HMM
        assert result.recommended_duration_hours == 4.0
        assert result.is_valid is True

    @patch("modules.detect_regime_change.regime_duration_analyzer.detect_change_points_pelt")
    @patch("modules.detect_regime_change.regime_duration_analyzer.estimate_hmm_regime_duration")
    def test_analyze_hmm_exception(self, mock_hmm, mock_pelt):
        """Test analyzer when HMM raises exception."""
        from modules.detect_regime_change.models import ChangePoint, RegimeSegment
        mock_pelt.return_value = (
            [ChangePoint(index=100, timestamp=None)],
            [
                RegimeSegment(
                    start_index=0, end_index=100,
                    duration_seconds=3600*4, duration_hours=4.0,
                    mean_return=0.01, volatility=0.02,
                ),
            ],
        )
        
        # HMM raises exception
        mock_hmm.side_effect = Exception("HMM error")
        
        analyzer = RegimeDurationAnalyzer()
        df = self.create_ohlcv_df()
        
        result = analyzer.analyze(df, symbol="BTC/USDT", timeframe="15m")
        
        # HMM fails but PELT succeeds → use PELT
        assert result.recommended_duration_hours == 4.0
        assert result.is_valid is True

    @patch("modules.detect_regime_change.regime_duration_analyzer.detect_change_points_pelt")
    @patch("modules.detect_regime_change.regime_duration_analyzer.estimate_hmm_regime_duration")
    def test_analyze_total_failure(self, mock_hmm, mock_pelt):
        """Test analyzer when everything fails."""
        mock_pelt.side_effect = Exception("PELT error")
        mock_hmm.side_effect = Exception("HMM error")
        
        analyzer = RegimeDurationAnalyzer()
        df = self.create_ohlcv_df()
        
        result = analyzer.analyze(df, symbol="BTC/USDT", timeframe="15m")
        
        # Everything fails → error in result
        assert result.recommended_duration_hours is None
        assert result.error is not None
        assert result.is_valid is False

    def test_custom_weights(self):
        """Test analyzer with custom weight configuration."""
        analyzer = RegimeDurationAnalyzer(
            w_pelt_high_conf=0.5,
            w_hmm_high_conf=0.5,
            w_pelt_low_conf=0.8,
            w_hmm_low_conf=0.2,
        )
        
        result = RegimeDurationResult(
            symbol="TEST",
            timeframe="15m",
            pelt_avg_duration_hours=3.0,
            hmm_next_state_duration_hours=5.0,
            hmm_state_probability=0.8,  # High confidence
        )
        
        combined = analyzer._combine_results(result)
        
        # High confidence with custom weights: 0.5 * 3.0 + 0.5 * 5.0 = 4.0
        assert combined == 4.0

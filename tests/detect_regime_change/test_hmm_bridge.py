"""
tests/detect_regime_change/test_hmm_bridge.py
=============================================
Tests for detect_regime_change.hmm_regime_bridge module.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from modules.detect_regime_change.hmm_regime_bridge import estimate_hmm_regime_duration


class TestEstimateHmmRegimeDuration:
    def create_ohlcv_df(self, interval_minutes: int = 15, days: int = 30) -> pd.DataFrame:
        """Create a sample OHLCV DataFrame."""
        periods = int(days * 24 * 60 / interval_minutes)
        index = pd.date_range(
            end=datetime.now(),
            periods=periods,
            freq=f"{interval_minutes}min",
            tz="UTC",
        )
        
        np.random.seed(42)
        base_price = 50000
        
        data = []
        for i in range(periods):
            price = base_price + np.random.randn() * 100
            data.append({
                "open": price - 10,
                "high": price + 20,
                "low": price - 20,
                "close": price,
                "volume": np.random.randint(1000, 10000),
            })
        
        df = pd.DataFrame(data, index=index)
        return df

    @patch("modules.detect_regime_change.hmm_regime_bridge.hmm_swings")
    def test_hmm_success_hourly_candles(self, mock_hmm_swings):
        """Test HMM bridge with hourly candles."""
        # Mock HMM result
        mock_result = MagicMock()
        mock_result.next_state_duration = 4  # 4 hours
        mock_result.next_state_with_high_order_hmm = 1  # BULLISH
        mock_result.next_state_probability = 0.85
        mock_hmm_swings.return_value = mock_result

        # Create hourly DataFrame
        df = self.create_ohlcv_df(interval_minutes=60, days=30)
        
        duration, state, prob = estimate_hmm_regime_duration(df, train_ratio=0.8)
        
        # Hourly candles → duration already in hours
        assert duration == 4.0
        assert state == 1
        assert prob == 0.85

    @patch("modules.detect_regime_change.hmm_regime_bridge.hmm_swings")
    def test_hmm_success_minute_candles(self, mock_hmm_swings):
        """Test HMM bridge with minute candles."""
        mock_result = MagicMock()
        mock_result.next_state_duration = 60  # 60 minutes
        mock_result.next_state_with_high_order_hmm = 0  # NEUTRAL
        mock_result.next_state_probability = 0.75
        mock_hmm_swings.return_value = mock_result

        # Create 15-minute DataFrame
        df = self.create_ohlcv_df(interval_minutes=15, days=30)
        
        duration, state, prob = estimate_hmm_regime_duration(df, train_ratio=0.8)
        
        # 15-minute candles → duration converted from minutes to hours
        assert duration == 1.0  # 60 minutes = 1 hour
        assert state == 0
        assert prob == 0.75

    @patch("modules.detect_regime_change.hmm_regime_bridge.hmm_swings")
    def test_hmm_success_second_candles(self, mock_hmm_swings):
        """Test HMM bridge with second candles."""
        mock_result = MagicMock()
        mock_result.next_state_duration = 7200  # 7200 seconds
        mock_result.next_state_with_high_order_hmm = -1  # BEARISH
        mock_result.next_state_probability = 0.65
        mock_hmm_swings.return_value = mock_result

        # Create DataFrame with second interval
        df = self.create_ohlcv_df(interval_minutes=1, days=1)  # Use 1m as base
        # Override to make it look like seconds
        df.index = pd.date_range(end=datetime.now(), periods=len(df), freq="1s", tz="UTC")
        
        duration, state, prob = estimate_hmm_regime_duration(df, train_ratio=0.8)
        
        # Second candles → duration converted from seconds to hours
        assert duration == 2.0  # 7200 seconds = 2 hours
        assert state == -1

    @patch("modules.detect_regime_change.hmm_regime_bridge.hmm_swings")
    def test_hmm_with_non_datetime_index(self, mock_hmm_swings):
        """Test HMM bridge fallback when index is not DatetimeIndex."""
        mock_result = MagicMock()
        mock_result.next_state_duration = 2  # 2 units
        mock_result.next_state_with_high_order_hmm = 1
        mock_result.next_state_probability = 0.8
        mock_hmm_swings.return_value = mock_result

        # Create DataFrame with RangeIndex
        df = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [105, 106, 107],
            "low": [98, 99, 100],
            "close": [102, 103, 104],
            "volume": [1000, 2000, 3000],
        })
        
        duration, state, prob = estimate_hmm_regime_duration(df, train_ratio=0.8)
        
        # Fallback: assumes 15m candles = 900s → 2 * 900 / 3600 = 0.5 hours
        assert duration == 0.5

    def test_hmm_import_failure(self):
        """Test HMM bridge when hmm module import fails."""
        with patch.dict("sys.modules", {"modules.hmm": None}):
            df = self.create_ohlcv_df(interval_minutes=15, days=30)
            
            duration, state, prob = estimate_hmm_regime_duration(df, train_ratio=0.8)
            
            # Should return None values on failure
            assert duration is None
            assert state is None
            assert prob is None

    @patch("modules.detect_regime_change.hmm_regime_bridge.hmm_swings")
    def test_hmm_exception_handling(self, mock_hmm_swings):
        """Test HMM bridge exception handling."""
        mock_hmm_swings.side_effect = Exception("HMM analysis failed")

        df = self.create_ohlcv_df(interval_minutes=15, days=30)
        
        duration, state, prob = estimate_hmm_regime_duration(df, train_ratio=0.8)
        
        # Should return None values on exception
        assert duration is None
        assert state is None
        assert prob is None

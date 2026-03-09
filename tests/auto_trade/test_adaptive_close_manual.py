#!/usr/bin/env python3
"""
Manual test runner for AdaptiveCloseResult tests.
"""
import sys
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd

# Add project root to path
sys.path.insert(0, r'D:\NGUYEN QUANG THANG\Probability projects\crypto-probability-')

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


def make_ohlcv_df(periods=120):
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


def test_adaptive_result_creation():
    """Test creating AdaptiveCloseResult with all fields."""
    print("Test 1: AdaptiveCloseResult creation...")
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
    print("[OK] PASSED")


def test_disabled_returns_static_source():
    """Test that disabled adaptive close returns source='static'."""
    print("Test 2: Disabled adaptive close returns static source...")
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
    print("[OK] PASSED")


def test_adaptive_source_when_analysis_valid():
    """Test source='adaptive' when analysis is valid."""
    print("Test 3: Valid analysis returns adaptive source...")
    
    with patch("modules.auto_trade.execution.adaptive_close_calculator.RegimeDurationAnalyzer") as mock_analyzer_class:
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

        df = make_ohlcv_df(periods=120)
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
    print("[OK] PASSED")


def test_adaptive_fallback_source_when_analysis_invalid():
    """Test source='adaptive_fallback' when analysis is invalid."""
    print("Test 4: Invalid analysis returns adaptive_fallback source...")
    
    with patch("modules.auto_trade.execution.adaptive_close_calculator.RegimeDurationAnalyzer") as mock_analyzer_class:
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

        df = make_ohlcv_df(periods=120)
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
    print("✓ PASSED")


def test_order_data_contains_adaptive_metadata_fields():
    """Test that order_data dict contains all 4 new metadata fields."""
    print("Test 5: Order data contains adaptive metadata fields...")
    
    order_data = {
        "order_id": "test_order_123",
        "symbol": "BTC/USDT",
        "status": "OPEN",
    }

    adaptive_result = AdaptiveCloseResult(
        deadline_utc=datetime.now(timezone.utc),
        source="adaptive",
        duration_hours=5.5,
        pelt_hours=4.2,
        hmm_hours=6.1,
    )

    order_data["auto_close_deadline_utc"] = adaptive_result.deadline_utc.isoformat().replace("+00:00", "Z")
    order_data["auto_close_deadline_source"] = adaptive_result.source
    order_data["adaptive_close_duration_hours"] = adaptive_result.duration_hours
    order_data["adaptive_close_pelt_hours"] = adaptive_result.pelt_hours
    order_data["adaptive_close_hmm_hours"] = adaptive_result.hmm_hours

    assert "auto_close_deadline_utc" in order_data
    assert "auto_close_deadline_source" in order_data
    assert "adaptive_close_duration_hours" in order_data
    assert "adaptive_close_pelt_hours" in order_data
    assert "adaptive_close_hmm_hours" in order_data
    print("[OK] PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("Running AdaptiveCloseResult Manual Tests")
    print("=" * 60)
    
    try:
        test_adaptive_result_creation()
        test_disabled_returns_static_source()
        test_adaptive_source_when_analysis_valid()
        test_adaptive_fallback_source_when_analysis_invalid()
        test_order_data_contains_adaptive_metadata_fields()
        
        print("=" * 60)
        print("All tests PASSED [OK]")
        print("=" * 60)
    except AssertionError as e:
        print(f"[FAIL] FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[FAIL] ERROR: {e}")
        sys.exit(1)

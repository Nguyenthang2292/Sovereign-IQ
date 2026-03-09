"""
tests/detect_regime_change/test_models.py
=========================================
Tests for detect_regime_change.models module.
"""

import pytest

from modules.detect_regime_change.models import (
    ChangePoint,
    RegimeDurationResult,
    RegimeSegment,
)


class TestChangePoint:
    def test_basic_creation(self):
        cp = ChangePoint(index=10, timestamp="2026-03-09T12:00:00Z")
        assert cp.index == 10
        assert cp.timestamp == "2026-03-09T12:00:00Z"

    def test_no_timestamp(self):
        cp = ChangePoint(index=5, timestamp=None)
        assert cp.index == 5
        assert cp.timestamp is None


class TestRegimeSegment:
    def test_basic_creation(self):
        seg = RegimeSegment(
            start_index=0,
            end_index=100,
            duration_seconds=3600,
            duration_hours=1.0,
            mean_return=0.05,
            volatility=0.02,
        )
        assert seg.start_index == 0
        assert seg.end_index == 100
        assert seg.duration_seconds == 3600
        assert seg.duration_hours == 1.0
        assert seg.mean_return == 0.05
        assert seg.volatility == 0.02

    def test_optional_fields_none(self):
        seg = RegimeSegment(
            start_index=0,
            end_index=50,
            duration_seconds=1800,
            duration_hours=0.5,
            mean_return=None,
            volatility=None,
        )
        assert seg.mean_return is None
        assert seg.volatility is None


class TestRegimeDurationResult:
    def test_basic_creation(self):
        result = RegimeDurationResult(
            symbol="BTC/USDT",
            timeframe="15m",
        )
        assert result.symbol == "BTC/USDT"
        assert result.timeframe == "15m"
        assert result.pelt_change_points == []
        assert result.pelt_segments == []
        assert result.pelt_avg_duration_hours is None
        assert result.pelt_median_duration_hours is None
        assert result.hmm_next_state_duration_hours is None
        assert result.hmm_state is None
        assert result.hmm_state_probability is None
        assert result.recommended_duration_hours is None
        assert result.data_points_analyzed == 0
        assert result.analysis_timestamp is None
        assert result.computation_time_ms is None
        assert result.error is None

    def test_is_valid_false_initially(self):
        result = RegimeDurationResult(symbol="TEST", timeframe="15m")
        assert result.is_valid is False

    def test_is_valid_true_with_recommendation(self):
        result = RegimeDurationResult(
            symbol="TEST",
            timeframe="15m",
            recommended_duration_hours=4.5,
        )
        assert result.is_valid is True

    def test_is_valid_false_when_error(self):
        result = RegimeDurationResult(
            symbol="TEST",
            timeframe="15m",
            recommended_duration_hours=4.5,
            error="Some error occurred",
        )
        assert result.is_valid is False

    def test_is_valid_false_when_no_recommendation(self):
        result = RegimeDurationResult(
            symbol="TEST",
            timeframe="15m",
            pelt_avg_duration_hours=3.0,
        )
        assert result.is_valid is False

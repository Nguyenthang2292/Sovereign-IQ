"""
tests/detect_regime_change/test_pelt_detector.py
================================================
Tests for detect_regime_change.pelt_detector module.
"""

import builtins

import numpy as np
import pytest

from modules.detect_regime_change.models import RegimeSegment
from modules.detect_regime_change.pelt_detector import (
    calculate_pelt_avg_duration,
    detect_change_points_pelt,
)


class TestDetectChangePointsPelt:
    def test_detect_on_synthetic_data(self):
        """Test PELT detection on synthetic data with known breakpoints."""
        np.random.seed(42)

        # Create synthetic returns with 3 clear regimes
        # Regime 1: low volatility (0-99)
        r1 = np.random.normal(0, 0.001, 100)
        # Regime 2: high volatility (100-199)
        r2 = np.random.normal(0, 0.01, 100)
        # Regime 3: medium volatility with drift (200-299)
        r3 = np.random.normal(0.001, 0.005, 100)

        returns = np.concatenate([r1, r2, r3])

        change_points, segments = detect_change_points_pelt(
            returns=returns,
            penalty=None,  # Auto BIC
            model="rbf",
            min_segment_length=20,
        )

        # Should detect at least 2 change points (3 regimes)
        assert len(change_points) >= 2
        assert len(segments) >= 3

        # Each segment should have valid duration
        for seg in segments:
            assert seg.duration_hours > 0
            assert seg.start_index < seg.end_index

    def test_insufficient_data(self):
        """Test that insufficient data returns empty results."""
        returns = np.random.normal(0, 0.01, 15)  # Too short

        change_points, segments = detect_change_points_pelt(
            returns=returns,
            min_segment_length=10,
        )

        assert change_points == []
        assert segments == []

    def test_with_timestamps(self):
        """Test PELT with timestamp array."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 200)
        timestamps = np.arange('2026-03-01', 200, dtype='datetime64[m]')

        change_points, segments = detect_change_points_pelt(
            returns=returns,
            timestamps=timestamps,
            min_segment_length=20,
        )

        # Should work with timestamps
        assert isinstance(change_points, list)
        assert isinstance(segments, list)

    def test_no_timestamps(self):
        """Test PELT without timestamp array (fallback to candle count)."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 200)

        change_points, segments = detect_change_points_pelt(
            returns=returns,
            timestamps=None,
            min_segment_length=20,
        )

        # Segments should have estimated durations
        for seg in segments:
            # Fallback assumes 15m candles = 900s
            expected_duration = (seg.end_index - seg.start_index) * 900 / 3600
            assert abs(seg.duration_hours - expected_duration) < 0.01

    def test_fallback_to_python_when_rust_missing(self, monkeypatch):
        """Test fallback to ruptures when Rust extension import fails."""
        np.random.seed(42)
        r1 = np.random.normal(0, 0.001, 100)
        r2 = np.random.normal(0, 0.01, 100)
        r3 = np.random.normal(0.001, 0.005, 100)
        returns = np.concatenate([r1, r2, r3])

        original_import = builtins.__import__

        def _mock_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in {"rust_extensions", "modules.detect_regime_change.rust_extensions"}:
                raise ImportError("simulated missing rust extension")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _mock_import)

        change_points, segments = detect_change_points_pelt(
            returns=returns,
            penalty=None,
            model="rbf",
            min_segment_length=20,
        )

        assert len(change_points) >= 2
        assert len(segments) >= 3

    def test_l2_works_without_ruptures_via_rust(self, monkeypatch):
        """Test L2 path still works when ruptures is unavailable, via Rust backend."""
        pytest.importorskip(
            "rust_extensions",
            reason="Rust extension required for this test",
        )

        np.random.seed(7)
        r1 = np.random.normal(0.0, 0.001, 120)
        r2 = np.random.normal(0.0, 0.01, 120)
        returns = np.concatenate([r1, r2])

        original_import = builtins.__import__

        def _mock_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "ruptures":
                raise ImportError("simulated missing ruptures")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _mock_import)

        change_points, segments = detect_change_points_pelt(
            returns=returns,
            penalty=None,
            model="l2",
            min_segment_length=20,
        )

        assert len(segments) >= 1
        assert isinstance(change_points, list)

    def test_normal_works_without_ruptures_via_rust(self, monkeypatch):
        """Test Normal path still works when ruptures is unavailable, via Rust backend."""
        pytest.importorskip(
            "rust_extensions",
            reason="Rust extension required for this test",
        )

        np.random.seed(11)
        r1 = np.random.normal(0.0, 0.002, 120)
        r2 = np.random.normal(0.0, 0.02, 120)
        returns = np.concatenate([r1, r2])

        original_import = builtins.__import__

        def _mock_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "ruptures":
                raise ImportError("simulated missing ruptures")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _mock_import)

        change_points, segments = detect_change_points_pelt(
            returns=returns,
            penalty=None,
            model="normal",
            min_segment_length=20,
        )

        assert len(segments) >= 1
        assert isinstance(change_points, list)


class TestCalculatePeltAvgDuration:
    def test_empty_segments(self):
        """Test with empty segments list."""
        avg, median = calculate_pelt_avg_duration([], trim_pct=0.1)
        assert avg is None
        assert median is None

    def test_single_segment(self):
        """Test with single segment."""
        segments = [
            RegimeSegment(
                start_index=0,
                end_index=100,
                duration_seconds=3600,
                duration_hours=1.0,
                mean_return=0.01,
                volatility=0.02,
            )
        ]
        avg, median = calculate_pelt_avg_duration(segments, trim_pct=0.1)
        assert avg == 1.0
        assert median == 1.0

    def test_multiple_segments(self):
        """Test with multiple segments - trimmed mean."""
        segments = [
            RegimeSegment(
                start_index=0,
                end_index=100,
                duration_seconds=3600 * 1,
                duration_hours=1.0,
                mean_return=0.01,
                volatility=0.02,
            ),
            RegimeSegment(
                start_index=100,
                end_index=200,
                duration_seconds=3600 * 2,
                duration_hours=2.0,
                mean_return=0.01,
                volatility=0.02,
            ),
            RegimeSegment(
                start_index=200,
                end_index=300,
                duration_seconds=3600 * 3,
                duration_hours=3.0,
                mean_return=0.01,
                volatility=0.02,
            ),
            RegimeSegment(
                start_index=300,
                end_index=400,
                duration_seconds=3600 * 4,
                duration_hours=4.0,
                mean_return=0.01,
                volatility=0.02,
            ),
            RegimeSegment(
                start_index=400,
                end_index=500,
                duration_seconds=3600 * 5,
                duration_hours=5.0,
                mean_return=0.01,
                volatility=0.02,
            ),
        ]
        avg, median = calculate_pelt_avg_duration(segments, trim_pct=0.1)

        # Median of [1, 2, 3, 4, 5] = 3
        assert median == 3.0

        # Trimmed mean with 10% removes 1 from each end → [2, 3, 4]
        # Mean of [2, 3, 4] = 3.0
        assert avg == 3.0

    def test_trimmed_mean_with_outliers(self):
        """Test that trimmed mean removes outliers."""
        segments = [
            RegimeSegment(
                start_index=0,
                end_index=10,
                duration_seconds=3600 * 0.5,  # Outlier: very short
                duration_hours=0.5,
                mean_return=0.01,
                volatility=0.02,
            ),
            RegimeSegment(
                start_index=10,
                end_index=110,
                duration_seconds=3600 * 2,
                duration_hours=2.0,
                mean_return=0.01,
                volatility=0.02,
            ),
            RegimeSegment(
                start_index=110,
                end_index=210,
                duration_seconds=3600 * 2.1,
                duration_hours=2.1,
                mean_return=0.01,
                volatility=0.02,
            ),
            RegimeSegment(
                start_index=210,
                end_index=310,
                duration_seconds=3600 * 1.9,
                duration_hours=1.9,
                mean_return=0.01,
                volatility=0.02,
            ),
            RegimeSegment(
                start_index=310,
                end_index=410,
                duration_seconds=3600 * 2.0,
                duration_hours=2.0,
                mean_return=0.01,
                volatility=0.02,
            ),
            RegimeSegment(
                start_index=410,
                end_index=420,
                duration_seconds=3600 * 10.0,  # Outlier: very long
                duration_hours=10.0,
                mean_return=0.01,
                volatility=0.02,
            ),
        ]
        avg, median = calculate_pelt_avg_duration(segments, trim_pct=0.1)

        # Trimmed mean should be close to 2.0 (without outliers)
        assert 1.5 < avg < 2.5

        # Median should be 2.0
        assert median == 2.0

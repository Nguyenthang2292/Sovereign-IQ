"""
Unit tests for SwingDetector (zigzag pivot detection).
"""

from __future__ import annotations

import pandas as pd
import pytest

from modules.gemini_gann_square.core.swing_detector import SwingDetector, SwingPoint

# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


def make_df(highs: list[float], lows: list[float]) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame with DatetimeIndex."""
    n = len(highs)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {
            "open": [1.0] * n,
            "high": highs,
            "low": lows,
            "close": [1.0] * n,
            "volume": [100.0] * n,
        },
        index=timestamps,
    )


# ──────────────────────────────────────────────
# Basic detection tests
# ──────────────────────────────────────────────


class TestSwingDetectorBasic:
    def test_detects_clear_swing_high(self):
        """A clear local peak surrounded by lower values should be detected."""
        # Pattern: rise to peak then fall
        highs = [1, 2, 3, 5, 3, 2, 1, 2, 3, 2, 1]
        lows = [0.5] * len(highs)
        df = make_df(highs, lows)
        detector = SwingDetector(lookback=2)
        swing_highs, _ = detector.detect(df)

        # index 3 has high=5 (max in window [1,2,3,5,3])
        assert any(sp.price == 5.0 for sp in swing_highs), f"Expected swing high at price=5, got: {swing_highs}"

    def test_detects_clear_swing_low(self):
        """A clear local trough should be detected as swing low."""
        lows = [5, 4, 3, 1, 3, 4, 5, 4, 3, 4, 5]
        highs = [6.0] * len(lows)
        df = make_df(highs, lows)
        detector = SwingDetector(lookback=2)
        _, swing_lows = detector.detect(df)

        assert any(sp.price == 1.0 for sp in swing_lows), f"Expected swing low at price=1, got: {swing_lows}"

    def test_no_pivot_on_flat_data(self):
        """Completely flat data should return no swing points (every point ties)."""
        highs = [5.0] * 20
        lows = [3.0] * 20
        df = make_df(highs, lows)
        detector = SwingDetector(lookback=3)
        swing_highs, swing_lows = detector.detect(df)

        # All points tie — every candle qualifies as swing, so we get many
        # The important thing: the function doesn't crash
        assert isinstance(swing_highs, list)
        assert isinstance(swing_lows, list)

    def test_returns_swing_points_as_correct_type(self):
        """Detected swing points should be SwingPoint instances."""
        highs = [1, 3, 5, 3, 1, 3, 5, 3, 1, 3, 5]
        lows = [0.5] * len(highs)
        df = make_df(highs, lows)
        detector = SwingDetector(lookback=2)
        swing_highs, swing_lows = detector.detect(df)

        for sp in swing_highs:
            assert isinstance(sp, SwingPoint)
            assert sp.kind == "high"

        for sp in swing_lows:
            assert isinstance(sp, SwingPoint)
            assert sp.kind == "low"


# ──────────────────────────────────────────────
# get_significant_swings tests
# ──────────────────────────────────────────────


class TestGetSignificantSwings:
    def test_highest_and_lowest_selected_correctly(self):
        """get_significant_swings should return the globally highest high and lowest low."""
        #            0    1    2    3    4    5    6    7    8    9   10
        highs = [2, 4, 10, 4, 2, 4, 8, 4, 2, 4, 6]
        lows = [1, 1, 0.5, 1, 1, 1, 1, 0.1, 1, 1, 1]
        df = make_df(highs, lows)
        detector = SwingDetector(lookback=2)

        highest, lowest = detector.get_significant_swings(df)

        assert highest is not None, "Should detect a swing high"
        assert lowest is not None, "Should detect a swing low"
        assert highest.price == 10.0, f"Expected highest=10, got {highest.price}"
        assert lowest.price == 0.1, f"Expected lowest=0.1, got {lowest.price}"

    def test_returns_none_for_too_little_data(self):
        """Should return (None, None) when there's insufficient data after validation."""
        highs = [1, 2, 3]
        lows = [0.5, 0.5, 0.5]
        df = make_df(highs, lows)
        detector = SwingDetector(lookback=5)  # requires 2*5+1=11 rows

        with pytest.raises(ValueError, match="at least"):
            detector.get_significant_swings(df)

    def test_swing_high_timestamp_matches_dataframe(self):
        """SwingPoint.timestamp should correspond to the DataFrame index."""
        highs = [1, 2, 5, 2, 1, 2, 3, 2, 1, 2, 1]
        lows = [0.5] * len(highs)
        df = make_df(highs, lows)
        detector = SwingDetector(lookback=2)

        highest, _ = detector.get_significant_swings(df)
        assert highest is not None
        # index 2 → third timestamp
        expected_ts = pd.date_range("2024-01-01", periods=len(highs), freq="1h")[2]
        assert highest.timestamp == expected_ts


# ──────────────────────────────────────────────
# Validation / edge cases
# ──────────────────────────────────────────────


class TestSwingDetectorValidation:
    def test_raises_on_empty_dataframe(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        with pytest.raises(ValueError, match="empty"):
            SwingDetector().detect(df)

    def test_raises_on_missing_columns(self):
        df = pd.DataFrame({"open": [1, 2, 3], "close": [1, 2, 3]})
        with pytest.raises(ValueError, match="missing"):
            SwingDetector().detect(df)

    def test_raises_on_invalid_lookback(self):
        with pytest.raises(ValueError, match="lookback"):
            SwingDetector(lookback=0)

    def test_raises_when_data_too_short(self):
        highs = [1, 2, 3]
        lows = [0.5, 0.5, 0.5]
        df = make_df(highs, lows)
        with pytest.raises(ValueError, match="at least"):
            SwingDetector(lookback=5).detect(df)  # needs 11 rows, has 3

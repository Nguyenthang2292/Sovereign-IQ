import pandas as pd
from modules.smart_money_concept.models.pivot import Pivot
from modules.smart_money_concept.core.trend import detect_trend, BULLISH, BEARISH, NEUTRAL

def test_detect_trend_bullish():
    dates = pd.date_range("2023-01-01", periods=3, tz="UTC")
    highs = [Pivot(10, dates[0]), Pivot(20, dates[1]), Pivot(30, dates[2])]
    lows = [Pivot(5, dates[0]), Pivot(15, dates[1]), Pivot(25, dates[2])]
    assert detect_trend(highs, lows) == BULLISH

def test_detect_trend_bearish():
    dates = pd.date_range("2023-01-01", periods=3, tz="UTC")
    highs = [Pivot(30, dates[0]), Pivot(20, dates[1]), Pivot(10, dates[2])]
    lows = [Pivot(25, dates[0]), Pivot(15, dates[1]), Pivot(5, dates[2])]
    assert detect_trend(highs, lows) == BEARISH

def test_detect_trend_neutral():
    dates = pd.date_range("2023-01-01", periods=3, tz="UTC")
    highs = [Pivot(10, dates[0]), Pivot(20, dates[1]), Pivot(30, dates[2])]
    lows = [Pivot(25, dates[0]), Pivot(15, dates[1]), Pivot(5, dates[2])]
    assert detect_trend(highs, lows) == NEUTRAL


def test_detect_trend_prefers_last_structure_break_when_provided():
    dates = pd.date_range("2023-01-01", periods=3, tz="UTC")
    highs = [Pivot(10, dates[0]), Pivot(20, dates[1]), Pivot(30, dates[2])]
    lows = [Pivot(5, dates[0]), Pivot(15, dates[1]), Pivot(25, dates[2])]

    assert detect_trend(highs, lows, last_structure_break=BEARISH) == BEARISH


def test_detect_trend_falls_back_to_pattern_without_structure_break():
    dates = pd.date_range("2023-01-01", periods=3, tz="UTC")
    highs = [Pivot(30, dates[0]), Pivot(20, dates[1]), Pivot(10, dates[2])]
    lows = [Pivot(25, dates[0]), Pivot(15, dates[1]), Pivot(5, dates[2])]

    assert detect_trend(highs, lows, last_structure_break=None) == BEARISH

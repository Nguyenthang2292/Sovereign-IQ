import pandas as pd
from modules.smart_money_concept.core.bos import identify_bos, identify_bos_choch
from modules.smart_money_concept.core.trend import BEARISH, BULLISH
from modules.smart_money_concept.models.pivot import Pivot

def test_identify_bos():
    dates = pd.date_range("2023-01-01", periods=10, tz="UTC")
    df = pd.DataFrame({
        "High": [10, 11, 12, 11, 15, 16, 17, 16, 20, 21],
        "Low":  [5,   6,  7,  6,  10, 11, 12, 11, 15, 16],
        "Open": [6]*10,
        "Close": [7, 7, 7, 8, 9, 10, 13, 14, 15, 16]
    }, index=dates)
    
    swing_highs = [Pivot(level=12.0, bar_time=dates[2]), Pivot(level=17.0, bar_time=dates[6])]
    swing_lows = [Pivot(level=7.0, bar_time=dates[2]), Pivot(level=12.0, bar_time=dates[6])]
    
    result = identify_bos(df, swing_highs, swing_lows)
    
    assert hasattr(result, "high_bos")
    assert hasattr(result, "low_bos")
    
    high_bos = result.high_bos
    assert not high_bos.empty
    assert "Pivot_bullishBos_Time" in high_bos.columns
    assert "Crossing_Time" in high_bos.columns


def test_identify_bos_choch_uses_close_crossover_only():
    dates = pd.date_range("2023-01-01", periods=8, tz="UTC")
    df = pd.DataFrame(
        {
            "Open": [9.5, 9.6, 9.7, 9.9, 10.2, 10.3, 10.4, 10.5],
            "High": [11.5, 11.2, 11.1, 11.4, 11.6, 11.5, 11.7, 11.8],
            "Low": [8.5, 8.7, 8.8, 8.9, 9.9, 10.1, 10.0, 10.1],
            "Close": [9.4, 9.5, 9.6, 9.7, 10.1, 10.2, 10.3, 10.4],
        },
        index=dates,
    )

    swing_highs = [Pivot(level=10.0, bar_time=dates[1]), Pivot(level=12.0, bar_time=dates[7])]
    swing_lows = [Pivot(level=8.8, bar_time=dates[1]), Pivot(level=9.2, bar_time=dates[7])]

    result = identify_bos_choch(df, swing_highs, swing_lows, initial_trend=BULLISH)

    assert len(result.bullish_bos) == 1
    assert result.bullish_choch.empty
    assert result.bullish_bos.iloc[0]["event_type"] == "BOS"


def test_identify_bos_choch_classifies_choch_when_break_against_trend():
    dates = pd.date_range("2023-01-01", periods=8, tz="UTC")
    df = pd.DataFrame(
        {
            "Open": [9.5, 9.6, 9.7, 9.9, 10.2, 10.3, 10.4, 10.5],
            "High": [11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7],
            "Low": [8.5, 8.6, 8.7, 8.8, 9.0, 9.1, 9.2, 9.3],
            "Close": [9.4, 9.5, 9.6, 9.7, 10.1, 10.2, 10.3, 10.4],
        },
        index=dates,
    )

    swing_highs = [Pivot(level=10.0, bar_time=dates[1]), Pivot(level=12.0, bar_time=dates[7])]
    swing_lows = [Pivot(level=8.8, bar_time=dates[1]), Pivot(level=9.2, bar_time=dates[7])]

    result = identify_bos_choch(df, swing_highs, swing_lows, initial_trend=BEARISH)

    assert result.bullish_bos.empty
    assert len(result.bullish_choch) == 1
    assert result.bullish_choch.iloc[0]["event_type"] == "CHoCH"

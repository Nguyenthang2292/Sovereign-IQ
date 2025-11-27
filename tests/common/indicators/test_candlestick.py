import pandas as pd

from modules.common.indicators.candlestick import CandlestickPatterns


def test_hammer_and_shooting_star_detection():
    df = pd.DataFrame(
        {
            "open": [10.0, 11.0],
            "high": [10.22, 12.0],
            "low": [9.0, 10.89],
            "close": [10.2, 10.9],
            "volume": [1000, 900],
        }
    )

    result, metadata = CandlestickPatterns.apply(df)

    assert result.loc[0, "HAMMER"] == 1
    assert result.loc[1, "SHOOTING_STAR"] == 1
    assert metadata["HAMMER"] == CandlestickPatterns.CATEGORY


def test_engulfing_patterns_detected():
    df = pd.DataFrame(
        {
            "open": [10.2, 9.6, 9.8, 10.7],
            "high": [10.4, 10.8, 10.2, 10.9],
            "low": [9.5, 9.5, 9.6, 9.4],
            "close": [9.7, 10.7, 10.5, 9.6],
            "volume": [1000, 1100, 1200, 1300],
        }
    )

    result, _ = CandlestickPatterns.apply(df)

    assert result.loc[1, "BULLISH_ENGULFING"] == 1
    assert result.loc[3, "BEARISH_ENGULFING"] == 1


def test_three_white_soldiers():
    df = pd.DataFrame(
        {
            "open": [10.0, 10.2, 10.5, 10.8],
            "high": [10.5, 10.8, 11.0, 11.2],
            "low": [9.8, 10.0, 10.3, 10.6],
            "close": [10.4, 10.7, 10.9, 11.1],
            "volume": [900, 950, 980, 1000],
        }
    )

    result, _ = CandlestickPatterns.apply(df)

    assert result.loc[2, "THREE_WHITE_SOLDIERS"] == 1


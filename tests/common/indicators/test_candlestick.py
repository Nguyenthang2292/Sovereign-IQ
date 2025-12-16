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


def test_piercing_pattern():
    """Test PIERCING pattern: bearish candle followed by bullish that gaps down but closes above midpoint."""
    df = pd.DataFrame(
        {
            "open": [10.0, 9.5],  # Row 0: bearish (10.0 > 9.8), Row 1: bullish (9.5 < 9.95)
            "high": [10.2, 10.0],
            "low": [9.5, 9.3],
            "close": [9.8, 9.95],  # Row 1: close (9.95) > midpoint (9.9) but < prev open (10.0)
            "volume": [1000, 1100],
        }
    )
    result, _ = CandlestickPatterns.apply(df)
    # Row 1 should be PIERCING: prev bearish, curr bullish, gap down, above midpoint, below prev open
    assert result.loc[1, "PIERCING"] == 1


def test_dark_cloud_pattern():
    """Test DARK_CLOUD pattern: bullish candle followed by bearish that gaps up but closes below midpoint."""
    df = pd.DataFrame(
        {
            "open": [9.5, 10.1],  # Row 0: bullish (9.5 < 10.0), Row 1: bearish (10.1 > 9.7)
            "high": [10.2, 10.5],
            "low": [9.3, 9.5],
            "close": [10.0, 9.7],  # Row 1: close (9.7) < midpoint (9.75) but > prev open (9.5)
            "volume": [1000, 1100],
        }
    )
    result, _ = CandlestickPatterns.apply(df)
    # Row 1 should be DARK_CLOUD: prev bullish, curr bearish, gap up, below midpoint, above prev open
    assert result.loc[1, "DARK_CLOUD"] == 1
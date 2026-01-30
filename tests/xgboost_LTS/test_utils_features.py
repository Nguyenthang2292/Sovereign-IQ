import numpy as np
import pandas as pd

from modules.xgboost_LTS.utils import features as feature_utils


def _sample_ohlcv_df(rows=10):
    df = pd.DataFrame(
        {
            "open": np.linspace(100, 109, rows),
            "high": np.linspace(101, 110, rows),
            "low": np.linspace(99, 108, rows),
            "close": np.linspace(100, 109, rows),
            "volume": np.linspace(10, 19, rows),
        }
    )
    return df


def test_add_price_derived_features_missing_columns():
    df = pd.DataFrame({"open": [1.0], "close": [1.0]})
    try:
        feature_utils.add_price_derived_features(df)
    except ValueError as e:
        assert "Missing required OHLCV columns" in str(e)
    else:
        raise AssertionError("Expected ValueError for missing columns")


def test_add_price_derived_features_python_values(monkeypatch):
    monkeypatch.setattr(feature_utils, "RUST_AVAILABLE", False)

    df = pd.DataFrame(
        {
            "open": [1.0, 2.0, 4.0],
            "high": [2.0, 3.0, 5.0],
            "low": [0.5, 1.5, 3.0],
            "close": [1.0, 2.0, 4.0],
            "volume": [0.0, 9.0, 0.0],
        }
    )

    result = feature_utils.add_price_derived_features(df)
    assert "returns_1" in result.columns
    assert "returns_5" in result.columns
    assert "log_volume" in result.columns
    assert "high_low_range" in result.columns
    assert "close_open_diff" in result.columns

    np.testing.assert_allclose(result["returns_1"].values, [0.0, 1.0, 1.0])
    np.testing.assert_allclose(result["returns_5"].values, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(result["log_volume"].values, np.log1p(df["volume"].values))


def test_add_advanced_features_python_adds_columns(monkeypatch):
    monkeypatch.setattr(feature_utils, "RUST_AVAILABLE", False)

    rows = 50
    df = _sample_ohlcv_df(rows)
    df["ATR_14"] = np.random.uniform(0.5, 2.0, rows)
    df["RSI_14"] = np.random.uniform(30, 70, rows)
    df["SMA_20"] = df["close"].rolling(20, min_periods=1).mean()
    df["SMA_50"] = df["close"].rolling(50, min_periods=1).mean()
    df["SMA_200"] = df["close"].rolling(50, min_periods=1).mean()
    df.index = pd.date_range("2024-01-01", periods=rows, freq="h")

    result = feature_utils.add_advanced_features(df)

    for col in [
        "roc_3",
        "atr_ratio",
        "price_to_SMA_20",
        "rolling_std_10",
        "rolling_skew_10",
        "returns_1_lag_1",
        "RSI_14_lag_1",
        "log_volume_lag_1",
        "atr_ratio_lag_1",
        "hour",
        "dayofweek",
        "month",
    ]:
        assert col in result.columns

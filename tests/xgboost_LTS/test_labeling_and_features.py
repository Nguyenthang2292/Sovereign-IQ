import numpy as np
import pandas as pd
import pytest

from modules.xgboost_LTS.core import labeling
from modules.xgboost_LTS.core.labeling import apply_directional_labels
from modules.xgboost_LTS.utils.features import add_price_derived_features


def _build_ohlcv(n_rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    base = 100 + np.cumsum(rng.normal(0, 0.4, n_rows))
    close = pd.Series(base).rolling(3, min_periods=1).mean().values
    open_price = close * (1 + rng.normal(0, 0.001, n_rows))
    high = np.maximum(open_price, close) * (1 + rng.uniform(0.0001, 0.002, n_rows))
    low = np.minimum(open_price, close) * (1 - rng.uniform(0.0001, 0.002, n_rows))
    volume = rng.uniform(1000, 5000, n_rows)

    return pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=pd.date_range("2026-01-01", periods=n_rows, freq="h"),
    )


@pytest.mark.unit
def test_apply_directional_labels_adds_expected_columns_and_tail_nan(monkeypatch: pytest.MonkeyPatch):
    df = _build_ohlcv(80)

    monkeypatch.setattr(labeling, "TARGET_HORIZON", 3)
    monkeypatch.setattr(labeling, "XGBOOST_VOLATILITY_ROLLING_WINDOW", 20)

    labeled = apply_directional_labels(df, use_cache=False)

    assert "TargetLabel" in labeled.columns
    assert "Target" in labeled.columns
    assert "DynamicThreshold" in labeled.columns

    assert labeled["Target"].iloc[-3:].isna().all()

    valid_targets = labeled["Target"].dropna()
    assert not valid_targets.empty
    assert set(valid_targets.unique()).issubset({0, 1, 2})


@pytest.mark.unit
def test_add_price_derived_features_outputs_expected_columns_and_finite_values():
    df = _build_ohlcv(64)

    out = add_price_derived_features(df)

    expected_columns = {
        "returns_1",
        "returns_5",
        "log_volume",
        "high_low_range",
        "close_open_diff",
    }
    assert expected_columns.issubset(set(out.columns))

    feature_matrix = out[list(expected_columns)].to_numpy(dtype=float)
    assert np.isfinite(feature_matrix).all()


@pytest.mark.unit
def test_add_price_derived_features_raises_for_missing_ohlcv_columns():
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="Missing required OHLCV columns"):
        add_price_derived_features(df)

import numpy as np
import pandas as pd

from config import (
    DYNAMIC_LOOKBACK_WEIGHTS_HIGH_VOL,
    DYNAMIC_LOOKBACK_WEIGHTS_LOW_VOL,
    DYNAMIC_LOOKBACK_WEIGHTS_MEDIUM_VOL,
)
from modules.xgboost_LTS.core import labeling


def test_calculate_lookback_weights_by_regime():
    volatility_multiplier = pd.Series([1.0, 2.0, 3.0])
    vol_low_threshold = pd.Series([1.5, 1.5, 1.5])
    vol_high_threshold = pd.Series([2.5, 2.5, 2.5])

    weight_short, weight_medium, weight_long = labeling._calculate_lookback_weights(
        volatility_multiplier,
        vol_low_threshold,
        vol_high_threshold,
    )

    # Low volatility row
    assert np.isclose(weight_short.iloc[0], DYNAMIC_LOOKBACK_WEIGHTS_LOW_VOL[0])
    assert np.isclose(weight_medium.iloc[0], DYNAMIC_LOOKBACK_WEIGHTS_LOW_VOL[1])
    assert np.isclose(weight_long.iloc[0], DYNAMIC_LOOKBACK_WEIGHTS_LOW_VOL[2])

    # Medium volatility row
    assert np.isclose(weight_short.iloc[1], DYNAMIC_LOOKBACK_WEIGHTS_MEDIUM_VOL[0])
    assert np.isclose(weight_medium.iloc[1], DYNAMIC_LOOKBACK_WEIGHTS_MEDIUM_VOL[1])
    assert np.isclose(weight_long.iloc[1], DYNAMIC_LOOKBACK_WEIGHTS_MEDIUM_VOL[2])

    # High volatility row
    assert np.isclose(weight_short.iloc[2], DYNAMIC_LOOKBACK_WEIGHTS_HIGH_VOL[0])
    assert np.isclose(weight_medium.iloc[2], DYNAMIC_LOOKBACK_WEIGHTS_HIGH_VOL[1])
    assert np.isclose(weight_long.iloc[2], DYNAMIC_LOOKBACK_WEIGHTS_HIGH_VOL[2])

    # Each row should sum to ~1
    totals = weight_short + weight_medium + weight_long
    np.testing.assert_allclose(totals.values, np.ones_like(totals.values), rtol=1e-8)


def test_calculate_volatility_multiplier_bounds_with_atr():
    rows = 200
    df = pd.DataFrame(
        {
            "close": np.random.uniform(100, 200, rows),
            "ATR_14": np.random.uniform(0.5, 5.0, rows),
        }
    )

    result = labeling._calculate_volatility_multiplier(df)
    assert len(result) == rows
    assert (result >= 1.5).all()
    assert (result <= 3.0).all()


def test_apply_directional_labels_empty_df():
    df = pd.DataFrame()
    result = labeling.apply_directional_labels(df, use_cache=False)

    assert "TargetLabel" in result.columns
    assert "Target" in result.columns
    assert "DynamicThreshold" in result.columns
    assert len(result) == 0

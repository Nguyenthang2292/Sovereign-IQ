
import numpy as np
import pandas as pd

from modules.common.indicators import calculate_kama
from modules.hmm.core.kama import HMM_KAMA, hmm_kama, prepare_observations
from modules.hmm.core.kama import HMM_KAMA, hmm_kama, prepare_observations




def _sample_close_dataframe(length: int = 150) -> pd.DataFrame:
    """Generate a simple increasing price series with hourly timestamps."""
    idx = pd.date_range("2024-01-01", periods=length, freq="h")
    prices = np.linspace(100.0, 120.0, length) + np.random.default_rng(42).normal(0, 0.5, length)
    return pd.DataFrame({"close": prices}, index=idx)


def test_calculate_kama_returns_finite_array():
    prices = np.linspace(1.0, 10.0, 40)
    kama = calculate_kama(prices, window=5, fast=2, slow=30)

    assert len(kama) == len(prices)
    assert np.isfinite(kama).all()


def test_prepare_observations_produces_three_features():
    df = _sample_close_dataframe(140)

    observations = prepare_observations(df, window_kama=12, fast_kama=2, slow_kama=35)

    assert observations.shape == (len(df), 3)
    assert np.isfinite(observations).all()


def test_hmm_kama_pipeline_returns_states():
    df = _sample_close_dataframe(160)

    result = hmm_kama(df, window_size=120)

    assert isinstance(result, HMM_KAMA)
    assert 0 <= result.next_state_with_hmm_kama <= 3

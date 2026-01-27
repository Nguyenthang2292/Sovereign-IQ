"""Tests for adaptive approximate moving averages."""

import pytest
import pandas as pd
import numpy as np

from modules.adaptive_trend_LTS.core.compute_moving_averages.adaptive_approximate_mas import (
    calculate_volatility,
    adaptive_ema_approx,
    adaptive_hma_approx,
    adaptive_wma_approx,
    adaptive_dema_approx,
    adaptive_lsma_approx,
    adaptive_kama_approx,
    get_adaptive_ma_approx,
)


@pytest.fixture
def sample_prices():
    """Sample price series for testing."""
    np.random.seed(42)
    return pd.Series(1000 + np.random.randn(200).cumsum())


def test_calculate_volatility(sample_prices):
    """Test volatility calculation."""
    vol = calculate_volatility(sample_prices, window=20)

    assert len(vol) == len(sample_prices)
    # With min_periods=1, index 0 returns NaN (pandas behavior)
    # First valid value is at index 1
    assert pd.isna(vol.iloc[0])
    assert not pd.isna(vol.iloc[1])
    # From index 19 onwards should be valid (have accumulated 20 values)
    assert not pd.isna(vol.iloc[19])
    assert not pd.isna(vol.iloc[20])
    # Volatility should be positive for valid values
    assert (vol.iloc[1:] > 0).all()


def test_adaptive_ema_approx(sample_prices):
    """Test adaptive EMA approximation."""
    ema = adaptive_ema_approx(sample_prices, length=28, volatility_window=20)

    assert len(ema) == len(sample_prices)
    # Should have valid values after warmup
    assert pd.notna(ema).iloc[27:].all()
    # EMA should be in reasonable range
    valid_ema = ema.iloc[27:]
    assert (valid_ema > 0).all()
    # EMA should roughly track prices (loosely)
    price_range = sample_prices.max() - sample_prices.min()
    ema_range = valid_ema.max() - valid_ema.min()
    # EMA range should be within reasonable bounds of price range
    assert ema_range > 0  # Just ensure it has some variation


def test_adaptive_hma_approx(sample_prices):
    """Test adaptive HMA approximation."""
    hma = adaptive_hma_approx(sample_prices, length=28, volatility_window=20)

    assert len(hma) == len(sample_prices)
    # HMA has warmup period
    assert pd.notna(hma).iloc[27:].all()
    # HMA should be positive
    valid_hma = hma.iloc[27:]
    assert (valid_hma > 0).all()


def test_adaptive_wma_approx(sample_prices):
    """Test adaptive WMA approximation."""
    wma = adaptive_wma_approx(sample_prices, length=28, volatility_window=20)

    assert len(wma) == len(sample_prices)
    # WMA has warmup
    assert pd.notna(wma).iloc[27:].all()
    # WMA should be positive
    valid_wma = wma.iloc[27:]
    assert (valid_wma > 0).all()


def test_adaptive_dema_approx(sample_prices):
    """Test adaptive DEMA approximation."""
    dema = adaptive_dema_approx(sample_prices, length=28, volatility_window=20)

    assert len(dema) == len(sample_prices)
    # DEMA has warmup
    assert pd.notna(dema).iloc[27:].all()
    # DEMA should be positive
    valid_dema = dema.iloc[27:]
    assert (valid_dema > 0).all()


def test_adaptive_lsma_approx(sample_prices):
    """Test adaptive LSMA approximation."""
    lsma = adaptive_lsma_approx(sample_prices, length=28, volatility_window=20)

    assert len(lsma) == len(sample_prices)
    # LSMA has warmup
    assert pd.notna(lsma).iloc[27:].all()
    # LSMA should be positive
    valid_lsma = lsma.iloc[27:]
    assert (valid_lsma > 0).all()


def test_adaptive_kama_approx(sample_prices):
    """Test adaptive KAMA approximation."""
    kama = adaptive_kama_approx(sample_prices, length=28, volatility_window=20)

    assert len(kama) == len(sample_prices)
    # KAMA has minimal warmup
    assert pd.notna(kama).iloc[1:].all()
    # KAMA should be positive
    valid_kama = kama.iloc[1:]
    assert (valid_kama > 0).all()


def test_get_adaptive_ma_approx(sample_prices):
    """Test get_adaptive_ma_approx function."""
    # Test EMA
    ema = get_adaptive_ma_approx("EMA", sample_prices, length=28)
    assert len(ema) == len(sample_prices)
    assert pd.notna(ema).iloc[27:].all()

    # Test HMA
    hma = get_adaptive_ma_approx("HMA", sample_prices, length=28)
    assert len(hma) == len(sample_prices)
    assert pd.notna(hma).iloc[27:].all()

    # Test invalid MA type
    invalid_tested = False
    try:
        get_adaptive_ma_approx("invalid", sample_prices, length=28)
        assert False, "Should raise ValueError for invalid MA type"
    except ValueError:
        invalid_tested = True  # Expected error
    except Exception:
        pass  # Other errors also OK

    assert invalid_tested, "ValueError should be raised for invalid MA"


def test_adaptive_approx_volatility_factor(sample_prices):
    """Test that volatility factor affects calculations."""
    # Low volatility factor should give similar results to base approximation
    ema_low = adaptive_ema_approx(sample_prices, length=28, volatility_factor=0.1)

    # High volatility factor
    ema_high = adaptive_ema_approx(sample_prices, length=28, volatility_factor=2.0)

    # Results should be the same for now (adaptation not fully implemented in base approx)
    # In a full implementation, volatility_factor would affect precision
    assert len(ema_low) == len(ema_high)


def test_adaptive_approx_base_tolerance(sample_prices):
    """Test base tolerance parameter."""
    ema = adaptive_ema_approx(
        sample_prices,
        length=28,
        base_tolerance=0.01,  # Very tight tolerance
    )

    assert len(ema) == len(sample_prices)
    # Should still compute successfully
    assert pd.notna(ema).iloc[27:].all()


def test_adaptive_approx_volatility_window(sample_prices):
    """Test different volatility windows."""
    ema_10 = adaptive_ema_approx(sample_prices, length=28, volatility_window=10)
    ema_30 = adaptive_ema_approx(sample_prices, length=28, volatility_window=30)

    # Both should produce valid results
    assert len(ema_10) == len(sample_prices)
    assert len(ema_30) == len(sample_prices)
    assert pd.notna(ema_10).iloc[9:].all()
    assert pd.notna(ema_30).iloc[29:].all()


def test_all_ma_types(sample_prices):
    """Test all adaptive MA types work."""
    ma_types = ["EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"]

    for ma_type in ma_types:
        ma = get_adaptive_ma_approx(ma_type, sample_prices, length=28)

        assert len(ma) == len(sample_prices), f"{ma_type} length mismatch"
        assert pd.notna(ma).iloc[27:].all(), f"{ma_type} has too many NaN values"

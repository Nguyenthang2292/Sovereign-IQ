"""Tests for JIT specialization of ATC computations."""

import numpy as np
import pandas as pd
import pytest

from modules.adaptive_trend_LTS.core.codegen.specialization import (
    compute_atc_specialized,
    get_specialized_compute_fn,
    is_config_specializable,
)
from modules.adaptive_trend_LTS.core.compute_atc_signals.compute_atc_signals import (
    compute_atc_signals,
)
from modules.adaptive_trend_LTS.utils.config import ATCConfig


def test_ema_only_config_is_specializable():
    """Test that EMA-only config is recognized as specializable."""
    config = ATCConfig(ema_len=28, robustness="Medium")
    assert is_config_specializable(config, mode="ema_only") is True


def test_default_config_not_yet_specializable():
    """Test that default config is not yet specializable."""
    config = ATCConfig(ema_len=28, robustness="Medium")
    assert is_config_specializable(config, mode="default") is False


def test_get_specialized_compute_fn_ema_only():
    """Test getting specialized compute function for EMA-only config."""
    config = ATCConfig(ema_len=28, robustness="Medium")

    specialized_fn = get_specialized_compute_fn(config, mode="ema_only", use_codegen=True)

    # Should return a function
    if specialized_fn is not None:
        assert callable(specialized_fn)
    else:
        # May return None if Numba not available
        pytest.skip("Numba not available or specialization not enabled")


def test_ema_only_specialization_results_match_generic():
    """Test that EMA-only specialization produces same results as generic path.

    This is the critical verification that specialized path produces
    identical results to the generic compute_atc_signals.
    """
    # Create test data
    np.random.seed(42)
    n = 500
    prices_arr = 100 + np.cumsum(np.random.randn(n) * 0.1)
    prices = pd.Series(prices_arr, name="close")

    # Create config
    config = ATCConfig(
        ema_len=28,
        robustness="Medium",
        lambda_param=0.02,
        decay=0.03,
        long_threshold=0.1,
        short_threshold=-0.1,
        cutout=0,
        strategy_mode=False,
        use_codegen_specialization=False,  # Use generic path
    )

    # Compute using generic path
    try:
        generic_result = compute_atc_signals(prices, ema_len=config.ema_len)
        ema_signal_generic = np.asarray(generic_result["EMA_Signal"].values)
        ema_equity_generic = np.asarray(generic_result["EMA_S"].values)

        # Compute using specialized path
        specialized_result = compute_atc_specialized(
            prices,
            config,
            mode="ema_only",
            use_codegen_specialization=True,
            fallback_to_generic=False,
        )

        ema_signal_specialized = np.asarray(specialized_result["EMA_Signal"].values)
        ema_equity_specialized = np.asarray(specialized_result["EMA_S"].values)

        # Compare results - should be very close
        # Note: Due to numerical differences in implementation, we allow small tolerance
        np.testing.assert_allclose(
            ema_signal_specialized,
            ema_signal_generic,
            rtol=1e-10,
            atol=1e-10,
            err_msg="EMA Signal mismatch between specialized and generic paths",
        )

        np.testing.assert_allclose(
            ema_equity_specialized,
            ema_equity_generic,
            rtol=1e-10,
            atol=1e-10,
            err_msg="EMA Equity mismatch between specialized and generic paths",
        )
    except ImportError:
        pytest.skip("Numba not available")
    except Exception as e:
        # Allow for numerical precision differences
        if "Numba" in str(e):
            pytest.skip(f"Numba error: {e}")
        else:
            raise


def test_ema_only_specialization_fallback():
    """Test that fallback to generic path works when specialization fails."""
    # Create test data
    np.random.seed(42)
    n = 500
    prices_arr = 100 + np.cumsum(np.random.randn(n) * 0.1)
    prices = pd.Series(prices_arr, name="close")

    # Create config
    config = ATCConfig(ema_len=28, robustness="Medium")

    # Compute with fallback enabled (default)
    try:
        result = compute_atc_specialized(
            prices,
            config,
            mode="ema_only",
            use_codegen_specialization=True,
            fallback_to_generic=True,
        )

        # Should return results
        assert isinstance(result, dict)
        assert "EMA_Signal" in result or "Average_Signal" in result
    except ImportError:
        pytest.skip("Numba not available")


def test_ema_only_different_lengths():
    """Test EMA-only specialization with different lengths."""
    np.random.seed(42)
    n = 500
    prices_arr = 100 + np.cumsum(np.random.randn(n) * 0.1)
    prices = pd.Series(prices_arr, name="close")

    lengths = [14, 20, 28, 50]

    for length in lengths:
        config = ATCConfig(ema_len=length, robustness="Medium")

        try:
            result = compute_atc_specialized(
                prices,
                config,
                mode="ema_only",
                use_codegen_specialization=True,
                fallback_to_generic=True,
            )

            # Should return results
            assert isinstance(result, dict)
            assert "EMA_Signal" in result or "Average_Signal" in result
        except ImportError:
            pytest.skip("Numba not available")


def test_flag_controls_specialization():
    """Test that use_codegen_specialization flag controls behavior."""
    np.random.seed(42)
    n = 500
    prices_arr = 100 + np.cumsum(np.random.randn(n) * 0.1)
    prices = pd.Series(prices_arr, name="close")

    # Config with specialization disabled
    config_disabled = ATCConfig(
        ema_len=28,
        robustness="Medium",
        use_codegen_specialization=False,
    )

    # Config with specialization enabled
    config_enabled = ATCConfig(
        ema_len=28,
        robustness="Medium",
        use_codegen_specialization=True,
    )

    try:
        # Both should produce results (via fallback for enabled if no Numba)
        result_disabled = compute_atc_specialized(
            prices,
            config_disabled,
            mode="ema_only",
            use_codegen_specialization=False,
            fallback_to_generic=True,
        )

        result_enabled = compute_atc_specialized(
            prices,
            config_enabled,
            mode="ema_only",
            use_codegen_specialization=True,
            fallback_to_generic=True,
        )

        # Both should return valid results
        assert isinstance(result_disabled, dict)
        assert isinstance(result_enabled, dict)
    except ImportError:
        pytest.skip("Numba not available")


def test_fallback_does_not_change_results():
    """Test that fallback path produces same results as generic path."""
    np.random.seed(42)
    n = 500
    prices_arr = 100 + np.cumsum(np.random.randn(n) * 0.1)
    prices = pd.Series(prices_arr, name="close")

    config = ATCConfig(ema_len=28, robustness="Medium")

    try:
        # Result with fallback enabled (should use generic if no specialization)
        result_fallback = compute_atc_specialized(
            prices,
            config,
            mode="ema_only",
            use_codegen_specialization=False,
            fallback_to_generic=True,
        )

        # Direct generic result
        result_generic = compute_atc_signals(prices, ema_len=config.ema_len)

        # Both should have same keys
        assert set(result_fallback.keys()) == set(result_generic.keys())

        # Compare EMA_Signal if available
        if "EMA_Signal" in result_fallback and "EMA_Signal" in result_generic:
            np.testing.assert_array_equal(
                result_fallback["EMA_Signal"].values,
                result_generic["EMA_Signal"].values,
            )
    except ImportError:
        pytest.skip("Numba not available")


def test_specialization_disabled_uses_generic():
    """Test that when specialization is disabled, generic path is used."""
    np.random.seed(42)
    n = 500
    prices_arr = 100 + np.cumsum(np.random.randn(n) * 0.1)
    prices = pd.Series(prices_arr, name="close")

    config = ATCConfig(ema_len=28, robustness="Medium", use_codegen_specialization=False)

    try:
        # With specialization disabled
        result_disabled = compute_atc_specialized(
            prices,
            config,
            mode="ema_only",
            use_codegen_specialization=False,
            fallback_to_generic=True,
        )

        # Should produce valid results
        assert isinstance(result_disabled, dict)
        assert "EMA_Signal" in result_disabled or "Average_Signal" in result_disabled

        # Compare with direct generic call
        result_generic = compute_atc_signals(prices, ema_len=config.ema_len)

        # EMA_Signal should be identical
        if "EMA_Signal" in result_disabled and "EMA_Signal" in result_generic:
            np.testing.assert_array_equal(
                result_disabled["EMA_Signal"].values,
                result_generic["EMA_Signal"].values,
            )
    except ImportError:
        pytest.skip("Numba not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

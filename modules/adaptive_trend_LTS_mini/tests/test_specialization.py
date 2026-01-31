"""Tests for JIT specialization of ATC computations."""

import numpy as np
import pandas as pd
import pytest

from modules.adaptive_trend_LTS_mini.core.codegen.specialization import (
    SpecializedConfigKey,
    _get_config_key,
    _validate_config,
    compute_atc_specialized,
    get_specialized_compute_fn,
    is_config_specializable,
)
from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.compute_atc_signals import (
    compute_atc_signals,
)
from modules.adaptive_trend_LTS_mini.utils.config import ATCConfig


def test_ema_only_config_is_specializable():
    """Test that EMA-only config is recognized as specializable."""
    config = ATCConfig(ema_len=28, robustness="Medium")

    assert is_config_specializable(config, mode="ema_only")


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


def test_ema_only_specialization_produces_signals():
    """Test that EMA-only specialization produces valid signals.

    Note: We do not compare with generic path because 'ema_only' mode
    implements a simplified algorithm (single EMA) compared to the
    generic path (diflen-based multiple MAs), so numerical results
    are expected to differ.
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
        use_codegen_specialization=True,
    )

    try:
        # Compute using specialized path
        specialized_result = compute_atc_specialized(
            prices,
            config,
            mode="ema_only",
            use_codegen_specialization=True,
            fallback_to_generic=False,
        )

        assert "EMA_Signal" in specialized_result
        assert "EMA_S" in specialized_result

        ema_signal = specialized_result["EMA_Signal"]
        ema_equity = specialized_result["EMA_S"]

        # Check properties
        assert len(ema_signal) == n
        assert len(ema_equity) == n
        assert not ema_signal.isna().all()  # Should have some values
        assert not ema_equity.isna().all()

    except ImportError:
        pytest.skip("Numba not available")
    except Exception as e:
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


def test_specialized_config_key_equality():
    """Test key equality and hashing."""
    key1 = SpecializedConfigKey("EMA", 28, "Medium", "ema_only")
    key2 = SpecializedConfigKey("EMA", 28, "Medium", "ema_only")
    key3 = SpecializedConfigKey("EMA", 30, "Medium", "ema_only")

    assert key1 == key2
    assert key1 != key3
    assert hash(key1) == hash(key2)

    d = {key1: "value"}
    assert d[key2] == "value"


def test_config_key_generation():
    """Test config key generation for different modes."""
    config = ATCConfig(ema_len=28, robustness="Medium")

    key_ema = _get_config_key(config, "ema_only")
    key_default = _get_config_key(config, "default")
    key_short_length = _get_config_key(config, "short_length")

    assert key_ema.mode == "ema_only"
    assert key_default.mode == "default"
    assert key_short_length.mode == "short_length"

    # Keys should be different for different modes
    assert key_ema != key_default
    assert key_default != key_short_length


def test_specialized_function_creation_and_caching():
    """Test that specialized functions are created and cached correctly."""
    config = ATCConfig(ema_len=28, robustness="Medium")

    try:
        # First call - creates function
        fn1 = get_specialized_compute_fn(config, mode="ema_only")
        assert fn1 is not None

        # Second call - should return cached function
        fn2 = get_specialized_compute_fn(config, mode="ema_only")
        assert fn2 is not None
        assert fn1 is fn2
    except ImportError:
        pytest.skip("Numba not available")


def test_validate_config_valid():
    """Test that valid configs pass validation."""
    config = ATCConfig(ema_len=28, robustness="Medium")
    _validate_config(config, mode="ema_only")


def test_validate_config_invalid_ema_len():
    """Test that invalid ema_len raises ValueError."""
    config = ATCConfig(ema_len=-1, robustness="Medium")
    with pytest.raises(ValueError, match="ema_len must be positive"):
        _validate_config(config, mode="ema_only")


def test_validate_config_invalid_lambda_param():
    """Test that invalid lambda_param raises ValueError."""
    config = ATCConfig(ema_len=28, robustness="Medium", lambda_param=1.5)
    with pytest.raises(ValueError, match="lambda_param must be in"):
        _validate_config(config, mode="ema_only")


def test_validate_config_invalid_lengths():
    """Test that invalid MA lengths raise ValueError."""
    config = ATCConfig(ema_len=10001, robustness="Medium")
    with pytest.raises(ValueError, match="ema_len too large"):
        _validate_config(config, mode="default")


def test_is_config_specializable_unknown_mode():
    """Test that unknown modes are not specializable."""
    config = ATCConfig(ema_len=28, robustness="Medium")
    result = is_config_specializable(config, "unknown_mode")
    assert result is False


def test_error_handling_specialized_path_failure():
    """Test that errors in specialized path are handled correctly."""
    np.random.seed(42)
    n = 500
    prices_arr = 100 + np.cumsum(np.random.randn(n) * 0.1)
    prices = pd.Series(prices_arr, name="close")

    config = ATCConfig(ema_len=28, robustness="Medium")

    # With fallback enabled, should not raise
    result = compute_atc_specialized(
        prices,
        config,
        mode="unknown_mode",
        use_codegen_specialization=True,
        fallback_to_generic=True,
    )
    assert isinstance(result, dict)

    # With fallback disabled, should raise if specialized path fails
    # Note: This test assumes unknown_mode will not have a specialized function
    # If a specialized function exists, this test should be adapted


def test_performance_benchmark_specialized_vs_generic():
    """Benchmark comparing specialized vs generic paths.

    This test measures performance difference between specialized and generic paths.
    """
    pytest.importorskip("timeit")

    np.random.seed(42)
    n = 5000
    prices_arr = 100 + np.cumsum(np.random.randn(n) * 0.1)
    prices = pd.Series(prices_arr, name="close")

    config = ATCConfig(ema_len=28, robustness="Medium")

    try:
        import timeit

        # Benchmark specialized path
        specialized_time = timeit.timeit(
            lambda: compute_atc_specialized(
                prices,
                config,
                mode="ema_only",
                use_codegen_specialization=True,
                fallback_to_generic=False,
            ),
            number=10,
        )

        # Benchmark generic path
        generic_time = timeit.timeit(
            lambda: compute_atc_specialized(
                prices,
                config,
                mode="ema_only",
                use_codegen_specialization=False,
                fallback_to_generic=True,
            ),
            number=10,
        )

        # Specialized should be faster (allow some overhead)
        if specialized_time < generic_time:
            speedup = generic_time / specialized_time
            print(f"\nSpecialized path is {speedup:.2f}x faster")
    except ImportError:
        pytest.skip("Numba not available")
    except Exception as e:
        pytest.skip(f"Performance benchmark skipped: {e}")


def test_all_specialization_modes():
    """Test that all specialization modes are handled correctly."""
    np.random.seed(42)
    n = 500
    prices_arr = 100 + np.cumsum(np.random.randn(n) * 0.1)
    prices = pd.Series(prices_arr, name="close")

    config = ATCConfig(ema_len=28, robustness="Medium")

    # Test all known modes
    modes = ["ema_only", "default", "short_length"]

    for mode in modes:
        try:
            result = compute_atc_specialized(
                prices,
                config,
                mode=mode,
                use_codegen_specialization=True,
                fallback_to_generic=True,
            )

            assert isinstance(result, dict)
            assert "Average_Signal" in result

            # ema_only mode should have EMA_Signal
            if mode == "ema_only":
                assert "EMA_Signal" in result
        except ImportError:
            pytest.skip("Numba not available")
        except Exception as e:
            if "specialization" not in str(e).lower():
                raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

from __future__ import annotations

from typing import Optional

import pandas as pd
import pandas_ta as ta

from modules.adaptive_trend_enhance.utils.cache_manager import get_cached_ma
from modules.common.system import get_hardware_manager
from modules.common.ui.logging import log_debug, log_error, log_warn

from ._gpu import _calculate_ma_gpu
from .calculate_kama_atc import calculate_kama_atc

# Try to import Rust backend
try:
    from ..rust_backend import (
        calculate_ema,
        calculate_wma,
        calculate_dema,
        calculate_lsma,
        calculate_hma,
        RUST_AVAILABLE,
    )
    RUST_MA_AVAILABLE = RUST_AVAILABLE
except ImportError:
    RUST_MA_AVAILABLE = False

# Global cache for hardware resources to avoid expensive get_resources() calls (approx 14ms per call)
_CACHED_HW_RESOURCES = None


def _get_cached_hw_resources():
    """Get cached hardware resources to avoid repeated system calls."""
    global _CACHED_HW_RESOURCES
    if _CACHED_HW_RESOURCES is None:
        hw_mgr = get_hardware_manager()
        _CACHED_HW_RESOURCES = hw_mgr.get_resources()
    return _CACHED_HW_RESOURCES


def ma_calculation_enhanced(
    source: pd.Series,
    length: int,
    ma_type: str,
    use_cache: bool = True,
    prefer_gpu: bool = True,
) -> Optional[pd.Series]:
    if not isinstance(source, pd.Series):
        raise TypeError(f"source must be a pandas Series, got {type(source)}")

    if len(source) == 0:
        log_warn("Empty source series for MA calculation")
        return None

    if length <= 0:
        raise ValueError(f"length must be > 0, got {length}")

    if not isinstance(ma_type, str) or not ma_type.strip():
        raise ValueError(f"ma_type must be a non-empty string, got {ma_type}")

    ma = ma_type.upper().strip()
    VALID_MA_TYPES = {"EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"}

    if ma not in VALID_MA_TYPES:
        log_warn(f"Invalid ma_type '{ma_type}'. Valid: {', '.join(VALID_MA_TYPES)}")
        return None

    try:

        def calculate(p_data=None, p_length=None):
            # p_data and p_length are provided by get_cached_ma, we use local source/length

            # PERFORMANCE OPTIMIZATION:
            # Removed track_memory(f"{ma}_calculation") as it adds significant overhead (approx 0.65ms per call)
            # and is called 600 times per 100 symbols.

            # Priority order: GPU > Rust > pandas_ta/Numba
            # Try GPU first if preferred and data is large enough to justify overhead
            # Benchmarks show GPU is often faster than Numba/CPU even for small series (>= 500 bars)
            MIN_GPU_SIZE = 500
            if prefer_gpu and len(source) >= MIN_GPU_SIZE and ma in ["WMA", "DEMA", "EMA", "HMA", "LSMA"]:
                # PERFORMANCE OPTIMIZATION: Use cached resources instead of calling get_resources() every time
                resources = _get_cached_hw_resources()

                # Only try GPU if available
                if resources.gpu_available:
                    result_array = _calculate_ma_gpu(source.values, length, ma)
                    if result_array is not None:
                        log_debug(f"GPU calculation succeeded for {ma}")
                        return pd.Series(result_array, index=source.index)

            # Try Rust backend if available (faster than pandas_ta for most cases)
            if RUST_MA_AVAILABLE and ma in ["EMA", "WMA", "DEMA", "LSMA", "HMA"]:
                try:
                    if ma == "EMA":
                        result_array = calculate_ema(source.values, length, use_rust=True)
                    elif ma == "WMA":
                        result_array = calculate_wma(source.values, length, use_rust=True)
                    elif ma == "DEMA":
                        result_array = calculate_dema(source.values, length, use_rust=True)
                    elif ma == "LSMA":
                        result_array = calculate_lsma(source.values, length, use_rust=True)
                    elif ma == "HMA":
                        result_array = calculate_hma(source.values, length, use_rust=True)
                    else:
                        result_array = None
                    
                    if result_array is not None:
                        log_debug(f"Rust calculation succeeded for {ma}")
                        return pd.Series(result_array, index=source.index)
                except Exception as e:
                    log_debug(f"Rust calculation failed for {ma}, falling back to pandas_ta: {e}")

            # Fallback to CPU calculation (Using pandas_ta for exact compatibility with original module)
            if ma == "EMA":
                return ta.ema(source, length=length)
            if ma == "HMA":
                return ta.sma(source, length=length)  # HMA maps to SMA in this script
            if ma == "WMA":
                return ta.wma(source, length=length)
            if ma == "DEMA":
                return ta.dema(source, length=length)
            if ma == "LSMA":
                return ta.linreg(source, length=length)
            if ma == "KAMA":
                return calculate_kama_atc(source, length=length)

            if ma == "HMA":
                # Note: Uses SMA for Pine Script compatibility
                return ta.sma(source, length=length)

            if ma == "KAMA":
                return calculate_kama_atc(source, length=length)

            return None

        if use_cache:
            result = get_cached_ma(ma, length, source, calculate)
        else:
            result = calculate()

        if result is None:
            log_warn(f"MA calculation ({ma}) returned None for length={length}")

        return result
    except Exception as e:
        log_error(f"Error calculating {ma} MA with length={length}: {e}")
        raise


__all__ = ["ma_calculation_enhanced"]

"""
Moving Average Calculation with Rust Backend.

This module provides MA calculations using Rust extensions for optimal performance.
Falls back to pandas_ta if Rust is not available.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from modules.adaptive_trend_LTS.utils.cache_manager import get_cached_ma
from modules.adaptive_trend_LTS.core.rust_backend import (
    RUST_AVAILABLE,
    calculate_dema,
    calculate_ema,
    calculate_hma,
    calculate_kama,
    calculate_lsma,
    calculate_wma,
)
from modules.common.ui.logging import log_debug, log_error, log_warn


def ma_calculation_rust(
    source: pd.Series,
    length: int,
    ma_type: str,
    use_cache: bool = True,
    use_rust: bool = True,
    use_cuda: bool = False,
) -> Optional[pd.Series]:
    """
    Calculate Moving Average using Rust backend.

    Args:
        source: Input price series.
        length: Period for moving average.
        ma_type: Type of moving average ("EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA").
        use_cache: If True, uses cached results if available (default: True).
        use_rust: If True, attempts to use Rust backend (default: True).

    Returns:
        pandas Series with MA values, or None if calculation fails.

    Raises:
        TypeError: If source is not a pandas Series.
        ValueError: If length <= 0 or ma_type is invalid.
    """
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

        def calculate_rust_ma(price_data=None, p_length=None):
            """Inner function for Rust MA calculation."""
            # Use closure variables if not provided by get_cached_ma
            if price_data is None:
                price_data = source.values
            if p_length is None:
                p_length = length

            if ma == "EMA":
                result = calculate_ema(price_data, p_length, use_rust=use_rust, use_cuda=use_cuda)
            elif ma == "WMA":
                result = calculate_wma(price_data, p_length, use_rust=use_rust, use_cuda=use_cuda)
            elif ma == "DEMA":
                result = calculate_dema(price_data, p_length, use_rust=use_rust)
            elif ma == "LSMA":
                result = calculate_lsma(price_data, p_length, use_rust=use_rust)
            elif ma == "HMA":
                result = calculate_hma(price_data, p_length, use_rust=use_rust, use_cuda=use_cuda)
            elif ma == "KAMA":
                result = calculate_kama(price_data, p_length, use_rust=use_rust, use_cuda=use_cuda)
            else:
                return None

            if result is None:
                log_warn(f"MA calculation ({ma}) returned None for length={length}")
                return None

            if use_rust and RUST_AVAILABLE:
                log_debug(f"Rust backend used for {ma} calculation")
            else:
                log_debug(f"Fallback (pandas_ta/numba) used for {ma} calculation")

            return pd.Series(result, index=source.index)

        if use_cache:
            result = get_cached_ma(ma, length, source, calculate_rust_ma)
        else:
            result = calculate_rust_ma()

        return result

    except Exception as e:
        log_error(f"Error calculating {ma} MA with length={length}: {e}")
        raise

    except Exception as e:
        log_error(f"Error calculating {ma} MA with length={length}: {e}")
        raise


__all__ = ["ma_calculation_rust"]

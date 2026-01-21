from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import pandas_ta as ta

from modules.adaptive_trend_enhance.utils.cache_manager import get_cached_ma
from modules.common.system import get_hardware_manager, track_memory
from modules.common.utils import log_error, log_warn

from ._gpu import _calculate_ma_gpu
from ._numba_cores import _calculate_dema_core, _calculate_lsma_core, _calculate_wma_core
from .calculate_kama_atc import calculate_kama_atc

logger = logging.getLogger(__name__)


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
            with track_memory(f"{ma}_calculation"):
                # Try GPU first if preferred
                if prefer_gpu and ma in ["WMA", "DEMA", "EMA"]:
                    hw_mgr = get_hardware_manager()
                    if hw_mgr.get_resources().gpu_available:
                        result_array = _calculate_ma_gpu(source.values, length, ma)
                        if result_array is not None:
                            logger.debug(f"GPU calculation succeeded for {ma}")
                            return pd.Series(result_array, index=source.index)

                # CPU calculation (Numba JIT or pandas_ta)
                if ma == "WMA":
                    result_array = _calculate_wma_core(source.values.astype("float64"), length)
                    return pd.Series(result_array, index=source.index)

                if ma == "DEMA":
                    result_array = _calculate_dema_core(source.values.astype("float64"), length)
                    return pd.Series(result_array, index=source.index)

                if ma == "LSMA":
                    result_array = _calculate_lsma_core(source.values.astype("float64"), length)
                    return pd.Series(result_array, index=source.index)

                if ma == "EMA":
                    return ta.ema(source, length=length)

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

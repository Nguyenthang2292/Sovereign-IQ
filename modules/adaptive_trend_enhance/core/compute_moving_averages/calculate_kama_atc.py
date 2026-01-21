from __future__ import annotations

from typing import Optional

import pandas as pd

from modules.common.system import track_memory
from modules.common.utils import log_error, log_warn

from ._numba_cores import _calculate_kama_atc_core


def calculate_kama_atc(
    prices: pd.Series,
    length: int = 28,
) -> Optional[pd.Series]:
    if not isinstance(prices, pd.Series):
        raise TypeError(f"prices must be a pandas Series, got {type(prices)}")

    if len(prices) == 0:
        log_warn("Empty prices series provided for KAMA calculation")
        return None

    if length <= 0:
        raise ValueError(f"length must be > 0, got {length}")

    try:
        with track_memory("KAMA_calculation"):
            prices_array = prices.values.astype("float64")
            kama_array = _calculate_kama_atc_core(prices_array, length)
            return pd.Series(kama_array, index=prices.index)
    except Exception as e:
        log_error(f"Error calculating KAMA: {e}")
        raise


__all__ = ["calculate_kama_atc"]

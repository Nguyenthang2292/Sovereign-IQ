import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple

from ..models.pivot import Pivot
from .trend import compute_atr
from .swing import _detect_swing_pivots
from modules.common.ui.logging import log_warn


@dataclass
class EqualHLResult:
    equal_highs: List[Tuple[Pivot, Pivot]]
    equal_lows: List[Tuple[Pivot, Pivot]]


def identify_equal_hl(
    df: pd.DataFrame,
    highs_arr: np.ndarray,
    lows_arr: np.ndarray,
    closes_arr: np.ndarray,
    equal_high_low_threshold: float = 0.1,
    size: int = 1,
    equal_length: int = 3,
) -> EqualHLResult:
    """
    Identify Equal High and Equal Low groups.
    """
    atr = compute_atr(highs_arr, lows_arr, closes_arr, period=200)
    if atr is None:
        log_warn("Unable to compute ATR, returning two empty lists.")
        return EqualHLResult([], [])

    threshold_value = equal_high_low_threshold * atr

    internal_highs, internal_lows = _detect_swing_pivots(df, equal_length, is_internal=True)

    equal_high_groups = []
    # Iterate over swing_highs, compare pivot[i] with pivot[i+size]
    for i in range(len(internal_highs) - size):
        current = internal_highs[i]
        compare_with = internal_highs[i + size]
        if abs(current.level - compare_with.level) < threshold_value:
            equal_high_groups.append((current, compare_with))

    equal_low_groups = []
    # Iterate over swing_lows, compare pivot[i] with pivot[i+size]
    for i in range(len(internal_lows) - size):
        current = internal_lows[i]
        compare_with = internal_lows[i + size]
        if abs(current.level - compare_with.level) < threshold_value:
            equal_low_groups.append((current, compare_with))

    return EqualHLResult(equal_high_groups, equal_low_groups)

"""Signal generation from Moving Average cross events.

Generates discrete trading signals from price/MA crossover events and persists
the latest signal state until an opposite event occurs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from modules.common.utils import log_error, log_warn

from .crossover import crossover
from .crossunder import crossunder

def generate_signal_from_ma(
    price: pd.Series,
    ma: pd.Series,
) -> pd.Series:
    """Generate discrete trading signals from price/MA crossover events.

    Behavior matches `modules/adaptive_trend`:
    1. Detect crossover/crossunder events.
    2. Emit +1/-1 on event bars.
    3. Persist latest non-zero state forward until opposite event.

    Args:
        price: Price series (typically close prices).
        ma: Moving Average series.

    Returns:
        Series with discrete signal values:
        - 1: Bullish state
        - -1: Bearish state
        - 0: Neutral state before first event

    Raises:
        ValueError: If price or ma are empty or have incompatible indices.
        TypeError: If inputs are not pandas Series.
    """
    if not isinstance(price, pd.Series):
        raise TypeError(f"price must be a pandas Series, got {type(price)}")

    if not isinstance(ma, pd.Series):
        raise TypeError(f"ma must be a pandas Series, got {type(ma)}")

    if len(price) == 0 or len(ma) == 0:
        log_warn("Empty price or MA series provided, returning empty signal series")
        return pd.Series(dtype="int8", index=price.index if len(price) > 0 else ma.index)

    try:
        # Align indices if needed
        if not price.index.equals(ma.index):
            log_warn("price and ma have different indices. Aligning to common indices.")
            common_index = price.index.intersection(ma.index)
            if len(common_index) == 0:
                log_warn("No common indices found between price and ma")
                return pd.Series(dtype="int8", index=price.index)
            price = price.loc[common_index]
            ma = ma.loc[common_index]

        # Check for excessive NaN values
        price_nan_count = price.isna().sum()
        ma_nan_count = ma.isna().sum()
        total_bars = len(price)

        if price_nan_count > 0:
            nan_pct = (price_nan_count / total_bars) * 100
            if nan_pct > 10:
                log_warn(
                    f"Price series contains {price_nan_count} NaN values ({nan_pct:.1f}%). "
                    f"This may affect signal generation."
                )

        if ma_nan_count > 0:
            nan_pct = (ma_nan_count / total_bars) * 100
            if nan_pct > 10:
                log_warn(
                    f"MA series contains {ma_nan_count} NaN values ({nan_pct:.1f}%). This may affect signal generation."
                )

        sig = pd.Series(0, index=price.index, dtype="int8")
        up = crossover(price, ma)
        down = crossunder(price, ma)

        sig.loc[up] = 1
        sig.loc[down] = -1

        conflict_mask = up & down
        if conflict_mask.any():
            sig.loc[conflict_mask] = 0

        # Persist last non-zero signal state, matching source-of-truth behavior.
        sig = sig.replace(0, np.nan).ffill().fillna(0).astype("int8")
        return sig

    except Exception as e:
        log_error(f"Error generating signal from MA: {e}")
        raise

"""Execution-layer helpers for strategy shift.

Core ATC computation must output raw causal `Average_Signal`.
Execution/backtest consumers can derive shifted output via this module.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def apply_execution_shift_series(raw_signal: pd.Series, fill_value: float = 0.0) -> pd.Series:
    """Return execution-view signal from raw signal.

    Formula:
        Average_Signal_Exec[t] = Average_Signal[t-1]
    """
    return raw_signal.shift(1).fillna(fill_value)


def apply_execution_shift_value(previous_raw: Optional[float], fill_value: float = 0.0) -> float:
    """Return execution-view scalar from previous raw value."""
    if previous_raw is None:
        return float(fill_value)

    value = float(previous_raw)
    if not np.isfinite(value):
        return float(fill_value)
    return value


__all__ = ["apply_execution_shift_series", "apply_execution_shift_value"]

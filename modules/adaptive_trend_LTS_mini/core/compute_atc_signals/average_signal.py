"""Average signal calculation for ATC final output.

Implements source-of-truth aggregation semantics from `modules/adaptive_trend`:
`Average_Signal = sum(cut_signal(L1) * Layer2_Equity) / sum(Layer2_Equity)`.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from modules.adaptive_trend_LTS_mini.core.process_layer1 import cut_signal

try:
    from modules.common.utils import log_debug, log_warn
except ImportError:
    def log_debug(msg: str, *args: object) -> None:  # pragma: no cover
        print(f"[DEBUG] {msg}")

    def log_warn(msg: str, *args: object) -> None:  # pragma: no cover
        print(f"[WARN] {msg}")


def calculate_average_signal(
    layer1_signals: Dict[str, pd.Series],
    layer2_equities: Dict[str, pd.Series],
    ma_configs: List[Tuple[str, int, float]],
    prices: pd.Series,
    long_threshold: float,
    short_threshold: float,
    cutout: int = 0,
    strategy_mode: bool = False,
    precision: str = "float64",
) -> pd.Series:
    """Calculate final raw `Average_Signal` from Layer1 signals and Layer2 equities.

    Note:
        `strategy_mode` is kept only for backward-compatible call signatures.
        Core aggregation no longer applies execution shift. Use adapter helpers
        to derive `Average_Signal_Exec` when needed.
    """
    log_debug("Computing Average_Signal (source parity)...")

    n_bars = len(prices)
    index = prices.index

    if cutout < 0:
        raise ValueError(f"cutout must be >= 0, got {cutout}")
    if cutout >= n_bars:
        log_warn(f"cutout={cutout} >= n_bars={n_bars}, returning zeros")
        return pd.Series(0.0, index=index, dtype="float64")

    nom_array = np.zeros(n_bars, dtype=np.float64)
    den_array = np.zeros(n_bars, dtype=np.float64)

    for ma_type, _, _ in ma_configs:
        if ma_type not in layer1_signals or ma_type not in layer2_equities:
            continue

        signal = layer1_signals[ma_type]
        equity = layer2_equities[ma_type]

        cut_sig = cut_signal(
            signal,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            cutout=cutout,
        )

        cut_sig_values = cut_sig.values
        equity_values = equity.values
        nom_array += cut_sig_values * equity_values
        den_array += equity_values

    with np.errstate(divide="ignore", invalid="ignore"):
        avg_signal_array = np.divide(nom_array, den_array)
        avg_signal_array = np.where(np.isfinite(avg_signal_array), avg_signal_array, 0.0)

    average_signal = pd.Series(avg_signal_array, index=index, dtype="float64")

    if strategy_mode:
        log_warn(
            "strategy_mode no longer shifts core Average_Signal. "
            "Use execution adapter output `Average_Signal_Exec` instead."
        )

    log_debug("Completed Average_Signal")
    return average_signal


__all__ = ["calculate_average_signal"]

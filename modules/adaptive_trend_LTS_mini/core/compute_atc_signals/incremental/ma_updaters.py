"""MA (Moving Average) updaters for incremental ATC computation.

This module contains all MA update implementations:
- EMA (Exponential Moving Average)
- WMA (Weighted Moving Average)
- HMA (Hull Moving Average)
- DEMA (Double Exponential Moving Average)
- LSMA (Least Squares Moving Average)
- KAMA (Kaufman Adaptive Moving Average)
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional

import numpy as np


def update_ema(
    state: Dict[str, Any],
    new_price: float,
    length: int,
    robustness: str,
    prev_emas: Optional[list] = None,
) -> list:
    """Update EMA incrementally for all 9 variations.

    Args:
        state: Current ATC state dictionary
        new_price: New price value
        length: Base EMA length
        robustness: Robustness setting ("Narrow", "Medium", "Wide")
        prev_emas: Previous EMA values for all 9 variations

    Returns:
        List of new EMA values for all 9 variations
    """
    from modules.adaptive_trend_LTS_mini.utils.diflen import diflen

    # Get 8 offset lengths based on robustness (strict_mode=True by default, so never None)
    _d = diflen(length, robustness=robustness)
    assert _d is not None
    L1, L2, L3, L4, L_1, L_2, L_3, L_4 = _d

    # All 9 lengths: base + 8 offsets
    lengths = [length, L1, L2, L3, L4, L_1, L_2, L_3, L_4]

    # Get previous EMAs for all 9 variations (or use new_price as initial)
    if not isinstance(prev_emas, list) or len(prev_emas) != 9:
        prev_emas = state["ma_values"].get("ema")
    if not isinstance(prev_emas, list) or len(prev_emas) != 9:
        prev_emas = [new_price] * 9

    # Calculate new EMA for each variation
    new_emas = []
    for i, ln in enumerate(lengths):
        alpha = 2.0 / (ln + 1.0)
        new_ema = alpha * new_price + (1 - alpha) * prev_emas[i]
        new_emas.append(new_ema)

    state["ma_values"]["ema"] = new_emas
    return new_emas


def update_wma(
    state: Dict[str, Any],
    new_price: float,
    length: int,
    robustness: str,
    o1_mas: Dict[str, Any],
    use_o1_mas: bool,
    ma_key: str = "wma",
) -> None:
    """Update WMA incrementally for all 9 variations.

    Args:
        state: Current ATC state dictionary
        new_price: New price value
        length: Base WMA length
        robustness: Robustness setting
        o1_mas: Dictionary of O(1) MA objects
        use_o1_mas: Whether to use O(1) MA implementations
        ma_key: Key for storing WMA values in state (default: "wma")
    """
    from modules.adaptive_trend_LTS_mini.utils.diflen import diflen

    # If using O(1) MAs for primary wma, we still need to calculate variations separately
    # O(1) implementation only handles the primary length
    _d = diflen(length, robustness=robustness)
    assert _d is not None
    L1, L2, L3, L4, L_1, L_2, L_3, L_4 = _d
    lengths = [length, L1, L2, L3, L4, L_1, L_2, L_3, L_4]

    prices = list(state["price_history"])

    # Get previous WMAs for all 9 variations
    prev_wmas = state["ma_values"].get(ma_key)
    if not isinstance(prev_wmas, list) or len(prev_wmas) != 9:
        prev_wmas = [new_price] * 9

    new_wmas = []
    for i, ln in enumerate(lengths):
        if len(prices) < ln:
            new_wmas.append(new_price)
            continue

        window = prices[-ln:]
        weights = np.arange(1, ln + 1)
        wma = np.dot(window, weights) / weights.sum()
        new_wmas.append(wma)

    state["ma_values"][ma_key] = new_wmas

    # Update O(1) MA if configured (for primary length only)
    if use_o1_mas and ma_key == "wma" and ma_key in o1_mas:
        # O(1) MA update happens in parallel but we use calculated values above
        o1_mas[ma_key].update(new_price)


def update_hma(state: Dict[str, Any], new_price: float, length: int, robustness: str) -> None:
    """Update HMA incrementally for all 9 variations.

    Args:
        state: Current ATC state dictionary
        new_price: New price value
        length: Base HMA length
        robustness: Robustness setting
    """
    from modules.adaptive_trend_LTS_mini.utils.diflen import diflen

    _d = diflen(length, robustness=robustness)
    assert _d is not None
    L1, L2, L3, L4, L_1, L_2, L_3, L_4 = _d
    lengths = [length, L1, L2, L3, L4, L_1, L_2, L_3, L_4]

    sqrt_lengths = [max(1, int(np.sqrt(ln))) for ln in lengths]
    half_lengths = [max(1, ln // 2) for ln in lengths]

    # Get previous HMAs
    prev_hmas = state["ma_values"].get("hma")
    if not isinstance(prev_hmas, list) or len(prev_hmas) != 9:
        prev_hmas = [new_price] * 9

    new_hmas = []
    for i, ln in enumerate(lengths):
        half_len = half_lengths[i]
        sqrt_len = sqrt_lengths[i]

        # Calculate WMA of half length
        prices = list(state["price_history"])
        if len(prices) < half_len:
            wma_half = new_price
        else:
            window = prices[-half_len:]
            weights = np.arange(1, half_len + 1)
            wma_half = np.dot(window, weights) / weights.sum()

        # Calculate WMA of full length
        if len(prices) < ln:
            wma_full = new_price
        else:
            window = prices[-ln:]
            weights = np.arange(1, ln + 1)
            wma_full = np.dot(window, weights) / weights.sum()

        hma_input_val = 2 * wma_half - wma_full

        # Store hma_input for each variation in separate history
        hma_hist_key = f"hma_input_history_{i}"
        if hma_hist_key not in state:
            state[hma_hist_key] = deque(maxlen=sqrt_len)
        state[hma_hist_key].append(hma_input_val)

        if len(state[hma_hist_key]) >= sqrt_len:
            weights = np.arange(1, sqrt_len + 1)
            hma = np.dot(list(state[hma_hist_key]), weights) / weights.sum()
        else:
            hma = hma_input_val

        new_hmas.append(hma)

    state["ma_values"]["hma"] = new_hmas


def update_dema(
    state: Dict[str, Any],
    new_price: float,
    length: int,
    robustness: str,
    prev_emas: Optional[list] = None,
    new_emas: Optional[list] = None,
) -> None:
    """Update DEMA incrementally for all 9 variations.

    Args:
        state: Current ATC state dictionary
        new_price: New price value
        length: Base DEMA length
        robustness: Robustness setting
        prev_emas: Previous EMA values for all 9 variations
        new_emas: New EMA values for all 9 variations (already calculated)
    """
    from modules.adaptive_trend_LTS_mini.utils.diflen import diflen

    _d = diflen(length, robustness=robustness)
    assert _d is not None
    L1, L2, L3, L4, L_1, L_2, L_3, L_4 = _d
    lengths = [length, L1, L2, L3, L4, L_1, L_2, L_3, L_4]

    # Get previous EMAs and EMA2s for all variations
    if not isinstance(prev_emas, list) or len(prev_emas) != 9:
        prev_emas = state["ma_values"].get("ema")
    if not isinstance(prev_emas, list) or len(prev_emas) != 9:
        prev_emas = [new_price] * 9

    prev_ema2s = state["ema2_values"].get("dema")
    if not isinstance(prev_ema2s, list) or len(prev_ema2s) != 9:
        prev_ema2s = prev_emas.copy()

    if not isinstance(new_emas, list) or len(new_emas) != 9:
        new_emas = []
    new_ema2s = []
    new_demas = []

    for i, ln in enumerate(lengths):
        alpha = 2.0 / (ln + 1.0)

        # Update first EMA
        if new_emas:
            new_ema = new_emas[i]
        else:
            new_ema = alpha * new_price + (1 - alpha) * prev_emas[i]
            new_emas.append(new_ema)

        # Update second EMA (EMA of EMA)
        new_ema2 = alpha * new_ema + (1 - alpha) * prev_ema2s[i]
        new_ema2s.append(new_ema2)

        # Calculate DEMA = 2*EMA - EMA2
        dema = 2 * new_ema - new_ema2
        new_demas.append(dema)

    state["ma_values"]["ema"] = new_emas
    state["ema2_values"]["dema"] = new_ema2s
    state["ma_values"]["dema"] = new_demas


def update_lsma(state: Dict[str, Any], new_price: float, length: int, robustness: str) -> None:
    """Update LSMA incrementally for all 9 variations.

    FIX #6: Improved numerical stability with epsilon-based comparison.
    """
    from modules.adaptive_trend_LTS_mini.utils.diflen import diflen

    try:
        from modules.common.utils import log_warn
    except ImportError:

        def log_warn(msg: str) -> None:
            print(f"[WARN] {msg}")

    _d = diflen(length, robustness=robustness)
    assert _d is not None
    L1, L2, L3, L4, L_1, L_2, L_3, L_4 = _d
    lengths = [length, L1, L2, L3, L4, L_1, L_2, L_3, L_4]

    prices = list(state["price_history"])

    # Get previous LSMAs
    prev_lsmas = state["ma_values"].get("lsma")
    if not isinstance(prev_lsmas, list) or len(prev_lsmas) != 9:
        prev_lsmas = [new_price] * 9

    new_lsmas = []
    EPSILON = 1e-10  # FIX #6: Use epsilon for float comparison

    for i, ln in enumerate(lengths):
        if len(prices) < ln:
            new_lsmas.append(new_price)
            continue

        window = prices[-ln:]
        x = np.arange(ln)
        y = np.array(window)

        n = ln
        sum_x = n * (n - 1) / 2
        sum_x2 = n * (n - 1) * (2 * n - 1) / 6
        sum_y = np.sum(y)
        sum_xy = np.dot(x, y)

        denom = n * sum_x2 - sum_x**2

        # FIX #6: Use epsilon-based comparison instead of == 0
        if abs(denom) < EPSILON:
            log_warn(f"LSMA denominator near zero ({denom:.2e}) for length {ln}, " f"using fallback (current price)")
            new_lsmas.append(new_price)
            continue

        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n
        lsma = intercept + slope * (n - 1)

        # FIX #6: Validate result is finite
        if not np.isfinite(lsma):
            log_warn(f"LSMA produced non-finite value: {lsma}, using previous value")
            lsma = prev_lsmas[i]

        new_lsmas.append(lsma)

    state["ma_values"]["lsma"] = new_lsmas


def update_kama(state: Dict[str, Any], new_price: float, length: int, robustness: str) -> None:
    """Update KAMA incrementally for all 9 variations.

    NOTE: Known Issue #4 - Floating Point Precision Drift
    KAMA calculation relies on `volatility` (sum of absolute differences) over a sliding window.
    The current incremental implementation recalculates volatility from the price history window
    (O(N) operation) rather than using a pure O(1) incremental update.
    Even so, slight differences in floating point operations between this incremental logic
    and the batch `pandas_ta` implementation can accumulate over time (approx 0.05 drift over 1000 bars).
    This is considered acceptable for trend classification but should be noted for high-precision requirements.
    Future optimization: Implement stable Kahan summation if drift becomes critical.
    """
    from modules.adaptive_trend_LTS_mini.utils.diflen import diflen

    _d = diflen(length, robustness=robustness)
    assert _d is not None
    L1, L2, L3, L4, L_1, L_2, L_3, L_4 = _d
    lengths = [length, L1, L2, L3, L4, L_1, L_2, L_3, L_4]

    prices = list(state["price_history"])

    # Get previous KAMAs
    prev_kamas = state["ma_values"].get("kama")
    if not isinstance(prev_kamas, list) or len(prev_kamas) != 9:
        prev_kamas = [new_price] * 9

    new_kamas = []
    for i, ln in enumerate(lengths):
        prev_kama = prev_kamas[i]

        if len(prices) < ln + 1:
            new_kamas.append(new_price)
            continue

        window = prices[-(ln + 1) :]
        change = abs(window[-1] - window[0])
        volatility = sum(abs(window[j] - window[j - 1]) for j in range(1, len(window)))

        er = change / volatility if volatility != 0 else 0
        fast_sc = 2 / (2.0 + 1)
        slow_sc = 2 / (30.0 + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        new_kama = prev_kama + sc * (new_price - prev_kama)
        new_kamas.append(new_kama)

    state["ma_values"]["kama"] = new_kamas

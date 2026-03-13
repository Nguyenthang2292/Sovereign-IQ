"""Signal calculation logic for incremental ATC."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .constants import calculate_growth_factor, get_initial_weights, get_scaled_params

try:
    from modules.common.utils import log_warn
except ImportError:

    def log_warn(msg: str, *args: object) -> None:
        print(f"[WARN] {msg % args if args else msg}")


def update_layer1_signals(
    state: Dict[str, Any],
    config: Dict[str, Any],
    prev_price: float,
    new_price: float,
    prev_ma_values: Dict[str, list],
) -> Dict[str, float]:
    """Update Layer 1 signals and equities.

    Args:
        state: Current ATC state dictionary
        config: ATC configuration
        prev_price: Previous price
        new_price: Current price
        prev_ma_values: Previous MA values

    Returns:
        Dict of Layer 1 signals per MA type
    """
    bar_index = int(state.get("bar_index") or 0)
    L_scaled, De_scaled = get_scaled_params(config)
    cutout = int(config.get("cutout", 0))

    # Rate of change and growth factor
    if prev_price is None or prev_price == 0 or np.isnan(prev_price) or np.isnan(new_price):
        r_raw = 0.0
    else:
        r_raw = (new_price - prev_price) / prev_price

    growth = calculate_growth_factor(bar_index, cutout, L_scaled)
    r_adj = r_raw * growth
    d = 1.0 - De_scaled

    ma_types_lower = ["ema", "hma", "wma", "dema", "lsma", "kama"]
    layer1_signals: Dict[str, float] = {}

    if state["equity"] is None:
        state["equity"] = {}

    # Update per-variation signals and Layer 1 equities
    for ma_key in ma_types_lower:
        ma_type = ma_key.upper()
        curr_ma_list = state["ma_values"].get(ma_key)
        if not isinstance(curr_ma_list, list) or len(curr_ma_list) != 9:
            continue

        prev_ma_list = prev_ma_values.get(ma_key)
        prev_signals = state["signals_l1"].get(ma_key, [0.0] * 9)
        prev_equities = state["equity_l1"].get(ma_key, [np.nan] * 9)

        new_signals = []
        new_equities = []
        numer = 0.0
        denom = 0.0

        for i in range(9):
            prev_ma = prev_ma_list[i] if isinstance(prev_ma_list, list) and len(prev_ma_list) == 9 else curr_ma_list[i]
            curr_ma = curr_ma_list[i]
            prev_sig = prev_signals[i] if i < len(prev_signals) else 0.0

            up = prev_price <= prev_ma and new_price > curr_ma
            down = prev_price >= prev_ma and new_price < curr_ma
            if up:
                sig = 1.0
            elif down:
                sig = -1.0
            else:
                sig = prev_sig

            prev_e = prev_equities[i] if i < len(prev_equities) else np.nan
            if np.isnan(prev_e):
                e_curr = 1.0
            else:
                if prev_sig > 0:
                    a = r_adj
                elif prev_sig < 0:
                    a = -r_adj
                else:
                    a = 0.0
                e_curr = (prev_e * d) * (1.0 + a)

            # FIX #3: Equity floor validation with negative equity detection
            if e_curr < 0.0:
                # Negative equity indicates extreme loss - log warning and reset
                log_warn(
                    f"Negative equity detected in {ma_key}[{i}]: {e_curr:.6f}. "
                    f"This indicates extreme loss (>100%). Resetting to floor value."
                )
                e_curr = 0.25
            elif e_curr < 0.25:
                e_curr = 0.25

            new_signals.append(sig)
            new_equities.append(e_curr)
            numer += sig * e_curr
            denom += e_curr

        state["signals_l1"][ma_key] = new_signals
        state["equity_l1"][ma_key] = new_equities

        # FIX: Add rounding to match batch behavior (weighted_signal rounds to 2 decimals)
        layer1_signals[ma_type] = round(numer / denom, 2) if denom != 0 else 0.0

    state["layer1_signals"] = layer1_signals
    return layer1_signals


def update_layer2_equities(
    state: Dict[str, Any],
    config: Dict[str, Any],
    prev_price: float,
    new_price: float,
    prev_layer1_signals: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Update Layer 2 equities based on Layer 1 signals.

    Args:
        state: Current ATC state dictionary
        config: ATC configuration
        prev_price: Previous price
        new_price: Current price
        prev_layer1_signals: Layer 1 signals from the PREVIOUS bar (T-1).
            Crucial for preventing look-ahead bias. If None, tries to use state (which might be updated).

    Returns:
        Dict of Layer 2 equities per MA type
    """
    bar_index = int(state.get("bar_index") or 0)
    L_scaled, De_scaled = get_scaled_params(config)
    cutout = int(config.get("cutout", 0))
    initial_weights = get_initial_weights(config)

    # Rate of change calculation (same as in update_layer1_signals)
    if prev_price is None or prev_price == 0 or np.isnan(prev_price) or np.isnan(new_price):
        r_raw = 0.0
    else:
        r_raw = (new_price - prev_price) / prev_price

    growth = calculate_growth_factor(bar_index, cutout, L_scaled)
    r_adj = r_raw * growth
    d = 1.0 - De_scaled

    # FIX #1: Prevent Look-ahead Bias
    # Use explicitly passed previous signals if available, otherwise fallback to state.
    # In incremental mode, state["layer1_signals"] contains the CURRENT bar's signals
    # if called after update_layer1_signals, so prev_layer1_signals must be passed.
    source_signals = prev_layer1_signals if prev_layer1_signals is not None else state.get("layer1_signals", {})

    new_layer2 = {}
    for ma_type in ["EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"]:
        prev_layer1 = source_signals.get(ma_type, 0.0)
        prev_eq2 = state["equity"].get(ma_type, np.nan) if state["equity"] else np.nan

        if bar_index < cutout:
            eq2 = np.nan
        else:
            if prev_layer1 > 0:
                a2 = r_adj
            elif prev_layer1 < 0:
                a2 = -r_adj
            else:
                a2 = 0.0

            if np.isnan(prev_eq2):
                eq2 = initial_weights.get(ma_type, 1.0)
            else:
                eq2 = (prev_eq2 * d) * (1.0 + a2)

            # FIX #3: Equity floor validation with negative equity detection
            # NOTE: This 0.25 floor is an intentional design choice to prevent
            # total bankruptcy (equity <= 0) which would permanently zero out a component.
            # It acts as a "refund" mechanism to keep all strategies in the game.
            if eq2 < 0.0:
                log_warn(f"Negative Layer 2 equity detected for {ma_type}: {eq2:.6f}. Resetting to floor value.")
                eq2 = 0.25
            elif eq2 < 0.25:
                eq2 = 0.25

        new_layer2[ma_type] = eq2

    state["equity"] = new_layer2
    return new_layer2


def calculate_average_signal(state: Dict[str, Any], config: Dict[str, Any]) -> float:
    """Compute final Average_Signal for current bar.

    Args:
        state: Current ATC state dictionary
        config: ATC configuration

    Returns:
        Final signal value
    """
    bar_index = int(state.get("bar_index") or 0)
    cutout = int(config.get("cutout", 0))
    long_threshold = float(config.get("long_threshold", 0.1))
    short_threshold = float(config.get("short_threshold", -0.1))

    layer1_signals = state["layer1_signals"]
    layer2_equities = state["equity"]

    nom = 0.0
    den = 0.0
    for ma_type, eq2 in layer2_equities.items():
        weight = eq2 if np.isfinite(eq2) else 0.0
        if weight == 0.0:
            continue
        sig_val = layer1_signals.get(ma_type, 0.0)
        if sig_val > long_threshold:
            c = 1.0
        elif sig_val < short_threshold:
            c = -1.0
        else:
            c = 0.0
        nom += c * weight
        den += weight

    avg_current = nom / den if den != 0 else 0.0
    # FIX: Return NaN for cutout period to match batch behavior
    # Batch returns np.nan for cutout period, not 0.0
    if bar_index < cutout:
        avg_current = np.nan

    return avg_current

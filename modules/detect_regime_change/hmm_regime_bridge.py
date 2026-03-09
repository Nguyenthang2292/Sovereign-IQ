"""
detect_regime_change/hmm_regime_bridge.py
==========================================
Bridge to existing modules/hmm for real-time regime state estimation.

Uses existing SwingsHMM to get:
- next_state_duration: expected duration of the next state
- next_state: predicted state (BULLISH/NEUTRAL/BEARISH)
- probability: prediction confidence
"""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

from modules.common.utils import log_error

# Keep a module-level symbol so tests can monkeypatch
# `modules.detect_regime_change.hmm_regime_bridge.hmm_swings` reliably.
hmm_swings = None


def _resolve_hmm_swings():
    """Resolve and cache hmm_swings lazily, while staying monkeypatch-friendly."""
    global hmm_swings
    if hmm_swings is not None:
        return hmm_swings
    from modules.hmm import hmm_swings as _hmm_swings
    hmm_swings = _hmm_swings
    return hmm_swings


def estimate_hmm_regime_duration(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
) -> Tuple[Optional[float], Optional[int], Optional[float]]:
    """
    Use existing HMM module to estimate current regime duration.

    Args:
        df: DataFrame with OHLCV columns and DatetimeIndex
        train_ratio: Train/test split ratio

    Returns:
        Tuple of (duration_hours, state, probability)
        - duration_hours: predicted next state duration in hours
        - state: -1 (BEARISH), 0 (NEUTRAL), 1 (BULLISH)
        - probability: confidence of prediction
    """
    try:
        hmm_swings_fn = _resolve_hmm_swings()

        result = hmm_swings_fn(df, train_ratio=train_ratio, eval_mode=False)

        # next_state_duration from HMM_SWINGS uses timeframe-dependent units.
        # Convert it to hours.
        duration_raw = result.next_state_duration
        state = result.next_state_with_high_order_hmm
        probability = result.next_state_probability

        # Determine timeframe from data
        has_datetime_index = isinstance(df.index, pd.DatetimeIndex) and len(df.index) >= 2
        if isinstance(df.index, pd.DatetimeIndex) and len(df.index) >= 2:
            interval_seconds = (df.index[1] - df.index[0]).total_seconds()
        else:
            interval_seconds = 900  # Default 15m

        # Convert duration to hours
        # duration_raw is the number of "units" (candles or swing distance).
        # _calculate_duration may convert to hours/minutes based on interval.
        # We still normalize here to guarantee output is always in hours.
        if not has_datetime_index:
            # Without DatetimeIndex, treat raw duration as candle units and
            # convert using default 15m candle interval.
            duration_hours = float(duration_raw) * float(interval_seconds) / 3600.0
        elif interval_seconds >= 3600:
            # Hourly candles -> duration_raw is already in hours.
            duration_hours = float(duration_raw)
        elif interval_seconds >= 60:
            # Minute candles -> duration_raw is in minutes.
            duration_hours = float(duration_raw) / 60.0
        else:
            duration_hours = float(duration_raw) / 3600.0

        return duration_hours, state, probability

    except Exception as e:
        log_error(f"HMM regime estimation failed: {e}")
        return None, None, None

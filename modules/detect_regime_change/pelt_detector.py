"""
detect_regime_change/pelt_detector.py
=====================================
Change Point Detection using PELT algorithm (ruptures library).

Analyze historical return/volatility series to detect breakpoints,
where each interval between two breakpoints is a regime segment.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple, cast

import numpy as np

from modules.detect_regime_change.models import ChangePoint, RegimeSegment

# Rust backend currently implements L2 and Normal cost.
# Keep RBF on ruptures for parity until a dedicated Rust implementation is added.
RUST_SUPPORTED_MODELS = {"l2", "normal"}


def detect_change_points_pelt(
    returns: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    penalty: Optional[float] = None,
    model: str = "rbf",
    min_segment_length: int = 10,
) -> Tuple[List[ChangePoint], List[RegimeSegment]]:
    """
    Detect regime change points using PELT algorithm.

    Args:
        returns: Array of log-returns or price changes
        timestamps: Optional array of timestamps (datetime64)
        penalty: PELT penalty parameter (beta). None = auto BIC.
        model: Cost model - "rbf" (recommended), "l2", "normal"
        min_segment_length: Minimum segment length

    Returns:
        Tuple of (change_points, segments)
    """
    n = len(returns)
    if n < min_segment_length * 2:
        return [], []

    normalized_model = model.lower()

    # Auto-penalty via BIC if not provided
    effective_penalty = float(np.log(n) * returns.var()) if penalty is None else float(penalty)

    breakpoints: Optional[List[int]] = None

    # Try Rust first for supported models.
    if normalized_model in RUST_SUPPORTED_MODELS:
        try:
            rust_module_name = "rust_extensions"
            rust_module = __import__(rust_module_name)
            detect_change_points_pelt_rs = getattr(rust_module, "detect_change_points_pelt_rs", None)
            if not callable(detect_change_points_pelt_rs):
                raise AttributeError("detect_change_points_pelt_rs not found")
            rust_detector = cast(
                Callable[[List[float], float, int, str], List[int]],
                detect_change_points_pelt_rs,
            )

            # Rust-only ABI (model-aware): (returns, penalty, min_size, model)
            breakpoints = rust_detector(
                returns.tolist(),
                effective_penalty,
                int(min_segment_length),
                normalized_model,
            )
        except ImportError:
            breakpoints = None
        except Exception:
            # Safety fallback: if Rust fails at runtime, use Python implementation.
            breakpoints = None

    if breakpoints is None:
        ruptures_module_name = "ruptures"
        rpt = __import__(ruptures_module_name)

        # Fallback to Python ruptures
        algo = rpt.Pelt(model=normalized_model, min_size=min_segment_length).fit(returns)
        breakpoints = algo.predict(pen=effective_penalty)

    if breakpoints is None:
        return [], []

    # Build change points
    change_points: List[ChangePoint] = []
    for bp in breakpoints[:-1]:  # Last element is always n
        ts = str(timestamps[bp]) if timestamps is not None and bp < len(timestamps) else None
        change_points.append(ChangePoint(index=bp, timestamp=ts))

    # Build regime segments
    segments: List[RegimeSegment] = []
    starts = [0] + breakpoints[:-1]
    ends = breakpoints

    for s, e in zip(starts, ends):
        seg_returns = returns[s:e]

        # Duration calculation
        if timestamps is not None and len(timestamps) > e - 1:
            t_start = np.datetime64(str(timestamps[s]), "s")
            t_end = np.datetime64(str(timestamps[min(e, len(timestamps)) - 1]), "s")
            duration_seconds = float((t_end - t_start) / np.timedelta64(1, "s"))
        else:
            # Fallback: estimate from candle count
            duration_seconds = float((e - s) * 900)  # Assume 15m candles = 900s

        segments.append(
            RegimeSegment(
                start_index=s,
                end_index=e,
                duration_seconds=duration_seconds,
                duration_hours=duration_seconds / 3600.0,
                mean_return=float(np.mean(seg_returns)) if len(seg_returns) > 0 else None,
                volatility=float(np.std(seg_returns)) if len(seg_returns) > 0 else None,
            )
        )

    return change_points, segments


def calculate_pelt_avg_duration(
    segments: List[RegimeSegment],
    trim_pct: float = 0.1,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate average and median regime duration from PELT segments.

    Args:
        segments: List of regime segments
        trim_pct: Percentage of extreme segments to trim (0.1 = 10% each side)

    Returns:
        Tuple of (trimmed_mean_hours, median_hours)
    """
    if not segments:
        return None, None

    durations = sorted([s.duration_hours for s in segments])

    # Trimmed mean - remove extreme outliers
    n = len(durations)
    if n >= 5:
        trim_count = max(1, int(n * trim_pct))
        trimmed = durations[trim_count:-trim_count]
    else:
        trimmed = durations

    avg = float(np.mean(trimmed)) if trimmed else None
    median = float(np.median(durations))

    return avg, median

"""
Adaptive Trend Classification (ATC) - Main computation module.

This module exposes the core function `compute_atc_signals`, which generates adaptive trend
signals using a layer-based approach and simulated equity curves for weighting.

Computation workflow:
1. Layer 1: Generates signals from multiple Moving Average (MA) types,
   evaluating each based on its equity curve.
2. Layer 2: Calculates adaptive weights for each MA-derived signal,
   leveraging the Layer 1 signal performance.
3. Final aggregation: Produces a consolidated Average_Signal by weighting all MA signals,
   aiming for robust trend detection.

Dependencies:
- utils: Core functions (e.g., rate_of_change, diflen, exp_growth)
- compute_moving_averages.py: Computation of various MAs (EMA, HMA, WMA, DEMA, LSMA, KAMA)
- signal_detection.py: Generation/classification of buy/sell signals
- compute_equity.py: Simulation of equity curves per MA
- process_layer1.py: Implements Layer 1 logic and signal post-processing

Performance/features:
- Uses vectorized NumPy operations for efficient final calculations.
- Employs rate_of_change result caching to improve speed.
- Logs computation progress and anomalies for debugging and monitoring purposes.

Typical ATC workflow:
    prices (pd.Series) →
        set_of_moving_averages →
        _layer1_signal_for_ma →
        equity_series →
        adaptive weighting →
        Average_Signal
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    from modules.common.utils import log_debug, log_error, log_info, log_warn
except ImportError:
    # Fallback logging if common utils not available
    def log_debug(msg: str) -> None:
        print(f"[DEBUG] {msg}")

    def log_info(msg: str) -> None:
        print(f"[INFO] {msg}")

    def log_warn(msg: str) -> None:
        print(f"[WARN] {msg}")

    def log_error(msg: str) -> None:
        print(f"[ERROR] {msg}")


from modules.adaptive_trend.utils import rate_of_change

from .compute_equity import equity_series
from .compute_moving_averages import set_of_moving_averages
from .process_layer1 import _layer1_signal_for_ma, cut_signal


def calculate_layer2_equities(
    layer1_signals: Dict[str, pd.Series],
    ma_configs: list,
    R: pd.Series,
    L: float,
    De: float,
    cutout: int,
) -> Dict[str, pd.Series]:
    """
    Calculate Layer 2 equity curves based on Layer 1 signal performance.

    Port of Pine Script Layer 2 calculation:
        EMA_S = eq(ema_w,  EMA_Signal,  R)
        HMA_S = eq(hma_w,  HMA_Signal,  R)
        ...

    This function calculates the equity curve for each MA type based on how well
    its Layer 1 signal performed. The equity curve serves as a dynamic weight
    in the final signal aggregation.

    Args:
        layer1_signals: Dictionary of Layer 1 signals keyed by MA type (e.g., "EMA", "HMA").
        ma_configs: List of (ma_type, length, initial_weight) tuples.
        R: Rate of change series (calculated once and reused).
        L: Lambda (growth rate) for exponential growth factor.
        De: Decay factor for equity calculations.
        cutout: Number of bars to skip at beginning.

    Returns:
        Dictionary of Layer 2 equity curves keyed by MA type.

    Raises:
        ValueError: If ma_configs contains invalid entries.
    """
    log_debug("Computing Layer 2 equity weights...")
    layer2_equities = {}

    for ma_type, _, initial_weight in ma_configs:
        if ma_type not in layer1_signals:
            log_warn(f"Layer 1 signal for {ma_type} not found, skipping")
            continue

        # Layer 2: Calculate equity curve based on Layer 1 signal performance
        # Port of Pine Script: EMA_S = eq(ema_w, EMA_Signal, R)
        # where ema_w is the initial weight (starting_equity)
        equity = equity_series(
            starting_equity=initial_weight,
            sig=layer1_signals[ma_type],
            R=R,
            L=L,
            De=De,
            cutout=cutout,
        )
        layer2_equities[ma_type] = equity

    log_debug("Completed Layer 2 equity weights")
    return layer2_equities


def compute_atc_signals(
    prices: pd.Series,
    src: Optional[pd.Series] = None,
    *,
    ema_len: int = 28,
    hull_len: int = 28,
    wma_len: int = 28,
    dema_len: int = 28,
    lsma_len: int = 28,
    kama_len: int = 28,
    ema_w: float = 1.0,
    hma_w: float = 1.0,
    wma_w: float = 1.0,
    dema_w: float = 1.0,
    lsma_w: float = 1.0,
    kama_w: float = 1.0,
    robustness: str = "Medium",
    La: float = 0.02,
    De: float = 0.03,
    cutout: int = 0,
    long_threshold: float = 0.1,
    short_threshold: float = -0.1,
) -> dict[str, pd.Series]:
    """
    Compute Adaptive Trend Classification (ATC) signals.

    This function orchestrates the entire ATC computation process:
    1. Compute Moving Averages with multiple lengths
    2. Compute Layer 1 signals for each MA type
    3. Compute Layer 2 weights from Layer 1 signals
    4. Combine all to create Average_Signal

    Args:
        prices: Price series (typically close) for computing rate of change and signals.
        src: Source series for computing MA. If None, uses prices.
        ema_len, hull_len, wma_len, dema_len, lsma_len, kama_len:
            Window length for each MA type.
        ema_w, hma_w, wma_w, dema_w, lsma_w, kama_w:
            Initial weights for each MA family at Layer 2.
        robustness: "Narrow" / "Medium" / "Wide" - width of length offsets.
        La: Lambda (growth rate) for exponential growth factor.
        De: Decay rate for equity calculations.
        cutout: Number of bars to skip at the beginning.
        long_threshold: Threshold for LONG signals (default: 0.1).
        short_threshold: Threshold for SHORT signals (default: -0.1).

    Returns:
        Dictionary containing:
            - EMA_Signal, HMA_Signal, WMA_Signal, DEMA_Signal, LSMA_Signal, KAMA_Signal:
              Layer 1 signals for each MA type
            - EMA_S, HMA_S, WMA_S, DEMA_S, LSMA_S, KAMA_S:
              Layer 2 equity weights
            - Average_Signal: Final combined signal

    Raises:
        ValueError: If prices is empty or invalid.
        ZeroDivisionError: If total equity weights equals 0 (very rare).
    """
    # Input validation
    log_debug(f"Starting ATC signal computation for {len(prices)} bars")

    if prices is None or len(prices) == 0:
        log_error("prices cannot be empty or None")
        raise ValueError("prices cannot be empty or None")

    if src is None:
        src = prices

    if len(src) == 0:
        log_error("src cannot be empty")
        raise ValueError("src cannot be empty")

    # Validate robustness
    if robustness not in ("Narrow", "Medium", "Wide"):
        log_warn(f"robustness '{robustness}' is invalid, using 'Medium'")
        robustness = "Medium"  # Default fallback

    # Validate cutout
    if cutout < 0:
        log_warn(f"cutout {cutout} < 0, setting to 0")
        cutout = 0
    if cutout >= len(prices):
        log_error(f"cutout ({cutout}) >= prices length ({len(prices)})")
        raise ValueError(f"cutout ({cutout}) must be less than prices length ({len(prices)})")

    log_info(f"Parameters: robustness={robustness}, La={La}, De={De}, cutout={cutout}")

    # Define configuration for each MA type
    ma_configs = [
        ("EMA", ema_len, ema_w),
        ("HMA", hull_len, hma_w),
        ("WMA", wma_len, wma_w),
        ("DEMA", dema_len, dema_w),
        ("LSMA", lsma_len, lsma_w),
        ("KAMA", kama_len, kama_w),
    ]

    # DECLARE MOVING AVERAGES (SetOfMovingAverages)
    log_debug("Computing Moving Averages...")
    ma_tuples = {}
    for ma_type, length, _ in ma_configs:
        ma_tuple = set_of_moving_averages(length, src, ma_type, robustness=robustness)
        if ma_tuple is None:
            log_error(f"Cannot compute {ma_type} with length={length}")
            raise ValueError(f"Cannot compute {ma_type} with length={length}")
        ma_tuples[ma_type] = ma_tuple
    log_debug(f"Computed {len(ma_tuples)} MA types")

    # MAIN CALCULATIONS - Adaptability Layer 1
    # Compute rate_of_change once and reuse for all MA types (performance optimization)
    log_debug("Computing rate_of_change (reused for Layer 1 and Layer 2)...")
    R = rate_of_change(prices)

    log_debug("Computing Layer 1 signals...")
    layer1_signals = {}
    for ma_type, _, _ in ma_configs:
        signal, _, _ = _layer1_signal_for_ma(prices, ma_tuples[ma_type], L=La, De=De, cutout=cutout, R=R)
        layer1_signals[ma_type] = signal
    log_debug("Completed Layer 1 signals")

    # Adaptability Layer 2
    # Calculate equity curves for Layer 2 weighting
    layer2_equities = calculate_layer2_equities(
        layer1_signals=layer1_signals,
        ma_configs=ma_configs,
        R=R,
        L=La,
        De=De,
        cutout=cutout,
    )

    # FINAL CALCULATIONS - Vectorized for performance
    log_debug("Computing Average_Signal (vectorized)...")

    # Vectorize: Compute all cut_signal at once and convert to numpy arrays
    # to leverage NumPy vectorization
    n_bars = len(prices)
    index = prices.index

    # Pre-allocate numpy arrays for better performance
    nom_array = np.zeros(n_bars, dtype=np.float64)
    den_array = np.zeros(n_bars, dtype=np.float64)

    # Vectorized calculation: compute all signals and equities at once
    for ma_type, _, _ in ma_configs:
        signal = layer1_signals[ma_type]
        equity = layer2_equities[ma_type]

        # Cut signal: vectorized operation
        cut_sig = cut_signal(
            signal,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            cutout=cutout,
        )

        # Convert to numpy arrays for faster computation
        cut_sig_values = cut_sig.values
        equity_values = equity.values

        # Vectorized addition (faster than pandas Series addition)
        nom_array += cut_sig_values * equity_values
        den_array += equity_values

    # Handle division by zero with NumPy (faster than pandas)
    # np.where to avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        avg_signal_array = np.divide(nom_array, den_array)
        # Replace inf and nan with 0 when den = 0
        avg_signal_array = np.where(np.isfinite(avg_signal_array), avg_signal_array, 0.0)

    Average_Signal = pd.Series(avg_signal_array, index=index, dtype="float64")

    # Check number of zero divisions (for logging)
    zero_divisions = np.sum(den_array == 0)
    if zero_divisions > 0:
        log_warn(f"Detected {zero_divisions} division by zero cases, replaced with 0")

    log_debug("Completed Average_Signal")

    # Build result dictionary
    result = {}
    for ma_type, _, _ in ma_configs:
        result[f"{ma_type}_Signal"] = layer1_signals[ma_type]
        result[f"{ma_type}_S"] = layer2_equities[ma_type]

    result["Average_Signal"] = Average_Signal

    log_info(f"Completed ATC signal computation for {len(prices)} bars")
    return result


__all__ = [
    "compute_atc_signals",
    "calculate_layer2_equities",
]

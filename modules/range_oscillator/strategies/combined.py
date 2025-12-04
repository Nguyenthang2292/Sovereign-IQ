"""
Range Oscillator Strategy 5: Combined.

This module provides the combined signal generation strategy that combines multiple methods.
"""

from typing import Optional, Tuple
import pandas as pd
import numpy as np
import os

from modules.range_oscillator.core.utils import get_oscillator_data
from modules.range_oscillator.strategies.basic import generate_signals_strategy1
from modules.range_oscillator.strategies.sustained import generate_signals_strategy2_sustained
from modules.range_oscillator.strategies.crossover import generate_signals_strategy3_crossover
from modules.range_oscillator.strategies.momentum import generate_signals_strategy4_momentum
from modules.common.utils import log_debug, log_analysis


def generate_signals_strategy5_combined(
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    close: Optional[pd.Series] = None,
    *,
    oscillator: Optional[pd.Series] = None,
    ma: Optional[pd.Series] = None,
    range_atr: Optional[pd.Series] = None,
    length: int = 50,
    mult: float = 2.0,
    use_sustained: bool = True,
    use_crossover: bool = True,
    use_momentum: bool = True,
    min_bars_sustained: int = 3,
    confirmation_bars: int = 2,
    momentum_period: int = 3,
    momentum_threshold: float = 5.0,
    enable_debug: bool = False,
) -> Tuple[pd.Series, pd.Series]:
    """
    Generate trading signals based on Range Oscillator Strategy 5: Combined.
    
    Strategy Logic:
    ---------------
    This strategy combines multiple signal generation methods:
    1. Sustained pressure (oscillator staying above/below 0)
    2. Zero line crossover with confirmation
    3. Momentum-based signals
    
    Signals are generated when at least one method confirms, with strength
    calculated as the average of all active methods.
    
    Args:
        high: High price series (required if oscillator not provided)
        low: Low price series (required if oscillator not provided)
        close: Close price series (required if oscillator not provided)
        oscillator: Pre-calculated oscillator series (optional, if provided with ma and range_atr)
        ma: Pre-calculated moving average series (optional)
        range_atr: Pre-calculated range ATR series (optional)
        length: Minimum range length for oscillator calculation (default: 50, ignored if oscillator provided)
        mult: Range width multiplier (default: 2.0, ignored if oscillator provided)
        use_sustained: Enable sustained pressure signals (default: True)
        use_crossover: Enable crossover signals (default: True)
        use_momentum: Enable momentum signals (default: True)
        min_bars_sustained: Minimum bars for sustained signal (default: 3)
        confirmation_bars: Bars for crossover confirmation (default: 2)
        momentum_period: Period for momentum calculation (default: 3)
        momentum_threshold: Threshold for momentum signal (default: 5.0)
        enable_debug: If True, enable debug logging (default: False)
    
    Returns:
        Tuple containing:
        - signals: Series with signal values (1 = LONG, -1 = SHORT, 0 = NEUTRAL)
        - signal_strength: Series with signal strength values (0.0 to 1.0)
    """
    # Validate parameters
    if use_sustained:
        if min_bars_sustained <= 0:
            raise ValueError(f"min_bars_sustained must be > 0, got {min_bars_sustained}")
    
    if use_crossover:
        if confirmation_bars <= 0:
            raise ValueError(f"confirmation_bars must be > 0, got {confirmation_bars}")
    
    if use_momentum:
        if momentum_period <= 0:
            raise ValueError(f"momentum_period must be > 0, got {momentum_period}")
        if momentum_threshold < 0:
            raise ValueError(f"momentum_threshold must be >= 0, got {momentum_threshold}")
    
    # Enable debug logging if requested
    debug_enabled = enable_debug or os.environ.get("RANGE_OSCILLATOR_DEBUG", "false").lower() == "true"
    
    if debug_enabled:
        log_analysis(f"[Strategy5] Starting combined signal generation")
        log_debug(f"[Strategy5] Enabled methods: sustained={use_sustained}, "
                 f"crossover={use_crossover}, momentum={use_momentum}")
        log_debug(f"[Strategy5] Parameters: min_bars_sustained={min_bars_sustained}, "
                 f"confirmation_bars={confirmation_bars}, momentum_period={momentum_period}, "
                 f"momentum_threshold={momentum_threshold}")
    
    # Calculate Range Oscillator ONCE (or use pre-calculated values)
    oscillator, ma, range_atr = get_oscillator_data(
        high=high, low=low, close=close,
        oscillator=oscillator, ma=ma, range_atr=range_atr,
        length=length, mult=mult
    )
    
    # Validate momentum_period against data length (if momentum strategy is enabled)
    if use_momentum and len(oscillator) > 0 and momentum_period >= len(oscillator):
        raise ValueError(f"momentum_period ({momentum_period}) must be < data length ({len(oscillator)})")
    
    if debug_enabled:
        log_debug(f"[Strategy5] Data shape: oscillator={len(oscillator)}")
    
    index = oscillator.index
    
    # Get signals from individual strategies (pass pre-calculated values to avoid recalculation)
    signal_votes = []
    strength_votes = []
    
    if use_sustained:
        sig_sustained, str_sustained = generate_signals_strategy2_sustained(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            min_bars_above_zero=min_bars_sustained,
            min_bars_below_zero=min_bars_sustained,
            enable_debug=False,  # Disable debug for sub-strategies to avoid spam
        )
        signal_votes.append(sig_sustained)
        strength_votes.append(str_sustained)
        if debug_enabled:
            log_debug(f"[Strategy5] Sustained strategy: LONG={int((sig_sustained == 1).sum())}, "
                     f"SHORT={int((sig_sustained == -1).sum())}")
    
    if use_crossover:
        sig_cross, str_cross = generate_signals_strategy3_crossover(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            confirmation_bars=confirmation_bars,
            enable_debug=False,  # Disable debug for sub-strategies to avoid spam
        )
        signal_votes.append(sig_cross)
        strength_votes.append(str_cross)
        if debug_enabled:
            log_debug(f"[Strategy5] Crossover strategy: LONG={int((sig_cross == 1).sum())}, "
                     f"SHORT={int((sig_cross == -1).sum())}")
    
    if use_momentum:
        sig_mom, str_mom = generate_signals_strategy4_momentum(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            momentum_period=momentum_period,
            momentum_threshold=momentum_threshold,
            enable_debug=False,  # Disable debug for sub-strategies to avoid spam
        )
        signal_votes.append(sig_mom)
        strength_votes.append(str_mom)
        if debug_enabled:
            log_debug(f"[Strategy5] Momentum strategy: LONG={int((sig_mom == 1).sum())}, "
                     f"SHORT={int((sig_mom == -1).sum())}")
    
    if not signal_votes:
        # Fallback to basic strategy if no methods enabled
        return generate_signals_strategy1(
            high=high, low=low, close=close,
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            length=length, mult=mult
        )
    
    # Combine signals: majority vote (OPTIMIZED with np.stack for vectorized calculation)
    # Stack all signals and strengths into arrays for vectorized operations
    signals_array = np.stack([sig.values for sig in signal_votes], axis=0)  # Shape: (n_strategies, n_bars)
    strengths_array = np.stack([str_vote.values for str_vote in strength_votes], axis=0)  # Shape: (n_strategies, n_bars)
    
    # Count votes vectorized using np.sum (much faster than loops)
    long_votes = np.sum(signals_array == 1, axis=0).astype(np.int8)  # Count LONG votes per bar
    short_votes = np.sum(signals_array == -1, axis=0).astype(np.int8)  # Count SHORT votes per bar
    
    # Determine signals based on votes (vectorized)
    # Majority vote: LONG if long_votes > short_votes, SHORT if short_votes > long_votes
    # If votes are equal, set to NEUTRAL (0) - this handles conflicts
    signals = np.zeros(len(index), dtype=np.int8)
    signals = np.where((long_votes > short_votes) & (long_votes > 0), 1, signals)
    signals = np.where((short_votes > long_votes) & (short_votes > 0), -1, signals)
    # Note: When long_votes == short_votes, signal remains 0 (NEUTRAL)
    
    if debug_enabled:
        total_long_votes = int(long_votes.sum())
        total_short_votes = int(short_votes.sum())
        tie_count = int(((long_votes == short_votes) & (long_votes > 0)).sum())
        log_debug(f"[Strategy5] Vote summary: total_long_votes={total_long_votes}, "
                 f"total_short_votes={total_short_votes}, ties={tie_count}")
    
    # Calculate average strength vectorized using np.stack
    # Create masks for LONG and SHORT signals
    long_mask = signals == 1
    short_mask = signals == -1
    
    # For LONG signals: average strength from strategies that voted LONG
    long_signal_mask = (signals_array == 1) & long_mask[np.newaxis, :]  # Shape: (n_strategies, n_bars)
    long_strength_sum = np.sum(strengths_array * long_signal_mask, axis=0)  # Sum strengths where LONG vote
    long_strength_count = np.sum(long_signal_mask, axis=0).astype(np.float64)  # Count LONG votes
    # Avoid division by zero warning by using np.divide with where
    long_strength_avg = np.divide(long_strength_sum, long_strength_count, out=np.zeros_like(long_strength_sum), where=long_strength_count > 0)
    
    # For SHORT signals: average strength from strategies that voted SHORT
    short_signal_mask = (signals_array == -1) & short_mask[np.newaxis, :]  # Shape: (n_strategies, n_bars)
    short_strength_sum = np.sum(strengths_array * short_signal_mask, axis=0)  # Sum strengths where SHORT vote
    short_strength_count = np.sum(short_signal_mask, axis=0).astype(np.float64)  # Count SHORT votes
    # Avoid division by zero warning by using np.divide with where
    short_strength_avg = np.divide(short_strength_sum, short_strength_count, out=np.zeros_like(short_strength_sum), where=short_strength_count > 0)
    
    # Combine strengths: use LONG strength where LONG signal, SHORT strength where SHORT signal
    signal_strength = np.where(long_mask, long_strength_avg, np.where(short_mask, short_strength_avg, 0.0))
    
    # Convert back to Series
    signals = pd.Series(signals, index=index, dtype="int8")
    signal_strength = pd.Series(signal_strength, index=index, dtype="float64")
    
    # Handle NaN values (optimized: combine all NaN checks)
    valid_mask = ~oscillator.isna()
    signals = signals.where(valid_mask, 0)
    signal_strength = signal_strength.where(valid_mask, 0.0)
    
    if debug_enabled:
        final_long = int((signals == 1).sum())
        final_short = int((signals == -1).sum())
        final_neutral = int((signals == 0).sum())
        avg_strength = float(signal_strength.mean())
        log_analysis(f"[Strategy5] Final combined signals: LONG={final_long}, SHORT={final_short}, "
                    f"NEUTRAL={final_neutral}, avg_strength={avg_strength:.3f}")
    
    return signals, signal_strength


__all__ = [
    "generate_signals_strategy5_combined",
]


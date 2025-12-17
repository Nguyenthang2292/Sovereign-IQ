"""
Display utilities for ATC + Range Oscillator CLI.

This module provides formatted display functions for combined signal results,
configuration, and summary information.
"""

import pandas as pd
from typing import Optional

from colorama import Fore, Style

from modules.common.utils import (
    color_text,
    format_price,
    log_progress,
)


def display_configuration(
    timeframe: str,
    limit: int,
    min_signal: float,
    max_workers: int,
    strategies: Optional[list],
    max_symbols: Optional[int] = None,
):
    """
    Display configuration information.
    
    Args:
        timeframe: Selected timeframe
        limit: Number of candles
        min_signal: Minimum signal strength
        max_workers: Number of parallel workers
        strategies: List of strategy numbers
        max_symbols: Maximum number of symbols to scan (optional)
    """
    print("\n" + color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    print(color_text("ATC + RANGE OSCILLATOR COMBINED SIGNAL FILTER", Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    print(color_text("Configuration:", Fore.WHITE))
    print(color_text(f"  Timeframe: {timeframe}", Fore.WHITE))
    print(color_text(f"  Limit: {limit} candles", Fore.WHITE))
    print(color_text(f"  Min Signal: {min_signal}", Fore.WHITE))
    print(color_text(f"  Parallel Workers: {max_workers}", Fore.WHITE))
    strategies_str = "Strategy 5 Combined (Dynamic Selection + Adaptive Weights)"
    print(color_text(f"  Oscillator Strategy: {strategies_str}", Fore.WHITE))
    print(color_text(f"  Mode: Dynamic Selection with Adaptive Weights", Fore.WHITE))
    if max_symbols:
        print(color_text(f"  Max Symbols: {max_symbols}", Fore.WHITE))
    print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))


def display_final_results(
    long_signals: pd.DataFrame,
    short_signals: pd.DataFrame,
    original_long_count: int,
    original_short_count: int,
    long_uses_fallback: bool = False,
    short_uses_fallback: bool = False,
):
    """
    Display final filtered results.
    
    Args:
        long_signals: Filtered LONG signals DataFrame
        short_signals: Filtered SHORT signals DataFrame
        original_long_count: Original number of LONG signals from ATC
        original_short_count: Original number of SHORT signals from ATC
        long_uses_fallback: True if LONG signals fallback to ATC only
        short_uses_fallback: True if SHORT signals fallback to ATC only
    """
    # DEBUG POINT: Display results entry - Check input DataFrames
    # Check: long_signals_empty, short_signals_empty, long_signals_shape, short_signals_shape
    # Check: long_columns, short_columns
    
    # Input validation
    if not isinstance(long_signals, pd.DataFrame):
        raise TypeError(f"long_signals must be a pandas DataFrame, got {type(long_signals)}")
    if not isinstance(short_signals, pd.DataFrame):
        raise TypeError(f"short_signals must be a pandas DataFrame, got {type(short_signals)}")
    if not isinstance(original_long_count, int) or original_long_count < 0:
        raise ValueError(f"original_long_count must be a non-negative integer, got {original_long_count}")
    if not isinstance(original_short_count, int) or original_short_count < 0:
        raise ValueError(f"original_short_count must be a non-negative integer, got {original_short_count}")
    
    # Required columns check
    required_columns = ['symbol', 'signal', 'price', 'exchange']
    missing_long = [col for col in required_columns if col not in long_signals.columns]
    missing_short = [col for col in required_columns if col not in short_signals.columns]
    if missing_long and not long_signals.empty:
        raise ValueError(f"long_signals missing required columns: {missing_long}")
    if missing_short and not short_signals.empty:
        raise ValueError(f"short_signals missing required columns: {missing_short}")
    print("\n" + color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    print(color_text("FINAL CONFIRMED SIGNALS (ATC + Range Oscillator)", Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))

    # LONG Signals
    if long_uses_fallback:
        print("\n" + color_text("LONG SIGNALS (ATC ONLY - Fallback)", Fore.YELLOW, Style.BRIGHT))
        print(color_text("  ⚠️  No signals confirmed by Range Oscillator, using ATC signals only", Fore.YELLOW))
    else:
        print("\n" + color_text("CONFIRMED LONG SIGNALS (ATC + Range Oscillator)", Fore.GREEN, Style.BRIGHT))
    print(color_text("-" * 80, Fore.CYAN))
    
    if long_signals.empty:
        print(color_text("  No LONG signals found", Fore.YELLOW))
    else:
        if long_uses_fallback:
            print(color_text(f"  Found {len(long_signals)} LONG signals from ATC (fallback - no oscillator confirmation)", Fore.YELLOW))
        else:
            print(color_text(f"  Found {len(long_signals)} confirmed LONG signals (from {original_long_count} ATC signals)", Fore.WHITE))
        print()
        # Check if osc_confidence column exists
        has_confidence = 'osc_confidence' in long_signals.columns
        if has_confidence and not long_uses_fallback:
            print(color_text(f"{'Symbol':<15} {'ATC Signal':>12} {'Price':>15} {'Exchange':<10} {'Confidence':>12}", Fore.MAGENTA))
        else:
            print(color_text(f"{'Symbol':<15} {'ATC Signal':>12} {'Price':>15} {'Exchange':<10}", Fore.MAGENTA))
        print(color_text("-" * 80, Fore.CYAN))
        
        for idx, row in long_signals.iterrows():
            # DEBUG POINT: Processing long signal row - Check row data availability
            # Check: index, has_symbol, has_signal, has_price, has_exchange, has_confidence
            
            try:
                signal_str = f"{row['signal']:+.6f}"
                price_str = format_price(row['price'])
                # Use yellow color for fallback signals, green for confirmed
                signal_color = Fore.YELLOW if long_uses_fallback else Fore.GREEN
                if has_confidence and not long_uses_fallback:
                    confidence = row.get('osc_confidence', 0.0)
                    confidence_str = f"{confidence:.3f}"
                    print(
                        color_text(
                            f"{row['symbol']:<15} {signal_str:>12} {price_str:>15} {row['exchange']:<10} {confidence_str:>12}",
                            signal_color,
                        )
                    )
                else:
                    print(
                        color_text(
                            f"{row['symbol']:<15} {signal_str:>12} {price_str:>15} {row['exchange']:<10}",
                            signal_color,
                        )
                    )
            except KeyError as e:
                # DEBUG POINT: KeyError in long signal row - Check missing keys
                # Check: index, missing_key, available_keys
                print(color_text(f"  Warning: Skipping row {idx} due to missing key: {e}", Fore.YELLOW))
                continue

    # SHORT Signals
    if short_uses_fallback:
        print("\n" + color_text("SHORT SIGNALS (ATC ONLY - Fallback)", Fore.YELLOW, Style.BRIGHT))
        print(color_text("  ⚠️  No signals confirmed by Range Oscillator, using ATC signals only", Fore.YELLOW))
    else:
        print("\n" + color_text("CONFIRMED SHORT SIGNALS (ATC + Range Oscillator)", Fore.RED, Style.BRIGHT))
    print(color_text("-" * 80, Fore.CYAN))
    
    if short_signals.empty:
        print(color_text("  No SHORT signals found", Fore.YELLOW))
    else:
        if short_uses_fallback:
            print(color_text(f"  Found {len(short_signals)} SHORT signals from ATC (fallback - no oscillator confirmation)", Fore.YELLOW))
        else:
            print(color_text(f"  Found {len(short_signals)} confirmed SHORT signals (from {original_short_count} ATC signals)", Fore.WHITE))
        print()
        # Check if osc_confidence column exists
        has_confidence = 'osc_confidence' in short_signals.columns
        if has_confidence and not short_uses_fallback:
            print(color_text(f"{'Symbol':<15} {'ATC Signal':>12} {'Price':>15} {'Exchange':<10} {'Confidence':>12}", Fore.MAGENTA))
        else:
            print(color_text(f"{'Symbol':<15} {'ATC Signal':>12} {'Price':>15} {'Exchange':<10}", Fore.MAGENTA))
        print(color_text("-" * 80, Fore.CYAN))
        
        for idx, row in short_signals.iterrows():
            try:
                signal_str = f"{row['signal']:+.6f}"
                price_str = format_price(row['price'])
                # Use yellow color for fallback signals, red for confirmed
                signal_color = Fore.YELLOW if short_uses_fallback else Fore.RED
                if has_confidence and not short_uses_fallback:
                    confidence = row.get('osc_confidence', 0.0)
                    confidence_str = f"{confidence:.3f}"
                    print(
                        color_text(
                            f"{row['symbol']:<15} {signal_str:>12} {price_str:>15} {row['exchange']:<10} {confidence_str:>12}",
                            signal_color,
                        )
                    )
                else:
                    print(
                        color_text(
                            f"{row['symbol']:<15} {signal_str:>12} {price_str:>15} {row['exchange']:<10}",
                            signal_color,
                        )
                    )
            except KeyError as e:
                print(color_text(f"  Warning: Skipping row {idx} due to missing key: {e}", Fore.YELLOW))
                continue

    # Calculate average confidence scores
    avg_long_confidence = 0.0
    avg_short_confidence = 0.0
    try:
        if not long_signals.empty and 'osc_confidence' in long_signals.columns:
            confidence_series = long_signals['osc_confidence']
            if len(confidence_series) > 0:
                avg_long_confidence = float(confidence_series.mean())
    except (ValueError, TypeError) as e:
        # DEBUG POINT: Error calculating avg_long_confidence - Check exception details
        # Check: exception_type, exception_msg
        pass  # Use default 0.0
    
    try:
        if not short_signals.empty and 'osc_confidence' in short_signals.columns:
            confidence_series = short_signals['osc_confidence']
            if len(confidence_series) > 0:
                avg_short_confidence = float(confidence_series.mean())
    except (ValueError, TypeError) as e:
        # DEBUG POINT: Error calculating avg_short_confidence - Check exception details
        # Check: exception_type, exception_msg
        pass  # Use default 0.0
    
    print("\n" + color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    print(color_text(f"Summary:", Fore.WHITE, Style.BRIGHT))
    print(color_text(f"  ATC Signals: {original_long_count} LONG + {original_short_count} SHORT = {original_long_count + original_short_count}", Fore.WHITE))
    
    # Build signal summary with source information
    long_source = "ATC Only (Fallback)" if long_uses_fallback else "ATC + Oscillator"
    short_source = "ATC Only (Fallback)" if short_uses_fallback else "ATC + Oscillator"
    
    print(color_text(f"  Final Signals: {len(long_signals)} LONG ({long_source}) + {len(short_signals)} SHORT ({short_source}) = {len(long_signals) + len(short_signals)}", Fore.WHITE, Style.BRIGHT))
    
    # Calculate confirmation rate only for confirmed signals (not fallback)
    confirmed_long = len(long_signals) if not long_uses_fallback else 0
    confirmed_short = len(short_signals) if not short_uses_fallback else 0
    total_confirmed = confirmed_long + confirmed_short
    
    if (original_long_count + original_short_count) > 0:
        confirmation_rate = total_confirmed / (original_long_count + original_short_count) * 100
        print(color_text(f"  Confirmation Rate: {confirmation_rate:.1f}% ({total_confirmed} confirmed by both ATC + Oscillator)", Fore.YELLOW))
        if long_uses_fallback or short_uses_fallback:
            fallback_count = (len(long_signals) if long_uses_fallback else 0) + (len(short_signals) if short_uses_fallback else 0)
            print(color_text(f"  Fallback Signals: {fallback_count} signals using ATC only (no oscillator confirmation)", Fore.YELLOW))
    else:
        print(color_text(f"  Confirmation Rate: N/A", Fore.YELLOW))
    
    # Display confidence scores
    if avg_long_confidence > 0 or avg_short_confidence > 0:
        print(color_text(f"  Average Confidence Score:", Fore.WHITE, Style.BRIGHT))
        if avg_long_confidence > 0:
            print(color_text(f"    LONG: {avg_long_confidence:.3f}", Fore.GREEN))
        if avg_short_confidence > 0:
            print(color_text(f"    SHORT: {avg_short_confidence:.3f}", Fore.RED))
        if avg_long_confidence > 0 and avg_short_confidence > 0:
            overall_avg = (avg_long_confidence + avg_short_confidence) / 2.0
            print(color_text(f"    Overall: {overall_avg:.3f}", Fore.YELLOW))
    
    print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))

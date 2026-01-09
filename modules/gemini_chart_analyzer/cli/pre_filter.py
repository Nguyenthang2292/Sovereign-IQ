"""
Pre-filter symbols using VotingAnalyzer or HybridAnalyzer.

This module provides functionality to filter symbols using VotingAnalyzer or HybridAnalyzer
before passing them to Gemini scan.
"""

import sys
import os
import argparse
import contextlib
from typing import List, Union, Optional, Set
import pandas as pd

from modules.common.core.exchange_manager import ExchangeManager
from modules.common.core.data_fetcher import DataFetcher
from modules.common.ui.logging import log_info, log_warn, log_error, log_success
from core.voting_analyzer import VotingAnalyzer
from core.hybrid_analyzer import HybridAnalyzer
from config import (
    DECISION_MATRIX_VOTING_THRESHOLD,
    DECISION_MATRIX_MIN_VOTES,
    SPC_LOOKBACK,
    SPC_P_LOW,
    SPC_P_HIGH,
    SPC_STRATEGY_PARAMETERS,
    RANGE_OSCILLATOR_LENGTH,
    RANGE_OSCILLATOR_MULTIPLIER,
    HMM_WINDOW_SIZE_DEFAULT,
    HMM_WINDOW_KAMA_DEFAULT,
    HMM_FAST_KAMA_DEFAULT,
    HMM_SLOW_KAMA_DEFAULT,
    HMM_HIGH_ORDER_ORDERS_ARGRELEXTREMA_DEFAULT,
    HMM_HIGH_ORDER_STRICT_MODE_DEFAULT,
)


def _extract_valid_symbols(df: pd.DataFrame, symbol_col: str, all_symbols_set: Set[str]) -> Set[str]:
    """
    Extract valid symbols from dataframe column using vectorized operations.
    
    Filters symbols that are:
    - Not null/NaN
    - Not empty strings
    - Present in the all_symbols_set
    
    Args:
        df: DataFrame containing the symbol column
        symbol_col: Name of the column containing symbols
        all_symbols_set: Set of all valid symbols to filter against
        
    Returns:
        Set of valid symbols extracted from the dataframe
    """
    if df.empty:
        return set()
    
    if symbol_col not in df.columns:
        return set()
    
    symbols = df[symbol_col]
    # Vectorized filtering: notna() & not empty & in all_symbols_set
    valid_symbols = symbols[
        symbols.notna() & 
        (symbols != '') & 
        symbols.isin(all_symbols_set)
    ]
    return set(valid_symbols)


def _create_analyzer_args(
    timeframe: str, 
    limit: int, 
    max_workers: Optional[int] = None
) -> argparse.Namespace:
    """
    Create a standardized argparse.Namespace with default values for analyzer configuration.
    
    Args:
        timeframe: Timeframe string for analysis
        limit: Number of candles to fetch per symbol
        max_workers: Maximum number of worker threads for parallel processing.
                    If None, automatically calculates based on CPU count:
                    - Uses min(10, max(1, cpu_count)) to balance performance
                    - Ensures at least 1 worker and caps at 10 to prevent resource exhaustion
                    - Defaults to 1 if cpu_count cannot be determined
        
    Returns:
        argparse.Namespace with all required analyzer configuration
    """
    # Calculate max_workers if not provided
    if max_workers is None:
        cpu_count = os.cpu_count()
        if cpu_count is None or cpu_count <= 0:
            # Fallback to 1 if CPU count cannot be determined
            max_workers = 1
        else:
            # Cap at 10 to prevent resource exhaustion, but scale with available CPUs
            # Formula: min(10, max(1, cpu_count)) ensures at least 1, at most 10
            max_workers = min(10, max(1, cpu_count))
        
    else:
        # Validate user-provided max_workers
        try:
            original_max_workers = int(max_workers)
        except (ValueError, TypeError):
            log_warn(f"Invalid max_workers value: {max_workers}. Defaulting to 1.")
            max_workers = 1
        else:
            # Clamp to allowed range [1, 10]
            max_workers = min(10, max(1, original_max_workers))
            if original_max_workers != max_workers:
                log_warn(f"max_workers capped from {original_max_workers} to {max_workers} (valid range: 1-10)")
    
    args = argparse.Namespace()
    args.timeframe = timeframe
    args.no_menu = True
    args.limit = limit
    args.max_workers = max_workers
    args.osc_length = RANGE_OSCILLATOR_LENGTH
    args.osc_mult = RANGE_OSCILLATOR_MULTIPLIER
    args.osc_strategies = None
    args.enable_spc = True
    args.spc_k = 2
    args.spc_lookback = SPC_LOOKBACK
    args.spc_p_low = SPC_P_LOW
    args.spc_p_high = SPC_P_HIGH
    args.spc_min_signal_strength = SPC_STRATEGY_PARAMETERS['cluster_transition']['min_signal_strength']
    args.spc_min_rel_pos_change = SPC_STRATEGY_PARAMETERS['cluster_transition']['min_rel_pos_change']
    args.spc_min_regime_strength = SPC_STRATEGY_PARAMETERS['regime_following']['min_regime_strength']
    args.spc_min_cluster_duration = SPC_STRATEGY_PARAMETERS['regime_following']['min_cluster_duration']
    args.spc_extreme_threshold = SPC_STRATEGY_PARAMETERS['mean_reversion']['extreme_threshold']
    args.spc_min_extreme_duration = SPC_STRATEGY_PARAMETERS['mean_reversion']['min_extreme_duration']
    args.spc_strategy = "all"
    args.enable_xgboost = True
    args.enable_hmm = True
    args.hmm_window_size = HMM_WINDOW_SIZE_DEFAULT
    args.hmm_window_kama = HMM_WINDOW_KAMA_DEFAULT
    args.hmm_fast_kama = HMM_FAST_KAMA_DEFAULT
    args.hmm_slow_kama = HMM_SLOW_KAMA_DEFAULT
    args.hmm_orders_argrelextrema = HMM_HIGH_ORDER_ORDERS_ARGRELEXTREMA_DEFAULT
    args.hmm_strict_mode = HMM_HIGH_ORDER_STRICT_MODE_DEFAULT
    args.enable_random_forest = True
    args.random_forest_model_path = None
    args.use_decision_matrix = True
    args.voting_threshold = DECISION_MATRIX_VOTING_THRESHOLD
    args.min_votes = DECISION_MATRIX_MIN_VOTES
    args.ema_len = 28
    args.hma_len = 28
    args.wma_len = 28
    args.dema_len = 28
    args.lsma_len = 28
    args.kama_len = 28
    args.robustness = "Medium"
    args.lambda_param = 0.5
    args.decay = 0.1
    args.cutout = 5
    args.min_signal = 0.01
    args.max_symbols = None
    return args


@contextlib.contextmanager
def _protect_stdin_windows():
    """
    Context manager to protect and restore stdin on Windows.
    
    Prevents "I/O operation on closed file" errors when analyzers
    may close stdin during initialization.
    
    Yields:
        None
    """
    saved_stdin = None
    if sys.platform == 'win32' and sys.stdin is not None:
        try:
            saved_stdin = sys.stdin
            if hasattr(sys.stdin, 'closed') and sys.stdin.closed:
                try:
                    sys.stdin = open('CON', 'r', encoding='utf-8', errors='replace')
                    saved_stdin = sys.stdin
                except (OSError, IOError):
                    pass
        except (AttributeError, ValueError, OSError, IOError):
            pass
    
    try:
        yield
    finally:
        # Always restore stdin after operation
        if sys.platform == 'win32' and saved_stdin is not None:
            try:
                if sys.stdin is None or (hasattr(sys.stdin, 'closed') and sys.stdin.closed):
                    if not (hasattr(saved_stdin, 'closed') and saved_stdin.closed):
                        sys.stdin = saved_stdin
                    else:
                        try:
                            sys.stdin = open('CON', 'r', encoding='utf-8', errors='replace')
                        except (OSError, IOError):
                            pass
            except (AttributeError, ValueError, OSError, IOError):
                pass


def _create_and_setup_analyzer(
    analyzer_class: Union[type[VotingAnalyzer], type[HybridAnalyzer]],
    args: argparse.Namespace,
    timeframe: str
) -> Union[VotingAnalyzer, HybridAnalyzer]:
    """
    Create analyzer instance and setup with timeframe.
    
    Args:
        analyzer_class: VotingAnalyzer or HybridAnalyzer class
        args: argparse.Namespace with analyzer configuration
        timeframe: Timeframe string for analysis
        
    Returns:
        Configured analyzer instance
    """
    exchange_manager = ExchangeManager()
    data_fetcher = DataFetcher(exchange_manager)
    analyzer = analyzer_class(args, data_fetcher)
    analyzer.selected_timeframe = timeframe
    analyzer.atc_analyzer.selected_timeframe = timeframe
    return analyzer


def pre_filter_symbols_with_voting(
    all_symbols: List[str],
    timeframe: str,
    limit: int
) -> List[str]:
    """
    Pre-filter symbols using VotingAnalyzer to select all symbols with signals.
    
    Runs VotingAnalyzer in-process to avoid stdin issues.
    Returns ALL symbols that have signals from VotingAnalyzer (LONG or SHORT),
    sorted by weighted_score descending.
    
    Args:
        all_symbols: List of all symbols to filter
        timeframe: Timeframe string for analysis
        limit: Number of candles to fetch per symbol
        
    Returns:
        List of all filtered symbols with signals, sorted by weighted_score descending.
        Returns all_symbols if filtering fails or no signals found.
    """
    if not all_symbols:
        return all_symbols
    
    total_symbols = len(all_symbols)
    log_info(f"Pre-filtering symbols using VotingAnalyzer (selecting all symbols with signals from {total_symbols} total symbols)...")
    
    with _protect_stdin_windows():
        try:
            args = _create_analyzer_args(timeframe, limit)
            analyzer = _create_and_setup_analyzer(VotingAnalyzer, args, timeframe)
            
            # Run ATC scan first to get initial signals
            log_info("Running ATC scan for pre-filtering...")
            if not analyzer.run_atc_scan():
                log_warn("No ATC signals found from VotingAnalyzer, cannot pre-filter")
                return all_symbols
            
            # Calculate signals for all indicators and apply voting system
            log_info("Calculating signals from all indicators...")
            analyzer.calculate_and_vote()
            
            # Extract symbols from long_signals_final and short_signals_final
            # Combine LONG and SHORT, but only include symbols that are in all_symbols
            # Sort by weighted_score descending
            all_symbols_set = set(all_symbols)  # For fast lookup
            all_signals_list = []
            
            if not analyzer.long_signals_final.empty:
                df = analyzer.long_signals_final
                mask = df['symbol'].notna() & (df['symbol'] != '') & df['symbol'].isin(all_symbols_set)
                filtered = df.loc[mask, ['symbol', 'weighted_score']].copy()
                filtered['direction'] = 'LONG'
                all_signals_list.extend(filtered.itertuples(index=False, name=None))
            
            if not analyzer.short_signals_final.empty:
                df = analyzer.short_signals_final
                mask = df['symbol'].notna() & (df['symbol'] != '') & df['symbol'].isin(all_symbols_set)
                filtered = df.loc[mask, ['symbol', 'weighted_score']].copy()
                filtered['direction'] = 'SHORT'
                all_signals_list.extend(filtered.itertuples(index=False, name=None))
            
            if not all_signals_list:
                log_warn("No signals found from VotingAnalyzer for the specified symbols, scanning all symbols instead")
                return all_symbols
            
            # Sort by weighted_score descending
            all_signals_list.sort(key=lambda x: x[1], reverse=True)
            
            # Get ALL symbols with signals (no percentage limit)
            filtered_symbols = [symbol for symbol, _, _ in all_signals_list]
            
            log_success(f"Pre-filter complete: {len(filtered_symbols)}/{total_symbols} symbols selected (all symbols with signals)")
            return filtered_symbols
            
        except Exception as e:
            log_error(f"Error during pre-filtering: {e}")
            log_warn("Falling back to scanning all symbols")
            import traceback
            traceback.print_exc()
            return all_symbols


def pre_filter_symbols_with_hybrid(
    all_symbols: List[str],
    timeframe: str,
    limit: int
) -> List[str]:
    """
    Pre-filter symbols using HybridAnalyzer to select all symbols with signals.
    
    Runs HybridAnalyzer in-process to avoid stdin issues.
    Returns ALL symbols that have signals from HybridAnalyzer (LONG or SHORT),
    after sequential filtering through ATC, Range Oscillator, SPC, and Decision Matrix.
    
    Args:
        all_symbols: List of all symbols to filter
        timeframe: Timeframe string for analysis
        limit: Number of candles to fetch per symbol
        
    Returns:
        List of all filtered symbols with signals.
        Returns all_symbols if filtering fails or no signals found.
    """
    if not all_symbols:
        return all_symbols
    
    total_symbols = len(all_symbols)
    log_info(f"Pre-filtering symbols using HybridAnalyzer (selecting all symbols with signals from {total_symbols} total symbols)...")
    
    with _protect_stdin_windows():
        try:
            args = _create_analyzer_args(timeframe, limit)
            analyzer = _create_and_setup_analyzer(HybridAnalyzer, args, timeframe)
            
            # Run ATC scan first to get initial signals
            log_info("Running ATC scan for pre-filtering...")
            if not analyzer.run_atc_scan():
                log_warn("No ATC signals found from HybridAnalyzer, cannot pre-filter")
                return all_symbols
            
            # Filter by Range Oscillator
            log_info("Filtering by Range Oscillator...")
            analyzer.filter_by_oscillator()
            
            # Calculate SPC signals if enabled
            if args.enable_spc:
                log_info("Calculating SPC signals...")
                analyzer.calculate_spc_signals_for_all()
            
            # Apply Decision Matrix voting if enabled
            if args.use_decision_matrix:
                log_info("Applying Decision Matrix voting system...")
                analyzer.filter_by_decision_matrix()
            
            # Extract symbols from long_signals_confirmed and short_signals_confirmed
            # Combine LONG and SHORT, but only include symbols that are in all_symbols
            # Use vectorized operations for better performance (O(n) instead of O(n²))
            all_symbols_set = set(all_symbols)  # For fast lookup
            
            # Extract valid symbols from both long and short signals
            filtered_symbols_set = _extract_valid_symbols(
                analyzer.long_signals_confirmed, 'symbol', all_symbols_set
            )
            filtered_symbols_set.update(_extract_valid_symbols(
                analyzer.short_signals_confirmed, 'symbol', all_symbols_set
            ))
            
            # Convert set to list for return value
            filtered_symbols = list(filtered_symbols_set)
            
            if not filtered_symbols:
                log_warn("No signals found from HybridAnalyzer for the specified symbols, scanning all symbols instead")
                return all_symbols
            
            log_success(f"Pre-filter complete: {len(filtered_symbols)}/{total_symbols} symbols selected (all symbols with signals)")
            return filtered_symbols
            
        except Exception as e:
            log_error(f"Error during pre-filtering: {e}")
            log_warn("Falling back to scanning all symbols")
            import traceback
            traceback.print_exc()
            return all_symbols

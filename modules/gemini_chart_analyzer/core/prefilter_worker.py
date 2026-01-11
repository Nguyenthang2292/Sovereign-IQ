
from pathlib import Path
from typing import List
import argparse
import json
import sys

"""
Pre-filter worker module for running VotingAnalyzer in subprocess.

This module provides a function that can be run in a separate process
to avoid stdin issues on Windows when running VotingAnalyzer.
"""


# Add project root to sys.path
# File is at: modules/gemini_chart_analyzer/core/prefilter_worker.py
# Project root is: modules/gemini_chart_analyzer/core/ -> modules/gemini_chart_analyzer/ -> modules/ -> project_root
if "__file__" in globals():
    current_file = Path(__file__).resolve()
    # Go up 4 levels: core -> gemini_chart_analyzer -> modules -> project_root
    project_root = current_file.parent.parent.parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

from config import (
    DECISION_MATRIX_MIN_VOTES,
    DECISION_MATRIX_VOTING_THRESHOLD,
    HMM_FAST_KAMA_DEFAULT,
    HMM_HIGH_ORDER_ORDERS_ARGRELEXTREMA_DEFAULT,
    HMM_HIGH_ORDER_STRICT_MODE_DEFAULT,
    HMM_SLOW_KAMA_DEFAULT,
    HMM_WINDOW_KAMA_DEFAULT,
    HMM_WINDOW_SIZE_DEFAULT,
    RANGE_OSCILLATOR_LENGTH,
    RANGE_OSCILLATOR_MULTIPLIER,
    SPC_LOOKBACK,
    SPC_P_HIGH,
    SPC_P_LOW,
    SPC_STRATEGY_PARAMETERS,
)
from core.voting_analyzer import VotingAnalyzer
from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager


def run_prefilter_worker(all_symbols: List[str], percentage: float, timeframe: str, limit: int) -> List[str]:
    """
    Run pre-filter in a worker function (can be called from subprocess).

    Args:
        all_symbols: List of all symbols to filter
        percentage: Percentage of symbols to select (0-100)
        timeframe: Timeframe string for analysis
        limit: Number of candles to fetch per symbol

    Returns:
        List of filtered symbols (top % by weighted_score), sorted by weighted_score descending
    """
    if not all_symbols or percentage <= 0.0:
        return all_symbols

    total_symbols = len(all_symbols)
    target_count = int(total_symbols * percentage / 100.0)

    # Validate target count
    if target_count <= 0:
        return all_symbols

    try:
        # Create args namespace for VotingAnalyzer with default values
        args = argparse.Namespace()
        args.timeframe = timeframe
        args.no_menu = True
        args.limit = limit
        args.max_workers = 10
        args.osc_length = RANGE_OSCILLATOR_LENGTH
        args.osc_mult = RANGE_OSCILLATOR_MULTIPLIER
        args.osc_strategies = None  # Use all strategies
        args.enable_spc = True
        args.spc_k = 2
        args.spc_lookback = SPC_LOOKBACK
        args.spc_p_low = SPC_P_LOW
        args.spc_p_high = SPC_P_HIGH
        args.spc_min_signal_strength = SPC_STRATEGY_PARAMETERS["cluster_transition"]["min_signal_strength"]
        args.spc_min_rel_pos_change = SPC_STRATEGY_PARAMETERS["cluster_transition"]["min_rel_pos_change"]
        args.spc_min_regime_strength = SPC_STRATEGY_PARAMETERS["regime_following"]["min_regime_strength"]
        args.spc_min_cluster_duration = SPC_STRATEGY_PARAMETERS["regime_following"]["min_cluster_duration"]
        args.spc_extreme_threshold = SPC_STRATEGY_PARAMETERS["mean_reversion"]["extreme_threshold"]
        args.spc_min_extreme_duration = SPC_STRATEGY_PARAMETERS["mean_reversion"]["min_extreme_duration"]
        args.spc_strategy = "all"
        args.enable_xgboost = True  # Default enabled
        args.enable_hmm = True  # Default enabled
        args.hmm_window_size = HMM_WINDOW_SIZE_DEFAULT
        args.hmm_window_kama = HMM_WINDOW_KAMA_DEFAULT
        args.hmm_fast_kama = HMM_FAST_KAMA_DEFAULT
        args.hmm_slow_kama = HMM_SLOW_KAMA_DEFAULT
        args.hmm_orders_argrelextrema = HMM_HIGH_ORDER_ORDERS_ARGRELEXTREMA_DEFAULT
        args.hmm_strict_mode = HMM_HIGH_ORDER_STRICT_MODE_DEFAULT
        args.enable_random_forest = True  # Default enabled
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

        # Create VotingAnalyzer instance
        exchange_manager = ExchangeManager()
        data_fetcher = DataFetcher(exchange_manager)
        analyzer = VotingAnalyzer(args, data_fetcher)
        analyzer.selected_timeframe = timeframe
        analyzer.atc_analyzer.selected_timeframe = timeframe

        # Run ATC scan first to get initial signals
        if not analyzer.run_atc_scan():
            return all_symbols  # No signals found, return all symbols

        # Calculate signals for all indicators and apply voting system
        analyzer.calculate_and_vote()

        # Extract symbols from long_signals_final and short_signals_final
        # Combine LONG and SHORT, but only include symbols that are in all_symbols
        # Sort by weighted_score descending
        all_symbols_set = set(all_symbols)  # For fast lookup
        all_signals_list = []

        if not analyzer.long_signals_final.empty:
            for _, row in analyzer.long_signals_final.iterrows():
                symbol = row.get("symbol", "")
                weighted_score = row.get("weighted_score", 0.0)
                if symbol and symbol in all_symbols_set:
                    all_signals_list.append((symbol, weighted_score, "LONG"))

        if not analyzer.short_signals_final.empty:
            for _, row in analyzer.short_signals_final.iterrows():
                symbol = row.get("symbol", "")
                weighted_score = row.get("weighted_score", 0.0)
                if symbol and symbol in all_symbols_set:
                    all_signals_list.append((symbol, weighted_score, "SHORT"))

        if not all_signals_list:
            return all_symbols  # No signals found, return all symbols

        # Sort by weighted_score descending
        all_signals_list.sort(key=lambda x: x[1], reverse=True)

        # Get top % symbols
        filtered_symbols = [symbol for symbol, _, _ in all_signals_list[:target_count]]

        return filtered_symbols

    except Exception:
        # On any error, return all symbols (fallback)
        return all_symbols


def main():
    """
    Main entry point for subprocess execution.
    Reads input from stdin (JSON) and writes output to stdout (JSON).
    """
    try:
        # Read input from stdin (JSON format)
        input_data = json.load(sys.stdin)

        all_symbols = input_data["all_symbols"]
        percentage = input_data["percentage"]
        timeframe = input_data["timeframe"]
        limit = input_data["limit"]

        # Run pre-filter
        filtered_symbols = run_prefilter_worker(
            all_symbols=all_symbols, percentage=percentage, timeframe=timeframe, limit=limit
        )

        # Write output to stdout (JSON format)
        output_data = {"filtered_symbols": filtered_symbols, "success": True}
        json.dump(output_data, sys.stdout)
        sys.stdout.flush()

    except Exception as e:
        # On error, return original symbols
        try:
            input_data = json.load(sys.stdin)
            all_symbols = input_data.get("all_symbols", [])
        except Exception:
            all_symbols = []

        output_data = {"filtered_symbols": all_symbols, "success": False, "error": str(e)}
        json.dump(output_data, sys.stdout)
        sys.stdout.flush()


if __name__ == "__main__":
    main()

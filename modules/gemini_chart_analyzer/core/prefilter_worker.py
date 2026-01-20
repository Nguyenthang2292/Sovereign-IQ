"""
Pre-filter worker module for running VotingAnalyzer.

This module provides a function to run VotingAnalyzer for pre-filtering symbols
before Gemini batch analysis. Can be called directly in the same process.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

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
from modules.common.ui.logging import log_error, log_info, log_success, log_warn


def _filter_stage_1_atc(
    analyzer: VotingAnalyzer,
    all_symbols: List[str],
) -> List[str]:
    """
    Filter Stage 1: ATC scan - keep 100% of symbols that pass ATC.

    Args:
        analyzer: VotingAnalyzer instance
        all_symbols: List of all symbols to filter

    Returns:
        List of symbols that passed ATC scan (100% of ATC results)
    """
    log_info("[Pre-filter Stage 1] Running ATC scan...")
    if not analyzer.run_atc_scan():
        log_warn("[Pre-filter Stage 1] No ATC signals found, returning all symbols")
        return all_symbols

    # Extract all symbols from ATC results (100%)
    all_symbols_set = set(all_symbols)
    stage1_symbols = []

    if not analyzer.long_signals_atc.empty:
        for _, row in analyzer.long_signals_atc.iterrows():
            symbol = row.get("symbol", "")
            if symbol and symbol in all_symbols_set:
                stage1_symbols.append(symbol)

    if not analyzer.short_signals_atc.empty:
        for _, row in analyzer.short_signals_atc.iterrows():
            symbol = row.get("symbol", "")
            if symbol and symbol in all_symbols_set and symbol not in stage1_symbols:
                stage1_symbols.append(symbol)

    log_success(f"[Pre-filter Stage 1] ATC filter: {len(stage1_symbols)}/{len(all_symbols)} symbols passed")
    return stage1_symbols if stage1_symbols else all_symbols


def _filter_stage_2_osc_spc(
    analyzer: VotingAnalyzer,
    stage1_symbols: List[str],
    timeframe: str,
    limit: int,
) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
    """
    Filter Stage 2: Range Oscillator + SPC → Voting → Decision Matrix.
    Keep 100% of symbols that pass voting.

    Args:
        analyzer: VotingAnalyzer instance
        stage1_symbols: List of symbols from Stage 1
        timeframe: Timeframe string for analysis
        limit: Number of candles to fetch per symbol

    Returns:
        Tuple of (list of symbols that passed Stage 2 voting, dict with stage2_signals DataFrames)
    """
    if not stage1_symbols:
        return [], {}

    log_info(f"[Pre-filter Stage 2] Calculating Range Oscillator + SPC signals for {len(stage1_symbols)} symbols...")

    # Combine long and short ATC signals for Stage 1 symbols
    stage1_atc_signals = pd.DataFrame()
    if not analyzer.long_signals_atc.empty:
        stage1_long = analyzer.long_signals_atc[analyzer.long_signals_atc["symbol"].isin(stage1_symbols)]
        if not stage1_long.empty:
            stage1_atc_signals = pd.concat([stage1_atc_signals, stage1_long], ignore_index=True)

    if not analyzer.short_signals_atc.empty:
        stage1_short = analyzer.short_signals_atc[analyzer.short_signals_atc["symbol"].isin(stage1_symbols)]
        if not stage1_short.empty:
            stage1_atc_signals = pd.concat([stage1_atc_signals, stage1_short], ignore_index=True)

    if stage1_atc_signals.empty:
        log_warn("[Pre-filter Stage 2] No ATC signals found for Stage 1 symbols")
        return stage1_symbols, {}

    # Calculate signals for Range Oscillator and SPC only
    long_signals = (
        stage1_atc_signals[stage1_atc_signals["signal"] > 0] if not stage1_atc_signals.empty else pd.DataFrame()
    )
    short_signals = (
        stage1_atc_signals[stage1_atc_signals["signal"] < 0] if not stage1_atc_signals.empty else pd.DataFrame()
    )

    stage2_symbols = []
    stage2_signals = {"long": pd.DataFrame(), "short": pd.DataFrame()}

    # Process LONG signals
    if not long_signals.empty:
        long_with_signals = analyzer.calculate_signals_for_all_indicators(
            atc_signals_df=long_signals,
            signal_type="LONG",
            indicators_to_calculate=["oscillator", "spc"],
        )
        if not long_with_signals.empty:
            # Apply voting system with only ATC + Range Osc + SPC
            long_final = analyzer.apply_voting_system(
                signals_df=long_with_signals,
                signal_type="LONG",
                indicators_to_include=["atc", "oscillator", "spc"],
            )
            stage2_signals["long"] = long_final
            if not long_final.empty:
                for _, row in long_final.iterrows():
                    symbol = row.get("symbol", "")
                    if symbol:
                        stage2_symbols.append(symbol)

    # Process SHORT signals
    if not short_signals.empty:
        short_with_signals = analyzer.calculate_signals_for_all_indicators(
            atc_signals_df=short_signals,
            signal_type="SHORT",
            indicators_to_calculate=["oscillator", "spc"],
        )
        if not short_with_signals.empty:
            # Apply voting system with only ATC + Range Osc + SPC
            short_final = analyzer.apply_voting_system(
                signals_df=short_with_signals,
                signal_type="SHORT",
                indicators_to_include=["atc", "oscillator", "spc"],
            )
            stage2_signals["short"] = short_final
            if not short_final.empty:
                for _, row in short_final.iterrows():
                    symbol = row.get("symbol", "")
                    if symbol and symbol not in stage2_symbols:
                        stage2_symbols.append(symbol)

    log_success(
        f"[Pre-filter Stage 2] Range Osc + SPC filter: {len(stage2_symbols)}/{len(stage1_symbols)} symbols passed"
    )
    return (stage2_symbols if stage2_symbols else stage1_symbols, stage2_signals)


def _filter_stage_3_ml_models(
    analyzer: VotingAnalyzer,
    stage2_symbols: List[str],
    stage2_signals: Dict[str, pd.DataFrame],
    timeframe: str,
    limit: int,
    rf_model_path: Optional[str] = None,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Filter Stage 3: XGBoost + HMM + RF → Voting → Decision Matrix.
    Keep 100% of symbols that pass voting.
    Note: In fast mode, this stage still calculates all ML models (ignores fast_mode flag).

    Stage 3 only calculates ML models and excludes Range Oscillator and SPC from voting.
    Voting includes: ATC + XGBoost + HMM + RF only.

    Args:
        analyzer: VotingAnalyzer instance
        stage2_symbols: List of symbols from Stage 2
        stage2_signals: Dict with 'long' and 'short' DataFrames from Stage 2 (with ATC, Range Osc, SPC signals)
                       Note: Range Osc and SPC signals are not used in Stage 3 voting
        timeframe: Timeframe string for analysis
        limit: Number of candles to fetch per symbol
        rf_model_path: Optional path to Random Forest model

    Returns:
        Tuple of (list of symbols that passed Stage 3 voting, dict of {symbol: weighted_score})
    """
    if not stage2_symbols:
        return [], {}

    log_info(f"[Pre-filter Stage 3] Calculating ML models (XGBoost + HMM + RF) for {len(stage2_symbols)} symbols...")

    # Temporarily enable ML models for Stage 3 (ignore fast_mode)
    original_enable_xgboost = getattr(analyzer.args, "enable_xgboost", False)
    original_enable_hmm = getattr(analyzer.args, "enable_hmm", False)
    original_enable_rf = getattr(analyzer.args, "enable_random_forest", False)

    analyzer.args.enable_xgboost = True
    analyzer.args.enable_hmm = True
    if rf_model_path:
        analyzer.args.enable_random_forest = True

    try:
        # Combine long and short ATC signals for Stage 2 symbols
        stage2_atc_signals = pd.DataFrame()
        if not analyzer.long_signals_atc.empty:
            stage2_long = analyzer.long_signals_atc[analyzer.long_signals_atc["symbol"].isin(stage2_symbols)]
            if not stage2_long.empty:
                stage2_atc_signals = pd.concat([stage2_atc_signals, stage2_long], ignore_index=True)

        if not analyzer.short_signals_atc.empty:
            stage2_short = analyzer.short_signals_atc[analyzer.short_signals_atc["symbol"].isin(stage2_symbols)]
            if not stage2_short.empty:
                stage2_atc_signals = pd.concat([stage2_atc_signals, stage2_short], ignore_index=True)

        if stage2_atc_signals.empty:
            log_warn("[Pre-filter Stage 3] No ATC signals found for Stage 2 symbols")
            return stage2_symbols, {}, {}

        # Calculate only ML models for Stage 2 symbols (XGBoost, HMM, RF)
        # Range Osc and SPC are excluded - they were already used in Stage 2
        long_signals = (
            stage2_atc_signals[stage2_atc_signals["signal"] > 0] if not stage2_atc_signals.empty else pd.DataFrame()
        )
        short_signals = (
            stage2_atc_signals[stage2_atc_signals["signal"] < 0] if not stage2_atc_signals.empty else pd.DataFrame()
        )

        stage3_symbols = []
        stage3_scores = {}  # {symbol: weighted_score}

        # Process LONG signals - calculate only ML models (XGBoost, HMM, RF)
        if not long_signals.empty:
            long_with_ml_signals = analyzer.calculate_signals_for_all_indicators(
                atc_signals_df=long_signals,
                signal_type="LONG",
                indicators_to_calculate=["xgboost", "hmm", "random_forest"],  # Only ML models
            )
            if not long_with_ml_signals.empty:
                # Apply voting system with ATC + ML models only (exclude Range Osc and SPC)
                long_final = analyzer.apply_voting_system(
                    signals_df=long_with_ml_signals,
                    signal_type="LONG",
                    indicators_to_include=["atc", "xgboost", "hmm", "random_forest"],  # ATC + ML models only
                )
                if not long_final.empty:
                    for _, row in long_final.iterrows():
                        symbol = row.get("symbol", "")
                        if symbol:
                            stage3_symbols.append(symbol)
                            stage3_scores[symbol] = row.get("weighted_score", 0.0)

        # Process SHORT signals - calculate only ML models (XGBoost, HMM, RF)
        if not short_signals.empty:
            short_with_ml_signals = analyzer.calculate_signals_for_all_indicators(
                atc_signals_df=short_signals,
                signal_type="SHORT",
                indicators_to_calculate=["xgboost", "hmm", "random_forest"],  # Only ML models
            )
            if not short_with_ml_signals.empty:
                # Apply voting system with ATC + ML models only (exclude Range Osc and SPC)
                short_final = analyzer.apply_voting_system(
                    signals_df=short_with_ml_signals,
                    signal_type="SHORT",
                    indicators_to_include=["atc", "xgboost", "hmm", "random_forest"],  # ATC + ML models only
                )
                if not short_final.empty:
                    for _, row in short_final.iterrows():
                        symbol = row.get("symbol", "")
                        if symbol:
                            if symbol not in stage3_symbols:
                                stage3_symbols.append(symbol)

                            # Use highest score if symbol appears in both long and short
                            stage3_scores[symbol] = max(stage3_scores.get(symbol, 0.0), row.get("weighted_score", 0.0))

        log_success(
            f"[Pre-filter Stage 3] ML models filter: {len(stage3_symbols)}/{len(stage2_symbols)} symbols passed"
        )
        return stage3_symbols if stage3_symbols else stage2_symbols, stage3_scores

    finally:
        # Restore original ML model flags
        analyzer.args.enable_xgboost = original_enable_xgboost
        analyzer.args.enable_hmm = original_enable_hmm
        analyzer.args.enable_random_forest = original_enable_rf


def run_prefilter_worker(
    all_symbols: List[str],
    percentage: float,
    timeframe: str,
    limit: int,
    mode: str = "voting",
    fast_mode: bool = True,
    spc_config: Optional[Dict[str, Any]] = None,
    rf_model_path: Optional[str] = None,
) -> List[str]:
    """
    Run pre-filter with 3-stage sequential filtering workflow.

    Stage 1: ATC scan → keep 100% of symbols that pass ATC
    Stage 2: Range Oscillator + SPC → Voting → Decision Matrix → keep 100% that pass
    Stage 3: XGBoost + HMM + RF → Voting → Decision Matrix → keep 100% that pass

    Args:
        all_symbols: List of all symbols to filter
        percentage: Percentage of symbols to select (0-100) - applied to final result if needed
        timeframe: Timeframe string for analysis
        limit: Number of candles to fetch per symbol
        mode: Pre-filter mode ('voting' or 'hybrid')
        fast_mode: Whether to run in fast mode (Stage 3 still calculates all ML models)
        spc_config: Optional SPC configuration
        rf_model_path: Optional path to Random Forest model

    Returns:
        List of filtered symbols from Stage 3 (or percentage of final result if percentage < 100)
    """
    if not all_symbols:
        return all_symbols

    total_symbols = len(all_symbols)

    try:
        log_info(f"[Pre-filter] Starting 3-stage pre-filter for {total_symbols} symbols")

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
        # Use spc_config if provided
        if spc_config:
            # Merging spc_config into args could be complex, but let's at least set what we can
            # For now, VotingAnalyzer might handle its own config if we pass it,
            # but let's assume we just need to set the flag.
            pass

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
        # In fast mode, ML models are disabled initially but enabled in Stage 3
        # This allows Stage 1-2 to run without ML overhead
        args.enable_xgboost = not fast_mode  # Stage 1-2 only, Stage 3 force-enables
        args.enable_hmm = not fast_mode  # Stage 1-2 only, Stage 3 force-enables
        args.hmm_window_size = HMM_WINDOW_SIZE_DEFAULT
        args.hmm_window_kama = HMM_WINDOW_KAMA_DEFAULT
        args.hmm_fast_kama = HMM_FAST_KAMA_DEFAULT
        args.hmm_slow_kama = HMM_SLOW_KAMA_DEFAULT
        args.hmm_orders_argrelextrema = HMM_HIGH_ORDER_ORDERS_ARGRELEXTREMA_DEFAULT
        args.hmm_strict_mode = HMM_HIGH_ORDER_STRICT_MODE_DEFAULT
        args.enable_random_forest = True if rf_model_path else False
        args.random_forest_model_path = rf_model_path
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
        log_info("[Pre-filter] Initializing VotingAnalyzer...")
        exchange_manager = ExchangeManager()
        data_fetcher = DataFetcher(exchange_manager)
        analyzer = VotingAnalyzer(args, data_fetcher)
        analyzer.selected_timeframe = timeframe
        analyzer.atc_analyzer.selected_timeframe = timeframe

        # Stage 1: ATC Filter - keep 100% of symbols that pass ATC
        stage1_symbols = _filter_stage_1_atc(analyzer, all_symbols)

        if not stage1_symbols:
            log_warn("[Pre-filter] Stage 1 returned no symbols, returning all symbols")
            return all_symbols

        # Stage 2: Range Oscillator + SPC Filter - keep 100% that pass voting
        stage2_symbols, stage2_signals = _filter_stage_2_osc_spc(analyzer, stage1_symbols, timeframe, limit)

        if not stage2_symbols:
            log_warn("[Pre-filter] Stage 2 returned no symbols, using Stage 1 results")
            stage2_symbols = stage1_symbols
            stage2_signals = {}

        # Stage 3: ML Models Filter (XGBoost + HMM + RF) - keep 100% that pass voting
        stage3_symbols, stage3_scores = _filter_stage_3_ml_models(
            analyzer, stage2_symbols, stage2_signals, timeframe, limit, rf_model_path
        )

        if not stage3_symbols:
            log_warn("[Pre-filter] Stage 3 returned no symbols, using Stage 2 results")
            stage3_symbols = stage2_symbols

        # Apply percentage filter to final result if needed (for backward compatibility)
        filtered_symbols = stage3_symbols
        if percentage > 0.0 and percentage < 100.0:
            target_count = int(len(stage3_symbols) * percentage / 100.0)
            if target_count > 0 and target_count < len(stage3_symbols):
                # Apply percentage filter with sorting by weighted score
                log_info(
                    f"[Pre-filter] Applying percentage filter: selecting top {percentage}% ({target_count}/{len(stage3_symbols)})"
                )

                # Sort symbols by their scores (descending)
                sorted_symbols = sorted(stage3_symbols, key=lambda s: stage3_scores.get(s, 0.0), reverse=True)
                filtered_symbols = sorted_symbols[:target_count]

        log_success(
            f"[Pre-filter] Completed 3-stage filtering: {len(filtered_symbols)}/{total_symbols} symbols selected"
        )

        return filtered_symbols

    except Exception as e:
        # On any error, return all symbols (fallback)
        log_error(f"[Pre-filter] ERROR: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return all_symbols


def main():
    """
    Main entry point for subprocess execution.
    Reads input from stdin (JSON) and writes output to stdout (JSON).
    """
    try:
        # Read all input from stdin (e.g., piped JSON)
        input_lines = []
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            input_lines.append(line.rstrip("\n\r"))

        input_text = "\n".join(input_lines)

        if not input_text.strip():
            raise ValueError("No input data received on stdin")

        input_data = json.loads(input_text)

        all_symbols = input_data["all_symbols"]
        percentage = input_data["percentage"]
        timeframe = input_data["timeframe"]
        limit = input_data["limit"]

        # Run pre-filter
        filtered_symbols = run_prefilter_worker(
            all_symbols=all_symbols,
            percentage=percentage,
            timeframe=timeframe,
            limit=limit,
            mode=input_data.get("mode", "voting"),
            fast_mode=input_data.get("fast_mode", True),
            spc_config=input_data.get("spc_config"),
            rf_model_path=input_data.get("rf_model_path"),
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

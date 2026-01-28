"""
Pre-filter workflow module for running VotingAnalyzer.

This module provides the core workflow logic for the 4-stage pre-filtering process.
It coordinates between ATC, oscillators, SPC, and ML models to filter symbols.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path
# File is at: modules/gemini_chart_analyzer/core/prefilter/workflow.py
if "__file__" in globals():
    current_file = Path(__file__).resolve()
    # Go up 5 levels for new location:
    # prefilter -> core -> gemini_chart_analyzer -> modules -> project_root
    project_root = current_file.parent.parent.parent.parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

from core.voting_analyzer import VotingAnalyzer
from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager
from modules.common.ui.logging import log_error, log_info, log_success, log_warn
from modules.gemini_chart_analyzer.core.prefilter.args_builder import build_voting_analyzer_args
from modules.gemini_chart_analyzer.core.prefilter.sampling import run_sampling_stage
from modules.gemini_chart_analyzer.core.prefilter.stages import (
    filter_stage_1_atc as _filter_stage_1_atc,
    filter_stage_2_osc_spc as _filter_stage_2_osc_spc,
    filter_stage_3_ml_models as _filter_stage_3_ml_models,
)


def run_prefilter_worker(
    all_symbols: List[str],
    percentage: float,
    timeframe: str,
    limit: int,
    mode: str = "voting",
    fast_mode: bool = True,
    spc_config: Optional[Dict[str, Any]] = None,
    rf_model_path: Optional[str] = None,
    stage0_sample_percentage: Optional[float] = None,
    stage0_sampling_strategy: str = "random",
    stage0_stratified_strata_count: int = 3,
    stage0_hybrid_top_percentage: float = 50.0,
    atc_performance: Optional[Dict[str, Any]] = None,
    approximate_ma_scanner: Optional[Dict[str, Any]] = None,
    auto_skip_threshold: int = 10,
    use_atc_performance: bool = True,
) -> List[str]:
    """
    Run pre-filter with 4-stage sequential filtering workflow.

    Stage 0: Sampling (optional) -> select a percentage of symbols using chosen strategy
    Stage 1: ATC scan -> keep 100% of symbols that pass ATC
    Stage 2: Range Oscillator + SPC -> Voting -> Decision Matrix -> keep 100% that pass
    Stage 3: XGBoost + HMM + RF -> Voting -> Decision Matrix -> keep 100% that pass

    Args:
        all_symbols: List of all symbols to filter
        percentage: Percentage of symbols to select (0-100) - applied to final result if needed
        timeframe: Timeframe string for analysis
        limit: Number of candles to fetch per symbol
        mode: Pre-filter mode ('voting' or 'hybrid')
        fast_mode: Whether to run in fast mode (Stage 3 still calculates all ML models)
        spc_config: Optional SPC configuration
        rf_model_path: Optional path to Random Forest model
        stage0_sample_percentage: Optional percentage of symbols to sample before Stage 1 (1-100).
                                 If None or 100, no sampling is performed.
        stage0_sampling_strategy: Sampling strategy to use ('random', 'volume_weighted', 'stratified',
                                 'top_n_hybrid', 'systematic', 'liquidity_weighted')
        stage0_stratified_strata_count: Number of strata for stratified sampling (default: 3)
        stage0_hybrid_top_percentage: For top_n_hybrid: percentage of sample from top volume (default: 50%)
        atc_performance: Optional ATC performance configuration
        auto_skip_threshold: Auto-skip percentage filter if Stage 3 returns fewer symbols than this (default: 10)

    Returns:
        List of filtered symbols from Stage 3 (or percentage of final result if percentage < 100)
    """
    if not all_symbols:
        return all_symbols

    total_symbols = len(all_symbols)

    symbols_to_process, data_cache = run_sampling_stage(
        all_symbols=all_symbols,
        stage0_sample_percentage=stage0_sample_percentage,
        stage0_sampling_strategy=stage0_sampling_strategy,
        stage0_stratified_strata_count=stage0_stratified_strata_count,
        stage0_hybrid_top_percentage=stage0_hybrid_top_percentage,
    )

    try:
        log_info(f"[Pre-filter] Starting 3-stage pre-filter for {len(symbols_to_process)} symbols")

        args = build_voting_analyzer_args(
            timeframe=timeframe,
            limit=limit,
            fast_mode=fast_mode,
            spc_config=spc_config,
            rf_model_path=rf_model_path,
            atc_performance=atc_performance,
            approximate_ma_scanner=approximate_ma_scanner,
            use_atc_performance=use_atc_performance,
        )

        log_info("[Pre-filter] Initializing VotingAnalyzer...")

        exchange_manager = ExchangeManager()
        data_fetcher = DataFetcher(exchange_manager)
        analyzer = VotingAnalyzer(args, data_fetcher, ohlcv_cache=data_cache)
        analyzer.selected_timeframe = timeframe
        analyzer.atc_analyzer.selected_timeframe = timeframe

        stage1_symbols = _filter_stage_1_atc(analyzer, symbols_to_process)

        if not stage1_symbols:
            log_warn("[Pre-filter] Stage 1 returned no symbols, returning all symbols")
            return all_symbols

        stage2_symbols, stage2_signals = _filter_stage_2_osc_spc(analyzer, stage1_symbols, timeframe, limit)

        if not stage2_symbols:
            log_warn("[Pre-filter] Stage 2 returned no symbols, using Stage 1 results")
            stage2_symbols = stage1_symbols
            stage2_signals = {}

        stage3_symbols, stage3_scores = _filter_stage_3_ml_models(
            analyzer, stage2_symbols, stage2_signals, timeframe, limit, rf_model_path
        )

        if not stage3_symbols:
            log_warn("[Pre-filter] Stage 3 returned no symbols, using Stage 2 results")
            stage3_symbols = stage2_symbols

        filtered_symbols = stage3_symbols
        if percentage > 0.0 and percentage < 100.0:
            if len(stage3_symbols) < auto_skip_threshold:
                log_info(
                    f"[Pre-filter] Skipping percentage filter: Stage 3 returned only {len(stage3_symbols)} symbols "
                    f"(threshold: {auto_skip_threshold}). Using all Stage 3 results."
                )
            else:
                target_count = int(len(stage3_symbols) * percentage / 100.0)
                if target_count > 0 and target_count < len(stage3_symbols):
                    log_info(
                        f"[Pre-filter] Applying percentage filter: selecting top {percentage}% "
                        f"({target_count}/{len(stage3_symbols)})"
                    )

                    sorted_symbols = sorted(stage3_symbols, key=lambda s: stage3_scores.get(s, 0.0), reverse=True)
                    filtered_symbols = sorted_symbols[:target_count]

        log_success(
            f"[Pre-filter] Completed 3-stage filtering: {len(filtered_symbols)}/{total_symbols} symbols selected"
        )

        return filtered_symbols

    except Exception as e:
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

        output_data = {"filtered_symbols": filtered_symbols, "success": True}
        json.dump(output_data, sys.stdout)
        sys.stdout.flush()

    except Exception as e:
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

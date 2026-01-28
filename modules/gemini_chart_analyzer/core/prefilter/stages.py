"""Stage filters for prefilter workflow."""

from typing import Any, Dict, List, Optional, Tuple, cast

import pandas as pd

from modules.common.ui.logging import log_info, log_success, log_warn


def filter_stage_1_atc(
    analyzer: Any,
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


def filter_stage_2_osc_spc(
    analyzer: Any,
    stage1_symbols: List[str],
    timeframe: str,
    limit: int,
) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
    """
    Filter Stage 2: Range Oscillator + SPC -> Voting -> Decision Matrix.
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

    long_signals = cast(pd.DataFrame, stage1_atc_signals.loc[stage1_atc_signals["signal"] > 0])
    short_signals = cast(pd.DataFrame, stage1_atc_signals.loc[stage1_atc_signals["signal"] < 0])

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


def filter_stage_3_ml_models(
    analyzer: Any,
    stage2_symbols: List[str],
    stage2_signals: Dict[str, pd.DataFrame],
    timeframe: str,
    limit: int,
    rf_model_path: Optional[str] = None,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Filter Stage 3: XGBoost + HMM + RF -> Voting -> Decision Matrix.
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

    original_enable_xgboost = getattr(analyzer.args, "enable_xgboost", False)
    original_enable_hmm = getattr(analyzer.args, "enable_hmm", False)
    original_enable_rf = getattr(analyzer.args, "enable_random_forest", False)

    analyzer.args.enable_xgboost = True
    analyzer.args.enable_hmm = True
    if rf_model_path:
        analyzer.args.enable_random_forest = True

    try:
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
            return stage2_symbols, {}

        long_signals = cast(pd.DataFrame, stage2_atc_signals.loc[stage2_atc_signals["signal"] > 0])
        short_signals = cast(pd.DataFrame, stage2_atc_signals.loc[stage2_atc_signals["signal"] < 0])

        stage3_symbols = []
        stage3_scores = {}

        # Process LONG signals - calculate only ML models (XGBoost, HMM, RF)
        if not long_signals.empty:
            long_with_ml_signals = analyzer.calculate_signals_for_all_indicators(
                atc_signals_df=long_signals,
                signal_type="LONG",
                indicators_to_calculate=["xgboost", "hmm", "random_forest"],
            )
            if not long_with_ml_signals.empty:
                long_final = analyzer.apply_voting_system(
                    signals_df=long_with_ml_signals,
                    signal_type="LONG",
                    indicators_to_include=["atc", "xgboost", "hmm", "random_forest"],
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
                indicators_to_calculate=["xgboost", "hmm", "random_forest"],
            )
            if not short_with_ml_signals.empty:
                short_final = analyzer.apply_voting_system(
                    signals_df=short_with_ml_signals,
                    signal_type="SHORT",
                    indicators_to_include=["atc", "xgboost", "hmm", "random_forest"],
                )
                if not short_final.empty:
                    for _, row in short_final.iterrows():
                        symbol = row.get("symbol", "")
                        if symbol:
                            if symbol not in stage3_symbols:
                                stage3_symbols.append(symbol)

                            weighted_score = row.get("weighted_score", 0.0) or 0.0
                            stage3_scores[symbol] = max(stage3_scores.get(symbol, 0.0), weighted_score)

        log_success(
            f"[Pre-filter Stage 3] ML models filter: {len(stage3_symbols)}/{len(stage2_symbols)} symbols passed"
        )
        return stage3_symbols if stage3_symbols else stage2_symbols, stage3_scores

    finally:
        analyzer.args.enable_xgboost = original_enable_xgboost
        analyzer.args.enable_hmm = original_enable_hmm
        analyzer.args.enable_random_forest = original_enable_rf

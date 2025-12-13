"""
Display utilities for ATC + Range Oscillator + SPC Hybrid and Pure Voting.

This module contains functions for displaying configuration and voting metadata.
"""

import pandas as pd
from typing import Any

from modules.common.utils import (
    log_progress,
    log_data,
)
from modules.range_oscillator.cli import display_configuration


def display_config(
    selected_timeframe: str,
    args: Any,
    get_oscillator_params: callable,
    get_spc_params: callable = None,
    mode: str = "hybrid",
) -> None:
    """
    Display configuration information.
    
    Args:
        selected_timeframe: Selected timeframe for analysis
        args: Arguments namespace object
        get_oscillator_params: Function to get oscillator parameters
        get_spc_params: Function to get SPC parameters (optional)
        mode: "hybrid" or "voting" - determines display format
    """
    osc_params = get_oscillator_params()
    display_configuration(
        timeframe=selected_timeframe,
        limit=args.limit,
        min_signal=args.min_signal,
        max_workers=osc_params["max_workers"],
        strategies=osc_params["strategies"],
        max_symbols=args.max_symbols,
    )
    
    if args.enable_spc:
        if get_spc_params:
            spc_params = get_spc_params()
            log_progress("\nSPC Configuration (All 3 strategies enabled):")
            log_data(f"  K: {spc_params['k']}")
            log_data(f"  Lookback: {spc_params['lookback']}")
            log_data(f"  Percentiles: {spc_params['p_low']}% - {spc_params['p_high']}%")
            log_data(f"  Strategies: Cluster Transition, Regime Following, Mean Reversion")
    
    if hasattr(args, 'enable_xgboost') and args.enable_xgboost:
        log_progress("\nXGBoost Configuration:")
        log_data(f"  XGBoost Prediction: Enabled")
    
    if mode == "hybrid":
        if hasattr(args, 'use_decision_matrix') and args.use_decision_matrix:
            log_progress("\nDecision Matrix Configuration:")
            log_data(f"  Voting Threshold: {args.voting_threshold}")
            log_data(f"  Min Votes: {args.min_votes}")
    else:  # voting mode
        log_progress("\nDecision Matrix Configuration (Pure Voting):")
        log_data(f"  Voting Threshold: {args.voting_threshold}")
        log_data(f"  Min Votes: {args.min_votes}")


def display_voting_metadata(
    signals_df: pd.DataFrame,
    signal_type: str,
    show_spc_debug: bool = False,
) -> None:
    """
    Display voting metadata for signals.
    
    Args:
        signals_df: DataFrame containing signals with voting metadata
        signal_type: "LONG" or "SHORT"
        show_spc_debug: If True, show debug info for SPC when contribution is 0%
    """
    if signals_df.empty:
        return
    
    log_progress(f"\n{signal_type} Signals - Voting Breakdown:")
    log_progress("-" * 80)
    
    for idx, row in signals_df.head(10).iterrows():
        symbol = row['symbol']
        weighted_score = row.get('weighted_score', 0.0)
        voting_breakdown = row.get('voting_breakdown', {})
        feature_importance = row.get('feature_importance', {})
        weighted_impact = row.get('weighted_impact', {})
        
        log_data(f"\nSymbol: {symbol}")
        log_data(f"  Weighted Score: {weighted_score:.2%}")
        log_data(f"  Voting Breakdown:")
        
        for indicator, vote_info in voting_breakdown.items():
            vote = vote_info['vote']
            weight = vote_info['weight']
            contribution = vote_info['contribution']
            importance = feature_importance.get(indicator, 0.0)
            impact = weighted_impact.get(indicator, 0.0)
            
            vote_str = "✓" if vote == 1 else "✗"
            log_data(
                f"    {indicator.upper()}: {vote_str} "
                f"(Weight: {weight:.1%}, Impact: {impact:.1%}, "
                f"Importance: {importance:.1%}, Contribution: {contribution:.2%})"
            )
            
            # Debug: Show SPC strategy signals if SPC contribution is 0%
            if show_spc_debug and indicator == 'spc' and abs(contribution) < 0.0001:
                spc_ct_signal = row.get('spc_cluster_transition_signal', 0)
                spc_rf_signal = row.get('spc_regime_following_signal', 0)
                spc_mr_signal = row.get('spc_mean_reversion_signal', 0)
                spc_ct_strength = row.get('spc_cluster_transition_strength', 0.0)
                spc_rf_strength = row.get('spc_regime_following_strength', 0.0)
                spc_mr_strength = row.get('spc_mean_reversion_strength', 0.0)
                log_data(
                    f"      [DEBUG] SPC Strategy Signals: "
                    f"CT={spc_ct_signal} (strength={spc_ct_strength:.2f}), "
                    f"RF={spc_rf_signal} (strength={spc_rf_strength:.2f}), "
                    f"MR={spc_mr_signal} (strength={spc_mr_strength:.2f})"
                )


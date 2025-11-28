"""
Pairs Trading Analysis Main Program

Analyzes futures pairs on Binance to identify pairs trading opportunities:
- Loads 1h candle data from all futures pairs
- Calculates top best and worst performers
- Identifies pairs trading opportunities (long worst, short best)
- Validates pairs using cointegration and quantitative metrics
"""

import warnings
import pandas as pd

from modules.common.utils import configure_windows_stdio

# Fix encoding issues on Windows for interactive CLI runs only
configure_windows_stdio()

from colorama import Fore, init as colorama_init

from modules.config import (
    PAIRS_TRADING_OPPORTUNITY_PRESETS,
)
from modules.common.utils import (
    color_text,
    normalize_symbol_key,
    log_warn,
    log_error,
    log_info,
    log_analysis,
    log_data,
    log_success,
    log_progress,
)
from modules.common.ExchangeManager import ExchangeManager
from modules.common.DataFetcher import DataFetcher
from modules.pairs_trading import (
    PerformanceAnalyzer,
    PairsTradingAnalyzer,
    display_performers,
    display_pairs_opportunities,
    select_top_unique_pairs,
    ensure_symbols_in_candidate_pools,
    select_pairs_for_symbols,
    parse_args,
    prompt_interactive_mode,
    parse_weights,
    parse_symbols,
    prompt_weight_preset_selection,
    prompt_kalman_preset_selection,
    prompt_opportunity_preset_selection,
    prompt_target_pairs,
    prompt_candidate_depth,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
colorama_init(autoreset=True)


def main() -> None:
    """
    Main function for pairs trading analysis.

    Orchestrates the complete pairs trading analysis workflow:
    1. Parse command-line arguments and interactive prompts
    2. Initialize components (ExchangeManager, DataFetcher, analyzers)
    3. Fetch futures symbols from Binance
    4. Analyze performance for all symbols
    5. Build candidate pools for long/short sides
    6. Analyze pairs trading opportunities
    7. Validate and display results
    """

    args = parse_args()

    if not args.no_menu:
        menu_result = prompt_interactive_mode()
        if menu_result["mode"] == "auto":
            args.symbols = None
        else:
            args.symbols = menu_result["symbols_raw"]
        if not args.weights:
            args.weight_preset = prompt_weight_preset_selection(args.weight_preset)
        args.opportunity_preset = prompt_opportunity_preset_selection(args.opportunity_preset)
        # Kalman preset selection
        args.kalman_delta, args.kalman_obs_cov, args.kalman_preset = prompt_kalman_preset_selection(
            args.kalman_delta,
            args.kalman_obs_cov,
        )
        # OLS fit intercept selection
        default_ols = "Y" if args.ols_fit_intercept else "N"
        ols_input = input(
            color_text(
                f"Use intercept for OLS hedge ratio? [Y/n] (default {default_ols}): ",
                Fore.YELLOW,
            )
        ).strip().lower()
        if ols_input in {"y", "yes"}:
            args.ols_fit_intercept = True
        elif ols_input in {"n", "no"}:
            args.ols_fit_intercept = False
        
        # Target pairs selection
        args.pairs_count = prompt_target_pairs(args.pairs_count)
        
        # Candidate depth selection
        args.candidate_depth = prompt_candidate_depth(args.candidate_depth)

    # Parse weights if provided
    weights = parse_weights(args.weights, args.weight_preset)

    # Parse target symbols if provided
    target_symbol_inputs, parsed_target_symbols = parse_symbols(args.symbols)
    target_symbols = []
    if target_symbol_inputs:
        if log_info:
            log_info(f"Manual mode enabled for symbols: {', '.join(target_symbol_inputs)}")

    if log_analysis:
        log_analysis("=" * 80)
        log_analysis("PAIRS TRADING ANALYSIS")
        log_analysis("=" * 80)
        log_analysis("Configuration:")
    if log_data:
        log_data(f"  Target pairs: {args.pairs_count}")
        log_data(f"  Candidate depth per side: {args.candidate_depth}")
        log_data(f"  Weight preset: {args.weight_preset}")
        log_data(f"  Weights: 1d={weights['1d']:.2f}, 3d={weights['3d']:.2f}, 1w={weights['1w']:.2f}")
        log_data(f"  Opportunity preset: {args.opportunity_preset}")
        if getattr(args, "kalman_preset", None):
            log_data(f"  Kalman preset: {args.kalman_preset}")
        log_data(f"  OLS fit intercept: {args.ols_fit_intercept}")
        log_data(f"  Kalman delta: {args.kalman_delta:.2e}, obs_cov: {args.kalman_obs_cov:.2f}")
        log_data(f"  Spread range: {args.min_spread*100:.2f}% - {args.max_spread*100:.2f}%")
        log_data(f"  Correlation range: {args.min_correlation:.2f} - {args.max_correlation:.2f}")
        if target_symbol_inputs:
            log_data(f"  Mode: MANUAL (requested {', '.join(target_symbol_inputs)})")
        else:
            log_data("  Mode: AUTO (optimize across all symbols)")

    # Initialize components
    if log_progress:
        log_progress("Initializing components...")
    exchange_manager = ExchangeManager()
    data_fetcher = DataFetcher(exchange_manager)
    performance_analyzer = PerformanceAnalyzer(weights=weights)
    opportunity_profile = PAIRS_TRADING_OPPORTUNITY_PRESETS.get(
        args.opportunity_preset,
        PAIRS_TRADING_OPPORTUNITY_PRESETS.get("balanced", {}),
    )
    pairs_analyzer = PairsTradingAnalyzer(
        min_spread=args.min_spread,
        max_spread=args.max_spread,
        min_correlation=args.min_correlation,
        max_correlation=args.max_correlation,
        require_cointegration=args.require_cointegration,
        max_half_life=args.max_half_life,
        min_quantitative_score=args.min_quantitative_score,
        ols_fit_intercept=args.ols_fit_intercept,
        kalman_delta=args.kalman_delta,
        kalman_obs_cov=args.kalman_obs_cov,
        scoring_multipliers=opportunity_profile,
    )

    # Step 1: Get list of futures symbols
    if log_progress:
        log_progress("[1/4] Fetching futures symbols from Binance...")

    try:
        symbols = data_fetcher.list_binance_futures_symbols(
            max_candidates=args.max_symbols,
            progress_label="Symbol Discovery",
        )
        if not symbols:
            if log_error:
                log_error("No symbols found. Please check your API connection.")
            return
        if log_success:
            log_success(f"Found {len(symbols)} futures symbols.")
    except Exception as e:
        if log_error:
            log_error(f"Error fetching symbols: {e}")
        return

    if target_symbol_inputs:
        available_lookup = {normalize_symbol_key(sym): sym for sym in symbols}
        missing_targets = []
        mapped_targets = []
        newly_added_symbols = []
        for sym in parsed_target_symbols:
            normalized_key = normalize_symbol_key(sym)
            actual_symbol = available_lookup.get(normalized_key)
            if actual_symbol:
                mapped_targets.append(actual_symbol)
            else:
                missing_targets.append(sym)
                mapped_targets.append(sym)
                if normalized_key not in available_lookup:
                    newly_added_symbols.append(sym)
                    available_lookup[normalized_key] = sym
        if newly_added_symbols:
            symbols = list(dict.fromkeys(symbols + newly_added_symbols))
            if log_info:
                log_info(f"Added manual symbols to analysis universe: {', '.join(newly_added_symbols)}")
        if missing_targets:
            if log_warn:
                log_warn(f"These symbols were not discovered automatically but will be fetched manually: {', '.join(missing_targets)}")
        target_symbols = mapped_targets
        if target_symbols:
            if log_info:
                log_info(f"Tracking manual symbols: {', '.join(target_symbols)}")
        if not target_symbols:
            if log_warn:
                log_warn("All requested symbols were unavailable. Reverting to AUTO mode.")
    # Step 2: Analyze performance
    if log_progress:
        log_progress(f"[2/4] Analyzing performance for {len(symbols)} symbols...")
    try:
        performance_df = performance_analyzer.analyze_all_symbols(
            symbols, data_fetcher, verbose=True
        )
        if performance_df.empty:
            if log_error:
                log_error("No valid performance data found. Please try again later.")
            return
    except KeyboardInterrupt:
        if log_warn:
            log_warn("Analysis interrupted by user.")
        return
    except Exception as e:
        if log_error:
            log_error(f"Error during performance analysis: {e}")
        return

    # Step 3: Build candidate pools for long/short sides
    if log_progress:
        log_progress("[3/4] Building candidate pools for auto pair selection...")
    candidate_depth = max(args.candidate_depth, args.pairs_count * 2)
    best_performers = (
        performance_df.sort_values("score", ascending=False)
        .head(candidate_depth)
        .reset_index(drop=True)
    )
    worst_performers = (
        performance_df.sort_values("score", ascending=True)
        .head(candidate_depth)
        .reset_index(drop=True)
    )

    if target_symbols:
        best_performers, worst_performers = ensure_symbols_in_candidate_pools(
            performance_df, best_performers, worst_performers, target_symbols
        )

    display_performers(best_performers, "SHORT CANDIDATES (Strong performers)", Fore.RED)
    display_performers(worst_performers, "LONG CANDIDATES (Weak performers)", Fore.GREEN)

    # Step 4: Analyze pairs trading opportunities
    if log_progress:
        log_progress("[4/4] Analyzing pairs trading opportunities...")
    try:
        pairs_df = pairs_analyzer.analyze_pairs_opportunity(
            best_performers, worst_performers, data_fetcher=data_fetcher, verbose=True
        )

        if pairs_df.empty:
            if log_warn:
                log_warn("No pairs opportunities found.")
            return

        # Validate pairs if requested
        if not args.no_validation:
            if log_progress:
                log_progress("Validating pairs...")
            pairs_df = pairs_analyzer.validate_pairs(pairs_df, data_fetcher, verbose=True)

        # Sort pairs by selected criteria
        sort_column = args.sort_by if args.sort_by in pairs_df.columns else "opportunity_score"
        if sort_column in pairs_df.columns:
            pairs_df = pairs_df.sort_values(sort_column, ascending=False).reset_index(drop=True)
            sort_display = "quantitative score" if sort_column == "quantitative_score" else "opportunity score"
            if log_info:
                log_info(f"Sorted pairs by {sort_display} (descending).")

        selected_pairs = None
        manual_pairs = None
        displayed_manual = False
        if target_symbols:
            manual_pairs = select_pairs_for_symbols(
                pairs_df,
                target_symbols,
                max_pairs=None,
            )
            matched_symbols = {
                sym
                for sym in target_symbols
                if not manual_pairs[
                    (manual_pairs["long_symbol"] == sym)
                    | (manual_pairs["short_symbol"] == sym)
                ].empty
            }
            missing_matches = [sym for sym in target_symbols if sym not in matched_symbols]
            if missing_matches:
                if log_warn:
                    log_warn(f"No valid pairs found for: {', '.join(missing_matches)}")
            if not manual_pairs.empty:
                if log_info:
                    log_info("Best pairs for requested symbols:")
                display_pairs_opportunities(
                    manual_pairs,
                    max_display=min(args.max_pairs, len(manual_pairs)),
                )
                selected_pairs = manual_pairs
                displayed_manual = True

        if selected_pairs is None:
            # Select target number of tradeable pairs (unique symbols when possible)
            selected_pairs = select_top_unique_pairs(pairs_df, args.pairs_count)

        if selected_pairs is None or selected_pairs.empty:
            if log_warn:
                log_warn("No qualifying pairs after selection.")
            return

        if not displayed_manual:
            display_pairs_opportunities(
                selected_pairs,
                max_display=min(args.max_pairs, len(selected_pairs)),
            )

        # Summary
        if log_analysis:
            log_analysis("=" * 80)
            log_analysis("SUMMARY")
            log_analysis("=" * 80)
        if log_data:
            log_data(f"Total symbols analyzed: {len(performance_df)}")
            log_data(f"Short candidates considered: {len(best_performers)}")
            log_data(f"Long candidates considered: {len(worst_performers)}")
            log_data(f"Valid pairs available: {len(pairs_df)}")
            log_data(f"Selected tradeable pairs: {len(selected_pairs)}")
        if not selected_pairs.empty:
            avg_spread = selected_pairs["spread"].mean() * 100
            if log_data:
                log_data(f"Average spread: {avg_spread:.2f}%")
            correlations = selected_pairs["correlation"].dropna()
            if not correlations.empty:
                avg_correlation = correlations.mean()
                if log_data:
                    log_data(f"Average correlation: {avg_correlation:.3f}")
            
            # Quantitative metrics statistics
            if log_analysis:
                log_analysis("Quantitative Metrics:")
            
            # Quantitative score
            quant_scores = selected_pairs["quantitative_score"].dropna()
            if not quant_scores.empty:
                avg_quant_score = quant_scores.mean()
                if log_data:
                    log_data(f"  Average quantitative score: {avg_quant_score:.1f}/100")
            
            # Cointegration rate
            is_cointegrated_col = selected_pairs.get("is_cointegrated")
            if is_cointegrated_col is not None:
                cointegrated_count = is_cointegrated_col.fillna(False).sum()
                cointegration_rate = (cointegrated_count / len(selected_pairs)) * 100
                if log_data:
                    log_data(f"  Cointegration rate: {cointegration_rate:.1f}% ({cointegrated_count}/{len(selected_pairs)})")
            
            # Average half-life
            half_lives = selected_pairs.get("half_life").dropna() if "half_life" in selected_pairs.columns else pd.Series()
            if not half_lives.empty:
                avg_half_life = half_lives.mean()
                if log_data:
                    log_data(f"  Average half-life: {avg_half_life:.1f} periods")
            
            # Average Sharpe ratio
            sharpe_ratios = selected_pairs.get("spread_sharpe").dropna() if "spread_sharpe" in selected_pairs.columns else pd.Series()
            if not sharpe_ratios.empty:
                avg_sharpe = sharpe_ratios.mean()
                if log_data:
                    log_data(f"  Average Sharpe ratio: {avg_sharpe:.2f}")
            
            # Average max drawdown
            max_dds = selected_pairs.get("max_drawdown").dropna() if "max_drawdown" in selected_pairs.columns else pd.Series()
            if not max_dds.empty:
                avg_max_dd = max_dds.mean() * 100
                if log_data:
                    log_data(f"  Average max drawdown: {avg_max_dd:.2f}%")
            
            # Average hedge ratios
            if log_analysis:
                log_analysis("Hedge Ratios:")
            
            # OLS hedge ratio
            hedge_ratios = selected_pairs.get("hedge_ratio").dropna() if "hedge_ratio" in selected_pairs.columns else pd.Series()
            if not hedge_ratios.empty:
                avg_hedge_ratio = hedge_ratios.mean()
                if log_data:
                    log_data(f"  Average OLS hedge ratio: {avg_hedge_ratio:.4f}")
            
            # Kalman hedge ratio
            kalman_ratios = selected_pairs.get("kalman_hedge_ratio").dropna() if "kalman_hedge_ratio" in selected_pairs.columns else pd.Series()
            if not kalman_ratios.empty:
                avg_kalman_ratio = kalman_ratios.mean()
                if log_data:
                    log_data(f"  Average Kalman hedge ratio: {avg_kalman_ratio:.4f}")
        if log_analysis:
            log_analysis("=" * 80)

    except KeyboardInterrupt:
        if log_warn:
            log_warn("Pairs analysis interrupted by user.")
    except Exception as e:
        if log_error:
            log_error(f"Error during pairs analysis: {e}")


if __name__ == "__main__":
    main()


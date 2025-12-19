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

from config import (
    PAIRS_TRADING_OPPORTUNITY_PRESETS,
    PAIRS_TRADING_HURST_THRESHOLD,
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
from modules.common.core.exchange_manager import ExchangeManager
from modules.common.core.data_fetcher import DataFetcher
# Import directly from submodules to avoid circular import
from modules.pairs_trading.analysis import PerformanceAnalyzer
from modules.pairs_trading.core import PairsTradingAnalyzer
from modules.pairs_trading.cli.display import (
    display_performers,
    display_pairs_opportunities,
)
from modules.pairs_trading.utils import (
    select_top_unique_pairs,
    ensure_symbols_in_candidate_pools,
    select_pairs_for_symbols,
)
from modules.pairs_trading.cli.argument_parser import parse_args
from modules.pairs_trading.cli.interactive_prompts import (
    prompt_interactive_mode,
    prompt_weight_preset_selection,
    prompt_kalman_preset_selection,
    prompt_opportunity_preset_selection,
    prompt_target_pairs,
    prompt_candidate_depth,
)
from modules.pairs_trading.cli.input_parsers import (
    parse_weights,
    parse_symbols,
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
    strategy = getattr(args, "strategy", "reversion")
    args.strategy = strategy

    if not args.no_menu:
        menu_result = prompt_interactive_mode()
        if menu_result["mode"] == "auto":
            args.symbols = None
        else:
            args.symbols = menu_result["symbols_raw"]
        if not args.weights:
            args.weight_preset = prompt_weight_preset_selection(args.weight_preset)
        args.opportunity_preset = prompt_opportunity_preset_selection(args.opportunity_preset)
        
        # Strategy selection
        print()
        print(color_text("=" * 60, Fore.CYAN))
        print(color_text("STRATEGY SELECTION", Fore.CYAN))
        print(color_text("=" * 60, Fore.CYAN))
        print(color_text("1. Mean Reversion", Fore.GREEN) + " - Long weak, Short strong (expect convergence)")
        print(color_text("2. Momentum", Fore.MAGENTA) + " - Long strong, Short weak (trend continuation)")
        print()
        
        current_strategy_num = "2" if strategy == "momentum" else "1"
        strategy_input = input(
            color_text(
                f"Select strategy [1/2] (default {current_strategy_num}): ",
                Fore.YELLOW,
            )
        ).strip()
        
        if strategy_input == "2":
            args.strategy = "momentum"
            strategy = "momentum"
            print(color_text("✓ Momentum strategy selected", Fore.MAGENTA))
        elif strategy_input == "1":
            args.strategy = "reversion"
            strategy = "reversion"
            print(color_text("✓ Mean Reversion strategy selected", Fore.GREEN))
        else:
            # Keep current/default
            print(color_text(f"✓ Using {strategy.capitalize()} strategy", Fore.CYAN))
        
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
        log_data(f"  Strategy: {strategy.capitalize()}")
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

    # Adjust validation thresholds based on strategy
    # 
    # Mean Reversion Strategy:
    #   - Requires mean-reverting behavior (Hurst < 0.5)
    #   - Needs cointegration for pairs to converge
    #   - Half-life should be short (< max_half_life) for quick reversion
    #
    # Momentum Strategy:
    #   - Requires trending behavior (Hurst > 0.5), so we disable mean-reversion checks
    #   - Cointegration is not required (pairs can diverge in trends)
    #   - Half-life is irrelevant (trends continue, not revert)
    max_half_life = args.max_half_life
    hurst_threshold = PAIRS_TRADING_HURST_THRESHOLD

    if strategy == "momentum":
        # Disable mean-reversion validation: Momentum pairs should trend (Hurst > 0.5),
        # not revert. Setting threshold to 1.0 effectively disables the check.
        hurst_threshold = 1.0 
        # Half-life is irrelevant for momentum: pairs don't need to revert quickly.
        # Setting to infinity disables the half-life constraint.
        max_half_life = float('inf')
        
        # Momentum strategy benefits from divergence, not convergence.
        # Cointegration requirement conflicts with momentum trading logic.
        if getattr(args, "require_cointegration", False):
            if log_warn:
                log_warn("Momentum strategy bỏ qua kiểm tra cointegration nghiêm ngặt. Tự động tắt --require-cointegration.")
            args.require_cointegration = False

    pairs_analyzer = PairsTradingAnalyzer(
        min_spread=args.min_spread,
        max_spread=args.max_spread,
        min_correlation=args.min_correlation,
        max_correlation=args.max_correlation,
        require_cointegration=args.require_cointegration,
        max_half_life=max_half_life,
        hurst_threshold=hurst_threshold,
        min_quantitative_score=args.min_quantitative_score,
        ols_fit_intercept=args.ols_fit_intercept,
        kalman_delta=args.kalman_delta,
        kalman_obs_cov=args.kalman_obs_cov,
        scoring_multipliers=opportunity_profile,
        strategy=strategy,
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
    except ConnectionError as e:
        if log_error:
            log_error(f"Connection error fetching symbols (check network/API): {e}")
        return
    except ValueError as e:
        if log_error:
            log_error(f"Invalid configuration error fetching symbols: {e}")
        return
    except Exception as e:
        if log_error:
            log_error(f"Unexpected error fetching symbols: {type(e).__name__}: {e}")
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
    except (pd.errors.EmptyDataError, ValueError) as e:
        if log_error:
            log_error(f"Data validation error during performance analysis: {e}")
        return
    except ConnectionError as e:
        if log_error:
            log_error(f"Connection error during performance analysis (check API/exchange): {e}")
        return
    except Exception as e:
        if log_error:
            log_error(f"Unexpected error during performance analysis: {type(e).__name__}: {e}")
            import traceback
            log_error(f"Traceback: {traceback.format_exc()}")
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

    long_candidates = worst_performers
    short_candidates = best_performers
    long_title = "LONG CANDIDATES (Weak performers)"
    short_title = "SHORT CANDIDATES (Strong performers)"

    if strategy == "momentum":
        long_candidates = best_performers
        short_candidates = worst_performers
        long_title = "LONG CANDIDATES (Momentum leaders)"
        short_title = "SHORT CANDIDATES (Lagging performers)"

    display_performers(short_candidates, short_title, Fore.RED)
    display_performers(long_candidates, long_title, Fore.GREEN)

    # Step 4: Analyze pairs trading opportunities
    # 
    # Spread Calculation:
    #   The spread is calculated as: spread = short_score - long_score
    #   - For Mean Reversion: long_score (worst performer) is negative, short_score (best) is positive
    #   - For Momentum: long_score (best performer) is positive, short_score (worst) is negative
    #   - Spread represents the performance gap between the two symbols
    #   - Larger spread indicates greater divergence, which is favorable for pairs trading
    #   See pairs_analyzer.calculate_spread() for implementation details.
    #
    # Performance Note:
    #   With large datasets, fetching OHLCV data for each pair can be slow.
    #   Consider adding estimated time remaining to ProgressBar for better user experience.
    if log_progress:
        log_progress("[4/4] Analyzing pairs trading opportunities...")
    try:
        pairs_df = pairs_analyzer.analyze_pairs_opportunity(
            short_candidates, long_candidates, data_fetcher=data_fetcher, verbose=True
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
            log_data(f"Short candidates considered: {len(short_candidates)}")
            log_data(f"Long candidates considered: {len(long_candidates)}")
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
            
            # Strategy-specific metrics
            if strategy == "momentum":
                # Momentum-specific statistics
                if log_analysis:
                    log_analysis("Momentum Metrics:")
                
                # ADX statistics
                if "long_adx" in selected_pairs.columns and "short_adx" in selected_pairs.columns:
                    long_adx_values = selected_pairs["long_adx"].dropna()
                    short_adx_values = selected_pairs["short_adx"].dropna()
                    
                    if not long_adx_values.empty and not short_adx_values.empty:
                        avg_long_adx = long_adx_values.mean()
                        avg_short_adx = short_adx_values.mean()
                        avg_combined_adx = (avg_long_adx + avg_short_adx) / 2
                        
                        # Count strong trends (ADX >= 25)
                        strong_long = (long_adx_values >= 25).sum()
                        strong_short = (short_adx_values >= 25).sum()
                        both_strong = ((selected_pairs["long_adx"] >= 25) & (selected_pairs["short_adx"] >= 25)).sum()
                        
                        if log_data:
                            log_data(f"  Average Long ADX: {avg_long_adx:.2f}")
                            log_data(f"  Average Short ADX: {avg_short_adx:.2f}")
                            log_data(f"  Average Combined ADX: {avg_combined_adx:.2f}")
                            log_data(f"  Strong trends (ADX≥25): {both_strong}/{len(selected_pairs)} pairs ({both_strong/len(selected_pairs)*100:.1f}%)")
                
                # Hurst exponent statistics (should be > 0.5 for momentum)
                hurst_values = selected_pairs.get("hurst_exponent").dropna() if "hurst_exponent" in selected_pairs.columns else pd.Series()
                if not hurst_values.empty:
                    avg_hurst = hurst_values.mean()
                    trending_count = (hurst_values > 0.5).sum()
                    trending_pct = (trending_count / len(hurst_values)) * 100
                    strong_trending = (hurst_values > 0.6).sum()
                    strong_trending_pct = (strong_trending / len(hurst_values)) * 100
                    
                    if log_data:
                        log_data(f"  Average Hurst exponent: {avg_hurst:.3f}")
                        log_data(f"  Trending pairs (H>0.5): {trending_count}/{len(hurst_values)} ({trending_pct:.1f}%)")
                        log_data(f"  Strong trending (H>0.6): {strong_trending}/{len(hurst_values)} ({strong_trending_pct:.1f}%)")
                
                # Z-Score divergence statistics
                zscore_values = selected_pairs.get("current_zscore").dropna() if "current_zscore" in selected_pairs.columns else pd.Series()
                if not zscore_values.empty:
                    avg_abs_zscore = zscore_values.abs().mean()
                    high_divergence = (zscore_values.abs() > 1.0).sum()
                    very_high_divergence = (zscore_values.abs() > 2.0).sum()
                    
                    if log_data:
                        log_data(f"  Average |Z-Score|: {avg_abs_zscore:.2f}")
                        log_data(f"  High divergence (|Z|>1): {high_divergence}/{len(zscore_values)} ({high_divergence/len(zscore_values)*100:.1f}%)")
                        log_data(f"  Very high divergence (|Z|>2): {very_high_divergence}/{len(zscore_values)} ({very_high_divergence/len(zscore_values)*100:.1f}%)")
            
            else:
                # Mean Reversion-specific statistics
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
                
                # Hurst exponent (should be < 0.5 for mean reversion)
                hurst_values = selected_pairs.get("hurst_exponent").dropna() if "hurst_exponent" in selected_pairs.columns else pd.Series()
                if not hurst_values.empty:
                    avg_hurst = hurst_values.mean()
                    mean_reverting = (hurst_values < 0.5).sum()
                    mean_reverting_pct = (mean_reverting / len(hurst_values)) * 100
                    if log_data:
                        log_data(f"  Average Hurst exponent: {avg_hurst:.3f}")
                        log_data(f"  Mean-reverting pairs (H<0.5): {mean_reverting}/{len(hurst_values)} ({mean_reverting_pct:.1f}%)")
            
            # Shared metrics (both strategies)
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
    except pd.errors.EmptyDataError as e:
        if log_error:
            log_error(f"Empty data error during pairs analysis: {e}")
    except (ValueError, KeyError) as e:
        if log_error:
            log_error(f"Data validation error during pairs analysis: {e}")
    except ConnectionError as e:
        if log_error:
            log_error(f"Connection error during pairs analysis (check API/exchange): {e}")
    except Exception as e:
        if log_error:
            log_error(f"Unexpected error during pairs analysis: {type(e).__name__}: {e}")
            import traceback
            log_error(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    main()


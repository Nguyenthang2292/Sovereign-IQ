"""
Voting Analyzer for ATC + Range Oscillator + SPC Pure Voting System.

This module contains the VotingAnalyzer class that combines signals from:
1. Adaptive Trend Classification (ATC)
2. Range Oscillator
3. Simplified Percentile Clustering (SPC)

Option 2: Completely replace sequential filtering with a voting system.
"""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from colorama import Fore, Style

# Import SPC configuration constants
from config.decision_matrix import DECISION_MATRIX_INDICATOR_ACCURACIES
from config.spc import (
    SPC_AGGREGATION_ADAPTIVE_PERFORMANCE_WINDOW,
    SPC_AGGREGATION_ENABLE_ADAPTIVE_WEIGHTS,
    SPC_AGGREGATION_ENABLE_SIMPLE_FALLBACK,
    SPC_AGGREGATION_MIN_SIGNAL_STRENGTH,
    SPC_AGGREGATION_MODE,
    SPC_AGGREGATION_SIMPLE_MIN_ACCURACY_TOTAL,
    SPC_AGGREGATION_STRATEGY_WEIGHTS,
    SPC_AGGREGATION_THRESHOLD,
    SPC_AGGREGATION_WEIGHTED_MIN_DIFF,
    SPC_AGGREGATION_WEIGHTED_MIN_TOTAL,
    SPC_STRATEGY_PARAMETERS,
)
from core.signal_calculators import (
    get_hmm_signal,
    get_random_forest_signal,
    get_range_oscillator_signal,
    get_spc_signal,
    get_xgboost_signal,
)
from modules.adaptive_trend_LTS.cli import (
    ATCAnalyzer,
    prompt_timeframe,
)
from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager
from modules.common.utils import (
    color_text,
    log_progress,
    log_success,
    log_warn,
)
from modules.decision_matrix.core.classifier import DecisionMatrixClassifier
from modules.range_oscillator.cli import (
    display_final_results,
)
from modules.simplified_percentile_clustering.aggregation import (
    SPCVoteAggregator,
)
from modules.simplified_percentile_clustering.config import (
    SPCAggregationConfig,
)
from modules.simplified_percentile_clustering.core.clustering import (
    ClusteringConfig,
)
from modules.simplified_percentile_clustering.core.features import FeatureConfig


class VotingAnalyzer:
    """
    ATC + Range Oscillator + SPC Pure Voting Analyzer.

    Option 2: Completely replace sequential filtering with a voting system.
    """

    def __init__(self, args, data_fetcher: DataFetcher, ohlcv_cache: Optional[Dict[str, pd.DataFrame]] = None):
        """Initialize analyzer."""
        self.args = args
        self.data_fetcher = data_fetcher
        self.ohlcv_cache = ohlcv_cache
        self.atc_analyzer = ATCAnalyzer(args, data_fetcher, ohlcv_cache=ohlcv_cache)
        self.selected_timeframe = args.timeframe
        self.atc_analyzer.selected_timeframe = args.timeframe

        # Initialize SPC Vote Aggregator
        aggregation_config = SPCAggregationConfig(
            mode=SPC_AGGREGATION_MODE,
            threshold=SPC_AGGREGATION_THRESHOLD,
            weighted_min_total=SPC_AGGREGATION_WEIGHTED_MIN_TOTAL,
            weighted_min_diff=SPC_AGGREGATION_WEIGHTED_MIN_DIFF,
            enable_adaptive_weights=SPC_AGGREGATION_ENABLE_ADAPTIVE_WEIGHTS,
            adaptive_performance_window=SPC_AGGREGATION_ADAPTIVE_PERFORMANCE_WINDOW,
            min_signal_strength=SPC_AGGREGATION_MIN_SIGNAL_STRENGTH,
            enable_simple_fallback=SPC_AGGREGATION_ENABLE_SIMPLE_FALLBACK,
            simple_min_accuracy_total=SPC_AGGREGATION_SIMPLE_MIN_ACCURACY_TOTAL,
            strategy_weights=SPC_AGGREGATION_STRATEGY_WEIGHTS,
        )
        self.spc_aggregator = SPCVoteAggregator(aggregation_config)

        # Thread-safe lock for mode changes
        self._mode_lock = threading.Lock()

        # Results storage
        self.long_signals_atc = pd.DataFrame()
        self.short_signals_atc = pd.DataFrame()
        self.long_signals_final = pd.DataFrame()
        self.short_signals_final = pd.DataFrame()

    def determine_timeframe(self) -> str:
        """Determine timeframe from arguments and interactive menu."""
        self.selected_timeframe = self.args.timeframe

        if not self.args.no_menu:
            print("\n" + color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
            print(color_text("TIMEFRAME SELECTION", Fore.CYAN, Style.BRIGHT))
            print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
            self.selected_timeframe = prompt_timeframe(default_timeframe=self.selected_timeframe)
            print(color_text(f"\nSelected timeframe: {self.selected_timeframe}", Fore.GREEN))

        self.atc_analyzer.selected_timeframe = self.selected_timeframe
        return self.selected_timeframe

    def get_oscillator_params(self) -> dict:
        """Extract Range Oscillator parameters."""
        return {
            "osc_length": self.args.osc_length,
            "osc_mult": self.args.osc_mult,
            "max_workers": self.args.max_workers,
            "strategies": self.args.osc_strategies,
        }

    def get_spc_params(self) -> dict:
        """Extract SPC parameters for all 3 strategies."""
        # Import enhancement parameters
        try:
            from config.spc_enhancements import (
                SPC_ENABLE_MTF,
                SPC_FLIP_CONFIDENCE_THRESHOLD,
                SPC_INTERPOLATION_MODE,
                SPC_MIN_FLIP_DURATION,
                SPC_MTF_REQUIRE_ALIGNMENT,
                SPC_MTF_TIMEFRAMES,
                SPC_PRESET_AGGRESSIVE,
                SPC_PRESET_BALANCED,
                SPC_PRESET_CONSERVATIVE,
                SPC_TIME_DECAY_FACTOR,
                SPC_USE_CORRELATION_WEIGHTS,
                SPC_VOLATILITY_ADJUSTMENT,
            )
        except ImportError:
            # Fallback to defaults if spc_enhancements module not available
            SPC_VOLATILITY_ADJUSTMENT = False
            SPC_USE_CORRELATION_WEIGHTS = False
            SPC_TIME_DECAY_FACTOR = 1.0
            SPC_INTERPOLATION_MODE = "linear"
            SPC_MIN_FLIP_DURATION = 3
            SPC_FLIP_CONFIDENCE_THRESHOLD = 0.6
            SPC_ENABLE_MTF = False
            SPC_MTF_TIMEFRAMES = ["1h", "4h"]
            SPC_MTF_REQUIRE_ALIGNMENT = True
            SPC_PRESET_CONSERVATIVE = {}
            SPC_PRESET_BALANCED = {}
            SPC_PRESET_AGGRESSIVE = {}

        # Check if preset is specified
        preset = getattr(self.args, "spc_preset", None)
        if preset:
            if preset == "conservative":
                preset_config = SPC_PRESET_CONSERVATIVE
            elif preset == "balanced":
                preset_config = SPC_PRESET_BALANCED
            elif preset == "aggressive":
                preset_config = SPC_PRESET_AGGRESSIVE
            else:
                preset_config = {}

            # Use preset values as defaults, but allow CLI overrides
            volatility_adjustment = preset_config.get("volatility_adjustment", SPC_VOLATILITY_ADJUSTMENT)
            use_correlation_weights = preset_config.get("use_correlation_weights", SPC_USE_CORRELATION_WEIGHTS)
            time_decay_factor = preset_config.get("time_decay_factor", SPC_TIME_DECAY_FACTOR)
            interpolation_mode = preset_config.get("interpolation_mode", SPC_INTERPOLATION_MODE)
            min_flip_duration = preset_config.get("min_flip_duration", SPC_MIN_FLIP_DURATION)
            flip_confidence_threshold = preset_config.get("flip_confidence_threshold", SPC_FLIP_CONFIDENCE_THRESHOLD)
        else:
            # Use config defaults
            volatility_adjustment = SPC_VOLATILITY_ADJUSTMENT
            use_correlation_weights = SPC_USE_CORRELATION_WEIGHTS
            time_decay_factor = SPC_TIME_DECAY_FACTOR
            interpolation_mode = SPC_INTERPOLATION_MODE
            min_flip_duration = SPC_MIN_FLIP_DURATION
            flip_confidence_threshold = SPC_FLIP_CONFIDENCE_THRESHOLD

        # Override with CLI arguments if provided
        if hasattr(self.args, "spc_volatility_adjustment") and self.args.spc_volatility_adjustment:
            volatility_adjustment = True
        if hasattr(self.args, "spc_use_correlation_weights") and self.args.spc_use_correlation_weights:
            use_correlation_weights = True
        if hasattr(self.args, "spc_time_decay_factor") and self.args.spc_time_decay_factor is not None:
            time_decay_factor = self.args.spc_time_decay_factor
        if hasattr(self.args, "spc_interpolation_mode") and self.args.spc_interpolation_mode is not None:
            interpolation_mode = self.args.spc_interpolation_mode
        if hasattr(self.args, "spc_min_flip_duration") and self.args.spc_min_flip_duration is not None:
            min_flip_duration = self.args.spc_min_flip_duration
        if hasattr(self.args, "spc_flip_confidence_threshold") and self.args.spc_flip_confidence_threshold is not None:
            flip_confidence_threshold = self.args.spc_flip_confidence_threshold

        # MTF parameters
        enable_mtf = getattr(self.args, "spc_enable_mtf", False) or SPC_ENABLE_MTF
        mtf_timeframes = getattr(self.args, "spc_mtf_timeframes", None) or SPC_MTF_TIMEFRAMES
        mtf_require_alignment = getattr(self.args, "spc_mtf_require_alignment", None)
        if mtf_require_alignment is None:
            mtf_require_alignment = SPC_MTF_REQUIRE_ALIGNMENT

        # Use values from config if not provided in args
        cluster_transition_params = SPC_STRATEGY_PARAMETERS["cluster_transition"].copy()
        regime_following_params = SPC_STRATEGY_PARAMETERS["regime_following"].copy()
        mean_reversion_params = SPC_STRATEGY_PARAMETERS["mean_reversion"].copy()

        # Override with args if provided (for command-line usage)
        if hasattr(self.args, "spc_min_signal_strength"):
            cluster_transition_params["min_signal_strength"] = self.args.spc_min_signal_strength
        if hasattr(self.args, "spc_min_rel_pos_change"):
            cluster_transition_params["min_rel_pos_change"] = self.args.spc_min_rel_pos_change
        if hasattr(self.args, "spc_min_regime_strength"):
            regime_following_params["min_regime_strength"] = self.args.spc_min_regime_strength
        if hasattr(self.args, "spc_min_cluster_duration"):
            regime_following_params["min_cluster_duration"] = self.args.spc_min_cluster_duration
        if hasattr(self.args, "spc_extreme_threshold"):
            mean_reversion_params["extreme_threshold"] = self.args.spc_extreme_threshold
        if hasattr(self.args, "spc_min_extreme_duration"):
            mean_reversion_params["min_extreme_duration"] = self.args.spc_min_extreme_duration

        return {
            "k": self.args.spc_k,
            "lookback": self.args.spc_lookback,
            "p_low": self.args.spc_p_low,
            "p_high": self.args.spc_p_high,
            "cluster_transition_params": cluster_transition_params,
            "regime_following_params": regime_following_params,
            "mean_reversion_params": mean_reversion_params,
            # Enhancement parameters (with CLI override support)
            "volatility_adjustment": volatility_adjustment,
            "use_correlation_weights": use_correlation_weights,
            "time_decay_factor": time_decay_factor,
            "interpolation_mode": interpolation_mode,
            "min_flip_duration": min_flip_duration,
            "flip_confidence_threshold": flip_confidence_threshold,
            # MTF parameters
            "enable_mtf": enable_mtf,
            "mtf_timeframes": mtf_timeframes,
            "mtf_require_alignment": mtf_require_alignment,
        }

    def display_config(self) -> None:
        """Display configuration information."""
        # Lazy import to avoid circular dependency
        from cli.display import display_config

        display_config(
            selected_timeframe=self.selected_timeframe,
            args=self.args,
            get_oscillator_params=self.get_oscillator_params,
            get_spc_params=self.get_spc_params,
            mode="voting",
        )

    def run_atc_scan(self) -> bool:
        """
        Run ATC auto scan to get LONG/SHORT signals.

        Returns:
            True if signals found, False otherwise
        """
        log_progress("\nStep 1: Running ATC auto scan...")
        log_progress("=" * 80)

        self.long_signals_atc, self.short_signals_atc = self.atc_analyzer.run_auto_scan()

        original_long_count = len(self.long_signals_atc)
        original_short_count = len(self.short_signals_atc)

        log_success(f"\nATC Scan Complete: Found {original_long_count} LONG + {original_short_count} SHORT signals")

        if self.long_signals_atc.empty and self.short_signals_atc.empty:
            log_warn("No ATC signals found. Cannot proceed with analysis.")
            log_warn("Please try:")
            log_warn("  - Different timeframe")
            log_warn("  - Different market conditions")
            log_warn("  - Check ATC configuration parameters")
            return False

        return True

    def _process_symbol_for_all_indicators(
        self,
        symbol_data: Dict[str, Any],
        exchange_manager: ExchangeManager,
        timeframe: str,
        limit: int,
        signal_type: str,
        osc_params: dict,
        spc_params: Optional[dict],
        indicators_to_calculate: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Worker function to calculate signals from all indicators in parallel.

        This is the key difference from hybrid approach - we calculate all signals
        at once instead of filtering sequentially.
        """
        try:
            data_fetcher = DataFetcher(exchange_manager)
            symbol = symbol_data["symbol"]
            expected_signal = 1 if signal_type == "LONG" else -1

            # Calculate all signals in parallel
            results = {
                "symbol": symbol,
                "signal": symbol_data["signal"],
                "trend": symbol_data["trend"],
                "price": symbol_data["price"],
                "exchange": symbol_data["exchange"],
            }

            # ATC signal (already have it from scan)
            # ATC signal is a percentage value (e.g., 94, -50), not just 1 or -1
            # If symbol is in long_signals_atc/short_signals_atc, it means it passed ATC scan
            # So we check the sign of the signal, not exact equality
            atc_signal = symbol_data["signal"]
            # Check sign: positive for LONG, negative for SHORT
            if signal_type == "LONG":
                atc_vote = 1 if atc_signal > 0 else 0
            else:  # SHORT
                atc_vote = 1 if atc_signal < 0 else 0
            atc_strength = abs(atc_signal) / 100.0 if atc_signal != 0 else 0.0
            results["atc_signal"] = atc_signal
            results["atc_vote"] = atc_vote
            results["atc_strength"] = min(atc_strength, 1.0)

            # Range Oscillator signal
            if indicators_to_calculate is None or "oscillator" in indicators_to_calculate:
                osc_result = get_range_oscillator_signal(
                    data_fetcher=data_fetcher,
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit,
                    osc_length=osc_params["osc_length"],
                    osc_mult=osc_params["osc_mult"],
                    strategies=osc_params["strategies"],
                )

                if osc_result is not None:
                    osc_signal, osc_confidence = osc_result
                    osc_vote = 1 if osc_signal == expected_signal else 0
                    results["osc_signal"] = osc_signal
                    results["osc_vote"] = osc_vote
                    results["osc_confidence"] = osc_confidence
                else:
                    results["osc_signal"] = 0
                    results["osc_vote"] = 0
                    results["osc_confidence"] = 0.0
            else:
                results["osc_signal"] = 0
                results["osc_vote"] = 0
                results["osc_confidence"] = 0.0

            # SPC signals from all 3 strategies (if enabled)
            if (
                (indicators_to_calculate is None or "spc" in indicators_to_calculate)
                and self.args.enable_spc
                and spc_params
            ):
                feature_config = FeatureConfig()
                clustering_config = ClusteringConfig(
                    k=spc_params["k"],
                    lookback=spc_params["lookback"],
                    p_low=spc_params["p_low"],
                    p_high=spc_params["p_high"],
                    main_plot="Clusters",
                    feature_config=feature_config,
                    # Enhancement parameters
                    volatility_adjustment=spc_params.get("volatility_adjustment", False),
                    use_correlation_weights=spc_params.get("use_correlation_weights", False),
                    time_decay_factor=spc_params.get("time_decay_factor", 1.0),
                    interpolation_mode=spc_params.get("interpolation_mode", "linear"),
                    min_flip_duration=spc_params.get("min_flip_duration", 3),
                    flip_confidence_threshold=spc_params.get("flip_confidence_threshold", 0.6),
                )

                # Calculate signals from all 3 strategies
                # Cluster Transition
                ct_result = get_spc_signal(
                    data_fetcher=data_fetcher,
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit,
                    strategy="cluster_transition",
                    strategy_params=spc_params["cluster_transition_params"],
                    clustering_config=clustering_config,
                )
                if ct_result is not None:
                    results["spc_cluster_transition_signal"] = ct_result[0]
                    results["spc_cluster_transition_strength"] = ct_result[1]
                else:
                    results["spc_cluster_transition_signal"] = 0
                    results["spc_cluster_transition_strength"] = 0.0

                # Regime Following
                rf_result = get_spc_signal(
                    data_fetcher=data_fetcher,
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit,
                    strategy="regime_following",
                    strategy_params=spc_params["regime_following_params"],
                    clustering_config=clustering_config,
                )
                if rf_result is not None:
                    results["spc_regime_following_signal"] = rf_result[0]
                    results["spc_regime_following_strength"] = rf_result[1]
                else:
                    results["spc_regime_following_signal"] = 0
                    results["spc_regime_following_strength"] = 0.0

                # Mean Reversion
                mr_result = get_spc_signal(
                    data_fetcher=data_fetcher,
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit,
                    strategy="mean_reversion",
                    strategy_params=spc_params["mean_reversion_params"],
                    clustering_config=clustering_config,
                )
                if mr_result is not None:
                    results["spc_mean_reversion_signal"] = mr_result[0]
                    results["spc_mean_reversion_strength"] = mr_result[1]
                else:
                    results["spc_mean_reversion_signal"] = 0
                    results["spc_mean_reversion_strength"] = 0.0

            # XGBoost prediction (if enabled)
            if (
                (indicators_to_calculate is None or "xgboost" in indicators_to_calculate)
                and hasattr(self.args, "enable_xgboost")
                and self.args.enable_xgboost
            ):
                xgb_result = get_xgboost_signal(
                    data_fetcher=data_fetcher,
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit,
                )
                if xgb_result is not None:
                    xgb_signal, xgb_confidence = xgb_result
                    xgb_vote = 1 if xgb_signal == expected_signal else 0
                    results["xgboost_signal"] = xgb_signal
                    results["xgboost_vote"] = xgb_vote
                    results["xgboost_confidence"] = xgb_confidence
                else:
                    results["xgboost_signal"] = 0
                    results["xgboost_vote"] = 0
                    results["xgboost_confidence"] = 0.0

            # HMM signal (if enabled)
            if (
                (indicators_to_calculate is None or "hmm" in indicators_to_calculate)
                and hasattr(self.args, "enable_hmm")
                and self.args.enable_hmm
            ):
                try:
                    hmm_result = get_hmm_signal(
                        data_fetcher=data_fetcher,
                        symbol=symbol,
                        timeframe=timeframe,
                        limit=limit,
                        window_size=getattr(self.args, "hmm_window_size", None),
                        window_kama=getattr(self.args, "hmm_window_kama", None),
                        fast_kama=getattr(self.args, "hmm_fast_kama", None),
                        slow_kama=getattr(self.args, "hmm_slow_kama", None),
                        orders_argrelextrema=getattr(self.args, "hmm_orders_argrelextrema", None),
                        strict_mode=getattr(self.args, "hmm_strict_mode", None),
                    )
                    if hmm_result is not None:
                        hmm_signal, hmm_confidence = hmm_result
                        hmm_vote = 1 if hmm_signal == expected_signal else 0
                        results["hmm_signal"] = hmm_signal
                        results["hmm_vote"] = hmm_vote
                        results["hmm_confidence"] = hmm_confidence
                    else:
                        results["hmm_signal"] = 0
                        results["hmm_vote"] = 0
                        results["hmm_confidence"] = 0.0
                except Exception as e:
                    # Log HMM errors but don't fail the entire process
                    # Sanitize error message to prevent information leakage
                    log_warn(f"HMM signal calculation failed for {symbol}: {type(e).__name__}")
                    results["hmm_signal"] = 0
                    results["hmm_vote"] = 0
                    results["hmm_confidence"] = 0.0

            # Random Forest prediction (if enabled)
            if (
                (indicators_to_calculate is None or "random_forest" in indicators_to_calculate)
                and hasattr(self.args, "enable_random_forest")
                and self.args.enable_random_forest
            ):
                try:
                    rf_result = get_random_forest_signal(
                        data_fetcher=data_fetcher,
                        symbol=symbol,
                        timeframe=timeframe,
                        limit=limit,
                        model_path=getattr(self.args, "random_forest_model_path", None),
                    )
                    if rf_result is not None:
                        rf_signal, rf_confidence = rf_result
                        rf_vote = 1 if rf_signal == expected_signal else 0
                        results["random_forest_signal"] = rf_signal
                        results["random_forest_vote"] = rf_vote
                        results["random_forest_confidence"] = rf_confidence
                    else:
                        results["random_forest_signal"] = 0
                        results["random_forest_vote"] = 0
                        results["random_forest_confidence"] = 0.0
                except Exception as e:
                    # Log Random Forest errors but don't fail the entire process
                    # Sanitize error message to prevent information leakage
                    log_warn(f"Random Forest signal calculation failed for {symbol}: {type(e).__name__}")
                    results["random_forest_signal"] = 0
                    results["random_forest_vote"] = 0
                    results["random_forest_confidence"] = 0.0

            return results

        except Exception as e:
            # Log error for debugging instead of silently swallowing
            # Sanitize error message to prevent information leakage
            from modules.common.utils import log_error

            symbol = symbol_data.get("symbol", "unknown")
            log_error(f"Error processing symbol {symbol} for all indicators: {type(e).__name__}")
            return None

    def calculate_signals_for_all_indicators(
        self,
        atc_signals_df: pd.DataFrame,
        signal_type: str,
        indicators_to_calculate: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Calculate signals from all indicators in parallel.

        This replaces sequential filtering with parallel calculation.

        Args:
            atc_signals_df: DataFrame with ATC signals
            signal_type: "LONG" or "SHORT"
            indicators_to_calculate: Optional list of indicator names to calculate.
                If None, calculates all enabled indicators.
                Valid names: "oscillator", "spc", "xgboost", "hmm", "random_forest"
        """
        if atc_signals_df.empty:
            return pd.DataFrame()

        osc_params = self.get_oscillator_params()
        spc_params = self.get_spc_params() if self.args.enable_spc else None
        total = len(atc_signals_df)

        # Build indicator list for logging
        indicators_list = ["ATC"]
        if indicators_to_calculate is None:
            # Calculate all enabled indicators
            indicators_list.append("Range Oscillator")
            if self.args.enable_spc:
                indicators_list.append("SPC")
            if hasattr(self.args, "enable_xgboost") and self.args.enable_xgboost:
                indicators_list.append("XGBoost")
            if hasattr(self.args, "enable_hmm") and self.args.enable_hmm:
                indicators_list.append("HMM")
            if hasattr(self.args, "enable_random_forest") and self.args.enable_random_forest:
                indicators_list.append("Random Forest")
        else:
            # Only calculate specified indicators
            if "oscillator" in indicators_to_calculate:
                indicators_list.append("Range Oscillator")
            if "spc" in indicators_to_calculate and self.args.enable_spc:
                indicators_list.append("SPC")
            if (
                "xgboost" in indicators_to_calculate
                and hasattr(self.args, "enable_xgboost")
                and self.args.enable_xgboost
            ):
                indicators_list.append("XGBoost")
            if "hmm" in indicators_to_calculate and hasattr(self.args, "enable_hmm") and self.args.enable_hmm:
                indicators_list.append("HMM")
            if (
                "random_forest" in indicators_to_calculate
                and hasattr(self.args, "enable_random_forest")
                and self.args.enable_random_forest
            ):
                indicators_list.append("Random Forest")

        log_progress(
            f"Calculating signals from indicators for {total} {signal_type} symbols "
            f"(workers: {osc_params['max_workers']})..."
        )
        log_progress(f"Indicators: {', '.join(indicators_list)}")

        exchange_manager = self.data_fetcher.exchange_manager

        symbol_data_list = [
            {
                "symbol": row["symbol"],
                "signal": row["signal"],
                "trend": row["trend"],
                "price": row["price"],
                "exchange": row["exchange"],
            }
            for _, row in atc_signals_df.iterrows()
        ]

        progress_lock = threading.Lock()
        checked_count = [0]
        processed_count = [0]

        results = []

        # ThreadPoolExecutor automatically handles resource cleanup via context manager
        # All threads are properly joined and resources released when exiting the 'with' block
        # No explicit cleanup needed - Python's context manager protocol ensures proper shutdown
        with ThreadPoolExecutor(max_workers=osc_params["max_workers"]) as executor:
            future_to_symbol = {
                executor.submit(
                    self._process_symbol_for_all_indicators,
                    symbol_data,
                    exchange_manager,
                    self.selected_timeframe,
                    self.args.limit,
                    signal_type,
                    osc_params,
                    spc_params,
                    indicators_to_calculate,
                ): symbol_data["symbol"]
                for symbol_data in symbol_data_list
            }

            for future in as_completed(future_to_symbol):
                _ = future_to_symbol[future]
                try:
                    result = future.result()
                    if result is not None:
                        with progress_lock:
                            processed_count[0] += 1
                            results.append(result)
                except Exception as e:
                    # Log error for debugging instead of silently swallowing
                    # Sanitize error message to prevent information leakage
                    from modules.common.utils import log_error

                    symbol = future_to_symbol.get(future, "unknown")
                    log_error(
                        f"Error processing symbol {symbol} in calculate_signals_for_all_indicators: {type(e).__name__}"
                    )
                finally:
                    with progress_lock:
                        checked_count[0] += 1
                        current_checked = checked_count[0]
                        current_processed = processed_count[0]

                        if current_checked % 10 == 0 or current_checked == total:
                            log_progress(
                                f"Processed {current_checked}/{total} symbols... "
                                f"Got {current_processed} with all indicator signals"
                            )

        if not results:
            return pd.DataFrame()

        result_df = pd.DataFrame(results)

        # Log HMM summary if enabled
        if hasattr(self.args, "enable_hmm") and self.args.enable_hmm:
            hmm_success_count = sum(1 for r in results if r.get("hmm_signal", 0) != 0)
            hmm_total = len(results)
            if hmm_total > 0:
                hmm_success_rate = (hmm_success_count / hmm_total) * 100
                log_progress(
                    f"HMM Status: {hmm_success_count}/{hmm_total} symbols with HMM signals "
                    f"({hmm_success_rate:.1f}% success rate)"
                )

        # Log Random Forest summary if enabled
        if hasattr(self.args, "enable_random_forest") and self.args.enable_random_forest:
            rf_success_count = sum(1 for r in results if r.get("random_forest_signal", 0) != 0)
            rf_total = len(results)
            if rf_total > 0:
                rf_success_rate = (rf_success_count / rf_total) * 100
                log_progress(
                    f"Random Forest Status: {rf_success_count}/{rf_total} symbols with Random Forest signals "
                    f"({rf_success_rate:.1f}% success rate)"
                )

        return result_df

    @contextmanager
    def _temporary_mode(self, new_mode: str):
        """Context manager to temporarily change spc_aggregator.config.mode (thread-safe)."""
        with self._mode_lock:
            original_mode = self.spc_aggregator.config.mode
            self.spc_aggregator.config.mode = new_mode
            try:
                yield
            finally:
                self.spc_aggregator.config.mode = original_mode

    def _aggregate_spc_votes(
        self,
        symbol_data: Dict[str, Any],
        signal_type: str,
        use_threshold_fallback: bool = False,
    ) -> Tuple[int, float]:
        """
        Aggregate 3 SPC strategy votes into a single vote.

        Uses SPCVoteAggregator with improved voting logic similar to Range Oscillator:
        - Separate LONG/SHORT weight calculation
        - Configurable consensus modes (threshold/weighted)
        - Optional adaptive weights based on performance
        - Signal strength filtering
        - Fallback to threshold mode if weighted mode gives no vote
        - Fallback to simple mode if both weighted and threshold give no vote

        Args:
            symbol_data: Symbol data with SPC signals
            signal_type: "LONG" or "SHORT"
            use_threshold_fallback: If True, force use threshold mode

        Returns:
            (vote, strength) where vote is 1 if matches expected signal_type, 0 otherwise
        """
        expected_signal = 1 if signal_type == "LONG" else -1

        # Use threshold mode if fallback requested
        if use_threshold_fallback or self.spc_aggregator.config.mode == "threshold":
            # Temporarily switch to threshold mode
            with self._temporary_mode("threshold"):
                vote, strength, _ = self.spc_aggregator.aggregate(symbol_data, signal_type)

            # If threshold mode also gives no vote, try simple mode fallback
            if vote == 0 and self.spc_aggregator.config.enable_simple_fallback:
                with self._temporary_mode("simple"):
                    vote, strength, _ = self.spc_aggregator.aggregate(symbol_data, signal_type)
        else:
            # Try weighted mode first
            vote, strength, _ = self.spc_aggregator.aggregate(symbol_data, signal_type)

            # If weighted mode gives no vote (vote = 0), fallback to threshold mode
            if vote == 0:
                with self._temporary_mode("threshold"):
                    vote, strength, _ = self.spc_aggregator.aggregate(symbol_data, signal_type)

                # If threshold mode also gives no vote, try simple mode fallback
                if vote == 0 and self.spc_aggregator.config.enable_simple_fallback:
                    with self._temporary_mode("simple"):
                        vote, strength, _ = self.spc_aggregator.aggregate(symbol_data, signal_type)

        # Convert vote to 1/0 format for Decision Matrix compatibility
        # Only accept vote if it matches the expected signal direction
        final_vote = 1 if vote == expected_signal else 0
        return (final_vote, strength)

    def _get_indicator_accuracy(self, indicator: str, signal_type: str) -> float:
        """Get historical accuracy for an indicator from config."""
        return DECISION_MATRIX_INDICATOR_ACCURACIES.get(indicator, 0.5)

    def apply_voting_system(
        self,
        signals_df: pd.DataFrame,
        signal_type: str,
        indicators_to_include: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Apply pure voting system to all signals.

        This is the core of Phương án 2 - no sequential filtering,
        just calculate all signals and vote.

        Args:
            signals_df: DataFrame with calculated signals
            signal_type: "LONG" or "SHORT"
            indicators_to_include: Optional list of indicator names to include in voting.
                If None, includes all enabled indicators.
                Valid names: "atc", "oscillator", "spc", "xgboost", "hmm", "random_forest"
        """
        if signals_df.empty:
            return pd.DataFrame()

        if indicators_to_include is None:
            # Include all enabled indicators
            indicators = ["atc", "oscillator"]
            if self.args.enable_spc:
                indicators.append("spc")
            if hasattr(self.args, "enable_xgboost") and self.args.enable_xgboost:
                indicators.append("xgboost")
            if hasattr(self.args, "enable_hmm") and self.args.enable_hmm:
                indicators.append("hmm")
            if hasattr(self.args, "enable_random_forest") and self.args.enable_random_forest:
                indicators.append("random_forest")
        else:
            # Only include specified indicators
            indicators = indicators_to_include.copy()

        results = []

        for _, row in signals_df.iterrows():
            # Build dynamic indicators list based on actual vote data availability
            # This prevents errors when an indicator is in the list but has no vote data
            available_indicators = []

            # Check which indicators actually have vote data for this row
            if "atc" in indicators and row.get("atc_vote") is not None:
                available_indicators.append("atc")
            if "oscillator" in indicators and row.get("osc_vote") is not None:
                available_indicators.append("oscillator")
            if "spc" in indicators and (
                row.get("spc_cluster_transition_signal") is not None
                or row.get("spc_regime_following_signal") is not None
                or row.get("spc_mean_reversion_signal") is not None
            ):
                available_indicators.append("spc")
            if "xgboost" in indicators and row.get("xgboost_vote") is not None:
                available_indicators.append("xgboost")
            if "hmm" in indicators and row.get("hmm_vote") is not None:
                available_indicators.append("hmm")
            if "random_forest" in indicators and row.get("random_forest_vote") is not None:
                available_indicators.append("random_forest")

            # Skip this row if no indicators have vote data
            if not available_indicators:
                continue

            classifier = DecisionMatrixClassifier(indicators=available_indicators)

            # Get votes from all indicators (only include those in available_indicators list)
            if "atc" in available_indicators:
                atc_vote = row.get("atc_vote", 0)
                atc_strength = row.get("atc_strength", 0.0)
                classifier.add_node_vote(
                    "atc", atc_vote, atc_strength, self._get_indicator_accuracy("atc", signal_type)
                )

            if "oscillator" in available_indicators:
                osc_vote = row.get("osc_vote", 0)
                osc_strength = row.get("osc_confidence", 0.0)
                classifier.add_node_vote(
                    "oscillator", osc_vote, osc_strength, self._get_indicator_accuracy("oscillator", signal_type)
                )

            if "spc" in available_indicators:
                # Aggregate SPC votes from 3 strategies
                spc_vote, spc_strength = self._aggregate_spc_votes(row.to_dict(), signal_type)
                classifier.add_node_vote(
                    "spc", spc_vote, spc_strength, self._get_indicator_accuracy("spc", signal_type)
                )

            if "xgboost" in available_indicators:
                # XGBoost vote
                xgb_vote = row.get("xgboost_vote", 0)
                xgb_strength = row.get("xgboost_confidence", 0.0)
                classifier.add_node_vote(
                    "xgboost", xgb_vote, xgb_strength, self._get_indicator_accuracy("xgboost", signal_type)
                )

            if "hmm" in available_indicators:
                # HMM vote
                hmm_vote = row.get("hmm_vote", 0)
                hmm_strength = row.get("hmm_confidence", 0.0)
                classifier.add_node_vote(
                    "hmm", hmm_vote, hmm_strength, self._get_indicator_accuracy("hmm", signal_type)
                )

            if "random_forest" in available_indicators:
                # Random Forest vote
                rf_vote = row.get("random_forest_vote", 0)
                rf_strength = row.get("random_forest_confidence", 0.0)
                classifier.add_node_vote(
                    "random_forest", rf_vote, rf_strength, self._get_indicator_accuracy("random_forest", signal_type)
                )

            classifier.calculate_weighted_impact()

            cumulative_vote, weighted_score, voting_breakdown = classifier.calculate_cumulative_vote(
                threshold=self.args.voting_threshold,
                min_votes=self.args.min_votes,
            )
            # Only keep if cumulative vote is positive
            if cumulative_vote == 1:
                result = row.to_dict()
                result["cumulative_vote"] = cumulative_vote
                result["weighted_score"] = weighted_score
                result["voting_breakdown"] = voting_breakdown

                metadata = classifier.get_metadata()
                result["feature_importance"] = metadata["feature_importance"]
                result["weighted_impact"] = metadata["weighted_impact"]
                result["independent_accuracy"] = metadata["independent_accuracy"]

                votes_count = sum(v for v in classifier.node_votes.values())
                if votes_count == len(indicators):
                    result["source"] = "ALL_INDICATORS"
                elif votes_count >= self.args.min_votes:
                    result["source"] = "MAJORITY_VOTE"
                else:
                    result["source"] = "WEIGHTED_VOTE"

                results.append(result)

        if not results:
            return pd.DataFrame()

        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values("weighted_score", ascending=False).reset_index(drop=True)

        return result_df

    def calculate_and_vote(self) -> None:
        """
        Calculate signals from all indicators and apply voting system.

        This is the main step for Phương án 2.
        """
        log_progress("\nStep 2: Calculating signals from all indicators...")
        log_progress("=" * 80)

        # Calculate all signals in parallel
        if not self.long_signals_atc.empty:
            long_with_signals = self.calculate_signals_for_all_indicators(
                atc_signals_df=self.long_signals_atc,
                signal_type="LONG",
            )

            # Apply voting system
            log_progress("\nStep 3: Applying voting system to LONG signals...")
            self.long_signals_final = self.apply_voting_system(long_with_signals, "LONG")
            log_progress(f"LONG signals: {len(self.long_signals_atc)} → {len(self.long_signals_final)} after voting")
        else:
            self.long_signals_final = pd.DataFrame()

        if not self.short_signals_atc.empty:
            short_with_signals = self.calculate_signals_for_all_indicators(
                atc_signals_df=self.short_signals_atc,
                signal_type="SHORT",
            )

            # Apply voting system
            log_progress("\nStep 3: Applying voting system to SHORT signals...")
            self.short_signals_final = self.apply_voting_system(short_with_signals, "SHORT")
            log_progress(f"SHORT signals: {len(self.short_signals_atc)} → {len(self.short_signals_final)} after voting")
        else:
            self.short_signals_final = pd.DataFrame()

    def display_results(self) -> None:
        """Display final results with voting metadata."""
        log_progress("\nStep 4: Displaying final results...")
        display_final_results(
            long_signals=self.long_signals_final,
            short_signals=self.short_signals_final,
            original_long_count=len(self.long_signals_atc),
            original_short_count=len(self.short_signals_atc),
            long_uses_fallback=False,
            short_uses_fallback=False,
        )

        # Display voting metadata
        if not self.long_signals_final.empty:
            self._display_voting_metadata(self.long_signals_final, "LONG")

        if not self.short_signals_final.empty:
            self._display_voting_metadata(self.short_signals_final, "SHORT")

    def _display_voting_metadata(self, signals_df: pd.DataFrame, signal_type: str) -> None:
        """Display voting metadata for signals."""
        # Lazy import to avoid circular dependency
        from cli.display import display_voting_metadata

        display_voting_metadata(
            signals_df=signals_df,
            signal_type=signal_type,
            show_spc_debug=False,  # No debug info for voting mode
        )

    def run(self) -> None:
        """
        Run the complete Pure Voting System workflow.

        Workflow:
        1. Determine timeframe
        2. Display configuration
        3. Run ATC auto scan
        4. Calculate signals from all indicators in parallel
        5. Apply voting system
        6. Display final results
        """
        self.determine_timeframe()
        self.display_config()
        log_progress("Initializing components...")

        # Run ATC scan - exit early if no signals found
        if not self.run_atc_scan():
            log_warn("\nAnalysis terminated: No ATC signals found.")
            return

        self.calculate_and_vote()
        self.display_results()

        log_success("\nAnalysis complete!")

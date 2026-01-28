"""Signal calculation helpers for VotingAnalyzer."""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import pandas as pd

from core.signal_calculators import (
    get_hmm_signal,
    get_random_forest_signal,
    get_range_oscillator_signal,
    get_spc_signal,
    get_xgboost_signal,
)
from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager
from modules.common.utils import log_progress, log_warn
from modules.simplified_percentile_clustering.core.clustering import ClusteringConfig
from modules.simplified_percentile_clustering.core.features import FeatureConfig


class VotingSignalCalculationMixin:
    """Mixin for parallel signal calculation across indicators."""

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

"""
Hybrid Analyzer for ATC + Range Oscillator + SPC approach.

This module contains the HybridAnalyzer class that combines signals from:
1. Adaptive Trend Classification (ATC)
2. Range Oscillator
3. Simplified Percentile Clustering (SPC)

Phương án 1: Kết hợp sequential filtering và voting system.
"""

import threading
from typing import Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import pandas as pd
import json
import time

from colorama import Fore, Style

from config import (
    DECISION_MATRIX_INDICATOR_ACCURACIES,
    SPC_STRATEGY_PARAMETERS,
    SPC_P_LOW,
    SPC_P_HIGH,
    SPC_AGGREGATION_MODE,
    SPC_AGGREGATION_THRESHOLD,
    SPC_AGGREGATION_WEIGHTED_MIN_TOTAL,
    SPC_AGGREGATION_WEIGHTED_MIN_DIFF,
    SPC_AGGREGATION_ENABLE_ADAPTIVE_WEIGHTS,
    SPC_AGGREGATION_ADAPTIVE_PERFORMANCE_WINDOW,
    SPC_AGGREGATION_MIN_SIGNAL_STRENGTH,
    SPC_AGGREGATION_ENABLE_SIMPLE_FALLBACK,
    SPC_AGGREGATION_SIMPLE_MIN_ACCURACY_TOTAL,
    SPC_AGGREGATION_STRATEGY_WEIGHTS,
)
from modules.common.utils import (
    color_text,
    log_progress,
    log_success,
    log_warn,
)
from modules.common.core.exchange_manager import ExchangeManager
from modules.common.core.data_fetcher import DataFetcher
from modules.adaptive_trend.cli import prompt_timeframe
from modules.adaptive_trend.cli.main import ATCAnalyzer
from modules.range_oscillator.cli import (
    display_final_results,
)
from cli.display import display_config, display_voting_metadata
from modules.simplified_percentile_clustering.core.clustering import (
    ClusteringConfig,
)
from modules.simplified_percentile_clustering.core.features import FeatureConfig
from modules.simplified_percentile_clustering.aggregation import (
    SPCVoteAggregator,
)
from modules.simplified_percentile_clustering.config import (
    SPCAggregationConfig,
)
from modules.decision_matrix.classifier import DecisionMatrixClassifier
from core.signal_calculators import (
    get_range_oscillator_signal,
    get_spc_signal,
    get_xgboost_signal,
    get_hmm_signal,
    get_random_forest_signal,
)


class HybridAnalyzer:
    """
    ATC + Range Oscillator + SPC Hybrid Analyzer.
    
    Phương án 1: Kết hợp sequential filtering và voting system.
    """
    
    def __init__(self, args, data_fetcher: DataFetcher):
        """Initialize analyzer."""
        self.args = args
        self.data_fetcher = data_fetcher
        self.atc_analyzer = ATCAnalyzer(args, data_fetcher)
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
        self.long_signals_confirmed = pd.DataFrame()
        self.short_signals_confirmed = pd.DataFrame()
        self.long_uses_fallback = False
        self.short_uses_fallback = False
    
    def determine_timeframe(self) -> str:
        """Determine timeframe from arguments and interactive menu."""
        self.selected_timeframe = self.args.timeframe
        
        if not self.args.no_menu:
            print("\n" + color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
            print(color_text("ATC PHASE - TIMEFRAME SELECTION", Fore.CYAN, Style.BRIGHT))
            print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
            self.selected_timeframe = prompt_timeframe(default_timeframe=self.selected_timeframe)
            print(color_text(f"\nSelected timeframe for ATC analysis: {self.selected_timeframe}", Fore.GREEN))
        
        self.atc_analyzer.selected_timeframe = self.selected_timeframe
        return self.selected_timeframe
    
    def get_oscillator_params(self) -> dict:
        """Extract Range Oscillator parameters from arguments."""
        return {
            "osc_length": self.args.osc_length,
            "osc_mult": self.args.osc_mult,
            "max_workers": self.args.max_workers,
            "strategies": self.args.osc_strategies,
        }
    
    def get_spc_params(self) -> dict:
        """Extract SPC parameters from arguments for all 3 strategies."""
        # Use values from config if not provided in args
        cluster_transition_params = SPC_STRATEGY_PARAMETERS['cluster_transition'].copy()
        regime_following_params = SPC_STRATEGY_PARAMETERS['regime_following'].copy()
        mean_reversion_params = SPC_STRATEGY_PARAMETERS['mean_reversion'].copy()
        
        # Override with args if provided (for command-line usage)
        if hasattr(self.args, 'spc_min_signal_strength'):
            cluster_transition_params['min_signal_strength'] = self.args.spc_min_signal_strength
        if hasattr(self.args, 'spc_min_rel_pos_change'):
            cluster_transition_params['min_rel_pos_change'] = self.args.spc_min_rel_pos_change
        if hasattr(self.args, 'spc_min_regime_strength'):
            regime_following_params['min_regime_strength'] = self.args.spc_min_regime_strength
        if hasattr(self.args, 'spc_min_cluster_duration'):
            regime_following_params['min_cluster_duration'] = self.args.spc_min_cluster_duration
        if hasattr(self.args, 'spc_extreme_threshold'):
            mean_reversion_params['extreme_threshold'] = self.args.spc_extreme_threshold
        if hasattr(self.args, 'spc_min_extreme_duration'):
            mean_reversion_params['min_extreme_duration'] = self.args.spc_min_extreme_duration
        
        return {
            "k": self.args.spc_k,
            "lookback": self.args.spc_lookback,
            "p_low": self.args.spc_p_low,
            "p_high": self.args.spc_p_high,
            "cluster_transition_params": cluster_transition_params,
            "regime_following_params": regime_following_params,
            "mean_reversion_params": mean_reversion_params,
        }
    
    def display_config(self) -> None:
        """Display configuration information."""
        display_config(
            selected_timeframe=self.selected_timeframe,
            args=self.args,
            get_oscillator_params=self.get_oscillator_params,
            get_spc_params=self.get_spc_params,
            mode="hybrid",
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
    
    def _process_symbol_for_oscillator(
        self,
        symbol_data: Dict[str, Any],
        exchange_manager: ExchangeManager,
        timeframe: str,
        limit: int,
        expected_osc_signal: int,
        osc_length: int,
        osc_mult: float,
        strategies: Optional[list] = None,
    ) -> Optional[Dict[str, Any]]:
        """Worker function to process a single symbol for Range Oscillator confirmation."""
        try:
            data_fetcher = DataFetcher(exchange_manager)
            symbol = symbol_data["symbol"]
            
            osc_result = get_range_oscillator_signal(
                data_fetcher=data_fetcher,
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                osc_length=osc_length,
                osc_mult=osc_mult,
                strategies=strategies,
            )

            if osc_result is None:
                return None
            
            osc_signal, osc_confidence = osc_result

            if osc_signal == expected_osc_signal:
                return {
                    "symbol": symbol,
                    "signal": symbol_data["signal"],
                    "trend": symbol_data["trend"],
                    "price": symbol_data["price"],
                    "exchange": symbol_data["exchange"],
                    "osc_signal": osc_signal,
                    "osc_confidence": osc_confidence,
                }
            
            return None
            
        except Exception as e:
            return None
    
    def _process_symbol_for_spc(
        self,
        symbol_data: Dict[str, Any],
        exchange_manager: ExchangeManager,
        timeframe: str,
        limit: int,
        spc_params: dict,
    ) -> Optional[Dict[str, Any]]:
        """Worker function to calculate SPC signals from all 3 strategies for a symbol."""
        try:
            data_fetcher = DataFetcher(exchange_manager)
            symbol = symbol_data["symbol"]
            
            feature_config = FeatureConfig()
            clustering_config = ClusteringConfig(
                k=spc_params["k"],
                lookback=spc_params["lookback"],
                p_low=spc_params["p_low"],
                p_high=spc_params["p_high"],
                main_plot="Clusters",
                feature_config=feature_config,
            )
            
            # Calculate signals from all 3 strategies
            result = {
                "symbol": symbol,
                "signal": symbol_data["signal"],
                "trend": symbol_data["trend"],
                "price": symbol_data["price"],
                "exchange": symbol_data["exchange"],
            }
            
            # Copy existing fields
            if "osc_signal" in symbol_data:
                result["osc_signal"] = symbol_data["osc_signal"]
                result["osc_confidence"] = symbol_data.get("osc_confidence", 0.0)
            if "source" in symbol_data:
                result["source"] = symbol_data["source"]
            
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
                result["spc_cluster_transition_signal"] = ct_result[0]
                result["spc_cluster_transition_strength"] = ct_result[1]
            else:
                result["spc_cluster_transition_signal"] = 0
                result["spc_cluster_transition_strength"] = 0.0
            
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
                result["spc_regime_following_signal"] = rf_result[0]
                result["spc_regime_following_strength"] = rf_result[1]
            else:
                result["spc_regime_following_signal"] = 0
                result["spc_regime_following_strength"] = 0.0
            
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
                result["spc_mean_reversion_signal"] = mr_result[0]
                result["spc_mean_reversion_strength"] = mr_result[1]
            else:
                result["spc_mean_reversion_signal"] = 0
                result["spc_mean_reversion_strength"] = 0.0
            
            return result
            
        except Exception as e:
            return None
    
    def filter_signals_by_range_oscillator(
        self,
        atc_signals_df: pd.DataFrame,
        signal_type: str,
    ) -> pd.DataFrame:
        """Filter ATC signals by checking Range Oscillator confirmation."""
        if atc_signals_df.empty:
            return pd.DataFrame()

        osc_params = self.get_oscillator_params()
        expected_osc_signal = 1 if signal_type == "LONG" else -1
        total = len(atc_signals_df)
        
        log_progress(
            f"Checking Range Oscillator signals for {total} {signal_type} symbols "
            f"(workers: {osc_params['max_workers']})..."
        )

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
        confirmed_count = [0]

        filtered_results = []
        
        with ThreadPoolExecutor(max_workers=osc_params["max_workers"]) as executor:
            future_to_symbol = {
                executor.submit(
                    self._process_symbol_for_oscillator,
                    symbol_data,
                    exchange_manager,
                    self.selected_timeframe,
                    self.args.limit,
                    expected_osc_signal,
                    osc_params["osc_length"],
                    osc_params["osc_mult"],
                    osc_params["strategies"],
                ): symbol_data["symbol"]
                for symbol_data in symbol_data_list
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result is not None:
                        with progress_lock:
                            confirmed_count[0] += 1
                            filtered_results.append(result)
                except Exception as e:
                    pass
                finally:
                    with progress_lock:
                        checked_count[0] += 1
                        current_checked = checked_count[0]
                        current_confirmed = confirmed_count[0]
                        
                        if current_checked % 10 == 0 or current_checked == total:
                            log_progress(
                                f"Checked {current_checked}/{total} symbols... "
                                f"Found {current_confirmed} confirmed {signal_type} signals"
                            )

        if not filtered_results:
            return pd.DataFrame()

        filtered_df = pd.DataFrame(filtered_results)
        
        if "osc_confidence" in filtered_df.columns:
            if signal_type == "LONG":
                filtered_df = filtered_df.sort_values(
                    ["osc_confidence", "signal"], 
                    ascending=[False, False]
                ).reset_index(drop=True)
            else:
                filtered_df = filtered_df.sort_values(
                    ["osc_confidence", "signal"], 
                    ascending=[False, True]
                ).reset_index(drop=True)
        else:
            if signal_type == "LONG":
                filtered_df = filtered_df.sort_values("signal", ascending=False).reset_index(drop=True)
            else:
                filtered_df = filtered_df.sort_values("signal", ascending=True).reset_index(drop=True)

        return filtered_df
    
    def calculate_spc_signals(
        self,
        signals_df: pd.DataFrame,
        signal_type: str,
    ) -> pd.DataFrame:
        """Calculate SPC signals from all 3 strategies for all symbols."""
        if signals_df.empty:
            return pd.DataFrame()

        spc_params = self.get_spc_params()
        total = len(signals_df)
        
        # Build indicator list for logging
        indicators_list = ["ATC", "Range Oscillator"]
        if self.args.enable_spc:
            indicators_list.append("SPC")
        if hasattr(self.args, 'enable_xgboost') and self.args.enable_xgboost:
            indicators_list.append("XGBoost")
        if hasattr(self.args, 'enable_hmm') and self.args.enable_hmm:
            indicators_list.append("HMM")
        
        log_progress(
            f"Calculating SPC signals (all 3 strategies) for {total} {signal_type} symbols "
            f"(workers: {self.args.max_workers})..."
        )
        log_progress(f"Active Indicators: {', '.join(indicators_list)}")

        exchange_manager = self.data_fetcher.exchange_manager
        
        symbol_data_list = [row.to_dict() for _, row in signals_df.iterrows()]

        progress_lock = threading.Lock()
        checked_count = [0]

        results = []
        
        with ThreadPoolExecutor(max_workers=self.args.max_workers) as executor:
            future_to_symbol = {
                executor.submit(
                    self._process_symbol_for_spc,
                    symbol_data,
                    exchange_manager,
                    self.selected_timeframe,
                    self.args.limit,
                    spc_params,
                ): symbol_data["symbol"]
                for symbol_data in symbol_data_list
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result is not None:
                        with progress_lock:
                            results.append(result)
                except Exception as e:
                    pass
                finally:
                    with progress_lock:
                        checked_count[0] += 1
                        current_checked = checked_count[0]
                        
                        if current_checked % 10 == 0 or current_checked == total:
                            log_progress(
                                f"Calculated SPC signals for {current_checked}/{total} symbols..."
                            )

        if not results:
            return pd.DataFrame()

        result_df = pd.DataFrame(results)
        
        # Calculate XGBoost signals if enabled
        if hasattr(self.args, 'enable_xgboost') and self.args.enable_xgboost and not result_df.empty:
            log_progress(f"Calculating XGBoost predictions for {len(result_df)} {signal_type} symbols...")
            xgb_results = []
            for _, row in result_df.iterrows():
                try:
                    xgb_result = get_xgboost_signal(
                        data_fetcher=self.data_fetcher,
                        symbol=row['symbol'],
                        timeframe=self.selected_timeframe,
                        limit=self.args.limit,
                    )
                    if xgb_result is not None:
                        row_dict = row.to_dict()
                        row_dict["xgboost_signal"] = xgb_result[0]
                        row_dict["xgboost_confidence"] = xgb_result[1]
                        xgb_results.append(row_dict)
                    else:
                        row_dict = row.to_dict()
                        row_dict["xgboost_signal"] = 0
                        row_dict["xgboost_confidence"] = 0.0
                        xgb_results.append(row_dict)
                except Exception as e:
                    row_dict = row.to_dict()
                    row_dict["xgboost_signal"] = 0
                    row_dict["xgboost_confidence"] = 0.0
                    xgb_results.append(row_dict)
            result_df = pd.DataFrame(xgb_results)
        
        # Calculate HMM signals if enabled
        if hasattr(self.args, 'enable_hmm') and self.args.enable_hmm and not result_df.empty:
            log_progress(f"Calculating HMM signals for {len(result_df)} {signal_type} symbols...")
            hmm_results = []
            for _, row in result_df.iterrows():
                try:
                    hmm_result = get_hmm_signal(
                        data_fetcher=self.data_fetcher,
                        symbol=row['symbol'],
                        timeframe=self.selected_timeframe,
                        limit=self.args.limit,
                        window_size=getattr(self.args, 'hmm_window_size', None),
                        window_kama=getattr(self.args, 'hmm_window_kama', None),
                        fast_kama=getattr(self.args, 'hmm_fast_kama', None),
                        slow_kama=getattr(self.args, 'hmm_slow_kama', None),
                        orders_argrelextrema=getattr(self.args, 'hmm_orders_argrelextrema', None),
                        strict_mode=getattr(self.args, 'hmm_strict_mode', None),
                    )
                    if hmm_result is not None:
                        row_dict = row.to_dict()
                        row_dict["hmm_signal"] = hmm_result[0]
                        row_dict["hmm_confidence"] = hmm_result[1]
                        hmm_results.append(row_dict)
                    else:
                        row_dict = row.to_dict()
                        row_dict["hmm_signal"] = 0
                        row_dict["hmm_confidence"] = 0.0
                        hmm_results.append(row_dict)
                except Exception as e:
                    # Log HMM errors but don't fail the entire process
                    log_warn(f"HMM signal calculation failed for {row['symbol']}: {type(e).__name__}: {e}")
                    row_dict = row.to_dict()
                    row_dict["hmm_signal"] = 0
                    row_dict["hmm_confidence"] = 0.0
                    hmm_results.append(row_dict)
            
            result_df = pd.DataFrame(hmm_results)
            
            # Log HMM summary
            if hmm_results:
                hmm_success_count = sum(1 for r in hmm_results if r.get('hmm_signal', 0) != 0)
                hmm_total = len(hmm_results)
                if hmm_total > 0:
                    hmm_success_rate = (hmm_success_count / hmm_total) * 100
                    log_progress(
                        f"HMM Status: {hmm_success_count}/{hmm_total} symbols with HMM signals "
                        f"({hmm_success_rate:.1f}% success rate)"
                    )
        
        # Calculate Random Forest signals if enabled
        if hasattr(self.args, 'enable_random_forest') and self.args.enable_random_forest and not result_df.empty:
            log_progress(f"Calculating Random Forest predictions for {len(result_df)} {signal_type} symbols...")
            rf_results = []
            for _, row in result_df.iterrows():
                try:
                    rf_result = get_random_forest_signal(
                        data_fetcher=self.data_fetcher,
                        symbol=row['symbol'],
                        timeframe=self.selected_timeframe,
                        limit=self.args.limit,
                        model_path=getattr(self.args, 'random_forest_model_path', None),
                    )
                    if rf_result is not None:
                        row_dict = row.to_dict()
                        row_dict["random_forest_signal"] = rf_result[0]
                        row_dict["random_forest_confidence"] = rf_result[1]
                        rf_results.append(row_dict)
                    else:
                        row_dict = row.to_dict()
                        row_dict["random_forest_signal"] = 0
                        row_dict["random_forest_confidence"] = 0.0
                        rf_results.append(row_dict)
                except Exception as e:
                    # Log Random Forest errors but don't fail the entire process
                    log_warn(f"Random Forest signal calculation failed for {row['symbol']}: {type(e).__name__}: {e}")
                    row_dict = row.to_dict()
                    row_dict["random_forest_signal"] = 0
                    row_dict["random_forest_confidence"] = 0.0
                    rf_results.append(row_dict)
            result_df = pd.DataFrame(rf_results)
        
        # Sort by signal strength (use average of all 3 strategies)
        if signal_type == "LONG":
            result_df = result_df.sort_values("signal", ascending=False).reset_index(drop=True)
        else:
            result_df = result_df.sort_values("signal", ascending=True).reset_index(drop=True)

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
        # Note: If vote is opposite direction (e.g., -1 when expected is 1),
        # we return 0 to indicate no vote for this direction
        final_vote = 1 if vote == expected_signal else 0
        
        return (final_vote, strength)
    
    def calculate_indicator_votes(
        self,
        symbol_data: Dict[str, Any],
        signal_type: str,
    ) -> Dict[str, Tuple[int, float]]:
        """Calculate votes from all indicators for a symbol (SPC votes aggregated into 1)."""
        expected_signal = 1 if signal_type == "LONG" else -1
        votes = {}
        
        # ATC vote (always 1 if symbol passed ATC scan)
        # ATC signal is a percentage value (e.g., 94, -50), not just 1 or -1
        # If symbol is in long_signals_atc/short_signals_atc, it means it passed ATC scan
        # So we check the sign of the signal, not exact equality
        atc_signal = symbol_data.get('signal', 0)
        # Check sign: positive for LONG, negative for SHORT
        if signal_type == "LONG":
            atc_vote = 1 if atc_signal > 0 else 0
        else:  # SHORT
            atc_vote = 1 if atc_signal < 0 else 0
        atc_strength = abs(atc_signal) / 100.0 if atc_signal != 0 else 0.0
        votes['atc'] = (atc_vote, min(atc_strength, 1.0))
        
        # Range Oscillator vote
        osc_signal = symbol_data.get('osc_signal', 0)
        osc_vote = 1 if osc_signal == expected_signal else 0
        osc_strength = symbol_data.get('osc_confidence', 0.0)
        votes['oscillator'] = (osc_vote, osc_strength)
        
        # SPC vote (aggregated from all 3 strategies)
        if self.args.enable_spc:
            spc_vote, spc_strength = self._aggregate_spc_votes(symbol_data, signal_type)
            votes['spc'] = (spc_vote, spc_strength)
        
        # XGBoost vote (if enabled)
        if hasattr(self.args, 'enable_xgboost') and self.args.enable_xgboost:
            xgb_signal = symbol_data.get('xgboost_signal', 0)
            xgb_vote = 1 if xgb_signal == expected_signal else 0
            xgb_strength = symbol_data.get('xgboost_confidence', 0.0)
            votes['xgboost'] = (xgb_vote, xgb_strength)
        
        # HMM vote (if enabled)
        if hasattr(self.args, 'enable_hmm') and self.args.enable_hmm:
            hmm_signal = symbol_data.get('hmm_signal', 0)
            hmm_vote = 1 if hmm_signal == expected_signal else 0
            hmm_strength = symbol_data.get('hmm_confidence', 0.0)
            votes['hmm'] = (hmm_vote, hmm_strength)
        
        # Random Forest vote (if enabled)
        if hasattr(self.args, 'enable_random_forest') and self.args.enable_random_forest:
            rf_signal = symbol_data.get('random_forest_signal', 0)
            rf_vote = 1 if rf_signal == expected_signal else 0
            rf_strength = symbol_data.get('random_forest_confidence', 0.0)
            votes['random_forest'] = (rf_vote, rf_strength)
        
        return votes
    
    def _get_indicator_accuracy(self, indicator: str, signal_type: str) -> float:
        """Get historical accuracy for an indicator from config."""
        return DECISION_MATRIX_INDICATOR_ACCURACIES.get(indicator, 0.5)
    
    def apply_decision_matrix(
        self,
        signals_df: pd.DataFrame,
        signal_type: str,
    ) -> pd.DataFrame:
        """Apply decision matrix voting system to filter signals."""
        if signals_df.empty:
            return pd.DataFrame()
        
        # Build indicators list: ATC, Oscillator, aggregated SPC, XGBoost, HMM, and Random Forest
        indicators = ['atc', 'oscillator']
        indicators_display = ['ATC', 'Range Oscillator']
        if self.args.enable_spc:
            indicators.append('spc')
            indicators_display.append('SPC')
        if hasattr(self.args, 'enable_xgboost') and self.args.enable_xgboost:
            indicators.append('xgboost')
            indicators_display.append('XGBoost')
        if hasattr(self.args, 'enable_hmm') and self.args.enable_hmm:
            indicators.append('hmm')
            indicators_display.append('HMM')
        if hasattr(self.args, 'enable_random_forest') and self.args.enable_random_forest:
            indicators.append('random_forest')
            indicators_display.append('Random Forest')
        
        log_progress(f"Decision Matrix Indicators: {', '.join(indicators_display)}")
        
        results = []
        
        for _, row in signals_df.iterrows():
            classifier = DecisionMatrixClassifier(indicators=indicators)
            
            votes = self.calculate_indicator_votes(row.to_dict(), signal_type)
            
            for indicator, (vote, strength) in votes.items():
                accuracy = self._get_indicator_accuracy(indicator, signal_type)
                classifier.add_node_vote(indicator, vote, strength, accuracy)
            
            classifier.calculate_weighted_impact()
            
            cumulative_vote, weighted_score, voting_breakdown = classifier.calculate_cumulative_vote(
                threshold=self.args.voting_threshold,
                min_votes=self.args.min_votes,
            )
            if cumulative_vote == 1:
                result = row.to_dict()
                result['cumulative_vote'] = cumulative_vote
                result['weighted_score'] = weighted_score
                result['voting_breakdown'] = voting_breakdown
                
                metadata = classifier.get_metadata()
                result['feature_importance'] = metadata['feature_importance']
                result['weighted_impact'] = metadata['weighted_impact']
                result['independent_accuracy'] = metadata['independent_accuracy']
                
                votes_count = sum(v for v in classifier.node_votes.values())
                if votes_count == len(indicators):
                    result['source'] = 'ALL_INDICATORS'
                elif votes_count >= 2:  # At least 2 out of 3 indicators agree
                    result['source'] = 'MAJORITY_VOTE'
                else:
                    result['source'] = 'WEIGHTED_VOTE'
                
                results.append(result)
        
        if not results:
            return pd.DataFrame()
        
        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values('weighted_score', ascending=False).reset_index(drop=True)
        
        return result_df
    
    def filter_by_oscillator(self) -> None:
        """Filter ATC signals by Range Oscillator confirmation."""
        log_progress("\nStep 2: Filtering by Range Oscillator confirmation...")
        log_progress("=" * 80)

        if not self.long_signals_atc.empty:
            self.long_signals_confirmed = self.filter_signals_by_range_oscillator(
                atc_signals_df=self.long_signals_atc,
                signal_type="LONG",
            )
            
            if self.long_signals_confirmed.empty:
                log_warn("No LONG signals confirmed by Range Oscillator. Falling back to ATC signals only.")
                self.long_signals_confirmed = self.long_signals_atc.copy()
                self.long_signals_confirmed['source'] = 'ATC_ONLY'
                self.long_uses_fallback = True
            else:
                self.long_signals_confirmed['source'] = 'ATC_OSCILLATOR'
                self.long_uses_fallback = False
        else:
            self.long_signals_confirmed = pd.DataFrame()
            self.long_uses_fallback = False

        if not self.short_signals_atc.empty:
            self.short_signals_confirmed = self.filter_signals_by_range_oscillator(
                atc_signals_df=self.short_signals_atc,
                signal_type="SHORT",
            )
            
            if self.short_signals_confirmed.empty:
                log_warn("No SHORT signals confirmed by Range Oscillator. Falling back to ATC signals only.")
                self.short_signals_confirmed = self.short_signals_atc.copy()
                self.short_signals_confirmed['source'] = 'ATC_ONLY'
                self.short_uses_fallback = True
            else:
                self.short_signals_confirmed['source'] = 'ATC_OSCILLATOR'
                self.short_uses_fallback = False
        else:
            self.short_signals_confirmed = pd.DataFrame()
            self.short_uses_fallback = False
    
    def calculate_spc_signals_for_all(self) -> None:
        """Calculate SPC signals from all 3 strategies for all confirmed signals."""
        log_progress("\nStep 3: Calculating SPC signals (all 3 strategies)...")
        log_progress("=" * 80)

        if not self.long_signals_confirmed.empty:
            self.long_signals_confirmed = self.calculate_spc_signals(
                signals_df=self.long_signals_confirmed,
                signal_type="LONG",
            )
            log_progress(f"Calculated SPC signals for {len(self.long_signals_confirmed)} LONG symbols")

        if not self.short_signals_confirmed.empty:
            self.short_signals_confirmed = self.calculate_spc_signals(
                signals_df=self.short_signals_confirmed,
                signal_type="SHORT",
            )
            log_progress(f"Calculated SPC signals for {len(self.short_signals_confirmed)} SHORT symbols")
    
    def filter_by_decision_matrix(self) -> None:
        """Filter signals using decision matrix voting system."""
        log_progress("\nStep 4: Applying Decision Matrix voting system...")
        log_progress("=" * 80)

        if not self.long_signals_confirmed.empty:
            long_before = len(self.long_signals_confirmed)
            self.long_signals_confirmed = self.apply_decision_matrix(
                self.long_signals_confirmed,
                "LONG",
            )
            long_after = len(self.long_signals_confirmed)
            log_progress(f"LONG signals: {long_before} → {long_after} after voting")
        else:
            self.long_signals_confirmed = pd.DataFrame()

        if not self.short_signals_confirmed.empty:
            short_before = len(self.short_signals_confirmed)
            self.short_signals_confirmed = self.apply_decision_matrix(
                self.short_signals_confirmed,
                "SHORT",
            )
            short_after = len(self.short_signals_confirmed)
            log_progress(f"SHORT signals: {short_before} → {short_after} after voting")
        else:
            self.short_signals_confirmed = pd.DataFrame()
    
    def display_results(self) -> None:
        """Display final filtered results."""
        log_progress("\nStep 5: Displaying final results...")
        display_final_results(
            long_signals=self.long_signals_confirmed,
            short_signals=self.short_signals_confirmed,
            original_long_count=len(self.long_signals_atc),
            original_short_count=len(self.short_signals_atc),
            long_uses_fallback=self.long_uses_fallback,
            short_uses_fallback=self.short_uses_fallback,
        )
        
        # Display voting metadata if decision matrix was used
        if self.args.use_decision_matrix and not self.long_signals_confirmed.empty:
            self._display_voting_metadata(self.long_signals_confirmed, "LONG")
        
        if self.args.use_decision_matrix and not self.short_signals_confirmed.empty:
            self._display_voting_metadata(self.short_signals_confirmed, "SHORT")
    
    def _display_voting_metadata(self, signals_df: pd.DataFrame, signal_type: str) -> None:
        """Display voting metadata for signals."""
        display_voting_metadata(
            signals_df=signals_df,
            signal_type=signal_type,
            show_spc_debug=True,  # Show debug info for hybrid mode
        )
    
    def run(self) -> None:
        """
        Run the complete ATC + Range Oscillator + SPC Hybrid workflow.
        
        Workflow:
        1. Determine timeframe
        2. Display configuration
        3. Run ATC auto scan
        4. Filter by Range Oscillator confirmation
        5. Filter by SPC confirmation (if enabled)
        6. Apply Decision Matrix voting (if enabled)
        7. Display final results
        """
        self.determine_timeframe()
        self.display_config()
        log_progress("Initializing components...")
        
        # Run ATC scan - exit early if no signals found
        if not self.run_atc_scan():
            log_warn("\nAnalysis terminated: No ATC signals found.")
            return
        
        self.filter_by_oscillator()
        
        if self.args.enable_spc:
            self.calculate_spc_signals_for_all()
        
        if self.args.use_decision_matrix:
            self.filter_by_decision_matrix()
        
        self.display_results()
        log_success("\nAnalysis complete!")


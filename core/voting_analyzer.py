"""
Voting Analyzer for ATC + Range Oscillator + SPC Pure Voting System.

This module contains the VotingAnalyzer class that combines signals from:
1. Adaptive Trend Classification (ATC)
2. Range Oscillator
3. Simplified Percentile Clustering (SPC)

Phương án 2: Thay thế hoàn toàn sequential filtering bằng voting system.
"""

import threading
from typing import Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

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
from modules.common.ExchangeManager import ExchangeManager
from modules.common.DataFetcher import DataFetcher
from modules.adaptive_trend.cli import prompt_timeframe
from main.main_atc import ATCAnalyzer
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
)


class VotingAnalyzer:
    """
    ATC + Range Oscillator + SPC Pure Voting Analyzer.
    
    Phương án 2: Thay thế hoàn toàn sequential filtering bằng voting system.
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
            mode="voting",
        )
    
    def run_atc_scan(self) -> None:
        """Run ATC auto scan to get LONG/SHORT signals."""
        log_progress("\nStep 1: Running ATC auto scan...")
        log_progress("=" * 80)
        
        self.long_signals_atc, self.short_signals_atc = self.atc_analyzer.run_auto_scan()
        
        original_long_count = len(self.long_signals_atc)
        original_short_count = len(self.short_signals_atc)
        
        log_success(f"\nATC Scan Complete: Found {original_long_count} LONG + {original_short_count} SHORT signals")
        
        if self.long_signals_atc.empty and self.short_signals_atc.empty:
            log_warn("No ATC signals found. Exiting.")
            raise ValueError("No ATC signals found")
    
    def _process_symbol_for_all_indicators(
        self,
        symbol_data: Dict[str, Any],
        exchange_manager: ExchangeManager,
        timeframe: str,
        limit: int,
        signal_type: str,
        osc_params: dict,
        spc_params: Optional[dict],
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
            atc_signal = symbol_data["signal"]
            atc_vote = 1 if atc_signal == expected_signal else 0
            atc_strength = abs(atc_signal) / 100.0 if atc_signal != 0 else 0.0
            results['atc_signal'] = atc_signal
            results['atc_vote'] = atc_vote
            results['atc_strength'] = min(atc_strength, 1.0)
            
            # Range Oscillator signal
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
                results['osc_signal'] = osc_signal
                results['osc_vote'] = osc_vote
                results['osc_confidence'] = osc_confidence
            else:
                results['osc_signal'] = 0
                results['osc_vote'] = 0
                results['osc_confidence'] = 0.0
            
            # SPC signals from all 3 strategies (if enabled)
            if self.args.enable_spc and spc_params:
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
                    results['spc_cluster_transition_signal'] = ct_result[0]
                    results['spc_cluster_transition_strength'] = ct_result[1]
                else:
                    results['spc_cluster_transition_signal'] = 0
                    results['spc_cluster_transition_strength'] = 0.0
                
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
                    results['spc_regime_following_signal'] = rf_result[0]
                    results['spc_regime_following_strength'] = rf_result[1]
                else:
                    results['spc_regime_following_signal'] = 0
                    results['spc_regime_following_strength'] = 0.0
                
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
                    results['spc_mean_reversion_signal'] = mr_result[0]
                    results['spc_mean_reversion_strength'] = mr_result[1]
                else:
                    results['spc_mean_reversion_signal'] = 0
                    results['spc_mean_reversion_strength'] = 0.0
            
            # XGBoost prediction (if enabled)
            if hasattr(self.args, 'enable_xgboost') and self.args.enable_xgboost:
                xgb_result = get_xgboost_signal(
                    data_fetcher=data_fetcher,
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit,
                )
                if xgb_result is not None:
                    xgb_signal, xgb_confidence = xgb_result
                    xgb_vote = 1 if xgb_signal == expected_signal else 0
                    results['xgboost_signal'] = xgb_signal
                    results['xgboost_vote'] = xgb_vote
                    results['xgboost_confidence'] = xgb_confidence
                else:
                    results['xgboost_signal'] = 0
                    results['xgboost_vote'] = 0
                    results['xgboost_confidence'] = 0.0
            
            return results
            
        except Exception as e:
            return None
    
    def calculate_signals_for_all_indicators(
        self,
        atc_signals_df: pd.DataFrame,
        signal_type: str,
    ) -> pd.DataFrame:
        """
        Calculate signals from all indicators in parallel.
        
        This replaces sequential filtering with parallel calculation.
        """
        if atc_signals_df.empty:
            return pd.DataFrame()

        osc_params = self.get_oscillator_params()
        spc_params = self.get_spc_params() if self.args.enable_spc else None
        total = len(atc_signals_df)
        
        log_progress(
            f"Calculating signals from all indicators for {total} {signal_type} symbols "
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
        processed_count = [0]

        results = []
        
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
                ): symbol_data["symbol"]
                for symbol_data in symbol_data_list
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result is not None:
                        with progress_lock:
                            processed_count[0] += 1
                            results.append(result)
                except Exception as e:
                    pass
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

        return pd.DataFrame(results)
    
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
            original_mode = self.spc_aggregator.config.mode
            self.spc_aggregator.config.mode = "threshold"
            try:
                vote, strength, _ = self.spc_aggregator.aggregate(symbol_data, signal_type)
            finally:
                self.spc_aggregator.config.mode = original_mode
            
            # If threshold mode also gives no vote, try simple mode fallback
            if vote == 0 and self.spc_aggregator.config.enable_simple_fallback:
                original_mode = self.spc_aggregator.config.mode
                self.spc_aggregator.config.mode = "simple"
                try:
                    vote, strength, _ = self.spc_aggregator.aggregate(symbol_data, signal_type)
                finally:
                    self.spc_aggregator.config.mode = original_mode
        else:
            # Try weighted mode first
            vote, strength, _ = self.spc_aggregator.aggregate(symbol_data, signal_type)
            
            # If weighted mode gives no vote (vote = 0), fallback to threshold mode
            if vote == 0:
                original_mode = self.spc_aggregator.config.mode
                self.spc_aggregator.config.mode = "threshold"
                try:
                    vote, strength, _ = self.spc_aggregator.aggregate(symbol_data, signal_type)
                finally:
                    self.spc_aggregator.config.mode = original_mode
                
                # If threshold mode also gives no vote, try simple mode fallback
                if vote == 0 and self.spc_aggregator.config.enable_simple_fallback:
                    original_mode = self.spc_aggregator.config.mode
                    self.spc_aggregator.config.mode = "simple"
                    try:
                        vote, strength, _ = self.spc_aggregator.aggregate(symbol_data, signal_type)
                    finally:
                        self.spc_aggregator.config.mode = original_mode
        
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
    ) -> pd.DataFrame:
        """
        Apply pure voting system to all signals.
        
        This is the core of Phương án 2 - no sequential filtering,
        just calculate all signals and vote.
        """
        if signals_df.empty:
            return pd.DataFrame()
        
        indicators = ['atc', 'oscillator']
        if self.args.enable_spc:
            indicators.append('spc')
        if hasattr(self.args, 'enable_xgboost') and self.args.enable_xgboost:
            indicators.append('xgboost')
        
        results = []
        
        for _, row in signals_df.iterrows():
            classifier = DecisionMatrixClassifier(indicators=indicators)
            
            # Get votes from all indicators
            atc_vote = row.get('atc_vote', 0)
            atc_strength = row.get('atc_strength', 0.0)
            classifier.add_node_vote('atc', atc_vote, atc_strength, 
                self._get_indicator_accuracy('atc', signal_type))
            
            osc_vote = row.get('osc_vote', 0)
            osc_strength = row.get('osc_confidence', 0.0)
            classifier.add_node_vote('oscillator', osc_vote, osc_strength,
                self._get_indicator_accuracy('oscillator', signal_type))
            
            if self.args.enable_spc:
                # Aggregate 3 SPC votes into 1
                spc_vote, spc_strength = self._aggregate_spc_votes(row.to_dict(), signal_type)
                classifier.add_node_vote('spc', spc_vote, spc_strength,
                    self._get_indicator_accuracy('spc', signal_type))
            
            if hasattr(self.args, 'enable_xgboost') and self.args.enable_xgboost:
                # XGBoost vote
                xgb_vote = row.get('xgboost_vote', 0)
                xgb_strength = row.get('xgboost_confidence', 0.0)
                classifier.add_node_vote('xgboost', xgb_vote, xgb_strength,
                    self._get_indicator_accuracy('xgboost', signal_type))
            
            classifier.calculate_weighted_impact()
            
            cumulative_vote, weighted_score, voting_breakdown = classifier.calculate_cumulative_vote(
                threshold=self.args.voting_threshold,
                min_votes=self.args.min_votes,
            )
            
            # Only keep if cumulative vote is positive
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
                elif votes_count >= self.args.min_votes:
                    result['source'] = 'MAJORITY_VOTE'
                else:
                    result['source'] = 'WEIGHTED_VOTE'
                
                results.append(result)
        
        if not results:
            return pd.DataFrame()
        
        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values('weighted_score', ascending=False).reset_index(drop=True)
        
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
        
        self.run_atc_scan()
        self.calculate_and_vote()
        self.display_results()
        
        log_success("\nAnalysis complete!")


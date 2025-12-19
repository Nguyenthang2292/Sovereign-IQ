"""
Hybrid Signal Calculator for Position Sizing.

This module combines signals from multiple indicators (Range Oscillator, SPC,
XGBoost, HMM, Random Forest) using majority vote or weighted voting approach.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from modules.common.core.data_fetcher import DataFetcher
from core.signal_calculators import (
    get_range_oscillator_signal,
    get_spc_signal,
    get_xgboost_signal,
    get_hmm_signal,
    get_random_forest_signal,
)
from modules.common.utils import (
    log_error,
    log_warn,
    log_progress,
)
from config.position_sizing import (
    ENABLE_MULTITHREADING,
)


class HybridSignalCalculator:
    """
    Calculates hybrid signals by combining multiple indicators.
    
    Uses rolling window approach: for each period, calculates signals using
    only historical data up to that period (walk-forward testing).
    """
    
    def __init__(
        self,
        data_fetcher: DataFetcher,
        enabled_indicators: Optional[List[str]] = None,
        use_confidence_weighting: bool = True,
        min_indicators_agreement: int = 3,
    ):
        """
        Initialize Hybrid Signal Calculator.
        
        Args:
            data_fetcher: DataFetcher instance for fetching OHLCV data
            enabled_indicators: List of enabled indicators (default: all)
                Options: 'range_oscillator', 'spc', 'xgboost', 'hmm', 'random_forest'
            use_confidence_weighting: Whether to weight votes by confidence scores
            min_indicators_agreement: Minimum number of indicators that must agree
        """
        self.data_fetcher = data_fetcher
        self.use_confidence_weighting = use_confidence_weighting
        self.min_indicators_agreement = min_indicators_agreement
        
        # Default: enable all indicators
        if enabled_indicators is None:
            enabled_indicators = [
                'range_oscillator',
                'spc',
                'xgboost',
                'hmm',
                'random_forest',
            ]
        
        self.enabled_indicators = enabled_indicators
        
        # Cache for signal calculations (to avoid recalculating for nearby periods)
        # Key: (symbol, period_index, signal_type), Value: (signal, confidence)
        self._signal_cache: Dict[Tuple[str, int, str], Tuple[int, float]] = {}
        self._cache_max_size = 200  # Limit cache size
        
        # Cache for data fetching (to avoid redundant fetches)
        # Key: (symbol, limit, timeframe), Value: DataFrame
        self._data_cache: Dict[Tuple[str, int, str], pd.DataFrame] = {}
        self._data_cache_max_size = 10  # Limit data cache size
        
        # Cache for intermediate indicator results
        # Key: (symbol, period_index, indicator_name), Value: Dict with signal and confidence
        self._indicator_cache: Dict[Tuple[str, int, str], Dict] = {}
        self._indicator_cache_max_size = 500  # Larger cache for indicators
    
    def calculate_hybrid_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        period_index: int,
        signal_type: str,
        osc_length: int = 50,
        osc_mult: float = 2.0,
        osc_strategies: Optional[List[int]] = None,
        spc_params: Optional[Dict] = None,
    ) -> Tuple[int, float]:
        """
        Calculate hybrid signal for a specific period using rolling window.
        
        Args:
            df: Full DataFrame with OHLCV data
            symbol: Trading pair symbol
            timeframe: Timeframe string
            period_index: Index of the period to calculate signal for (0-based)
            signal_type: "LONG" or "SHORT"
            osc_length: Range Oscillator length parameter
            osc_mult: Range Oscillator multiplier parameter
            osc_strategies: Range Oscillator strategies to use
            spc_params: SPC parameters dictionary
            
        Returns:
            Tuple of (signal, confidence) where:
            - signal: 1 (LONG), -1 (SHORT), or 0 (HOLD/NEUTRAL)
            - confidence: Combined confidence score (0.0 to 1.0)
        """
        try:
            # Check cache first
            cache_key = (symbol, period_index, signal_type.upper())
            if cache_key in self._signal_cache:
                return self._signal_cache[cache_key]
            
            # Use only data up to current period (rolling window)
            if period_index >= len(df):
                return (0, 0.0)
            
            # Get historical data up to current period (inclusive)
            historical_df = df.iloc[:period_index + 1].copy()
            
            # Need at least some data to calculate signals
            if len(historical_df) < 10:
                result = (0, 0.0)
                self._cache_result(cache_key, result)
                return result
            
            # Calculate limit (number of candles to use)
            limit = min(len(historical_df), 1500)  # API limit
            
            # Calculate signals from all enabled indicators
            if ENABLE_MULTITHREADING:
                indicator_signals = self._calculate_indicators_parallel(
                    symbol, timeframe, limit, osc_length, osc_mult, osc_strategies, spc_params, len(historical_df), period_index
                )
            else:
                indicator_signals = self._calculate_indicators_sequential(
                    symbol, timeframe, limit, osc_length, osc_mult, osc_strategies, spc_params, len(historical_df), period_index
                )
            
            # Combine signals using majority vote
            if not indicator_signals:
                return (0, 0.0)
            
            combined_signal, combined_confidence = self.combine_signals_majority_vote(
                indicator_signals,
                expected_signal_type=signal_type,
            )
            
            result = (combined_signal, combined_confidence)
            self._cache_result(cache_key, result)
            return result
            
        except Exception as e:
            log_error(f"Error calculating hybrid signal: {e}")
            log_error(f"Traceback: {traceback.format_exc()}")
            result = (0, 0.0)
            self._cache_result(cache_key, result)
            return result
    
    def _cache_result(self, cache_key: Tuple[str, int, str], result: Tuple[int, float]):
        """Cache a result, removing oldest entries if cache is full."""
        if len(self._signal_cache) >= self._cache_max_size:
            # Remove oldest entry (FIFO - simple approach)
            oldest_key = next(iter(self._signal_cache))
            del self._signal_cache[oldest_key]
        self._signal_cache[cache_key] = result
    
    def combine_signals_majority_vote(
        self,
        indicator_signals: List[Dict],
        expected_signal_type: str = "LONG",
    ) -> Tuple[int, float]:
        """
        Combine signals from multiple indicators using majority vote.
        
        Args:
            indicator_signals: List of dicts with 'indicator', 'signal', 'confidence'
            expected_signal_type: "LONG" or "SHORT" - the expected signal direction
            
        Returns:
            Tuple of (combined_signal, combined_confidence)
        """
        if not indicator_signals:
            return (0, 0.0)
        
        # Convert expected signal type to int
        expected_signal = 1 if expected_signal_type.upper() == "LONG" else -1
        
        # Count votes for each signal direction
        long_votes = 0
        short_votes = 0
        neutral_votes = 0
        
        long_confidence_sum = 0.0
        short_confidence_sum = 0.0
        neutral_confidence_sum = 0.0
        
        for indicator in indicator_signals:
            signal = indicator['signal']
            confidence = indicator.get('confidence', 0.5)
            
            if signal == 1:  # LONG
                long_votes += 1
                if self.use_confidence_weighting:
                    long_confidence_sum += confidence
                else:
                    long_confidence_sum += 1.0
            elif signal == -1:  # SHORT
                short_votes += 1
                if self.use_confidence_weighting:
                    short_confidence_sum += confidence
                else:
                    short_confidence_sum += 1.0
            else:  # NEUTRAL/HOLD (0)
                neutral_votes += 1
                if self.use_confidence_weighting:
                    neutral_confidence_sum += confidence
                else:
                    neutral_confidence_sum += 1.0
        
        # Determine majority signal
        total_votes = len(indicator_signals)
        
        # Check if we have minimum agreement
        max_votes = max(long_votes, short_votes, neutral_votes)
        if max_votes < self.min_indicators_agreement:
            # Not enough agreement, return neutral
            return (0, 0.0)
        
        # Majority vote
        if long_votes > short_votes and long_votes > neutral_votes:
            combined_signal = 1
            combined_confidence = long_confidence_sum / max(long_votes, 1) if self.use_confidence_weighting else long_votes / total_votes
        elif short_votes > long_votes and short_votes > neutral_votes:
            combined_signal = -1
            combined_confidence = short_confidence_sum / max(short_votes, 1) if self.use_confidence_weighting else short_votes / total_votes
        else:
            # Neutral wins or tie
            combined_signal = 0
            combined_confidence = neutral_confidence_sum / max(neutral_votes, 1) if self.use_confidence_weighting else neutral_votes / total_votes
        
        # Filter by expected signal type: only return signal if it matches expected direction
        if expected_signal == 1 and combined_signal != 1:
            return (0, 0.0)  # Expected LONG but got something else
        elif expected_signal == -1 and combined_signal != -1:
            return (0, 0.0)  # Expected SHORT but got something else
        
        return (combined_signal, min(combined_confidence, 1.0))
    
    def _calculate_indicators_parallel(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        osc_length: int,
        osc_mult: float,
        osc_strategies: Optional[List[int]],
        spc_params: Optional[Dict],
        historical_df_len: int,
        period_index: int,
    ) -> List[Dict]:
        """Calculate indicators in parallel using ThreadPoolExecutor."""
        indicator_signals = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {}
            
            # 1. Range Oscillator
            if 'range_oscillator' in self.enabled_indicators:
                futures['range_oscillator'] = executor.submit(
                    self._calc_range_oscillator,
                    symbol, timeframe, limit, osc_length, osc_mult, osc_strategies, period_index
                )
            
            # 2. SPC (3 strategies)
            if 'spc' in self.enabled_indicators:
                spc_strategies = ['cluster_transition', 'regime_following', 'mean_reversion']
                for strategy in spc_strategies:
                    futures[f'spc_{strategy}'] = executor.submit(
                        self._calc_spc,
                        symbol, timeframe, limit, strategy, spc_params, period_index
                    )
            
            # 3. XGBoost
            if 'xgboost' in self.enabled_indicators and historical_df_len >= 50:
                futures['xgboost'] = executor.submit(
                    self._calc_xgboost,
                    symbol, timeframe, limit, period_index
                )
            
            # 4. HMM
            if 'hmm' in self.enabled_indicators:
                futures['hmm'] = executor.submit(
                    self._calc_hmm,
                    symbol, timeframe, limit, period_index
                )
            
            # 5. Random Forest
            if 'random_forest' in self.enabled_indicators:
                futures['random_forest'] = executor.submit(
                    self._calc_random_forest,
                    symbol, timeframe, limit, period_index
                )
            
            # Collect results
            for indicator_name, future in futures.items():
                try:
                    result = future.result(timeout=60)  # 60 second timeout per indicator
                    if result:
                        indicator_signals.append(result)
                except Exception as e:
                    log_warn(f"{indicator_name} signal calculation failed: {e}")
        
        return indicator_signals
    
    def _calculate_indicators_sequential(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        osc_length: int,
        osc_mult: float,
        osc_strategies: Optional[List[int]],
        spc_params: Optional[Dict],
        historical_df_len: int,
        period_index: int,
    ) -> List[Dict]:
        """Calculate indicators sequentially (original implementation)."""
        indicator_signals = []
        
        # 1. Range Oscillator
        if 'range_oscillator' in self.enabled_indicators:
            result = self._calc_range_oscillator(
                symbol, timeframe, limit, osc_length, osc_mult, osc_strategies, period_index
            )
            if result:
                indicator_signals.append(result)
        
        # 2. SPC (Simplified Percentile Clustering) - 3 strategies
        if 'spc' in self.enabled_indicators:
            spc_strategies = ['cluster_transition', 'regime_following', 'mean_reversion']
            for strategy in spc_strategies:
                result = self._calc_spc(
                    symbol, timeframe, limit, strategy, spc_params, period_index
                )
                if result:
                    indicator_signals.append(result)
        
        # 3. XGBoost
        if 'xgboost' in self.enabled_indicators and historical_df_len >= 50:
            result = self._calc_xgboost(
                symbol, timeframe, limit, period_index
            )
            if result:
                indicator_signals.append(result)
        
        # 4. HMM
        if 'hmm' in self.enabled_indicators:
            result = self._calc_hmm(
                symbol, timeframe, limit, period_index
            )
            if result:
                indicator_signals.append(result)
        
        # 5. Random Forest
        if 'random_forest' in self.enabled_indicators:
            result = self._calc_random_forest(
                symbol, timeframe, limit, period_index
            )
            if result:
                indicator_signals.append(result)
        
        return indicator_signals
    
    def _calc_range_oscillator(
        self, symbol: str, timeframe: str, limit: int,
        osc_length: int, osc_mult: float, osc_strategies: Optional[List[int]],
        period_index: Optional[int] = None
    ) -> Optional[Dict]:
        """Calculate Range Oscillator signal with caching."""
        # Check cache first
        if period_index is not None:
            cache_key = (symbol, period_index, 'range_oscillator')
            if cache_key in self._indicator_cache:
                return self._indicator_cache[cache_key]
        
        try:
            osc_result = get_range_oscillator_signal(
                data_fetcher=self.data_fetcher,
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                osc_length=osc_length,
                osc_mult=osc_mult,
                strategies=osc_strategies,
            )
            if osc_result is not None:
                osc_signal, osc_confidence = osc_result
                result = {
                    'indicator': 'range_oscillator',
                    'signal': osc_signal,
                    'confidence': osc_confidence,
                }
                # Cache result
                if period_index is not None:
                    self._cache_indicator_result(cache_key, result)
                return result
        except Exception as e:
            log_warn(f"Range Oscillator signal calculation failed: {e}")
        return None
    
    def _calc_spc(
        self, symbol: str, timeframe: str, limit: int,
        strategy: str, spc_params: Optional[Dict],
        period_index: Optional[int] = None
    ) -> Optional[Dict]:
        """Calculate SPC signal for a specific strategy with caching."""
        # Check cache first
        if period_index is not None:
            cache_key = (symbol, period_index, f'spc_{strategy}')
            if cache_key in self._indicator_cache:
                return self._indicator_cache[cache_key]
        
        try:
            spc_result = get_spc_signal(
                data_fetcher=self.data_fetcher,
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                strategy=strategy,
                strategy_params=spc_params.get(strategy, {}) if spc_params else None,
            )
            if spc_result is not None:
                spc_signal, spc_confidence = spc_result
                result = {
                    'indicator': f'spc_{strategy}',
                    'signal': spc_signal,
                    'confidence': spc_confidence,
                }
                # Cache result
                if period_index is not None:
                    self._cache_indicator_result(cache_key, result)
                return result
        except Exception as e:
            log_warn(f"SPC {strategy} signal calculation failed: {e}")
        return None
    
    def _calc_xgboost(
        self, symbol: str, timeframe: str, limit: int,
        period_index: Optional[int] = None
    ) -> Optional[Dict]:
        """Calculate XGBoost signal with caching."""
        # Check cache first
        if period_index is not None:
            cache_key = (symbol, period_index, 'xgboost')
            if cache_key in self._indicator_cache:
                return self._indicator_cache[cache_key]
        
        try:
            xgb_result = get_xgboost_signal(
                data_fetcher=self.data_fetcher,
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
            )
            if xgb_result is not None:
                xgb_signal, xgb_confidence = xgb_result
                result = {
                    'indicator': 'xgboost',
                    'signal': xgb_signal,
                    'confidence': xgb_confidence,
                }
                # Cache result
                if period_index is not None:
                    self._cache_indicator_result(cache_key, result)
                return result
        except Exception as e:
            log_warn(f"XGBoost signal calculation failed: {e}")
        return None
    
    def _calc_hmm(
        self, symbol: str, timeframe: str, limit: int,
        period_index: Optional[int] = None
    ) -> Optional[Dict]:
        """Calculate HMM signal with caching."""
        # Check cache first
        if period_index is not None:
            cache_key = (symbol, period_index, 'hmm')
            if cache_key in self._indicator_cache:
                return self._indicator_cache[cache_key]
        
        try:
            hmm_result = get_hmm_signal(
                data_fetcher=self.data_fetcher,
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
            )
            if hmm_result is not None:
                hmm_signal, hmm_confidence = hmm_result
                result = {
                    'indicator': 'hmm',
                    'signal': hmm_signal,
                    'confidence': hmm_confidence,
                }
                # Cache result
                if period_index is not None:
                    self._cache_indicator_result(cache_key, result)
                return result
        except Exception as e:
            log_warn(f"HMM signal calculation failed: {e}")
        return None
    
    def _calc_random_forest(
        self, symbol: str, timeframe: str, limit: int,
        period_index: Optional[int] = None
    ) -> Optional[Dict]:
        """Calculate Random Forest signal with caching."""
        # Check cache first
        if period_index is not None:
            cache_key = (symbol, period_index, 'random_forest')
            if cache_key in self._indicator_cache:
                return self._indicator_cache[cache_key]
        
        try:
            rf_result = get_random_forest_signal(
                data_fetcher=self.data_fetcher,
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
            )
            if rf_result is not None:
                rf_signal, rf_confidence = rf_result
                result = {
                    'indicator': 'random_forest',
                    'signal': rf_signal,
                    'confidence': rf_confidence,
                }
                # Cache result
                if period_index is not None:
                    self._cache_indicator_result(cache_key, result)
                return result
        except Exception as e:
            log_warn(f"Random Forest signal calculation failed: {e}")
        return None
    
    def _cache_indicator_result(self, cache_key: Tuple[str, int, str], result: Dict):
        """Cache an indicator result, removing oldest entries if cache is full."""
        if len(self._indicator_cache) >= self._indicator_cache_max_size:
            # Remove oldest entry (FIFO - simple approach)
            oldest_key = next(iter(self._indicator_cache))
            del self._indicator_cache[oldest_key]
        self._indicator_cache[cache_key] = result
    
    def clear_cache(self):
        """Clear all caches."""
        self._signal_cache.clear()
        self._data_cache.clear()
        self._indicator_cache.clear()
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics for monitoring."""
        return {
            'signal_cache_size': len(self._signal_cache),
            'signal_cache_max_size': self._cache_max_size,
            'data_cache_size': len(self._data_cache),
            'data_cache_max_size': self._data_cache_max_size,
        }


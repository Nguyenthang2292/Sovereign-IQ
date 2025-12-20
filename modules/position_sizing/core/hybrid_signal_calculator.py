"""
Hybrid Signal Calculator for Position Sizing.

This module combines signals from multiple indicators (Range Oscillator, SPC,
XGBoost, HMM, Random Forest) using majority vote or weighted voting approach.
"""

from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
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
    SIGNAL_CACHE_MAX_SIZE,
    INDICATOR_CACHE_MAX_SIZE,
    DATA_CACHE_MAX_SIZE,
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
        # Using OrderedDict for LRU cache implementation
        self._signal_cache: OrderedDict[Tuple[str, int, str], Tuple[int, float]] = OrderedDict()
        self._cache_max_size = SIGNAL_CACHE_MAX_SIZE
        
        # Cache for data fetching (to avoid redundant fetches)
        # Key: (symbol, limit, timeframe), Value: DataFrame
        # Using OrderedDict for LRU cache implementation
        self._data_cache: OrderedDict[Tuple[str, int, str], pd.DataFrame] = OrderedDict()
        self._data_cache_max_size = DATA_CACHE_MAX_SIZE
        
        # Cache for intermediate indicator results
        # Key: (symbol, period_index, indicator_name), Value: Dict with signal and confidence
        # Using OrderedDict for LRU cache implementation
        self._indicator_cache: OrderedDict[Tuple[str, int, str], Dict] = OrderedDict()
        self._indicator_cache_max_size = INDICATOR_CACHE_MAX_SIZE
        
        # Cache statistics for monitoring
        self._cache_hits = 0
        self._cache_misses = 0
    
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
            # Check cache first (LRU: move to end if found)
            cache_key = (symbol, period_index, signal_type.upper())
            if cache_key in self._signal_cache:
                # Move to end (most recently used)
                self._signal_cache.move_to_end(cache_key)
                self._cache_hits += 1
                return self._signal_cache[cache_key]
            self._cache_misses += 1
            
            # Use only data up to current period (rolling window)
            if period_index >= len(df):
                return (0, 0.0)
            
            # Get historical data up to current period (inclusive)
            # Use view instead of copy for better performance (only copy when needed)
            historical_df = df.iloc[:period_index + 1]
            
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
                    symbol, timeframe, limit, osc_length, osc_mult, osc_strategies, spc_params, len(historical_df), period_index, historical_df
                )
            else:
                indicator_signals = self._calculate_indicators_sequential(
                    symbol, timeframe, limit, osc_length, osc_mult, osc_strategies, spc_params, len(historical_df), period_index, historical_df
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
        """Cache a result using LRU strategy, removing oldest entries if cache is full."""
        if cache_key in self._signal_cache:
            # Update existing entry (move to end)
            self._signal_cache.move_to_end(cache_key)
        else:
            # Add new entry
            if len(self._signal_cache) >= self._cache_max_size:
                # Remove oldest entry (LRU - first item)
                self._signal_cache.popitem(last=False)
            self._signal_cache[cache_key] = result
        # Move to end to mark as most recently used
        self._signal_cache.move_to_end(cache_key)
    
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
    
    def calculate_single_signal_highest_confidence(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        period_index: int,
        osc_length: int = 50,
        osc_mult: float = 2.0,
        osc_strategies: Optional[List[int]] = None,
        spc_params: Optional[Dict] = None,
    ) -> Tuple[int, float]:
        """
        Calculate signal by selecting the indicator with highest confidence.
        
        Unlike calculate_hybrid_signal(), this method:
        - Does NOT require majority vote
        - Does NOT filter by expected signal_type
        - Simply selects the signal with highest confidence from all indicators
        - If multiple signals have same confidence, prefers LONG over SHORT
        
        Args:
            df: Full DataFrame with OHLCV data
            symbol: Trading pair symbol
            timeframe: Timeframe string
            period_index: Index of the period to calculate signal for (0-based)
            osc_length: Range Oscillator length parameter
            osc_mult: Range Oscillator multiplier parameter
            osc_strategies: Range Oscillator strategies to use
            spc_params: SPC parameters dictionary
            
        Returns:
            Tuple of (signal, confidence) where:
            - signal: 1 (LONG), -1 (SHORT), or 0 (HOLD/NEUTRAL)
            - confidence: Confidence score of selected signal (0.0 to 1.0)
        """
        try:
            # Use only data up to current period (rolling window)
            if period_index >= len(df):
                return (0, 0.0)
            
            # Get historical data up to current period (inclusive)
            historical_df = df.iloc[:period_index + 1]
            
            # Need at least some data to calculate signals
            if len(historical_df) < 10:
                return (0, 0.0)
            
            # Calculate limit (number of candles to use)
            limit = min(len(historical_df), 1500)  # API limit
            
            # Calculate signals from all enabled indicators
            if ENABLE_MULTITHREADING:
                indicator_signals = self._calculate_indicators_parallel(
                    symbol, timeframe, limit, osc_length, osc_mult, osc_strategies, spc_params, len(historical_df), period_index, historical_df
                )
            else:
                indicator_signals = self._calculate_indicators_sequential(
                    symbol, timeframe, limit, osc_length, osc_mult, osc_strategies, spc_params, len(historical_df), period_index, historical_df
                )
            
            # Filter out neutral signals (keep only LONG=1 or SHORT=-1)
            non_zero_signals = [
                sig for sig in indicator_signals
                if sig.get('signal', 0) != 0
            ]
            
            if not non_zero_signals:
                return (0, 0.0)
            
            # Select signal with highest confidence
            # If tie, prefer LONG (1) over SHORT (-1)
            best_signal = None
            best_confidence = -1.0
            
            for sig in non_zero_signals:
                signal_val = sig.get('signal', 0)
                confidence_val = sig.get('confidence', 0.0)
                
                if confidence_val > best_confidence:
                    best_signal = signal_val
                    best_confidence = confidence_val
                elif confidence_val == best_confidence and signal_val == 1 and best_signal == -1:
                    # Tie in confidence, prefer LONG over SHORT
                    best_signal = signal_val
                    best_confidence = confidence_val
            
            if best_signal is None:
                return (0, 0.0)
            
            return (best_signal, best_confidence)
            
        except Exception as e:
            log_error(f"Error calculating single signal (highest confidence): {e}")
            log_error(f"Traceback: {traceback.format_exc()}")
            return (0, 0.0)
    
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
        df: pd.DataFrame,
    ) -> List[Dict]:
        """Calculate indicators in parallel using ThreadPoolExecutor."""
        indicator_signals = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {}
            
            # 1. Range Oscillator
            if 'range_oscillator' in self.enabled_indicators:
                futures['range_oscillator'] = executor.submit(
                    self._calc_range_oscillator,
                    symbol, timeframe, limit, osc_length, osc_mult, osc_strategies, period_index, df
                )
            
            # 2. SPC (3 strategies)
            if 'spc' in self.enabled_indicators:
                spc_strategies = ['cluster_transition', 'regime_following', 'mean_reversion']
                for strategy in spc_strategies:
                    futures[f'spc_{strategy}'] = executor.submit(
                        self._calc_spc,
                        symbol, timeframe, limit, strategy, spc_params, period_index, df
                    )
            
            # 3. XGBoost
            if 'xgboost' in self.enabled_indicators and historical_df_len >= 50:
                futures['xgboost'] = executor.submit(
                    self._calc_xgboost,
                    symbol, timeframe, limit, period_index, df
                )
            
            # 4. HMM
            if 'hmm' in self.enabled_indicators:
                futures['hmm'] = executor.submit(
                    self._calc_hmm,
                    symbol, timeframe, limit, period_index, df
                )
            
            # 5. Random Forest
            if 'random_forest' in self.enabled_indicators:
                futures['random_forest'] = executor.submit(
                    self._calc_random_forest,
                    symbol, timeframe, limit, period_index, df
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
        df: pd.DataFrame,
    ) -> List[Dict]:
        """Calculate indicators sequentially (original implementation)."""
        indicator_signals = []
        
        # 1. Range Oscillator
        if 'range_oscillator' in self.enabled_indicators:
            result = self._calc_range_oscillator(
                symbol, timeframe, limit, osc_length, osc_mult, osc_strategies, period_index, df
            )
            if result:
                indicator_signals.append(result)
        
        # 2. SPC (Simplified Percentile Clustering) - 3 strategies
        if 'spc' in self.enabled_indicators:
            spc_strategies = ['cluster_transition', 'regime_following', 'mean_reversion']
            for strategy in spc_strategies:
                result = self._calc_spc(
                    symbol, timeframe, limit, strategy, spc_params, period_index, df
                )
                if result:
                    indicator_signals.append(result)
        
        # 3. XGBoost
        if 'xgboost' in self.enabled_indicators and historical_df_len >= 50:
            result = self._calc_xgboost(
                symbol, timeframe, limit, period_index, df
            )
            if result:
                indicator_signals.append(result)
        
        # 4. HMM
        if 'hmm' in self.enabled_indicators:
            result = self._calc_hmm(
                symbol, timeframe, limit, period_index, df
            )
            if result:
                indicator_signals.append(result)
        
        # 5. Random Forest
        if 'random_forest' in self.enabled_indicators:
            result = self._calc_random_forest(
                symbol, timeframe, limit, period_index, df
            )
            if result:
                indicator_signals.append(result)
        
        return indicator_signals
    
    def _calc_range_oscillator(
        self, symbol: str, timeframe: str, limit: int,
        osc_length: int, osc_mult: float, osc_strategies: Optional[List[int]],
        period_index: Optional[int] = None,
        df: Optional[pd.DataFrame] = None
    ) -> Optional[Dict]:
        """Calculate Range Oscillator signal with caching."""
        # Check cache first (LRU: move to end if found)
        if period_index is not None:
            cache_key = (symbol, period_index, 'range_oscillator')
            if cache_key in self._indicator_cache:
                # Move to end (most recently used)
                self._indicator_cache.move_to_end(cache_key)
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
                df=df,
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
        period_index: Optional[int] = None,
        df: Optional[pd.DataFrame] = None
    ) -> Optional[Dict]:
        """Calculate SPC signal for a specific strategy with caching."""
        # Check cache first (LRU: move to end if found)
        if period_index is not None:
            cache_key = (symbol, period_index, f'spc_{strategy}')
            if cache_key in self._indicator_cache:
                # Move to end (most recently used)
                self._indicator_cache.move_to_end(cache_key)
                return self._indicator_cache[cache_key]
        
        try:
            spc_result = get_spc_signal(
                data_fetcher=self.data_fetcher,
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                strategy=strategy,
                strategy_params=spc_params.get(strategy, {}) if spc_params else None,
                df=df,
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
        period_index: Optional[int] = None,
        df: Optional[pd.DataFrame] = None
    ) -> Optional[Dict]:
        """Calculate XGBoost signal with caching."""
        # Check cache first (LRU: move to end if found)
        if period_index is not None:
            cache_key = (symbol, period_index, 'xgboost')
            if cache_key in self._indicator_cache:
                # Move to end (most recently used)
                self._indicator_cache.move_to_end(cache_key)
                return self._indicator_cache[cache_key]
        
        try:
            xgb_result = get_xgboost_signal(
                data_fetcher=self.data_fetcher,
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                df=df,
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
        period_index: Optional[int] = None,
        df: Optional[pd.DataFrame] = None
    ) -> Optional[Dict]:
        """Calculate HMM signal with caching."""
        # Check cache first (LRU: move to end if found)
        if period_index is not None:
            cache_key = (symbol, period_index, 'hmm')
            if cache_key in self._indicator_cache:
                # Move to end (most recently used)
                self._indicator_cache.move_to_end(cache_key)
                return self._indicator_cache[cache_key]
        
        try:
            hmm_result = get_hmm_signal(
                data_fetcher=self.data_fetcher,
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                df=df,
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
        period_index: Optional[int] = None,
        df: Optional[pd.DataFrame] = None
    ) -> Optional[Dict]:
        """Calculate Random Forest signal with caching."""
        # Check cache first (LRU: move to end if found)
        if period_index is not None:
            cache_key = (symbol, period_index, 'random_forest')
            if cache_key in self._indicator_cache:
                # Move to end (most recently used)
                self._indicator_cache.move_to_end(cache_key)
                return self._indicator_cache[cache_key]
        
        try:
            rf_result = get_random_forest_signal(
                data_fetcher=self.data_fetcher,
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                df=df,
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
        """Cache an indicator result using LRU strategy, removing oldest entries if cache is full."""
        if cache_key in self._indicator_cache:
            # Update existing entry (move to end)
            self._indicator_cache.move_to_end(cache_key)
        else:
            # Add new entry
            if len(self._indicator_cache) >= self._indicator_cache_max_size:
                # Remove oldest entry (LRU - first item)
                self._indicator_cache.popitem(last=False)
            self._indicator_cache[cache_key] = result
        # Move to end to mark as most recently used
        self._indicator_cache.move_to_end(cache_key)
    
    def clear_cache(self):
        """Clear all caches."""
        self._signal_cache.clear()
        self._data_cache.clear()
        self._indicator_cache.clear()
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics for monitoring."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
        return {
            'signal_cache_size': len(self._signal_cache),
            'signal_cache_max_size': self._cache_max_size,
            'data_cache_size': len(self._data_cache),
            'data_cache_max_size': self._data_cache_max_size,
            'indicator_cache_size': len(self._indicator_cache),
            'indicator_cache_max_size': self._indicator_cache_max_size,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_hit_rate': hit_rate,
        }
    
    def precompute_all_indicators_vectorized(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        osc_length: int,
        osc_mult: float,
        osc_strategies: Optional[List[int]],
        spc_params: Optional[Dict],
    ) -> Dict[str, pd.DataFrame]:
        """
        Pre-compute all indicators for entire DataFrame using vectorization.
        
        This method calculates indicators once for the entire DataFrame, which is
        much faster than calculating them for each period separately. The results
        are stored in a dictionary where each key is an indicator name and each
        value is a DataFrame with 'signal' and 'confidence' columns for all periods.
        
        Args:
            df: Full DataFrame with OHLCV data
            symbol: Trading pair symbol
            timeframe: Timeframe string
            osc_length: Range Oscillator length parameter
            osc_mult: Range Oscillator multiplier parameter
            osc_strategies: Range Oscillator strategies to use
            spc_params: SPC parameters dictionary
            
        Returns:
            Dict mapping indicator_name -> DataFrame with columns ['signal', 'confidence']
            Each DataFrame has the same index as df, with NaN for periods where
            the indicator cannot be calculated (e.g., insufficient data).
        """
        indicators = {}
        
        # Range Oscillator - vectorized
        if 'range_oscillator' in self.enabled_indicators:
            try:
                indicators['range_oscillator'] = self._calc_range_oscillator_vectorized(
                    df, osc_length, osc_mult, osc_strategies
                )
            except Exception as e:
                log_warn(f"Range Oscillator vectorized calculation failed: {e}")
                indicators['range_oscillator'] = pd.DataFrame({
                    'signal': [0] * len(df),
                    'confidence': [0.0] * len(df)
                }, index=df.index)
        
        # SPC - vectorized where possible (3 strategies)
        if 'spc' in self.enabled_indicators:
            spc_strategies = ['cluster_transition', 'regime_following', 'mean_reversion']
            for strategy in spc_strategies:
                try:
                    indicators[f'spc_{strategy}'] = self._calc_spc_vectorized(
                        df, strategy, spc_params
                    )
                except Exception as e:
                    log_warn(f"SPC {strategy} vectorized calculation failed: {e}")
                    indicators[f'spc_{strategy}'] = pd.DataFrame({
                        'signal': [0] * len(df),
                        'confidence': [0.0] * len(df)
                    }, index=df.index)
        
        # ML models - batch processing
        if 'xgboost' in self.enabled_indicators:
            try:
                indicators['xgboost'] = self._calc_xgboost_batch(df, symbol, timeframe)
            except Exception as e:
                log_warn(f"XGBoost batch calculation failed: {e}")
                indicators['xgboost'] = pd.DataFrame({
                    'signal': [0] * len(df),
                    'confidence': [0.0] * len(df)
                }, index=df.index)
        
        if 'hmm' in self.enabled_indicators:
            try:
                indicators['hmm'] = self._calc_hmm_batch(df, symbol, timeframe)
            except Exception as e:
                log_warn(f"HMM batch calculation failed: {e}")
                indicators['hmm'] = pd.DataFrame({
                    'signal': [0] * len(df),
                    'confidence': [0.0] * len(df)
                }, index=df.index)
        
        if 'random_forest' in self.enabled_indicators:
            try:
                indicators['random_forest'] = self._calc_random_forest_batch(df, symbol, timeframe)
            except Exception as e:
                log_warn(f"Random Forest batch calculation failed: {e}")
                indicators['random_forest'] = pd.DataFrame({
                    'signal': [0] * len(df),
                    'confidence': [0.0] * len(df)
                }, index=df.index)
        
        return indicators
    
    def _calc_range_oscillator_vectorized(
        self,
        df: pd.DataFrame,
        osc_length: int,
        osc_mult: float,
        osc_strategies: Optional[List[int]],
    ) -> pd.DataFrame:
        """
        Calculate Range Oscillator signals for entire DataFrame using vectorization.
        
        Returns DataFrame with 'signal' and 'confidence' columns for all periods.
        """
        from modules.range_oscillator.config import CombinedStrategyConfig
        from modules.range_oscillator.strategies.combined import generate_signals_combined_all_strategy
        
        # Validate required columns
        required_columns = ["high", "low", "close"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return pd.DataFrame({
                'signal': [0] * len(df),
                'confidence': [0.0] * len(df)
            }, index=df.index)
        
        high = df["high"].copy()
        low = df["low"].copy()
        close = df["close"].copy()
        
        if osc_strategies is None:
            enabled_strategies = [2, 3, 4, 6, 7, 8, 9]
        else:
            if 5 in osc_strategies:
                enabled_strategies = [2, 3, 4, 6, 7, 8, 9]
            else:
                enabled_strategies = osc_strategies
        
        config = CombinedStrategyConfig()
        config.enabled_strategies = enabled_strategies
        config.return_confidence_score = True
        config.dynamic.enabled = True
        config.dynamic.lookback = 20
        config.dynamic.volatility_threshold = 0.6
        config.dynamic.trend_threshold = 0.5
        config.consensus.mode = "weighted"
        config.consensus.adaptive_weights = False
        config.consensus.performance_window = 10
        
        try:
            result = generate_signals_combined_all_strategy(
                high=high,
                low=low,
                close=close,
                length=osc_length,
                mult=osc_mult,
                config=config,
            )
        except Exception as e:
            # Retry without adaptive weights on classification errors
            error_str = str(e)
            is_value_or_type_error = isinstance(e, (ValueError, TypeError))
            matches_classification = ("Classification metrics" in error_str or 
                                    "Number of classes" in error_str or 
                                    "Invalid classes" in error_str)
            if is_value_or_type_error and matches_classification:
                config.consensus.adaptive_weights = False
                result = generate_signals_combined_all_strategy(
                    high=high,
                    low=low,
                    close=close,
                    length=osc_length,
                    mult=osc_mult,
                    config=config,
                )
            else:
                raise
        
        signals = result[0]  # pd.Series with signals
        confidence = result[3] if len(result) > 3 else None  # pd.Series with confidence
        
        # Create DataFrame with same index as df
        result_df = pd.DataFrame(index=df.index)
        result_df['signal'] = signals.reindex(df.index, fill_value=0)
        if confidence is not None and not confidence.empty:
            result_df['confidence'] = confidence.reindex(df.index, fill_value=0.0)
        else:
            result_df['confidence'] = 0.0
        
        # Fill NaN with 0
        result_df['signal'] = result_df['signal'].fillna(0).astype(int)
        result_df['confidence'] = result_df['confidence'].fillna(0.0).astype(float)
        
        return result_df
    
    def _calc_spc_vectorized(
        self,
        df: pd.DataFrame,
        strategy: str,
        spc_params: Optional[Dict],
    ) -> pd.DataFrame:
        """
        Calculate SPC signals for entire DataFrame using vectorization.
        
        Returns DataFrame with 'signal' and 'confidence' columns for all periods.
        """
        from modules.simplified_percentile_clustering.core.clustering import (
            SimplifiedPercentileClustering,
            ClusteringConfig,
        )
        from modules.simplified_percentile_clustering.core.features import FeatureConfig
        from modules.simplified_percentile_clustering.strategies import (
            generate_signals_cluster_transition,
            generate_signals_regime_following,
            generate_signals_mean_reversion,
        )
        from modules.simplified_percentile_clustering.config import (
            ClusterTransitionConfig,
            RegimeFollowingConfig,
            MeanReversionConfig,
        )
        from config import SPC_P_LOW, SPC_P_HIGH
        
        # Validate required columns
        required_columns = ["high", "low", "close"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return pd.DataFrame({
                'signal': [0] * len(df),
                'confidence': [0.0] * len(df)
            }, index=df.index)
        
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        # Compute clustering once for entire DataFrame
        feature_config = FeatureConfig()
        clustering_config = ClusteringConfig(
            k=2,
            lookback=len(df),
            p_low=SPC_P_LOW,
            p_high=SPC_P_HIGH,
            main_plot="Clusters",
            feature_config=feature_config,
        )
        
        clustering = SimplifiedPercentileClustering(clustering_config)
        clustering_result = clustering.compute(high, low, close)
        
        strategy_params = (spc_params.get(strategy, {}) if spc_params else {}) or {}
        
        if strategy == "cluster_transition":
            strategy_config = ClusterTransitionConfig(
                min_signal_strength=strategy_params.get("min_signal_strength", 0.3),
                min_rel_pos_change=strategy_params.get("min_rel_pos_change", 0.1),
                clustering_config=clustering_config,
            )
            signals, signal_strength, metadata = generate_signals_cluster_transition(
                high=high,
                low=low,
                close=close,
                clustering_result=clustering_result,
                config=strategy_config,
            )
        elif strategy == "regime_following":
            strategy_config = RegimeFollowingConfig(
                min_regime_strength=strategy_params.get("min_regime_strength", 0.7),
                min_cluster_duration=strategy_params.get("min_cluster_duration", 2),
                clustering_config=clustering_config,
            )
            signals, signal_strength, metadata = generate_signals_regime_following(
                high=high,
                low=low,
                close=close,
                clustering_result=clustering_result,
                config=strategy_config,
            )
        elif strategy == "mean_reversion":
            strategy_config = MeanReversionConfig(
                extreme_threshold=strategy_params.get("extreme_threshold", 0.2),
                min_extreme_duration=strategy_params.get("min_extreme_duration", 3),
                clustering_config=clustering_config,
            )
            signals, signal_strength, metadata = generate_signals_mean_reversion(
                high=high,
                low=low,
                close=close,
                clustering_result=clustering_result,
                config=strategy_config,
            )
        else:
            return pd.DataFrame({
                'signal': [0] * len(df),
                'confidence': [0.0] * len(df)
            }, index=df.index)
        
        # Create DataFrame with same index as df
        result_df = pd.DataFrame(index=df.index)
        if signals is not None and not signals.empty:
            result_df['signal'] = signals.reindex(df.index, fill_value=0)
        else:
            result_df['signal'] = 0
        
        if signal_strength is not None and not signal_strength.empty:
            result_df['confidence'] = signal_strength.reindex(df.index, fill_value=0.0)
        else:
            result_df['confidence'] = 0.0
        
        # Fill NaN with 0
        result_df['signal'] = result_df['signal'].fillna(0).astype(int)
        result_df['confidence'] = result_df['confidence'].fillna(0.0).astype(float)
        
        return result_df
    
    def _calc_xgboost_batch(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> pd.DataFrame:
        """
        Calculate XGBoost signals for entire DataFrame using batch processing.
        
        For XGBoost, we still need to process incrementally (walk-forward) but
        we can batch the feature calculation. However, prediction needs to be
        done incrementally to respect walk-forward semantics.
        
        Returns DataFrame with 'signal' and 'confidence' columns for all periods.
        """
        from modules.common.core.indicator_engine import (
            IndicatorConfig,
            IndicatorEngine,
            IndicatorProfile,
        )
        from modules.xgboost.labeling import apply_directional_labels
        from modules.xgboost.model import train_and_predict, predict_next_move
        from config import TARGET_BASE_THRESHOLD, ID_TO_LABEL
        
        # Validate required columns
        required_columns = ["high", "low", "close"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return pd.DataFrame({
                'signal': [0] * len(df),
                'confidence': [0.0] * len(df)
            }, index=df.index)
        
        # Initialize result DataFrame
        result_df = pd.DataFrame({
            'signal': [0] * len(df),
            'confidence': [0.0] * len(df)
        }, index=df.index)
        
        # XGBoost needs at least 50 periods to train
        min_periods = 50
        if len(df) < min_periods:
            return result_df
        
        # Process incrementally (walk-forward) for each period
        # This respects walk-forward semantics: only use data up to current period
        indicator_engine = IndicatorEngine(
            IndicatorConfig.for_profile(IndicatorProfile.XGBOOST)
        )
        
        for i in range(min_periods, len(df)):
            try:
                # Get historical data up to current period (inclusive)
                historical_df = df.iloc[:i + 1].copy()
                
                # Calculate features
                historical_df = indicator_engine.compute_features(historical_df)
                
                # Save latest data before applying labels
                latest_data = historical_df.iloc[-1:].copy()
                latest_data = latest_data.ffill().bfill()
                
                # Apply labels and drop NaN
                historical_df = apply_directional_labels(historical_df)
                latest_threshold = (
                    historical_df["DynamicThreshold"].iloc[-1]
                    if len(historical_df) > 0 and "DynamicThreshold" in historical_df.columns
                    else TARGET_BASE_THRESHOLD
                )
                historical_df.dropna(inplace=True)
                
                if len(historical_df) < min_periods:
                    continue
                
                # Predict next move
                prediction_result = predict_next_move(
                    historical_df, latest_data, latest_threshold
                )
                
                if prediction_result:
                    predicted_label_id, predicted_label, confidence = prediction_result
                    # Convert label to signal
                    if predicted_label == "UP":
                        signal = 1
                    elif predicted_label == "DOWN":
                        signal = -1
                    else:
                        signal = 0
                    
                    result_df.loc[df.index[i], 'signal'] = signal
                    result_df.loc[df.index[i], 'confidence'] = confidence
            except Exception as e:
                log_warn(f"XGBoost batch calculation failed for period {i}: {e}")
                continue
        
        return result_df
    
    def _calc_hmm_batch(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> pd.DataFrame:
        """
        Calculate HMM signals for entire DataFrame using batch processing.
        
        Similar to XGBoost, HMM needs to respect walk-forward semantics.
        
        Returns DataFrame with 'signal' and 'confidence' columns for all periods.
        """
        from core.signal_calculators import get_hmm_signal
        
        # Validate required columns
        required_columns = ["high", "low", "close"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return pd.DataFrame({
                'signal': [0] * len(df),
                'confidence': [0.0] * len(df)
            }, index=df.index)
        
        # Initialize result DataFrame
        result_df = pd.DataFrame({
            'signal': [0] * len(df),
            'confidence': [0.0] * len(df)
        }, index=df.index)
        
        # Process incrementally (walk-forward) for each period
        for i in range(len(df)):
            try:
                # Get historical data up to current period (inclusive)
                historical_df = df.iloc[:i + 1].copy()
                
                # Calculate HMM signal
                hmm_result = get_hmm_signal(
                    data_fetcher=self.data_fetcher,
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=len(historical_df),
                    df=historical_df,
                )
                
                if hmm_result is not None:
                    signal, confidence = hmm_result
                    result_df.loc[df.index[i], 'signal'] = signal
                    result_df.loc[df.index[i], 'confidence'] = confidence
            except Exception as e:
                log_warn(f"HMM batch calculation failed for period {i}: {e}")
                continue
        
        return result_df
    
    def _calc_random_forest_batch(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> pd.DataFrame:
        """
        Calculate Random Forest signals for entire DataFrame using batch processing.
        
        Similar to XGBoost and HMM, Random Forest needs to respect walk-forward semantics.
        
        Returns DataFrame with 'signal' and 'confidence' columns for all periods.
        """
        from core.signal_calculators import get_random_forest_signal
        
        # Validate required columns
        required_columns = ["high", "low", "close"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return pd.DataFrame({
                'signal': [0] * len(df),
                'confidence': [0.0] * len(df)
            }, index=df.index)
        
        # Initialize result DataFrame
        result_df = pd.DataFrame({
            'signal': [0] * len(df),
            'confidence': [0.0] * len(df)
        }, index=df.index)
        
        # Process incrementally (walk-forward) for each period
        for i in range(len(df)):
            try:
                # Get historical data up to current period (inclusive)
                historical_df = df.iloc[:i + 1].copy()
                
                # Calculate Random Forest signal
                rf_result = get_random_forest_signal(
                    data_fetcher=self.data_fetcher,
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=len(historical_df),
                    df=historical_df,
                )
                
                if rf_result is not None:
                    signal, confidence = rf_result
                    result_df.loc[df.index[i], 'signal'] = signal
                    result_df.loc[df.index[i], 'confidence'] = confidence
            except Exception as e:
                log_warn(f"Random Forest batch calculation failed for period {i}: {e}")
                continue
        
        return result_df
    
    def calculate_signal_from_precomputed(
        self,
        precomputed_indicators: Dict[str, pd.DataFrame],
        period_index: int,
        signal_type: str,
    ) -> Tuple[int, float]:
        """
        Calculate signal using pre-computed indicators.
        
        This method extracts indicators for a specific period from pre-computed
        indicator data and combines them using majority vote. It maintains
        walk-forward semantics by only using indicators up to period_index.
        
        Args:
            precomputed_indicators: Dict mapping indicator_name -> DataFrame with
                columns ['signal', 'confidence'] for all periods
            period_index: Index of the period to calculate signal for (0-based)
            signal_type: "LONG" or "SHORT"
            
        Returns:
            Tuple of (signal, confidence) where:
            - signal: 1 (LONG), -1 (SHORT), or 0 (HOLD/NEUTRAL)
            - confidence: Combined confidence score (0.0 to 1.0)
        """
        if period_index < 0:
            return (0, 0.0)
        
        # Extract indicators for this period (respecting walk-forward)
        indicator_signals = []
        for indicator_name, indicator_data in precomputed_indicators.items():
            if indicator_data is None or indicator_data.empty:
                continue
            
            # Check if we have data for this period
            if period_index < len(indicator_data):
                try:
                    # Try to access by integer position first (faster)
                    signal_val = indicator_data.iloc[period_index]['signal']
                    confidence_val = indicator_data.iloc[period_index]['confidence']
                    
                    # Only add if signal is valid (not NaN)
                    if pd.notna(signal_val) and pd.notna(confidence_val):
                        indicator_signals.append({
                            'indicator': indicator_name,
                            'signal': int(signal_val),
                            'confidence': float(confidence_val),
                        })
                except (IndexError, KeyError):
                    # Fallback: skip this indicator for this period
                    continue
        
        # Combine signals using majority vote
        if not indicator_signals:
            return (0, 0.0)
        
        combined_signal, combined_confidence = self.combine_signals_majority_vote(
            indicator_signals,
            expected_signal_type=signal_type,
        )
        
        return (combined_signal, combined_confidence)
    
    def calculate_single_signal_from_precomputed(
        self,
        precomputed_indicators: Dict[str, pd.DataFrame],
        period_index: int,
    ) -> Tuple[int, float]:
        """
        Calculate signal from pre-computed indicators by selecting highest confidence.
        
        Unlike calculate_signal_from_precomputed(), this method:
        - Does NOT require majority vote
        - Does NOT filter by expected signal_type
        - Simply selects the signal with highest confidence from all indicators
        - If multiple signals have same confidence, prefers LONG over SHORT
        
        Args:
            precomputed_indicators: Dict mapping indicator_name -> DataFrame with
                columns ['signal', 'confidence'] for all periods
            period_index: Index of the period to calculate signal for (0-based)
            
        Returns:
            Tuple of (signal, confidence) where:
            - signal: 1 (LONG), -1 (SHORT), or 0 (HOLD/NEUTRAL)
            - confidence: Confidence score of selected signal (0.0 to 1.0)
        """
        if period_index < 0:
            return (0, 0.0)
        
        # Extract indicators for this period (respecting walk-forward)
        indicator_signals = []
        for indicator_name, indicator_data in precomputed_indicators.items():
            if indicator_data is None or indicator_data.empty:
                continue
            
            # Check if we have data for this period
            if period_index < len(indicator_data):
                try:
                    # Try to access by integer position first (faster)
                    signal_val = indicator_data.iloc[period_index]['signal']
                    confidence_val = indicator_data.iloc[period_index]['confidence']
                    
                    # Only add if signal is valid (not NaN)
                    if pd.notna(signal_val) and pd.notna(confidence_val):
                        indicator_signals.append({
                            'indicator': indicator_name,
                            'signal': int(signal_val),
                            'confidence': float(confidence_val),
                        })
                except (IndexError, KeyError):
                    # Fallback: skip this indicator for this period
                    continue
        
        # Filter out neutral signals (keep only LONG=1 or SHORT=-1)
        non_zero_signals = [
            sig for sig in indicator_signals
            if sig.get('signal', 0) != 0
        ]
        
        if not non_zero_signals:
            return (0, 0.0)
        
        # Select signal with highest confidence
        # If tie, prefer LONG (1) over SHORT (-1)
        best_signal = None
        best_confidence = -1.0
        
        for sig in non_zero_signals:
            signal_val = sig.get('signal', 0)
            confidence_val = sig.get('confidence', 0.0)
            
            if confidence_val > best_confidence:
                best_signal = signal_val
                best_confidence = confidence_val
            elif confidence_val == best_confidence and signal_val == 1 and best_signal == -1:
                # Tie in confidence, prefer LONG over SHORT
                best_signal = signal_val
                best_confidence = confidence_val
        
        if best_signal is None:
            return (0, 0.0)
        
        return (best_signal, best_confidence)


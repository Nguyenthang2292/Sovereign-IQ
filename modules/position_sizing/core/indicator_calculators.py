"""
Indicator Calculators Mixin for Hybrid Signal Calculator.

This module provides individual indicator calculation methods for:
- Range Oscillator
- SPC (Simplified Percentile Clustering)
- XGBoost
- HMM (Hidden Markov Model)
- Random Forest
"""

from typing import Dict, List, Optional, Callable, Tuple, Any
import pandas as pd

from core.signal_calculators import (
    get_range_oscillator_signal,
    get_spc_signal,
    get_xgboost_signal,
    get_hmm_signal,
    get_random_forest_signal,
)
from modules.common.utils import log_warn


class IndicatorCalculatorsMixin:
    """
    Mixin class providing individual indicator calculation methods.
    
    Each method calculates a signal from a specific indicator with caching support.
    
    Required Interface:
        The host class must provide the following attributes and methods:
        
        Attributes:
            _indicator_cache (OrderedDict[Tuple[str, int, str], Dict]):
                LRU cache for indicator results. Must support:
                - Dictionary-like access (__contains__, __getitem__, __setitem__)
                - move_to_end(key) method for LRU ordering
                - popitem(last=False) for removing oldest entries
                - Key format: (symbol: str, period_index: int, indicator_name: str)
                - Value format: Dict with 'indicator', 'signal', and 'confidence' keys
                
            data_fetcher (DataFetcher):
                Object providing data access methods for fetching OHLCV data.
                Must be compatible with signal calculation functions from
                core.signal_calculators module (get_range_oscillator_signal,
                get_spc_signal, get_xgboost_signal, get_hmm_signal,
                get_random_forest_signal).
        
        Methods:
            _cache_indicator_result(cache_key: Tuple[str, int, str], result: Dict) -> None:
                Persist an indicator calculation result to the cache.
                Should implement LRU eviction when cache reaches maximum size.
                Args:
                    cache_key: Tuple of (symbol, period_index, indicator_name)
                    result: Dict with 'indicator', 'signal', and 'confidence' keys
        
        Thread Safety:
            This mixin does not provide thread-safety guarantees. If the host class
            is used in a multi-threaded context, the host class must ensure thread-safe
            access to _indicator_cache and _cache_indicator_result.
        
        Lifetime:
            _indicator_cache and data_fetcher must remain valid for the lifetime
            of all indicator calculation method calls. The cache is expected to be
            initialized before any calculation methods are invoked.
    """
    
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
    
    def _calc_indicator_generic(
        self,
        indicator_name: str,
        compute_func: Callable[..., Optional[Tuple[Any, float]]],
        symbol: str,
        timeframe: str,
        limit: int,
        period_index: Optional[int] = None,
        df: Optional[pd.DataFrame] = None,
        **compute_kwargs: Any
    ) -> Optional[Dict]:
        """
        Generic helper for calculating indicator signals with caching.
        
        Args:
            indicator_name: Name of the indicator (used for cache key and result dict)
            compute_func: Function to compute the signal (returns tuple of (signal, confidence) or None)
            symbol: Trading symbol
            timeframe: Timeframe string
            limit: Number of candles to fetch
            period_index: Optional period index for caching
            df: Optional pre-fetched DataFrame
            **compute_kwargs: Additional keyword arguments to pass to compute_func
        
        Returns:
            Dict with 'indicator', 'signal', and 'confidence' keys, or None on failure
        """
        # Check cache first (LRU: move to end if found)
        if period_index is not None:
            cache_key = (symbol, period_index, indicator_name)
            if cache_key in self._indicator_cache:
                # Move to end (most recently used)
                self._indicator_cache.move_to_end(cache_key)
                return self._indicator_cache[cache_key]
        
        try:
            # Invoke compute function with common params and any additional kwargs
            result_tuple = compute_func(
                data_fetcher=self.data_fetcher,
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                df=df,
                **compute_kwargs
            )
            if result_tuple is not None:
                signal, confidence = result_tuple
                result = {
                    'indicator': indicator_name,
                    'signal': signal,
                    'confidence': confidence,
                }
                # Cache result
                if period_index is not None:
                    self._cache_indicator_result(cache_key, result)
                return result
        except Exception as e:
            log_warn(f"{indicator_name.replace('_', ' ').title()} signal calculation failed: {e}")
        return None
    
    def _calc_random_forest(
        self, symbol: str, timeframe: str, limit: int,
        period_index: Optional[int] = None,
        df: Optional[pd.DataFrame] = None
    ) -> Optional[Dict]:
        """Calculate Random Forest signal with caching."""
        return self._calc_indicator_generic(
            indicator_name='random_forest',
            compute_func=get_random_forest_signal,
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
            period_index=period_index,
            df=df
        )


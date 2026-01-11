
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple
import json
import traceback

import numpy as np
import pandas as pd

from config.position_sizing import ENABLE_MULTITHREADING
from modules.common.core.data_fetcher import DataFetcher
from modules.common.utils import (
from modules.common.core.data_fetcher import DataFetcher
from modules.common.utils import (

"""
Hybrid Signal Calculator for Position Sizing.

This module combines signals from multiple indicators (Range Oscillator, SPC,
XGBoost, HMM, Random Forest) using majority vote or weighted voting approach.
"""



    log_error,
    log_warn,
)
from modules.position_sizing.core.batch_calculators import BatchCalculatorsMixin
from modules.position_sizing.core.cache_manager import CacheManagerMixin
from modules.position_sizing.core.indicator_calculators import IndicatorCalculatorsMixin
from modules.position_sizing.core.signal_combiner import SignalCombinerMixin


class HybridSignalCalculator(
    CacheManagerMixin,
    SignalCombinerMixin,
    IndicatorCalculatorsMixin,
    BatchCalculatorsMixin,
):
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
        default_confidence: Optional[float] = None,
    ):
        """
        Initialize Hybrid Signal Calculator.

        Args:
            data_fetcher: DataFetcher instance for fetching OHLCV data
            enabled_indicators: List of enabled indicators (default: all)
                Options: 'range_oscillator', 'spc', 'xgboost', 'hmm', 'random_forest'
            use_confidence_weighting: Whether to weight votes by confidence scores
            min_indicators_agreement: Minimum number of indicators that must agree
            default_confidence: Default confidence value (0.0-1.0) used when indicators
                don't provide a confidence score. If None, uses SignalCombinerMixin.DEFAULT_CONFIDENCE (0.5).
                The midpoint (0.5) represents neutral confidence and can be adjusted to tune
                behavior when indicators lack explicit confidence values.
        """
        self.data_fetcher = data_fetcher
        self.use_confidence_weighting = use_confidence_weighting

        # Set default_confidence if provided, otherwise it will use class constant
        if default_confidence is not None:
            if not isinstance(default_confidence, (int, float)) or not (0.0 <= default_confidence <= 1.0):
                raise ValueError(f"default_confidence must be a float between 0.0 and 1.0, got {default_confidence}")
            self.default_confidence = float(default_confidence)

        # Default: enable all indicators
        if enabled_indicators is None:
            enabled_indicators = [
                "range_oscillator",
                "spc",
                "xgboost",
                "hmm",
                "random_forest",
            ]

        self.enabled_indicators = enabled_indicators

        # Validate min_indicators_agreement
        if not isinstance(min_indicators_agreement, int):
            raise ValueError(f"min_indicators_agreement must be an int, got {type(min_indicators_agreement).__name__}")

        num_enabled = len(self.enabled_indicators)
        if not (1 <= min_indicators_agreement <= num_enabled):
            raise ValueError(
                f"min_indicators_agreement must be an int between 1 and {num_enabled} "
                f"(number of enabled indicators), got {min_indicators_agreement}"
            )

        self.min_indicators_agreement = min_indicators_agreement

        # Initialize cache structures (from CacheManagerMixin)
        self._init_cache()

        # Track indicator calculation errors
        self.indicator_errors = {}

    def _validate_indicator_schema(
        self,
        indicator_name: str,
        indicator_data: pd.DataFrame,
    ) -> None:
        """
        Validate that indicator DataFrame contains required columns.

        Args:
            indicator_name: Name of the indicator (for error messages)
            indicator_data: DataFrame to validate

        Raises:
            ValueError: If required columns are missing
        """
        required_columns = ["signal", "confidence"]
        missing_columns = [col for col in required_columns if col not in indicator_data.columns]
        if missing_columns:
            raise ValueError(
                f"Indicator '{indicator_name}' DataFrame is missing required columns: {missing_columns}. "
                f"Available columns: {list(indicator_data.columns)}"
            )

    def _fetch_ohlcv_with_cache(
        self,
        symbol: str,
        limit: int,
        timeframe: str,
        check_freshness: bool = False,
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Fetch OHLCV data with caching support.

        Checks the data cache first. If not found, fetches from data_fetcher
        and populates the cache. Returns the DataFrame and exchange_id.

        Args:
            symbol: Trading pair symbol
            limit: Number of candles to fetch
            timeframe: Timeframe string
            check_freshness: Whether to check data freshness

        Returns:
            Tuple of (DataFrame, exchange_id) or (None, None) if fetch fails
        """
        cache_key = (symbol.upper(), limit, timeframe)

        # Check cache first (only if not checking freshness)
        if not check_freshness:
            if cache_key in self._data_cache:
                # Move to end (most recently used)
                self._data_cache.move_to_end(cache_key)
                cached_df, cached_exchange_id = self._data_cache[cache_key]
                # Return cached data with stored exchange_id
                return cached_df.copy(), cached_exchange_id

        # Fetch from data_fetcher
        df, exchange_id = self.data_fetcher.fetch_ohlcv_with_fallback_exchange(
            symbol,
            limit=limit,
            timeframe=timeframe,
            check_freshness=check_freshness,
        )

        # Cache the result if fetch was successful
        if df is not None and not df.empty:
            self._cache_data_result(cache_key, df, exchange_id)

        return df, exchange_id

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
        # Validate and normalize signal_type before try block to ensure cache_key is always defined
        if signal_type is None:
            signal_type = "LONG"  # Default fallback
        signal_type = str(signal_type).upper()

        # Compute cache_key before try block so it's always defined
        # Include all parameters that affect the calculation
        cache_key = (
            symbol,
            period_index,
            signal_type,
            osc_length,
            osc_mult,
            tuple(osc_strategies) if osc_strategies else None,
            json.dumps(spc_params, sort_keys=True, separators=(",", ":")) if spc_params else None,
        )
        try:
            # Check cache first (LRU: move to end if found)
            if cache_key in self._signal_cache:
                # Move to end (most recently used)
                self._signal_cache.move_to_end(cache_key)
                self._cache_hits += 1
                return self._signal_cache[cache_key]
            self._cache_misses += 1

            # Use only data up to current period (rolling window)
            if period_index >= len(df):
                result = (0, 0.0)
                self._cache_result(cache_key, result)
                return result

            # Get historical data up to current period (inclusive)
            # Use view instead of copy for better performance (only copy when needed)
            historical_df = df.iloc[: period_index + 1]

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
                    symbol,
                    timeframe,
                    limit,
                    osc_length,
                    osc_mult,
                    osc_strategies,
                    spc_params,
                    len(historical_df),
                    period_index,
                    historical_df,
                )
            else:
                indicator_signals = self._calculate_indicators_sequential(
                    symbol,
                    timeframe,
                    limit,
                    osc_length,
                    osc_mult,
                    osc_strategies,
                    spc_params,
                    len(historical_df),
                    period_index,
                    historical_df,
                )

            # Combine signals using majority vote
            if not indicator_signals:
                result = (0, 0.0)
                self._cache_result(cache_key, result)
                return result

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
            historical_df = df.iloc[: period_index + 1]

            # Need at least some data to calculate signals
            if len(historical_df) < 10:
                return (0, 0.0)

            # Calculate limit (number of candles to use)
            limit = min(len(historical_df), 1500)  # API limit

            # Calculate signals from all enabled indicators
            if ENABLE_MULTITHREADING:
                indicator_signals = self._calculate_indicators_parallel(
                    symbol,
                    timeframe,
                    limit,
                    osc_length,
                    osc_mult,
                    osc_strategies,
                    spc_params,
                    len(historical_df),
                    period_index,
                    historical_df,
                )
            else:
                indicator_signals = self._calculate_indicators_sequential(
                    symbol,
                    timeframe,
                    limit,
                    osc_length,
                    osc_mult,
                    osc_strategies,
                    spc_params,
                    len(historical_df),
                    period_index,
                    historical_df,
                )

            # Filter out neutral signals (keep only LONG=1 or SHORT=-1)
            non_zero_signals = [sig for sig in indicator_signals if sig.get("signal", 0) != 0]

            return self._select_highest_confidence_signal(non_zero_signals)

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
            if "range_oscillator" in self.enabled_indicators:
                futures["range_oscillator"] = executor.submit(
                    self._calc_range_oscillator,
                    symbol,
                    timeframe,
                    limit,
                    osc_length,
                    osc_mult,
                    osc_strategies,
                    period_index,
                    df,
                )

            # 2. SPC (3 strategies)
            if "spc" in self.enabled_indicators:
                spc_strategies = ["cluster_transition", "regime_following", "mean_reversion"]
                for strategy in spc_strategies:
                    futures[f"spc_{strategy}"] = executor.submit(
                        self._calc_spc, symbol, timeframe, limit, strategy, spc_params, period_index, df
                    )

            # 3. XGBoost
            if "xgboost" in self.enabled_indicators and historical_df_len >= 50:
                futures["xgboost"] = executor.submit(self._calc_xgboost, symbol, timeframe, limit, period_index, df)

            # 4. HMM
            if "hmm" in self.enabled_indicators:
                futures["hmm"] = executor.submit(self._calc_hmm, symbol, timeframe, limit, period_index, df)

            # 5. Random Forest
            if "random_forest" in self.enabled_indicators:
                futures["random_forest"] = executor.submit(
                    self._calc_random_forest, symbol, timeframe, limit, period_index, df
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
        if "range_oscillator" in self.enabled_indicators:
            try:
                result = self._calc_range_oscillator(
                    symbol, timeframe, limit, osc_length, osc_mult, osc_strategies, period_index, df
                )
                if result:
                    indicator_signals.append(result)
            except Exception as e:
                log_warn(
                    f"range_oscillator signal calculation failed: symbol={symbol}, timeframe={timeframe}, period_index={period_index}, error={e}"
                )

        # 2. SPC (Simplified Percentile Clustering) - 3 strategies
        if "spc" in self.enabled_indicators:
            spc_strategies = ["cluster_transition", "regime_following", "mean_reversion"]
            for strategy in spc_strategies:
                try:
                    result = self._calc_spc(symbol, timeframe, limit, strategy, spc_params, period_index, df)
                    if result:
                        indicator_signals.append(result)
                except Exception as e:
                    indicator_name = f"spc_{strategy}"
                    log_warn(
                        f"{indicator_name} signal calculation failed: symbol={symbol}, timeframe={timeframe}, period_index={period_index}, error={e}"
                    )

        # 3. XGBoost
        if "xgboost" in self.enabled_indicators and historical_df_len >= 50:
            try:
                result = self._calc_xgboost(symbol, timeframe, limit, period_index, df)
                if result:
                    indicator_signals.append(result)
            except Exception as e:
                log_warn(
                    f"xgboost signal calculation failed: symbol={symbol}, timeframe={timeframe}, period_index={period_index}, error={e}"
                )

        # 4. HMM
        if "hmm" in self.enabled_indicators:
            try:
                result = self._calc_hmm(symbol, timeframe, limit, period_index, df)
                if result:
                    indicator_signals.append(result)
            except Exception as e:
                log_warn(
                    f"hmm signal calculation failed: symbol={symbol}, timeframe={timeframe}, period_index={period_index}, error={e}"
                )

        # 5. Random Forest
        if "random_forest" in self.enabled_indicators:
            try:
                result = self._calc_random_forest(symbol, timeframe, limit, period_index, df)
                if result:
                    indicator_signals.append(result)
            except Exception as e:
                log_warn(
                    f"random_forest signal calculation failed: symbol={symbol}, timeframe={timeframe}, period_index={period_index}, error={e}"
                )

        return indicator_signals

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
        self.indicator_errors = {}

        # Range Oscillator - vectorized
        if "range_oscillator" in self.enabled_indicators:
            try:
                indicators["range_oscillator"] = self._calc_range_oscillator_vectorized(
                    df, osc_length, osc_mult, osc_strategies
                )
            except Exception as e:
                error_msg = str(e)
                log_warn(f"Range Oscillator vectorized calculation failed: {error_msg}")
                self.indicator_errors["range_oscillator"] = error_msg
                indicators["range_oscillator"] = pd.DataFrame({"signal": np.nan, "confidence": np.nan}, index=df.index)

        # SPC - vectorized where possible (3 strategies)
        if "spc" in self.enabled_indicators:
            spc_strategies = ["cluster_transition", "regime_following", "mean_reversion"]
            for strategy in spc_strategies:
                try:
                    indicators[f"spc_{strategy}"] = self._calc_spc_vectorized(df, strategy, spc_params)
                except Exception as e:
                    error_msg = str(e)
                    indicator_name = f"spc_{strategy}"
                    log_warn(f"SPC {strategy} vectorized calculation failed: {error_msg}")
                    self.indicator_errors[indicator_name] = error_msg
                    indicators[indicator_name] = pd.DataFrame({"signal": np.nan, "confidence": np.nan}, index=df.index)

        # ML models - batch processing
        if "xgboost" in self.enabled_indicators:
            try:
                indicators["xgboost"] = self._calc_xgboost_batch(df, symbol, timeframe)
            except Exception as e:
                error_msg = str(e)
                log_warn(f"XGBoost batch calculation failed: {error_msg}")
                self.indicator_errors["xgboost"] = error_msg
                indicators["xgboost"] = pd.DataFrame({"signal": np.nan, "confidence": np.nan}, index=df.index)

        if "hmm" in self.enabled_indicators:
            try:
                indicators["hmm"] = self._calc_hmm_batch(df, symbol, timeframe)
            except Exception as e:
                error_msg = str(e)
                log_warn(f"HMM batch calculation failed: {error_msg}")
                self.indicator_errors["hmm"] = error_msg
                indicators["hmm"] = pd.DataFrame({"signal": np.nan, "confidence": np.nan}, index=df.index)

        if "random_forest" in self.enabled_indicators:
            try:
                indicators["random_forest"] = self._calc_random_forest_batch(df, symbol, timeframe)
            except Exception as e:
                error_msg = str(e)
                log_warn(f"Random Forest batch calculation failed: {error_msg}")
                self.indicator_errors["random_forest"] = error_msg
                indicators["random_forest"] = pd.DataFrame({"signal": np.nan, "confidence": np.nan}, index=df.index)

        return indicators

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

            # Schema validation: check required columns before access
            self._validate_indicator_schema(indicator_name, indicator_data)

            # Check if we have data for this period
            if period_index < len(indicator_data):
                try:
                    # Try to access by integer position first (faster)
                    signal_val = indicator_data.iloc[period_index]["signal"]
                    confidence_val = indicator_data.iloc[period_index]["confidence"]

                    # Only add if signal is valid (not NaN)
                    if pd.notna(signal_val) and pd.notna(confidence_val):
                        indicator_signals.append(
                            {
                                "indicator": indicator_name,
                                "signal": int(signal_val),
                                "confidence": float(confidence_val),
                            }
                        )
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

            # Schema validation: check required columns before access
            self._validate_indicator_schema(indicator_name, indicator_data)

            # Check if we have data for this period
            if period_index < len(indicator_data):
                try:
                    # Try to access by integer position first (faster)
                    signal_val = indicator_data.iloc[period_index]["signal"]
                    confidence_val = indicator_data.iloc[period_index]["confidence"]

                    # Only add if signal is valid (not NaN)
                    if pd.notna(signal_val) and pd.notna(confidence_val):
                        indicator_signals.append(
                            {
                                "indicator": indicator_name,
                                "signal": int(signal_val),
                                "confidence": float(confidence_val),
                            }
                        )
                except (IndexError, KeyError):
                    # Fallback: skip this indicator for this period
                    continue

        # Filter out neutral signals (keep only LONG=1 or SHORT=-1)
        non_zero_signals = [sig for sig in indicator_signals if sig.get("signal", 0) != 0]

        return self._select_highest_confidence_signal(non_zero_signals)

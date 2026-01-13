"""
Batch Calculators Mixin for Hybrid Signal Calculator.

This module provides batch/vectorized calculation methods for indicators:
- Range Oscillator (vectorized)
- SPC (vectorized)
- XGBoost (batch)
- HMM (batch)
- Random Forest (batch)
"""

from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from modules.common.utils import log_warn

if TYPE_CHECKING:
    from modules.common.core.data_fetcher import DataFetcher


class BatchCalculatorsMixin:
    """
    Mixin class providing batch/vectorized indicator calculation methods.

    These methods calculate indicators for entire DataFrames at once, which is
    much faster than calculating them for each period separately.

    Requirements:
        Consuming classes must provide a `data_fetcher` attribute of type
        `DataFetcher`. This attribute is used by `_calc_hmm_batch` and
        `_calc_random_forest_batch` methods to fetch historical market data
        for signal calculation. The `data_fetcher` should be initialized with
        an `ExchangeManager` and is responsible for retrieving OHLCV data from
        cryptocurrency exchanges.
    """

    # Type hint for required attribute (not enforced at runtime, but helps linters and IDEs)
    data_fetcher: "DataFetcher"

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
            return pd.DataFrame({"signal": [0] * len(df), "confidence": [0.0] * len(df)}, index=df.index)

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
        config.consensus.adaptive_weights = True
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
            matches_classification = (
                "Classification metrics" in error_str
                or "Number of classes" in error_str
                or "Invalid classes" in error_str
            )
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
        result_df["signal"] = signals.reindex(df.index, fill_value=0)
        if confidence is not None and not confidence.empty:
            result_df["confidence"] = confidence.reindex(df.index, fill_value=0.0)
        else:
            result_df["confidence"] = 0.0

        # Fill NaN with 0
        result_df["signal"] = result_df["signal"].fillna(0).astype(int)
        result_df["confidence"] = result_df["confidence"].fillna(0.0).astype(float)

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
        from config import SPC_P_HIGH, SPC_P_LOW
        from modules.simplified_percentile_clustering.config import (
            ClusterTransitionConfig,
            MeanReversionConfig,
            RegimeFollowingConfig,
        )
        from modules.simplified_percentile_clustering.core.clustering import (
            ClusteringConfig,
            SimplifiedPercentileClustering,
        )
        from modules.simplified_percentile_clustering.core.features import FeatureConfig
        from modules.simplified_percentile_clustering.strategies import (
            generate_signals_cluster_transition,
            generate_signals_mean_reversion,
            generate_signals_regime_following,
        )

        # Validate required columns
        required_columns = ["high", "low", "close"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return pd.DataFrame({"signal": [0] * len(df), "confidence": [0.0] * len(df)}, index=df.index)

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
            return pd.DataFrame({"signal": [0] * len(df), "confidence": [0.0] * len(df)}, index=df.index)

        # Create DataFrame with same index as df
        result_df = pd.DataFrame(index=df.index)
        if signals is not None and not signals.empty:
            result_df["signal"] = signals.reindex(df.index, fill_value=0)
        else:
            result_df["signal"] = 0

        if signal_strength is not None and not signal_strength.empty:
            result_df["confidence"] = signal_strength.reindex(df.index, fill_value=0.0)
        else:
            result_df["confidence"] = 0.0

        # Fill NaN with 0
        result_df["signal"] = result_df["signal"].fillna(0).astype(int)
        result_df["confidence"] = result_df["confidence"].fillna(0.0).astype(float)

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
        from config import ID_TO_LABEL
        from modules.common.core.indicator_engine import (
            IndicatorConfig,
            IndicatorEngine,
            IndicatorProfile,
        )
        from modules.xgboost.labeling import apply_directional_labels
        from modules.xgboost.model import ClassDiversityError, predict_next_move, train_and_predict

        # Validate required columns
        required_columns = ["high", "low", "close"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return pd.DataFrame({"signal": [0] * len(df), "confidence": [0.0] * len(df)}, index=df.index)

        # Initialize result DataFrame
        result_df = pd.DataFrame({"signal": [0] * len(df), "confidence": [0.0] * len(df)}, index=df.index)

        # XGBoost needs at least 50 periods to train
        min_periods = 50
        if len(df) < min_periods:
            return result_df

        # Process incrementally (walk-forward) for each period
        # This respects walk-forward semantics: only use data up to current period
        indicator_engine = IndicatorEngine(IndicatorConfig.for_profile(IndicatorProfile.XGBOOST))

        for i in range(min_periods, len(df)):
            try:
                # Get historical data up to current period (inclusive)
                historical_df = df.iloc[: i + 1].copy()

                # Calculate features
                historical_df = indicator_engine.compute_features(historical_df)

                # Save latest data before applying labels
                latest_data = historical_df.iloc[-1:].copy()
                latest_data = latest_data.ffill().bfill()

                # Apply labels and drop NaN
                historical_df = apply_directional_labels(historical_df)
                historical_df.dropna(inplace=True)

                if len(historical_df) < min_periods:
                    continue

                # Check class diversity - XGBoost requires at least 2 classes for training
                if "Target" not in historical_df.columns:
                    continue
                unique_classes = historical_df["Target"].dropna().unique()
                # XGBoost requires at least 2 classes, but ideally 3 (UP, DOWN, NEUTRAL)
                # Since train_and_predict does train/test split, we need to be conservative
                # and require at least 2 classes to reduce risk of post-split having only 1 class
                if len(unique_classes) < 2:
                    # Skip if insufficient class diversity (XGBoost needs at least 2 classes)
                    continue
                # Additional check: if we only have 2 classes but model expects 3, it might fail
                # But we'll let train_and_predict handle this and catch the exception

                # Train model and predict next move
                try:
                    model = train_and_predict(historical_df)
                except ClassDiversityError:
                    # Skip this period if training fails due to class diversity issues
                    continue
                except Exception:
                    # Re-raise all other exceptions unchanged
                    raise

                proba = predict_next_move(model, latest_data)

                # Get prediction: UP=1, DOWN=-1, NEUTRAL=0
                best_idx = int(np.argmax(proba))
                direction = ID_TO_LABEL.get(best_idx, "NEUTRAL")

                # Convert to signal format: UP -> 1, DOWN -> -1, NEUTRAL -> 0
                if direction == "UP":
                    signal = 1
                elif direction == "DOWN":
                    signal = -1
                else:
                    signal = 0

                # Use probability as confidence/strength
                confidence = float(proba[best_idx])

                result_df.loc[df.index[i], "signal"] = signal
                result_df.loc[df.index[i], "confidence"] = confidence
            except Exception as e:
                log_warn(f"XGBoost batch calculation failed for period {i}: {e}")
                continue

        return result_df

    def _calc_batch_walkforward(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        signal_fn: Callable[[pd.DataFrame, str, str], Optional[Tuple[int, float]]],
        min_periods: Optional[int] = None,
        indicator_name: str = "Indicator",
    ) -> pd.DataFrame:
        """
        Generic walk-forward batch calculation helper.

        This helper encapsulates the common walk-forward pattern used by HMM and
        Random Forest batch calculations. It handles:
        - Column validation
        - Result DataFrame initialization
        - Walk-forward loop with historical data slicing
        - Error handling and logging
        - Result assignment

        Args:
            df: DataFrame with OHLC data
            symbol: Trading symbol
            timeframe: Timeframe string
            signal_fn: Callable that takes (df, symbol, timeframe) and returns
                      (signal, confidence) tuple or None
            min_periods: Minimum number of periods required (optional)
            indicator_name: Name of indicator for logging purposes

        Returns:
            DataFrame with 'signal' and 'confidence' columns matching df.index
        """
        # Validate required columns
        required_columns = ["high", "low", "close"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return pd.DataFrame({"signal": [0] * len(df), "confidence": [0.0] * len(df)}, index=df.index)

        # Initialize result DataFrame with same index and columns
        result_df = pd.DataFrame({"signal": [0] * len(df), "confidence": [0.0] * len(df)}, index=df.index)

        # Check minimum periods if specified
        if min_periods is not None and len(df) < min_periods:
            return result_df

        # Determine start index for walk-forward loop
        start_idx = min_periods if min_periods is not None else 0

        # Process incrementally (walk-forward) for each period
        for i in range(start_idx, len(df)):
            try:
                # Get historical data up to current period (inclusive)
                historical_df = df.iloc[: i + 1].copy()

                # Calculate signal using provided function
                signal_result = signal_fn(historical_df, symbol, timeframe)

                if signal_result is not None:
                    signal, confidence = signal_result
                    result_df.loc[df.index[i], "signal"] = signal
                    result_df.loc[df.index[i], "confidence"] = confidence
            except Exception as e:
                log_warn(f"{indicator_name} batch calculation failed for period {i}: {e}")
                continue

        return result_df

    def _calc_hmm_batch(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        min_periods: int = 50,
    ) -> pd.DataFrame:
        """
        Calculate HMM signals for entire DataFrame using batch processing.

        Similar to XGBoost, HMM needs to respect walk-forward semantics.

        Args:
            df: DataFrame with OHLC data
            symbol: Trading symbol
            timeframe: Timeframe string
            min_periods: Minimum number of periods required for HMM calculation (default: 50)

        Returns DataFrame with 'signal' and 'confidence' columns for all periods.
        """
        from core.signal_calculators import get_hmm_signal

        def hmm_signal_fn(historical_df: pd.DataFrame, sym: str, tf: str) -> Optional[Tuple[int, float]]:
            """Wrapper function for get_hmm_signal to match helper signature."""
            return get_hmm_signal(
                data_fetcher=self.data_fetcher,
                symbol=sym,
                timeframe=tf,
                limit=len(historical_df),
                df=historical_df,
            )

        return self._calc_batch_walkforward(
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            signal_fn=hmm_signal_fn,
            min_periods=min_periods,
            indicator_name="HMM",
        )

    def _calc_random_forest_batch(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> pd.DataFrame:
        """
        Calculate Random Forest signals for entire DataFrame using batch processing.

        Similar to XGBoost and HMM, Random Forest needs to respect walk-forward semantics.
        Random Forest needs sufficient history to train.

        Returns DataFrame with 'signal' and 'confidence' columns for all periods.
        """
        from core.signal_calculators import get_random_forest_signal

        # Random Forest needs sufficient history to train
        min_periods = 30  # Adjust based on RF requirements

        def rf_signal_fn(historical_df: pd.DataFrame, sym: str, tf: str) -> Optional[Tuple[int, float]]:
            """Wrapper function for get_random_forest_signal to match helper signature."""
            return get_random_forest_signal(
                data_fetcher=self.data_fetcher,
                symbol=sym,
                timeframe=tf,
                limit=len(historical_df),
                df=historical_df,
            )

        return self._calc_batch_walkforward(
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            signal_fn=rf_signal_fn,
            min_periods=min_periods,
            indicator_name="Random Forest",
        )

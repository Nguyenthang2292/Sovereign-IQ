"""
Signal calculators for hybrid approach.

This module contains functions to calculate signals from:
1. Range Oscillator
2. Simplified Percentile Clustering (SPC)
3. XGBoost
4. HMM (Hidden Markov Model)
5. Random Forest
"""

from typing import Optional, Tuple

import pandas as pd

from config import (
    HMM_FAST_KAMA_DEFAULT,
    HMM_HIGH_ORDER_ORDERS_ARGRELEXTREMA_DEFAULT,
    HMM_HIGH_ORDER_STRICT_MODE_DEFAULT,
    HMM_SLOW_KAMA_DEFAULT,
    HMM_WINDOW_KAMA_DEFAULT,
    HMM_WINDOW_SIZE_DEFAULT,
    ID_TO_LABEL,
    SPC_P_HIGH,
    SPC_P_LOW,
    TARGET_BASE_THRESHOLD,
)

# Constants for Range Oscillator strategies
# Default enabled strategies when strategies parameter is None or when strategy 5 is included
DEFAULT_ENABLED_STRATEGIES = [2, 3, 4, 6, 7, 8, 9]

# Constants for XGBoost training
# Minimum number of samples required for training to prevent overfitting
XGBOOST_MIN_TRAINING_SAMPLES = 50
from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.indicator_engine import (
    IndicatorConfig,
    IndicatorEngine,
    IndicatorProfile,
)
from modules.hmm.signals.combiner import combine_signals
from modules.random_forest.core.model import load_random_forest_model
from modules.random_forest.core.signals import get_latest_random_forest_signal
from modules.range_oscillator.config import (
    CombinedStrategyConfig,
)
from modules.range_oscillator.strategies.combined import (
    generate_signals_combined_all_strategy,
)
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
from modules.xgboost.labeling import apply_directional_labels
from modules.xgboost.model import predict_next_move, train_and_predict


def get_range_oscillator_signal(
    data_fetcher: DataFetcher,
    symbol: str,
    timeframe: str,
    limit: int,
    osc_length: int = 50,
    osc_mult: float = 2.0,
    strategies: Optional[list] = None,
    df: Optional[pd.DataFrame] = None,
) -> Optional[Tuple[int, float]]:
    """Calculate Range Oscillator signal for a symbol."""
    # Wrap entire function in try-except to catch all exceptions
    try:
        # Validate numeric parameters to prevent DoS attacks
        if not isinstance(limit, int) or limit <= 0 or limit > 10000:
            from modules.common.ui.logging import log_error

            log_error(f"Invalid limit parameter: {limit}. Must be positive integer <= 10000")
            return None
        if not isinstance(osc_length, int) or osc_length <= 0 or osc_length > 1000:
            from modules.common.ui.logging import log_error

            log_error(f"Invalid osc_length parameter: {osc_length}. Must be positive integer <= 1000")
            return None
        if not isinstance(osc_mult, (int, float)) or osc_mult <= 0 or osc_mult > 100:
            from modules.common.ui.logging import log_error

            log_error(f"Invalid osc_mult parameter: {osc_mult}. Must be positive number <= 100")
            return None
        # Use provided DataFrame if available, otherwise fetch from API
        if df is None:
            # Use check_freshness=False to avoid multiple fetch attempts and inconsistent results
            # For signal calculation, we don't need the absolute latest data
            df, exchange_id = data_fetcher.fetch_ohlcv_with_fallback_exchange(
                symbol,
                limit=limit,
                timeframe=timeframe,
                check_freshness=False,
            )
        else:
            pass

        if df is None or df.empty:
            return None

        # Validate required columns
        required_columns = ["high", "low", "close"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            return None

        # Make a deep copy of only required columns to avoid modifying the original DataFrame
        # This prevents issues with shared DataFrame references in multi-threaded environments
        high = df["high"].copy()
        low = df["low"].copy()
        close = df["close"].copy()

        if strategies is None:
            enabled_strategies = DEFAULT_ENABLED_STRATEGIES
        else:
            if 5 in strategies:
                enabled_strategies = DEFAULT_ENABLED_STRATEGIES
            else:
                enabled_strategies = strategies

        config = CombinedStrategyConfig()
        config.enabled_strategies = enabled_strategies
        config.return_confidence_score = True
        config.dynamic.enabled = True
        config.dynamic.lookback = 20
        config.dynamic.volatility_threshold = 0.6
        config.dynamic.trend_threshold = 0.5
        config.consensus.mode = "weighted"
        # Disable adaptive weights by default to avoid classification metrics errors
        # These errors occur when data doesn't have enough classes for metrics calculation
        # Adaptive weights can be enabled if needed, but will be retried without if errors occur
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
            # Handle classification metrics errors by retrying without adaptive weights
            # These errors occur when data doesn't have enough classes for metrics calculation
            error_str = str(e)
            is_value_or_type_error = isinstance(e, (ValueError, TypeError))
            matches_classification = (
                "Classification metrics" in error_str
                or "Number of classes" in error_str
                or "Invalid classes" in error_str
            )

            # Only retry for ValueError/TypeError with classification metrics errors
            if is_value_or_type_error and matches_classification:
                # Retry without adaptive weights
                config.consensus.adaptive_weights = False
                try:
                    result = generate_signals_combined_all_strategy(
                        high=high,
                        low=low,
                        close=close,
                        length=osc_length,
                        mult=osc_mult,
                        config=config,
                    )
                except Exception:
                    raise
            else:
                # Not a classification metrics error, re-raise to let outer handler catch it
                raise

        signals = result[0]
        confidence = result[3]

        if signals is None or signals.empty:
            return None

        non_nan_mask = ~signals.isna()
        if not non_nan_mask.any():
            return None

        latest_idx = signals[non_nan_mask].index[-1]
        try:
            latest_signal = int(signals.loc[latest_idx])
            latest_confidence = (
                float(confidence.loc[latest_idx]) if confidence is not None and not confidence.empty else 0.0
            )
        except (ValueError, TypeError) as e:
            # Handle type conversion errors explicitly
            from modules.common.ui.logging import log_error

            log_error(f"Type conversion error in Range Oscillator signal for {symbol}: {type(e).__name__}")
            return None
        return (latest_signal, latest_confidence)

    except Exception as e:
        # Log error for debugging instead of silently swallowing
        # Sanitize error message to prevent information leakage
        from modules.common.ui.logging import log_error

        log_error(f"Error calculating Range Oscillator signal for {symbol}: {type(e).__name__}")
        return None


def get_spc_signal(
    data_fetcher: DataFetcher,
    symbol: str,
    timeframe: str,
    limit: int,
    strategy: str = "cluster_transition",
    strategy_params: Optional[dict] = None,
    clustering_config: Optional[ClusteringConfig] = None,
    df: Optional[pd.DataFrame] = None,
) -> Optional[Tuple[int, float]]:
    """Calculate SPC signal for a symbol."""
    try:
        # Validate numeric parameters to prevent DoS attacks
        if not isinstance(limit, int) or limit <= 0 or limit > 10000:
            from modules.common.ui.logging import log_error

            log_error(f"Invalid limit parameter: {limit}. Must be positive integer <= 10000")
            return None
        # Use provided DataFrame if available, otherwise fetch from API
        if df is None:
            df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
                symbol,
                limit=limit,
                timeframe=timeframe,
                check_freshness=False,
            )

        if df is None or df.empty:
            return None

        # Validate required columns (standardized validation)
        required_columns = ["high", "low", "close"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return None

        high = df["high"]
        low = df["low"]
        close = df["close"]

        if clustering_config is None:
            feature_config = FeatureConfig()
            clustering_config = ClusteringConfig(
                k=2,
                lookback=limit,
                p_low=SPC_P_LOW,
                p_high=SPC_P_HIGH,
                main_plot="Clusters",
                feature_config=feature_config,
            )

        clustering = SimplifiedPercentileClustering(clustering_config)
        clustering_result = clustering.compute(high, low, close)

        strategy_params = strategy_params or {}

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
            return None

        if signals is None or signals.empty:
            return None

        non_nan_mask = ~signals.isna()
        if not non_nan_mask.any():
            return None

        latest_idx = signals[non_nan_mask].index[-1]
        try:
            latest_signal = int(signals.loc[latest_idx])
            latest_strength = float(signal_strength.loc[latest_idx]) if not signal_strength.empty else 0.0
        except (ValueError, TypeError) as e:
            # Handle type conversion errors explicitly
            from modules.common.ui.logging import log_error

            log_error(f"Type conversion error in SPC signal for {symbol} (strategy: {strategy}): {type(e).__name__}")
            return None
        return (latest_signal, latest_strength)

    except Exception as e:
        # Log error for debugging instead of silently swallowing
        # Sanitize error message to prevent information leakage
        from modules.common.ui.logging import log_error

        log_error(f"Error calculating SPC signal for {symbol} (strategy: {strategy}): {type(e).__name__}")
        return None


def get_xgboost_signal(
    data_fetcher: DataFetcher,
    symbol: str,
    timeframe: str,
    limit: int,
    df: Optional[pd.DataFrame] = None,
) -> Optional[Tuple[int, float]]:
    """Calculate XGBoost prediction signal for a symbol."""
    try:
        # Validate numeric parameters to prevent DoS attacks
        if not isinstance(limit, int) or limit <= 0 or limit > 10000:
            from modules.common.ui.logging import log_error

            log_error(f"Invalid limit parameter: {limit}. Must be positive integer <= 10000")
            return None

        # Use provided DataFrame if available, otherwise fetch from API
        if df is None:
            df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
                symbol,
                limit=limit,
                timeframe=timeframe,
                check_freshness=False,
            )

        if df is None or df.empty:
            return None

        # Validate required columns (standardized validation)
        required_columns = ["high", "low", "close"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return None

        # Check data size BEFORE computing expensive features to prevent resource waste
        if len(df) < XGBOOST_MIN_TRAINING_SAMPLES:
            return None

        # Initialize IndicatorEngine for XGBoost features
        indicator_engine = IndicatorEngine(IndicatorConfig.for_profile(IndicatorProfile.XGBOOST))

        # Calculate indicators without labels first (to preserve latest_data)
        df = indicator_engine.compute_features(df)

        # Save latest data before applying labels and dropping NaN
        latest_data = df.iloc[-1:].copy()
        # Fill any remaining NaN in latest_data with forward fill then backward fill
        latest_data = latest_data.ffill().bfill()

        # Apply directional labels and drop NaN for training data
        df = apply_directional_labels(df)
        latest_threshold = (
            df["DynamicThreshold"].iloc[-1]
            if len(df) > 0 and "DynamicThreshold" in df.columns
            else TARGET_BASE_THRESHOLD
        )
        df.dropna(inplace=True)
        latest_data["DynamicThreshold"] = latest_threshold

        # Double-check after dropping NaN (shouldn't fail if initial check passed with sufficient margin)
        if len(df) < XGBOOST_MIN_TRAINING_SAMPLES:
            return None

        # Train model and predict
        model = train_and_predict(df)
        proba = predict_next_move(model, latest_data)

        # Get prediction: UP=1, DOWN=-1, NEUTRAL=0
        try:
            best_idx = int(proba.argmax())
            direction = ID_TO_LABEL[best_idx]

            # Convert to signal format: UP -> 1, DOWN -> -1, NEUTRAL -> 0
            if direction == "UP":
                signal = 1
            elif direction == "DOWN":
                signal = -1
            else:
                signal = 0

            # Use probability as confidence/strength
            confidence = float(proba[best_idx])
        except (ValueError, TypeError, KeyError, IndexError) as e:
            # Handle type conversion errors explicitly
            from modules.common.ui.logging import log_error

            log_error(f"Type conversion error in XGBoost signal for {symbol}: {type(e).__name__}")
            return None
        return (signal, confidence)

    except Exception as e:
        # Log error for debugging instead of silently swallowing
        # Sanitize error message to prevent information leakage
        from modules.common.ui.logging import log_error

        log_error(f"Error calculating XGBoost signal for {symbol}: {type(e).__name__}")
        return None


def get_hmm_signal(
    data_fetcher: DataFetcher,
    symbol: str,
    timeframe: str,
    limit: int,
    window_size: Optional[int] = None,
    window_kama: Optional[int] = None,
    fast_kama: Optional[int] = None,
    slow_kama: Optional[int] = None,
    orders_argrelextrema: Optional[int] = None,
    strict_mode: Optional[bool] = None,
    df: Optional[pd.DataFrame] = None,
) -> Optional[Tuple[int, float]]:
    """
    Calculate HMM signal for a symbol.

    Combines High-Order HMM and HMM-KAMA signals using the same logic as main_hmm.py:
    - If both signals agree and not HOLD -> use that signal
    - If conflict (one LONG, one SHORT) -> HOLD
    - If one is HOLD -> use the other signal

    Args:
        data_fetcher: DataFetcher instance
        symbol: Trading pair symbol
        timeframe: Timeframe for data
        limit: Number of candles to fetch
        window_size: Rolling window size (default: from config)
        window_kama: KAMA window size (default: from config)
        fast_kama: Fast KAMA parameter (default: from config)
        slow_kama: Slow KAMA parameter (default: from config)
        orders_argrelextrema: Order for swing detection (default: from config)
        strict_mode: Use strict mode for swing-to-state conversion (default: from config)
        df: Optional DataFrame to use instead of fetching from API

    Returns:
        Tuple of (signal, confidence) where:
        - signal: 1 (LONG), -1 (SHORT), or 0 (HOLD)
        - confidence: Signal confidence (0.0 to 1.0)
    """
    try:
        # Validate numeric parameters to prevent DoS attacks
        if not isinstance(limit, int) or limit <= 0 or limit > 10000:
            from modules.common.ui.logging import log_error

            log_error(f"Invalid limit parameter: {limit}. Must be positive integer <= 10000")
            return None
        if window_size is not None and (not isinstance(window_size, int) or window_size <= 0 or window_size > 1000):
            from modules.common.ui.logging import log_error

            log_error(f"Invalid window_size parameter: {window_size}. Must be positive integer <= 1000")
            return None
        if window_kama is not None and (not isinstance(window_kama, int) or window_kama <= 0 or window_kama > 1000):
            from modules.common.ui.logging import log_error

            log_error(f"Invalid window_kama parameter: {window_kama}. Must be positive integer <= 1000")
            return None

        # Use provided DataFrame if available, otherwise fetch from API
        if df is None:
            df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
                symbol,
                limit=limit,
                timeframe=timeframe,
                check_freshness=False,
            )

        if df is None or df.empty:
            return None

        # Validate required columns (standardized validation)
        required_columns = ["high", "low", "close"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return None

        # Get HMM signals using new combiner
        result = combine_signals(
            df,
            window_kama=window_kama if window_kama is not None else HMM_WINDOW_KAMA_DEFAULT,
            fast_kama=fast_kama if fast_kama is not None else HMM_FAST_KAMA_DEFAULT,
            slow_kama=slow_kama if slow_kama is not None else HMM_SLOW_KAMA_DEFAULT,
            window_size=window_size if window_size is not None else HMM_WINDOW_SIZE_DEFAULT,
            orders_argrelextrema=orders_argrelextrema
            if orders_argrelextrema is not None
            else HMM_HIGH_ORDER_ORDERS_ARGRELEXTREMA_DEFAULT,
            strict_mode=strict_mode if strict_mode is not None else HMM_HIGH_ORDER_STRICT_MODE_DEFAULT,
        )

        # Extract combined signal and confidence
        combined_signal = result["combined_signal"]
        confidence = result["confidence"]

        # Convert Signal type (Literal[-1, 0, 1]) to int
        try:
            signal_value = int(combined_signal)
        except (ValueError, TypeError) as e:
            # Handle type conversion errors explicitly
            from modules.common.ui.logging import log_error

            log_error(f"Type conversion error in HMM signal for {symbol}: {type(e).__name__}")
            return None
        return (signal_value, confidence)

    except Exception as e:
        # Log error for debugging instead of silently swallowing
        # Sanitize error message to prevent information leakage
        from modules.common.ui.logging import log_error

        log_error(f"Error calculating HMM signal for {symbol}: {type(e).__name__}")
        return None


def get_random_forest_signal(
    data_fetcher: DataFetcher,
    symbol: str,
    timeframe: str,
    limit: int,
    model_path: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
) -> Optional[Tuple[int, float]]:
    """
    Calculate Random Forest signal for a symbol.

    Args:
        data_fetcher: DataFetcher instance
        symbol: Trading pair symbol
        timeframe: Timeframe for data
        limit: Number of candles to fetch
        model_path: Optional path to model file (default: uses default path)
        df: Optional DataFrame to use instead of fetching from API

    Returns:
        Tuple of (signal, confidence) where:
        - signal: 1 (LONG), -1 (SHORT), or 0 (NEUTRAL)
        - confidence: Signal confidence (0.0 to 1.0)
    """
    try:
        # Validate numeric parameters to prevent DoS attacks
        if not isinstance(limit, int) or limit <= 0 or limit > 10000:
            from modules.common.ui.logging import log_error

            log_error(f"Invalid limit parameter: {limit}. Must be positive integer <= 10000")
            return None
        # Load model
        from pathlib import Path

        from config.random_forest import MODELS_DIR
        from modules.common.ui.logging import log_error

        # Validate and sanitize model_path to prevent path traversal attacks
        if model_path:
            path_obj = Path(model_path)

            # Resolve to absolute path to normalize any traversal sequences (../, ..\, etc.)
            try:
                resolved_path = path_obj.resolve()
            except (OSError, RuntimeError) as e:
                # Sanitize error message to prevent information leakage
                # Only log path validation errors, not full exception details
                log_error(f"Invalid model path (cannot resolve): {type(e).__name__}")
                return None

            # Ensure path is within allowed directory (MODELS_DIR)
            # Resolve MODELS_DIR to absolute path for comparison
            try:
                allowed_dir = Path(MODELS_DIR).resolve()
            except (OSError, RuntimeError):
                # If MODELS_DIR cannot be resolved, use it as-is (might be relative)
                allowed_dir = Path(MODELS_DIR).absolute()

            # Check if resolved path is within allowed directory
            try:
                # This will raise ValueError if path is outside allowed_dir
                resolved_path.relative_to(allowed_dir)
            except ValueError:
                # Path is outside allowed directory - security violation
                log_error(
                    f"Path traversal detected or invalid model path: {model_path}. "
                    f"Resolved path: {resolved_path}, Allowed directory: {allowed_dir}"
                )
                return None

            validated_path = resolved_path
        else:
            validated_path = None

        model = load_random_forest_model(validated_path)
        if model is None:
            return None

        # Use provided DataFrame if available, otherwise fetch from API
        if df is None:
            df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
                symbol,
                limit=limit,
                timeframe=timeframe,
                check_freshness=False,
            )

        if df is None or df.empty:
            return None

        # Validate required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            return None

        # Get signal from Random Forest model
        signal_str, confidence = get_latest_random_forest_signal(df, model)

        # Convert signal string to int: "LONG" -> 1, "SHORT" -> -1, "NEUTRAL" -> 0
        if signal_str == "LONG":
            signal = 1
        elif signal_str == "SHORT":
            signal = -1
        else:
            signal = 0

        return (signal, confidence)

    except Exception as e:
        # Log error for debugging instead of silently swallowing
        # Sanitize error message to prevent information leakage
        from modules.common.ui.logging import log_error

        log_error(f"Error calculating Random Forest signal for {symbol}: {type(e).__name__}")
        return None

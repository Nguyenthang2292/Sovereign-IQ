"""
Signal calculators for hybrid approach.

This module contains functions to calculate signals from:
1. Range Oscillator
2. Simplified Percentile Clustering (SPC)
3. XGBoost
4. HMM (Hidden Markov Model)
"""

from typing import Optional, Tuple
import json
import os

from modules.common.DataFetcher import DataFetcher
from modules.range_oscillator.analysis.combined import (
    generate_signals_combined_all_strategy,
)
from modules.range_oscillator.config import (
    CombinedStrategyConfig,
)
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
from modules.common.IndicatorEngine import (
    IndicatorConfig,
    IndicatorEngine,
    IndicatorProfile,
)
from modules.xgboost.labeling import apply_directional_labels
from modules.xgboost.model import train_and_predict, predict_next_move
from modules.hmm.signals.combiner import combine_signals
from modules.hmm.signals.resolution import LONG, HOLD, SHORT
from config import (
    SPC_P_LOW,
    SPC_P_HIGH,
    TARGET_BASE_THRESHOLD,
    ID_TO_LABEL,
    HMM_WINDOW_SIZE_DEFAULT,
    HMM_WINDOW_KAMA_DEFAULT,
    HMM_FAST_KAMA_DEFAULT,
    HMM_SLOW_KAMA_DEFAULT,
    HMM_HIGH_ORDER_ORDERS_ARGRELEXTREMA_DEFAULT,
    HMM_HIGH_ORDER_STRICT_MODE_DEFAULT,
)


def get_range_oscillator_signal(
    data_fetcher: DataFetcher,
    symbol: str,
    timeframe: str,
    limit: int,
    osc_length: int = 50,
    osc_mult: float = 2.0,
    strategies: Optional[list] = None,
) -> Optional[Tuple[int, float]]:
    """Calculate Range Oscillator signal for a symbol."""
    # Wrap entire function in try-except to catch all exceptions
    try:
        df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
            symbol,
            limit=limit,
            timeframe=timeframe,
            check_freshness=True,
        )

        if df is None or df.empty:
            # #region agent log
            try:
                log_path = r"d:\NGUYEN QUANG THANG\Probability projects\crypto-probability-\.cursor\debug.log"
                with open(log_path, 'a', encoding='utf-8') as f:
                    json.dump({
                        "id": f"log_get_range_oscillator_empty_df_{id(symbol)}",
                        "timestamp": int(__import__('time').time() * 1000),
                        "location": "signal_calculators.py:78",
                        "message": "Empty or None DataFrame in get_range_oscillator_signal",
                        "data": {"symbol": symbol, "df_is_none": df is None, "df_empty": df.empty if df is not None else None},
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "C"
                    }, f, ensure_ascii=False)
                    f.write('\n')
            except:
                pass
            # #endregion
            return None

        if "high" not in df.columns or "low" not in df.columns or "close" not in df.columns:
            # #region agent log
            try:
                log_path = r"d:\NGUYEN QUANG THANG\Probability projects\crypto-probability-\.cursor\debug.log"
                with open(log_path, 'a', encoding='utf-8') as f:
                    json.dump({
                        "id": f"log_get_range_oscillator_missing_cols_{id(symbol)}",
                        "timestamp": int(__import__('time').time() * 1000),
                        "location": "signal_calculators.py:81",
                        "message": "Missing required columns in get_range_oscillator_signal",
                        "data": {"symbol": symbol, "columns": list(df.columns) if df is not None else []},
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "C"
                    }, f, ensure_ascii=False)
                    f.write('\n')
            except:
                pass
            # #endregion
            return None

        high = df["high"]
        low = df["low"]
        close = df["close"]

        if strategies is None:
            enabled_strategies = [2, 3, 4, 6, 7, 8, 9]
        else:
            if 5 in strategies:
                enabled_strategies = [2, 3, 4, 6, 7, 8, 9]
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
        
        # #region agent log
        try:
            log_path = r"d:\NGUYEN QUANG THANG\Probability projects\crypto-probability-\.cursor\debug.log"
            with open(log_path, 'a', encoding='utf-8') as f:
                json.dump({
                    "id": f"log_get_range_oscillator_before_try_{id(symbol)}",
                    "timestamp": int(__import__('time').time() * 1000),
                    "location": "signal_calculators.py:143",
                    "message": "Before calling generate_signals_combined_all_strategy",
                    "data": {"symbol": symbol, "adaptive_weights": config.consensus.adaptive_weights},
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "A"
                }, f, ensure_ascii=False)
                f.write('\n')
        except:
            pass
        # #endregion
        
        try:
            # #region agent log
            try:
                log_path = r"d:\NGUYEN QUANG THANG\Probability projects\crypto-probability-\.cursor\debug.log"
                with open(log_path, 'a', encoding='utf-8') as f:
                    json.dump({
                        "id": f"log_get_range_oscillator_about_to_call_{id(symbol)}",
                        "timestamp": int(__import__('time').time() * 1000),
                        "location": "signal_calculators.py:163",
                        "message": "About to call generate_signals_combined_all_strategy",
                        "data": {"symbol": symbol},
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "A"
                    }, f, ensure_ascii=False)
                    f.write('\n')
            except:
                pass
            # #endregion
            
            result = generate_signals_combined_all_strategy(
                high=high,
                low=low,
                close=close,
                length=osc_length,
                mult=osc_mult,
                config=config,
            )
            
            # #region agent log
            try:
                log_path = r"d:\NGUYEN QUANG THANG\Probability projects\crypto-probability-\.cursor\debug.log"
                with open(log_path, 'a', encoding='utf-8') as f:
                    json.dump({
                        "id": f"log_get_range_oscillator_call_success_{id(symbol)}",
                        "timestamp": int(__import__('time').time() * 1000),
                        "location": "signal_calculators.py:171",
                        "message": "generate_signals_combined_all_strategy returned successfully",
                        "data": {"symbol": symbol, "result_type": type(result).__name__},
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "A"
                    }, f, ensure_ascii=False)
                    f.write('\n')
            except:
                pass
            # #endregion
        except Exception as e:
            # Handle classification metrics errors by retrying without adaptive weights
            # These errors occur when data doesn't have enough classes for metrics calculation
            error_str = str(e)
            error_type = type(e).__name__
            is_value_or_type_error = isinstance(e, (ValueError, TypeError))
            matches_classification = ("Classification metrics" in error_str or 
                                    "Number of classes" in error_str or 
                                    "Invalid classes" in error_str)
            
            # #region agent log
            try:
                log_path = r"d:\NGUYEN QUANG THANG\Probability projects\crypto-probability-\.cursor\debug.log"
                with open(log_path, 'a', encoding='utf-8') as f:
                    json.dump({
                        "id": f"log_get_range_oscillator_caught_{id(symbol)}",
                        "timestamp": int(__import__('time').time() * 1000),
                        "location": "signal_calculators.py:172",
                        "message": "Caught exception in get_range_oscillator_signal (inner handler)",
                        "data": {
                            "symbol": symbol, 
                            "error": error_str, 
                            "error_type": error_type,
                            "is_value_error": isinstance(e, ValueError),
                            "is_type_error": isinstance(e, TypeError),
                            "is_value_or_type_error": is_value_or_type_error,
                            "matches_classification": matches_classification,
                            "will_retry": is_value_or_type_error and matches_classification
                        },
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "A"
                    }, f, ensure_ascii=False)
                    f.write('\n')
            except:
                pass
            # #endregion
            
            # Only retry for ValueError/TypeError with classification metrics errors
            if is_value_or_type_error and matches_classification:
                # #region agent log
                try:
                    log_path = r"d:\NGUYEN QUANG THANG\Probability projects\crypto-probability-\.cursor\debug.log"
                    with open(log_path, 'a', encoding='utf-8') as f:
                        json.dump({
                            "id": f"log_get_range_oscillator_retry_{id(symbol)}",
                            "timestamp": int(__import__('time').time() * 1000),
                            "location": "signal_calculators.py:175",
                            "message": "Retrying without adaptive weights due to classification metrics error",
                            "data": {"symbol": symbol, "error": error_str, "error_type": type(e).__name__},
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "A"
                        }, f, ensure_ascii=False)
                        f.write('\n')
                except:
                    pass
                # #endregion
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
                except Exception as retry_e:
                    # #region agent log
                    try:
                        log_path = r"d:\NGUYEN QUANG THANG\Probability projects\crypto-probability-\.cursor\debug.log"
                        with open(log_path, 'a', encoding='utf-8') as f:
                            json.dump({
                                "id": f"log_get_range_oscillator_retry_failed_{id(symbol)}",
                                "timestamp": int(__import__('time').time() * 1000),
                                "location": "signal_calculators.py:195",
                                "message": "Retry failed, raising exception",
                                "data": {"symbol": symbol, "retry_error": str(retry_e), "retry_error_type": type(retry_e).__name__},
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "A"
                            }, f, ensure_ascii=False)
                            f.write('\n')
                    except:
                        pass
                    # #endregion
                    raise
            else:
                # Not a classification metrics error, re-raise to let outer handler catch it
                raise

        signals = result[0]
        confidence = result[3]

        if signals is None or signals.empty:
            # #region agent log
            try:
                log_path = r"d:\NGUYEN QUANG THANG\Probability projects\crypto-probability-\.cursor\debug.log"
                with open(log_path, 'a', encoding='utf-8') as f:
                    json.dump({
                        "id": f"log_get_range_oscillator_empty_signals_{id(symbol)}",
                        "timestamp": int(__import__('time').time() * 1000),
                        "location": "signal_calculators.py:119",
                        "message": "Empty signals in get_range_oscillator_signal",
                        "data": {"symbol": symbol, "signals_is_none": signals is None, "signals_empty": signals.empty if signals is not None else None},
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "D"
                    }, f, ensure_ascii=False)
                    f.write('\n')
            except:
                pass
            # #endregion
            return None

        non_nan_mask = ~signals.isna()
        if not non_nan_mask.any():
            # #region agent log
            try:
                log_path = r"d:\NGUYEN QUANG THANG\Probability projects\crypto-probability-\.cursor\debug.log"
                with open(log_path, 'a', encoding='utf-8') as f:
                    json.dump({
                        "id": f"log_get_range_oscillator_all_nan_{id(symbol)}",
                        "timestamp": int(__import__('time').time() * 1000),
                        "location": "signal_calculators.py:123",
                        "message": "All signals are NaN in get_range_oscillator_signal",
                        "data": {"symbol": symbol},
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "D"
                    }, f, ensure_ascii=False)
                    f.write('\n')
            except:
                pass
            # #endregion
            return None

        latest_idx = signals[non_nan_mask].index[-1]
        latest_signal = int(signals.loc[latest_idx])
        latest_confidence = float(confidence.loc[latest_idx]) if confidence is not None and not confidence.empty else 0.0

        return (latest_signal, latest_confidence)

    except Exception as e:
        # #region agent log
        try:
            log_path = r"d:\NGUYEN QUANG THANG\Probability projects\crypto-probability-\.cursor\debug.log"
            error_type = type(e).__name__
            error_str = str(e)
            is_value_error = isinstance(e, (ValueError, TypeError))
            with open(log_path, 'a', encoding='utf-8') as f:
                json.dump({
                    "id": f"log_get_range_oscillator_exception_{id(e)}",
                    "timestamp": int(__import__('time').time() * 1000),
                    "location": "signal_calculators.py:239",
                    "message": "Exception in get_range_oscillator_signal (outer handler)",
                    "data": {
                        "symbol": symbol if 'symbol' in locals() else "unknown", 
                        "error": error_str, 
                        "error_type": error_type,
                        "is_value_error": is_value_error,
                        "is_type_error": isinstance(e, TypeError),
                        "matches_classification": "Classification metrics" in error_str or "Number of classes" in error_str
                    },
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "A"
                }, f, ensure_ascii=False)
                f.write('\n')
        except:
            pass
        # #endregion
        return None


def get_spc_signal(
    data_fetcher: DataFetcher,
    symbol: str,
    timeframe: str,
    limit: int,
    strategy: str = "cluster_transition",
    strategy_params: Optional[dict] = None,
    clustering_config: Optional[ClusteringConfig] = None,
) -> Optional[Tuple[int, float]]:
    """Calculate SPC signal for a symbol."""
    try:
        df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
            symbol,
            limit=limit,
            timeframe=timeframe,
            check_freshness=True,
        )

        if df is None or df.empty:
            return None

        if "high" not in df.columns or "low" not in df.columns or "close" not in df.columns:
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
        latest_signal = int(signals.loc[latest_idx])
        latest_strength = float(signal_strength.loc[latest_idx]) if not signal_strength.empty else 0.0

        return (latest_signal, latest_strength)

    except Exception as e:
        # #region agent log
        try:
            log_path = r"d:\NGUYEN QUANG THANG\Probability projects\crypto-probability-\.cursor\debug.log"
            with open(log_path, 'a', encoding='utf-8') as f:
                json.dump({
                    "id": f"log_get_range_oscillator_exception_{id(e)}",
                    "timestamp": int(__import__('time').time() * 1000),
                    "location": "signal_calculators.py:130",
                    "message": "Exception in get_range_oscillator_signal",
                    "data": {"symbol": symbol if 'symbol' in locals() else "unknown", "error": str(e), "error_type": type(e).__name__},
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "A"
                }, f, ensure_ascii=False)
                f.write('\n')
        except:
            pass
        # #endregion
        return None


def get_xgboost_signal(
    data_fetcher: DataFetcher,
    symbol: str,
    timeframe: str,
    limit: int,
) -> Optional[Tuple[int, float]]:
    """Calculate XGBoost prediction signal for a symbol."""
    try:
        df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
            symbol,
            limit=limit,
            timeframe=timeframe,
            check_freshness=True,
        )

        if df is None or df.empty:
            return None

        if "high" not in df.columns or "low" not in df.columns or "close" not in df.columns:
            return None

        # Initialize IndicatorEngine for XGBoost features
        indicator_engine = IndicatorEngine(
            IndicatorConfig.for_profile(IndicatorProfile.XGBOOST)
        )

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

        if len(df) < 50:  # Need minimum samples for training
            return None

        # Train model and predict
        model = train_and_predict(df)
        proba = predict_next_move(model, latest_data)

        # Get prediction: UP=1, DOWN=-1, NEUTRAL=0
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

        return (signal, confidence)

    except Exception as e:
        # #region agent log
        try:
            log_path = r"d:\NGUYEN QUANG THANG\Probability projects\crypto-probability-\.cursor\debug.log"
            with open(log_path, 'a', encoding='utf-8') as f:
                json.dump({
                    "id": f"log_get_range_oscillator_exception_{id(e)}",
                    "timestamp": int(__import__('time').time() * 1000),
                    "location": "signal_calculators.py:130",
                    "message": "Exception in get_range_oscillator_signal",
                    "data": {"symbol": symbol if 'symbol' in locals() else "unknown", "error": str(e), "error_type": type(e).__name__},
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "A"
                }, f, ensure_ascii=False)
                f.write('\n')
        except:
            pass
        # #endregion
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
    
    Returns:
        Tuple of (signal, confidence) where:
        - signal: 1 (LONG), -1 (SHORT), or 0 (HOLD)
        - confidence: Signal confidence (0.0 to 1.0)
    """
    try:
        df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
            symbol,
            limit=limit,
            timeframe=timeframe,
            check_freshness=True,
        )

        if df is None or df.empty:
            return None

        if "high" not in df.columns or "low" not in df.columns or "close" not in df.columns:
            return None

        # Get HMM signals using new combiner
        result = combine_signals(
            df,
            window_kama=window_kama if window_kama is not None else HMM_WINDOW_KAMA_DEFAULT,
            fast_kama=fast_kama if fast_kama is not None else HMM_FAST_KAMA_DEFAULT,
            slow_kama=slow_kama if slow_kama is not None else HMM_SLOW_KAMA_DEFAULT,
            window_size=window_size if window_size is not None else HMM_WINDOW_SIZE_DEFAULT,
            orders_argrelextrema=orders_argrelextrema if orders_argrelextrema is not None else HMM_HIGH_ORDER_ORDERS_ARGRELEXTREMA_DEFAULT,
            strict_mode=strict_mode if strict_mode is not None else HMM_HIGH_ORDER_STRICT_MODE_DEFAULT,
        )
        
        # Extract combined signal and confidence
        combined_signal = result["combined_signal"]
        confidence = result["confidence"]
        
        # Convert Signal type (Literal[-1, 0, 1]) to int
        signal_value = int(combined_signal)
        
        return (signal_value, confidence)

    except Exception as e:
        # #region agent log
        try:
            log_path = r"d:\NGUYEN QUANG THANG\Probability projects\crypto-probability-\.cursor\debug.log"
            with open(log_path, 'a', encoding='utf-8') as f:
                json.dump({
                    "id": f"log_get_range_oscillator_exception_{id(e)}",
                    "timestamp": int(__import__('time').time() * 1000),
                    "location": "signal_calculators.py:130",
                    "message": "Exception in get_range_oscillator_signal",
                    "data": {"symbol": symbol if 'symbol' in locals() else "unknown", "error": str(e), "error_type": type(e).__name__},
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "A"
                }, f, ensure_ascii=False)
                f.write('\n')
        except:
            pass
        # #endregion
        return None


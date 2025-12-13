"""
Signal calculators for hybrid approach.

This module contains functions to calculate signals from:
1. Range Oscillator
2. Simplified Percentile Clustering (SPC)
3. XGBoost
"""

from typing import Optional, Tuple

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
from config import (
    SPC_P_LOW,
    SPC_P_HIGH,
    TARGET_BASE_THRESHOLD,
    ID_TO_LABEL,
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
        config.consensus.adaptive_weights = True
        config.consensus.performance_window = 10
        
        result = generate_signals_combined_all_strategy(
            high=high,
            low=low,
            close=close,
            length=osc_length,
            mult=osc_mult,
            config=config,
        )

        signals = result[0]
        confidence = result[3]

        if signals is None or signals.empty:
            return None

        non_nan_mask = ~signals.isna()
        if not non_nan_mask.any():
            return None

        latest_idx = signals[non_nan_mask].index[-1]
        latest_signal = int(signals.loc[latest_idx])
        latest_confidence = float(confidence.loc[latest_idx]) if confidence is not None and not confidence.empty else 0.0

        return (latest_signal, latest_confidence)

    except Exception as e:
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
        return None


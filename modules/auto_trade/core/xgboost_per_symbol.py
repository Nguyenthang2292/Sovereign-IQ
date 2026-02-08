"""
XGBoost Per-Symbol Filter

Trains a fresh XGBoost model for each ATC-filtered symbol and validates
that the XGBoost prediction confirms the ATC signal direction.

Workflow:
1. Receive ATC signals (LONG/SHORT)
2. For each signal, fetch historical data for that specific symbol
3. Train a fresh XGBoost model on that symbol's data
4. Predict the direction using the trained model
5. If XGBoost confirms ATC direction with sufficient confidence, pass the signal

Usage:
    filter = XGBoostPerSymbolFilter(
        data_fetcher=data_fetcher,
        config={
            "min_confidence": 0.55,
            "training_timeframe": "1h",
            "training_limit": 1500,
        }
    )
    validated_signals = filter.filter_signals(atc_signals)
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from time import time
from typing import Any, Dict, List, Optional, TypedDict, cast

import pandas as pd

from config import XGBOOST_PER_SYMBOL_DEFAULTS
from modules.auto_trade.core.atc_scanner import SignalResult
from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.indicator_engine import (
    IndicatorConfig,
    IndicatorEngine,
    IndicatorProfile,
)
from modules.common.ui.logging import log_debug, log_error, log_info, log_warn
from modules.xgboost.core.labeling import apply_directional_labels
from modules.xgboost_LTS.core.model import (
    ClassDiversityError,
    predict_next_move,
    train_and_predict,
)
from modules.xgboost_LTS.utils.features import add_advanced_features


class XGBoostPerSymbolConfig(TypedDict, total=False):
    """Configuration for XGBoostPerSymbolFilter."""

    min_confidence: float  # Minimum confidence threshold (0.0 to 1.0)
    training_timeframe: str  # Timeframe for training data ("5m", "15m", "1h", etc.)
    training_limit: int  # Number of historical candles to fetch for training
    min_required_candles: int  # Minimum candles required for training
    on_error: str  # Error handling policy: "drop", "pass"
    max_workers: int  # Max parallel workers for training (default: 4)
    use_cache: bool  # Whether to use model caching (default: False for fresh training)
    handle_class_imbalance: bool  # Use class weights to handle imbalance
    skip_if_imbalanced: bool  # Skip symbols with >80% in one class


@dataclass
class TrainingResult:
    """Result of per-symbol XGBoost training."""

    symbol: str
    success: bool
    confidence: float
    direction: str  # "UP", "DOWN", "NEUTRAL"
    error: Optional[str] = None
    training_time: float = 0.0


class XGBoostPerSymbolFilter:
    """Filters signals by training a fresh XGBoost model per symbol.

    This filter validates ATC signals by training a symbol-specific XGBoost model
    and checking if the model's prediction confirms the ATC signal direction.

    The approach:
    - ATC provides the primary signal direction (LONG/SHORT)
    - XGBoost is trained on the symbol's historical data
    - XGBoost must confirm the direction with sufficient confidence
    """

    data_fetcher: DataFetcher
    config: XGBoostPerSymbolConfig
    min_confidence: float
    training_timeframe: str
    training_limit: int
    min_required_candles: int
    on_error: str
    max_workers: int
    use_cache: bool
    handle_class_imbalance: bool
    skip_if_imbalanced: bool
    indicator_engine: IndicatorEngine
    _stats: Dict[str, int]

    def __init__(
        self,
        data_fetcher: DataFetcher,
        config: Optional[XGBoostPerSymbolConfig] = None,
    ) -> None:
        """
        Initialize XGBoostPerSymbolFilter.

        Args:
            data_fetcher: DataFetcher instance for market data
            config: Configuration dictionary

        Raises:
            ValueError: If configuration parameters are invalid
        """
        self.data_fetcher = data_fetcher
        # Merge with defaults from config
        merged: Dict[str, Any] = {**dict(XGBOOST_PER_SYMBOL_DEFAULTS), **(config or {})}
        self.config = cast(XGBoostPerSymbolConfig, merged)

        # Validate and extract configuration
        self._validate_config()

        # Initialize Indicator Engine for XGBoost features
        self.indicator_engine = IndicatorEngine(
            IndicatorConfig.for_profile(IndicatorProfile.XGBOOST)
        )

        # Statistics tracking
        self._stats = {
            "total_processed": 0,
            "training_success": 0,
            "training_failed": 0,
            "skipped_imbalanced": 0,
            "confirmed": 0,
            "rejected": 0,
        }

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If any configuration parameter is invalid
        """
        self.min_confidence = self.config.get("min_confidence", 0.55)
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError(
                f"min_confidence must be between 0 and 1, got {self.min_confidence}"
            )

        self.training_timeframe = self.config.get("training_timeframe", "1h")
        valid_timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        if self.training_timeframe not in valid_timeframes:
            raise ValueError(
                f"Invalid training_timeframe: {self.training_timeframe}. "
                f"Must be one of {valid_timeframes}"
            )

        self.training_limit = self.config.get("training_limit", 1500)
        if self.training_limit < 100:
            raise ValueError(
                f"training_limit must be at least 100, got {self.training_limit}"
            )

        self.min_required_candles = self.config.get("min_required_candles", 200)
        if self.min_required_candles < 50:
            raise ValueError(
                f"min_required_candles must be at least 50, got {self.min_required_candles}"
            )

        self.on_error = self.config.get("on_error", "drop")
        if self.on_error not in ["drop", "pass"]:
            raise ValueError(f"on_error must be 'drop' or 'pass', got {self.on_error}")

        self.max_workers = self.config.get("max_workers", 4)
        self.use_cache = self.config.get("use_cache", False)
        self.handle_class_imbalance = self.config.get("handle_class_imbalance", True)
        self.skip_if_imbalanced = self.config.get("skip_if_imbalanced", True)

    def get_stats(self) -> Dict[str, int]:
        """Get filter statistics."""
        return self._stats.copy()

    def reset_stats(self) -> None:
        """Reset filter statistics."""
        for key in self._stats:
            self._stats[key] = 0

    def _train_and_predict_symbol(self, symbol: str) -> TrainingResult:
        """
        Train XGBoost model for a specific symbol and get prediction.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")

        Returns:
            TrainingResult with prediction confidence and direction
        """
        start_time = time()

        try:
            # 1. Fetch historical data
            df = self.data_fetcher.fetch_ohlcv(
                symbol,
                timeframe=self.training_timeframe,
                limit=self.training_limit,
            )

            if df is None or df.empty:
                return TrainingResult(
                    symbol=symbol,
                    success=False,
                    confidence=0.0,
                    direction="NEUTRAL",
                    error="No data available",
                    training_time=time() - start_time,
                )

            if len(df) < self.min_required_candles:
                return TrainingResult(
                    symbol=symbol,
                    success=False,
                    confidence=0.0,
                    direction="NEUTRAL",
                    error=f"Insufficient data: {len(df)}/{self.min_required_candles}",
                    training_time=time() - start_time,
                )

            # 2. Compute features
            df = self.indicator_engine.compute_features(df)
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                return TrainingResult(
                    symbol=symbol,
                    success=False,
                    confidence=0.0,
                    direction="NEUTRAL",
                    error="Feature computation failed",
                    training_time=time() - start_time,
                )

            # 3. Add advanced features
            df = add_advanced_features(df)
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                return TrainingResult(
                    symbol=symbol,
                    success=False,
                    confidence=0.0,
                    direction="NEUTRAL",
                    error="Advanced feature computation failed",
                    training_time=time() - start_time,
                )

            # 4. Create labels
            df = apply_directional_labels(df)

            # 5. Store latest row before dropping NaN
            latest_row = df.iloc[-1:].copy()

            # 6. Drop NaN rows for training
            df = df.dropna()

            if len(df) < 50:
                return TrainingResult(
                    symbol=symbol,
                    success=False,
                    confidence=0.0,
                    direction="NEUTRAL",
                    error=f"Insufficient valid data after cleaning: {len(df)}",
                    training_time=time() - start_time,
                )

            # Check class distribution before training
            if "Target" in df.columns:
                class_counts = df["Target"].value_counts().to_dict()
                total_samples = len(df)
                class_dist = {
                    "DOWN": class_counts.get(0, 0),
                    "NEUTRAL": class_counts.get(1, 0),
                    "UP": class_counts.get(2, 0),
                }

                # Log class distribution
                log_info(
                    f"{symbol} class distribution: "
                    f"DOWN={class_dist['DOWN']} ({class_dist['DOWN']/total_samples*100:.1f}%), "
                    f"NEUTRAL={class_dist['NEUTRAL']} ({class_dist['NEUTRAL']/total_samples*100:.1f}%), "
                    f"UP={class_dist['UP']} ({class_dist['UP']/total_samples*100:.1f}%)"
                )

                # Check for extreme class imbalance (>80% in one class)
                max_class_pct = max(class_dist.values()) / total_samples
                if max_class_pct > 0.8:
                    dominant_class = max(class_dist, key=lambda k: class_dist[k])
                    log_warn(
                        f"{symbol}: Severe class imbalance - {dominant_class} dominates with "
                        f"{max_class_pct*100:.1f}% of samples"
                    )

                    # Skip training if configured to do so
                    if self.skip_if_imbalanced:
                        return TrainingResult(
                            symbol=symbol,
                            success=False,
                            confidence=0.0,
                            direction="NEUTRAL",
                            error=f"Skipped due to class imbalance - {dominant_class} has {max_class_pct*100:.1f}%",
                            training_time=time() - start_time,
                        )

            # 7. Train model
            try:
                model = train_and_predict(df, use_cache=self.use_cache)
            except ClassDiversityError as e:
                return TrainingResult(
                    symbol=symbol,
                    success=False,
                    confidence=0.0,
                    direction="NEUTRAL",
                    error=f"Class diversity error: {str(e)}",
                    training_time=time() - start_time,
                )

            # 8. Predict on latest data
            proba = predict_next_move(model, latest_row)

            # proba = [prob_down, prob_neutral, prob_up]
            prob_down = float(proba[0])
            prob_neutral = float(proba[1])
            prob_up = float(proba[2])

            # Determine direction
            max_prob = max(prob_up, prob_down, prob_neutral)
            if max_prob == prob_up:
                direction = "UP"
                confidence = prob_up
            elif max_prob == prob_down:
                direction = "DOWN"
                confidence = prob_down
            else:
                direction = "NEUTRAL"
                confidence = prob_neutral

            return TrainingResult(
                symbol=symbol,
                success=True,
                confidence=confidence,
                direction=direction,
                training_time=time() - start_time,
            )

        except ClassDiversityError as e:
            return TrainingResult(
                symbol=symbol,
                success=False,
                confidence=0.0,
                direction="NEUTRAL",
                error=f"Class diversity error: {e}",
                training_time=time() - start_time,
            )
        except Exception as e:
            return TrainingResult(
                symbol=symbol,
                success=False,
                confidence=0.0,
                direction="NEUTRAL",
                error=str(e),
                training_time=time() - start_time,
            )

    def filter_signals(self, signals: List[SignalResult]) -> List[SignalResult]:
        """
        Filter signals by training XGBoost models and confirming ATC direction.

        Args:
            signals: List of SignalResult objects from ATC Scanner

        Returns:
            List of filtered SignalResult objects that passed validation
        """
        if not signals:
            log_info("No signals to filter")
            return []

        log_info(f"XGBoost Per-Symbol Filter: Processing {len(signals)} signals...")

        filtered_signals: List[SignalResult] = []
        training_results: Dict[str, TrainingResult] = {}

        # Train models in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_symbol = {
                executor.submit(self._train_and_predict_symbol, signal.symbol): signal
                for signal in signals
            }

            for future in as_completed(future_to_symbol):
                signal = future_to_symbol[future]
                try:
                    result = future.result()
                    training_results[signal.symbol] = result
                    self._stats["total_processed"] += 1

                    if result.success:
                        self._stats["training_success"] += 1
                        log_debug(
                            f"XGBoost trained for {signal.symbol}: "
                            f"{result.direction} ({result.confidence:.2%}) "
                            f"in {result.training_time:.1f}s"
                        )
                    else:
                        # Check if it was skipped due to class imbalance
                        if result.error and "class imbalance" in result.error.lower():
                            self._stats["skipped_imbalanced"] += 1
                        else:
                            self._stats["training_failed"] += 1
                        log_warn(
                            f"XGBoost training failed for {signal.symbol}: {result.error}"
                        )

                except Exception as e:
                    self._stats["training_failed"] += 1
                    log_error(f"Error processing {signal.symbol}: {e}")
                    training_results[signal.symbol] = TrainingResult(
                        symbol=signal.symbol,
                        success=False,
                        confidence=0.0,
                        direction="NEUTRAL",
                        error=str(e),
                    )

        # Validate signals against training results
        for signal in signals:
            training_result = training_results.get(signal.symbol)
            if training_result is None:
                continue

            atc_type = signal.signal_type  # LONG or SHORT

            # Check if XGBoost confirms ATC direction
            confirmed = False
            if training_result.success and training_result.confidence >= self.min_confidence:
                if atc_type == "LONG" and training_result.direction == "UP":
                    confirmed = True
                elif atc_type == "SHORT" and training_result.direction == "DOWN":
                    confirmed = True

            if confirmed:
                self._stats["confirmed"] += 1

                # Add XGBoost details to signal
                new_details = signal.details.copy()
                new_details["xgboost_conf"] = training_result.confidence
                new_details["xgboost_dir"] = training_result.direction
                new_details["xgboost_validated"] = True
                new_details["xgboost_training_time"] = training_result.training_time

                filtered_signals.append(
                    SignalResult(
                        symbol=signal.symbol,
                        score=signal.score,
                        signal_type=signal.signal_type,
                        details=new_details,
                        strengths=signal.strengths,
                    )
                )
                log_debug(
                    f"XGBoost CONFIRMED {signal.symbol} ({atc_type}): "
                    f"Model predicted {training_result.direction} ({training_result.confidence:.2%})"
                )
            else:
                self._stats["rejected"] += 1
                reason = "training failed" if not training_result.success else (
                    f"direction mismatch ({training_result.direction})"
                    if training_result.confidence >= self.min_confidence
                    else f"low confidence ({training_result.confidence:.2%})"
                )

                log_debug(
                    f"XGBoost REJECTED {signal.symbol} ({atc_type}): {reason}"
                )

                # Handle error policy
                if self.on_error == "pass" and not training_result.success:
                    filtered_signals.append(signal)

        log_info(
            f"XGBoost Per-Symbol Filter: {len(signals)} -> {len(filtered_signals)} signals passed "
            f"(confirmed: {self._stats['confirmed']}, rejected: {self._stats['rejected']}, "
            f"skipped_imbalanced: {self._stats['skipped_imbalanced']}, "
            f"failed: {self._stats['training_failed']})"
        )

        return filtered_signals

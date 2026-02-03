"""
XGBoost Signal Filter

Responsible for:
- Validating signals using a pre-trained XGBoost model.
- Fetching historical data and computing features for candidates.
- Filtering signals based on model confidence threshold.

Usage:
    # Initialize filter
    filter = XGBoostFilter(
        data_fetcher=data_fetcher,
        model_path="models/xgboost_model.joblib",
        config={
            "min_confidence": 0.6,
            "prediction_timeframe": "5m",
            "on_error": "drop",
            "require_model": True
        }
    )

    # Filter signals
    atc_signals = [SignalResult("BTC/USDT", 0.9, "LONG", {})]
    validated_signals = filter.filter_signals(atc_signals)
"""

import hashlib
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import joblib
import numpy as np

from config import XGBOOST_FILTER_DEFAULTS
from modules.auto_trade.core.atc_scanner import SignalResult
from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.indicator_engine import (
    IndicatorConfig,
    IndicatorEngine,
    IndicatorProfile,
)
from modules.common.ui.logging import log_debug, log_error, log_info, log_warn
from modules.xgboost_LTS.core.model import predict_next_move
from modules.xgboost_LTS.utils.features import add_advanced_features


class XGBoostFilterConfig(TypedDict, total=False):
    """Configuration for XGBoostFilter."""

    min_confidence: float  # Minimum confidence threshold (0.0 to 1.0)
    history_limit: int  # Number of historical candles to fetch
    prediction_timeframe: str  # Timeframe for prediction ("5m", "15m", "1h", etc.)
    on_error: str  # Error handling policy: "drop", "pass", or "neutral"
    model_hash: str  # SHA256 hash for model integrity verification
    min_required_candles: int  # Minimum candles required for prediction
    cache_ttl: int  # Cache time-to-live in seconds
    require_model: bool  # Whether to fail-fast if model doesn't load
    max_consecutive_failures: int  # Circuit breaker threshold
    prob_sum_tolerance: float  # Tolerance for probability sum validation
    min_confidence_delta: float  # Minimum delta between prediction classes


class XGBoostFilter:
    """Filters signals using a pre-trained XGBoost model.

    This filter validates ATC signals using machine learning predictions,
    adding a second layer of confirmation to reduce false positives.
    """

    def __init__(
        self,
        data_fetcher: DataFetcher,
        model_path: str,
        config: Optional[XGBoostFilterConfig] = None,
    ):
        """
        Initialize XGBoostFilter.

        Args:
            data_fetcher: DataFetcher instance for market data
            model_path: Path to the pre-trained .joblib model file
            config: Configuration dictionary

        Raises:
            ValueError: If configuration parameters are invalid
            RuntimeError: If model fails to load and require_model=True
        """
        self.data_fetcher = data_fetcher
        self.model_path = model_path

        # Merge with defaults
        self.config: XGBoostFilterConfig = {**XGBOOST_FILTER_DEFAULTS, **(config or {})}

        # Validate and extract configuration
        self._validate_config()

        # Initialize Indicator Engine for XGBoost features
        self.indicator_engine = IndicatorEngine(
            IndicatorConfig.for_profile(IndicatorProfile.XGBOOST)
        )

        # Cache for predictions with timestamps (symbol -> (confidence, direction, timestamp))
        self._prediction_cache: Dict[str, Tuple[float, str, float]] = {}

        # Circuit breaker for feature computation failures
        self._feature_failure_count: Dict[str, int] = {}

        # Load Model with validation
        self.model = self._load_model()

        # Fail-fast if model is required but failed to load
        if not self.model and self.require_model:
            raise RuntimeError(
                f"XGBoost model failed to load from {model_path} and require_model=True. "
                "Cannot proceed without a valid model."
            )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If any configuration parameter is invalid
        """
        # Min confidence
        self.min_confidence = self.config.get("min_confidence", XGBOOST_FILTER_DEFAULTS["min_confidence"])
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError(
                f"min_confidence must be between 0 and 1, got {self.min_confidence}"
            )

        # History limit
        self.history_limit = self.config.get("history_limit", XGBOOST_FILTER_DEFAULTS["history_limit"])
        if self.history_limit <= 0:
            raise ValueError(
                f"history_limit must be positive, got {self.history_limit}"
            )

        # Prediction timeframe
        self.prediction_timeframe = self.config.get(
            "prediction_timeframe", XGBOOST_FILTER_DEFAULTS["prediction_timeframe"]
        )
        valid_timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        if self.prediction_timeframe not in valid_timeframes:
            raise ValueError(
                f"Invalid prediction_timeframe: {self.prediction_timeframe}. "
                f"Must be one of {valid_timeframes}"
            )

        # Error handling policy
        self.on_error = self.config.get("on_error", XGBOOST_FILTER_DEFAULTS["on_error"])
        if self.on_error not in ["drop", "pass", "neutral"]:
            raise ValueError(
                f"on_error must be 'drop', 'pass', or 'neutral', got {self.on_error}"
            )

        # Minimum required candles
        self.min_required_candles = self.config.get(
            "min_required_candles", XGBOOST_FILTER_DEFAULTS["min_required_candles"]
        )

        # Cache TTL
        self.cache_ttl = self.config.get("cache_ttl", XGBOOST_FILTER_DEFAULTS["cache_ttl"])

        # Require model
        self.require_model = self.config.get("require_model", XGBOOST_FILTER_DEFAULTS["require_model"])

        # Max consecutive failures (circuit breaker)
        self.max_consecutive_failures = self.config.get(
            "max_consecutive_failures", XGBOOST_FILTER_DEFAULTS["max_consecutive_failures"]
        )

        # Probability sum tolerance
        self.prob_sum_tolerance = self.config.get(
            "prob_sum_tolerance", XGBOOST_FILTER_DEFAULTS["prob_sum_tolerance"]
        )

        # Minimum confidence delta
        self.min_confidence_delta = self.config.get(
            "min_confidence_delta", XGBOOST_FILTER_DEFAULTS["min_confidence_delta"]
        )

    def _validate_model_integrity(self, path: Path) -> bool:
        """Validate model file hasn't been tampered with.

        Args:
            path: Path to model file

        Returns:
            True if validation passes or no hash configured, False if mismatch
        """
        expected_hash = self.config.get("model_hash")
        if not expected_hash:
            log_warn(
                "No model_hash configured - skipping integrity check. "
                "Consider adding model_hash to config for security."
            )
            return True

        try:
            with open(path, "rb") as f:
                actual_hash = hashlib.sha256(f.read()).hexdigest()

            if actual_hash != expected_hash:
                log_error(
                    f"Model integrity check failed!\n"
                    f"Expected: {expected_hash}\n"
                    f"Got:      {actual_hash}\n"
                    f"Possible tampering detected."
                )
                return False

            log_info("Model integrity verified ✓")
            return True

        except Exception as e:
            log_error(f"Error during integrity check: {e}")
            return False

    def _load_model(self) -> Optional[Any]:
        """Load and validate the XGBoost model from disk.

        Returns:
            Loaded XGBoost model or None if loading/validation fails
        """
        path = Path(self.model_path)
        if not path.exists():
            log_error(f"XGBoost model not found at {path}")
            return None

        # Validate model integrity
        if not self._validate_model_integrity(path):
            log_error("Model integrity check failed - refusing to load")
            return None

        try:
            log_info(f"Loading XGBoost model from {path}...")
            model = joblib.load(path)

            # Validate it's a proper classifier model
            if not hasattr(model, "predict_proba"):
                log_error("Loaded object is not a valid classifier model")
                return None

            # CRITICAL: Validate model has 3 classes (DOWN/NEUTRAL/UP)
            if hasattr(model, "n_classes_"):
                log_info(f"Model classes: {model.n_classes_}")
                if model.n_classes_ != 3:
                    log_error(
                        f"Model has {model.n_classes_} classes, expected 3 (DOWN/NEUTRAL/UP). "
                        "Refusing to load incompatible model."
                    )
                    return None

            if hasattr(model, "feature_names_in_"):
                log_info(f"Model expects {len(model.feature_names_in_)} features")
            elif hasattr(model, "n_features_in_"):
                log_info(f"Model expects {model.n_features_in_} features")

            log_info("Successfully loaded and validated XGBoost model ✓")
            return model

        except Exception as e:
            log_error(f"Failed to load XGBoost model: {e}")
            return None

    def clear_cache(self) -> None:
        """Clear the prediction cache.

        Useful for testing or when you want fresh predictions.
        """
        self._prediction_cache.clear()
        log_debug("Prediction cache cleared")

    def _get_cached_prediction(self, symbol: str) -> Optional[Tuple[float, str]]:
        """Get cached prediction if valid.

        Args:
            symbol: Trading pair symbol

        Returns:
            Tuple of (confidence, direction) if cache is valid, None otherwise
        """
        if symbol in self._prediction_cache:
            confidence, direction, timestamp = self._prediction_cache[symbol]
            if time() - timestamp < self.cache_ttl:
                log_debug(f"Using cached prediction for {symbol} (age: {time() - timestamp:.1f}s)")
                return confidence, direction
            else:
                log_debug(f"Cache expired for {symbol} (age: {time() - timestamp:.1f}s > {self.cache_ttl}s)")
                del self._prediction_cache[symbol]
        return None

    def filter_signals(self, signals: List[SignalResult]) -> List[SignalResult]:
        """
        Filter signals based on XGBoost model confidence.

        Uses cached predictions for duplicate symbols to avoid redundant computation.

        Args:
            signals: List of SignalResult objects from ATC Scanner

        Returns:
            List of filtered SignalResult objects that passed validation

        Raises:
            RuntimeError: If model is not loaded and require_model=True
        """
        if not self.model:
            error_msg = "XGBoost model not loaded. Filter is non-functional."
            if self.require_model:
                log_error(error_msg)
                raise RuntimeError(error_msg)
            else:
                log_warn(f"{error_msg} Returning all signals (require_model=False).")
                return signals

        if not signals:
            log_info("No signals to filter")
            return []

        filtered_signals = []
        rejected_count = 0
        error_count = 0

        for signal in signals:
            try:
                # Use cached prediction if available and valid
                cached = self._get_cached_prediction(signal.symbol)
                if cached:
                    confidence, direction = cached
                else:
                    confidence, direction = self._predict_signal(signal.symbol)
                    self._prediction_cache[signal.symbol] = (confidence, direction, time())

                # Check if model agrees with ATC signal
                atc_type = signal.signal_type  # LONG or SHORT

                model_confirms = False
                if (
                    atc_type == "LONG"
                    and direction == "UP"
                    and confidence >= self.min_confidence
                ):
                    model_confirms = True
                elif (
                    atc_type == "SHORT"
                    and direction == "DOWN"
                    and confidence >= self.min_confidence
                ):
                    model_confirms = True

                if model_confirms:
                    # Add model confidence to details (store as float, not string)
                    new_details = signal.details.copy()
                    new_details["xgboost_conf"] = confidence  # Store as float
                    new_details["xgboost_dir"] = direction
                    new_details["xgboost_validated"] = True  # Store as bool

                    filtered_signals.append(
                        SignalResult(
                            symbol=signal.symbol,
                            score=signal.score,
                            signal_type=signal.signal_type,
                            details=new_details,
                        )
                    )
                else:
                    rejected_count += 1
                    log_debug(
                        f"XGBoost rejected {signal.symbol} ({atc_type}): "
                        f"Model predicted {direction} (conf: {confidence:.2f}, "
                        f"threshold: {self.min_confidence:.2f})"
                    )

            except Exception as e:
                error_count += 1
                log_error(f"Error filtering {signal.symbol}: {e}")

                # Apply error handling policy
                if self.on_error == "pass":
                    # Include original signal without XGBoost validation
                    log_debug(f"Passing {signal.symbol} through despite error (on_error=pass)")
                    filtered_signals.append(signal)
                elif self.on_error == "neutral":
                    # Mark as neutral/uncertain
                    new_details = signal.details.copy()
                    new_details["xgboost_error"] = str(e)
                    log_debug(f"Marking {signal.symbol} as NEUTRAL due to error (on_error=neutral)")
                    filtered_signals.append(
                        SignalResult(
                            symbol=signal.symbol,
                            score=0.0,
                            signal_type="NEUTRAL",
                            details=new_details,
                        )
                    )
                # else: "drop" - do nothing (current behavior)

        log_info(
            f"XGBoost Filter: {len(signals)} -> {len(filtered_signals)} signals passed "
            f"({rejected_count} rejected, {error_count} errors)"
        )
        return filtered_signals

    def _predict_signal(self, symbol: str) -> Tuple[float, str]:
        """
        Predict direction for a symbol using XGBoost model.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")

        Returns:
            Tuple of (confidence, direction):
                - confidence: probability of the predicted class (0.0 to 1.0)
                - direction: "UP", "DOWN", or "NEUTRAL"

        Raises:
            Exception: If data fetching or feature computation fails
        """
        # 1. Fetch Data
        df = self.data_fetcher.fetch_ohlcv(
            symbol,
            timeframe=self.prediction_timeframe,
            limit=self.history_limit,
        )

        if df is None or df.empty:
            log_warn(f"No data available for {symbol}")
            return 0.0, "NEUTRAL"

        # Check if we have sufficient data
        if len(df) < self.min_required_candles:
            log_warn(
                f"Insufficient data for {symbol}: {len(df)}/{self.min_required_candles} candles. "
                f"Need at least {self.min_required_candles} for reliable predictions."
            )
            return 0.0, "NEUTRAL"

        # 2. Compute Features with circuit breaker
        try:
            # a. Standard Indicators
            df = self.indicator_engine.compute_features(df)
            if df is None or df.empty:
                # Track failure
                self._feature_failure_count[symbol] = self._feature_failure_count.get(symbol, 0) + 1

                if self._feature_failure_count[symbol] >= self.max_consecutive_failures:
                    log_error(
                        f"Feature computation failed {self.max_consecutive_failures} times "
                        f"consecutively for {symbol}. Possible data quality issue."
                    )
                else:
                    log_error(f"Feature computation failed for {symbol} "
                             f"({self._feature_failure_count[symbol]}/{self.max_consecutive_failures})")
                return 0.0, "NEUTRAL"

            # b. Advanced/Rust Features
            df = add_advanced_features(df)
            if df is None or df.empty:
                # Track failure
                self._feature_failure_count[symbol] = self._feature_failure_count.get(symbol, 0) + 1

                if self._feature_failure_count[symbol] >= self.max_consecutive_failures:
                    log_error(
                        f"Advanced feature computation failed {self.max_consecutive_failures} times "
                        f"consecutively for {symbol}. Possible data quality issue."
                    )
                else:
                    log_error(f"Advanced feature computation failed for {symbol} "
                             f"({self._feature_failure_count[symbol]}/{self.max_consecutive_failures})")
                return 0.0, "NEUTRAL"

            # Reset failure count on success
            self._feature_failure_count[symbol] = 0

        except Exception as e:
            # Track failure
            self._feature_failure_count[symbol] = self._feature_failure_count.get(symbol, 0) + 1
            log_error(f"Error computing features for {symbol}: {e} "
                     f"({self._feature_failure_count[symbol]}/{self.max_consecutive_failures})")
            return 0.0, "NEUTRAL"

        # 3. Predict on last row
        try:
            last_row = df.iloc[-1:]

            # predict_next_move returns [prob_down, prob_neutral, prob_up]
            probs = predict_next_move(self.model, last_row)

            # Validate prediction format
            if not isinstance(probs, (list, np.ndarray)) or len(probs) != 3:
                log_error(
                    f"Invalid prediction format for {symbol}: "
                    f"expected 3 probabilities, got {probs}"
                )
                return 0.0, "NEUTRAL"

            prob_down = float(probs[0])
            prob_neutral = float(probs[1])
            prob_up = float(probs[2])

            # Validate probabilities sum to ~1.0 (tighter tolerance)
            prob_sum = prob_down + prob_neutral + prob_up
            tolerance = self.prob_sum_tolerance
            if not (1.0 - tolerance <= prob_sum <= 1.0 + tolerance):
                log_warn(
                    f"Probabilities don't sum to ~1.0 for {symbol}: {prob_sum:.4f} "
                    f"[DOWN={prob_down:.4f}, NEUTRAL={prob_neutral:.4f}, UP={prob_up:.4f}]"
                )
                # Normalize probabilities
                norm_factor = 1.0 / prob_sum
                prob_down *= norm_factor
                prob_neutral *= norm_factor
                prob_up *= norm_factor
                log_debug(f"Normalized probabilities for {symbol}: "
                         f"[DOWN={prob_down:.4f}, NEUTRAL={prob_neutral:.4f}, UP={prob_up:.4f}]")

            # Determine direction with minimum confidence delta
            max_prob = max(prob_up, prob_down, prob_neutral)
            second_max = sorted([prob_up, prob_down, prob_neutral])[-2]
            confidence_delta = max_prob - second_max

            # If the delta is too small, it's uncertain
            if confidence_delta < self.min_confidence_delta:
                log_debug(
                    f"Uncertain prediction for {symbol}: max_prob={max_prob:.4f}, "
                    f"delta={confidence_delta:.4f} < threshold={self.min_confidence_delta}"
                )
                return max_prob, "NEUTRAL"

            # Determine direction based on highest probability
            if max_prob == prob_up:
                return prob_up, "UP"
            elif max_prob == prob_down:
                return prob_down, "DOWN"
            else:
                return prob_neutral, "NEUTRAL"

        except Exception as e:
            log_error(f"Error during prediction for {symbol}: {e}")
            return 0.0, "NEUTRAL"

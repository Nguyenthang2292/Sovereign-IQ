"""
XGBoost Signal Filter

Responsible for:
- Validating signals using a pre-trained XGBoost model.
- Fetching historical data and computing features for candidates.
- Filtering signals based on model confidence threshold.
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict

import joblib
import numpy as np

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
        """
        self.data_fetcher = data_fetcher
        self.model_path = model_path
        self.config: XGBoostFilterConfig = config or {}

        # Configuration with validation
        self.min_confidence = self.config.get("min_confidence", 0.6)
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError(
                f"min_confidence must be between 0 and 1, got {self.min_confidence}"
            )

        self.history_limit = self.config.get(
            "history_limit", 1500
        )  # Need enough for Lag-3 of SMA-200 etc.
        if self.history_limit <= 0:
            raise ValueError(
                f"history_limit must be positive, got {self.history_limit}"
            )

        # Configurable prediction timeframe
        self.prediction_timeframe = self.config.get("prediction_timeframe", "5m")
        valid_timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        if self.prediction_timeframe not in valid_timeframes:
            raise ValueError(
                f"Invalid prediction_timeframe: {self.prediction_timeframe}. "
                f"Must be one of {valid_timeframes}"
            )

        # Error handling policy
        self.on_error = self.config.get("on_error", "drop")
        if self.on_error not in ["drop", "pass", "neutral"]:
            raise ValueError(
                f"on_error must be 'drop', 'pass', or 'neutral', got {self.on_error}"
            )

        # Minimum required candles for prediction
        self.min_required_candles = self.config.get("min_required_candles", 250)

        # Initialize Indicator Engine for XGBoost features
        self.indicator_engine = IndicatorEngine(
            IndicatorConfig.for_profile(IndicatorProfile.XGBOOST)
        )

        # Load Model with validation
        self.model = self._load_model()

        # Cache for predictions (symbol -> (confidence, direction))
        self._prediction_cache: Dict[str, Tuple[float, str]] = {}

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

    def _load_model(self):
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

            # Log model information
            if hasattr(model, "n_classes_"):
                log_info(f"Model classes: {model.n_classes_}")
                if model.n_classes_ != 3:
                    log_warn(
                        f"Model has {model.n_classes_} classes, expected 3 (DOWN/NEUTRAL/UP)"
                    )

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

    def filter_signals(self, signals: List[SignalResult]) -> List[SignalResult]:
        """
        Filter signals based on XGBoost model confidence.

        Uses cached predictions for duplicate symbols to avoid redundant computation.

        Args:
            signals: List of SignalResult objects from ATC Scanner

        Returns:
            List of filtered SignalResult objects that passed validation
        """
        if not self.model:
            log_warn("XGBoost model not loaded. Skipping filter (returning all signals).")
            return signals

        if not signals:
            log_info("No signals to filter")
            return []

        filtered_signals = []
        rejected_count = 0
        error_count = 0

        for signal in signals:
            try:
                # Use cached prediction if available
                if signal.symbol in self._prediction_cache:
                    confidence, direction = self._prediction_cache[signal.symbol]
                    log_debug(f"Using cached prediction for {signal.symbol}")
                else:
                    confidence, direction = self._predict_signal(signal.symbol)
                    self._prediction_cache[signal.symbol] = (confidence, direction)

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
                    # Add model confidence to details
                    new_details = signal.details.copy()
                    new_details["xgboost_conf"] = f"{confidence:.2f}"
                    new_details["xgboost_dir"] = direction
                    new_details["xgboost_validated"] = "true"

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

        # 2. Compute Features
        try:
            # a. Standard Indicators
            df = self.indicator_engine.compute_features(df)
            if df is None or df.empty:
                log_error(f"Feature computation failed for {symbol}")
                return 0.0, "NEUTRAL"

            # b. Advanced/Rust Features
            df = add_advanced_features(df)
            if df is None or df.empty:
                log_error(f"Advanced feature computation failed for {symbol}")
                return 0.0, "NEUTRAL"

        except Exception as e:
            log_error(f"Error computing features for {symbol}: {e}")
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

            # Validate probabilities
            prob_sum = prob_down + prob_neutral + prob_up
            if not (0.95 <= prob_sum <= 1.05):
                log_warn(
                    f"Probabilities don't sum to ~1.0 for {symbol}: {prob_sum:.3f} "
                    f"({prob_down:.3f}, {prob_neutral:.3f}, {prob_up:.3f})"
                )

            # Determine direction based on highest probability
            if prob_up > prob_down and prob_up > prob_neutral:
                return prob_up, "UP"
            elif prob_down > prob_up and prob_down > prob_neutral:
                return prob_down, "DOWN"
            else:
                return prob_neutral, "NEUTRAL"

        except Exception as e:
            log_error(f"Error during prediction for {symbol}: {e}")
            return 0.0, "NEUTRAL"


from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
from pomegranate.hmm import DenseHMM
from config import (
from pomegranate.hmm import DenseHMM

"""
HMM-Swings Model Definitions.

This module contains the HMM_SWINGS dataclass, constants, and the SwingsHMM class.
"""



    HMM_HIGH_ORDER_ORDERS_ARGRELEXTREMA_DEFAULT,
    HMM_HIGH_ORDER_STRICT_MODE_DEFAULT,
    HMM_HIGH_ORDER_USE_DATA_DRIVEN_INIT,
)
from modules.common.utils import log_error, log_warn

# Constants
BULLISH, NEUTRAL, BEARISH = 1, 0, -1


@dataclass
class HMM_SWINGS:
    """Result dataclass for HMM-Swings analysis."""

    next_state_with_high_order_hmm: Literal[-1, 0, 1]
    next_state_duration: int
    next_state_probability: float


class SwingsHMM:
    """
    Swings Hidden Markov Model for market state prediction.

    Encapsulates HMM model creation, training, and prediction logic with
    data-driven initialization and optimized state selection.
    """

    def __init__(
        self,
        orders_argrelextrema: Optional[int] = None,
        strict_mode: Optional[bool] = None,
        use_data_driven: bool = True,
        train_ratio: float = 0.8,
        min_states: int = 2,
        max_states: int = 10,
        n_folds: int = 3,
        use_bic: bool = True,
    ):
        """
        Initialize Swings HMM analyzer.

        Args:
            orders_argrelextrema: Order parameter for swing detection (default: from config)
            strict_mode: Whether to use strict mode for swing-to-state conversion (default: from config)
            use_data_driven: Use data-driven initialization for HMM parameters
            train_ratio: Ratio of data to use for training
            min_states: Minimum number of hidden states for optimization
            max_states: Maximum number of hidden states for optimization
            n_folds: Number of folds for cross-validation
            use_bic: Use BIC for model selection (default: True)
        """
        self.orders_argrelextrema = (
            orders_argrelextrema if orders_argrelextrema is not None else HMM_HIGH_ORDER_ORDERS_ARGRELEXTREMA_DEFAULT
        )
        self.strict_mode = strict_mode if strict_mode is not None else HMM_HIGH_ORDER_STRICT_MODE_DEFAULT
        self.use_data_driven = use_data_driven
        self.train_ratio = train_ratio
        self.min_states = min_states
        self.max_states = max_states
        self.n_folds = n_folds
        self.use_bic = use_bic

        # Model state
        self.model: Optional[DenseHMM] = None
        self.optimal_n_states: Optional[int] = None
        self.swing_highs_info: Optional[pd.DataFrame] = None
        self.swing_lows_info: Optional[pd.DataFrame] = None
        self.states: Optional[List[float]] = None
        self.train_states: Optional[List[float]] = None
        self.test_states: Optional[List[float]] = None

    def _validate_dataframe(self, df: pd.DataFrame) -> bool:
        """Validate input DataFrame."""
        required_columns = ["open", "high", "low", "close"]
        if df is None or df.empty or not all(col in df.columns for col in required_columns):
            log_error("Invalid dataframe provided - missing required columns")
            return False

        try:
            for col in required_columns:
                pd.to_numeric(df[col], errors="raise")
        except (ValueError, TypeError):
            log_error("Invalid dataframe provided - non-numeric data detected")
            return False

        return True

    def _determine_interval(self, df: pd.DataFrame) -> str:
        """Determine data interval from DataFrame index."""
        if len(df) > 1 and isinstance(df.index, pd.DatetimeIndex):
            time_diff = df.index[1] - df.index[0]
            total_minutes = int(time_diff.total_seconds() / 60)
            return f"h{total_minutes // 60}" if total_minutes % 60 == 0 else f"m{total_minutes}"
        elif len(df) > 1:
            log_warn("DataFrame index is not DatetimeIndex. Using default interval.")
            return "h1"  # Default to 1 hour
        return "h1"

    def detect_swings(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Detect swing highs and lows from price data.

        Args:
            df: DataFrame with price data

        Returns:
            Tuple of (swing_highs_info, swing_lows_info) DataFrames
        """
        from scipy.signal import argrelextrema

        swing_highs = argrelextrema(df["high"].values, np.greater, order=self.orders_argrelextrema)[0]
        swing_lows = argrelextrema(df["low"].values, np.less, order=self.orders_argrelextrema)[0]

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            log_warn("Not enough swing points detected for reliable prediction")
            return pd.DataFrame(), pd.DataFrame()

        swing_highs_info = df.iloc[swing_highs][["open", "high", "low", "close"]]
        swing_lows_info = df.iloc[swing_lows][["open", "high", "low", "close"]]

        return swing_highs_info, swing_lows_info

    def convert_to_states(self, swing_highs_info: pd.DataFrame, swing_lows_info: pd.DataFrame) -> List[float]:
        """
        Convert swing points to state sequence.

        Args:
            swing_highs_info: DataFrame with swing highs
            swing_lows_info: DataFrame with swing lows

        Returns:
            List of state values (0, 1, or 2)
        """
        from modules.hmm.core.swings.state_conversion import convert_swing_to_state

        return convert_swing_to_state(swing_highs_info, swing_lows_info, strict_mode=self.strict_mode)

    def optimize_and_create_model(self, train_states: List[float]) -> DenseHMM:
        """
        Optimize number of states and create HMM model.

        Args:
            train_states: Training state sequence

        Returns:
            Trained HMM model
        """
        from modules.hmm.core.swings.model_creation import create_hmm_model, train_model
        from modules.hmm.core.swings.optimization import optimize_n_states

        train_observations = [np.array(train_states).reshape(-1, 1)]

        # Optimize number of states
        try:
            self.optimal_n_states = optimize_n_states(
                train_observations,
                min_states=self.min_states,
                max_states=self.max_states,
                n_folds=self.n_folds,
                use_bic=self.use_bic,
            )
        except Exception as e:
            log_warn(f"State optimization failed: {e}. Using default n_states=2.")
            self.optimal_n_states = 2

        # Create model with data-driven initialization
        model = create_hmm_model(
            n_symbols=3,
            n_states=self.optimal_n_states,
            states_data=train_states if self.use_data_driven else None,
            use_data_driven=self.use_data_driven and HMM_HIGH_ORDER_USE_DATA_DRIVEN_INIT,
        )

        # Train model
        model = train_model(model, train_observations)

        return model

    def predict_next_state(self, model: DenseHMM, states: List[float]) -> Tuple[int, float]:
        """
        Predict next state from current state sequence.

        Args:
            model: Trained HMM model
            states: Current state sequence

        Returns:
            Tuple of (predicted_state_index, probability)
        """
        from modules.hmm.core.swings.prediction import predict_next_observation

        full_observations = [np.array(states).reshape(-1, 1)]
        next_obs_proba = predict_next_observation(model, full_observations)
        next_obs_proba = np.nan_to_num(next_obs_proba, nan=1 / 3, posinf=1 / 3, neginf=1 / 3)

        if not np.isfinite(next_obs_proba).all() or np.sum(next_obs_proba) == 0:
            return 1, 0.33  # Default to NEUTRAL

        max_index = int(np.argmax(next_obs_proba))
        max_value = float(next_obs_proba[max_index])

        return max_index, max_value

    def _calculate_duration(
        self, swing_highs_info: pd.DataFrame, swing_lows_info: pd.DataFrame, interval_str: str
    ) -> int:
        """Calculate predicted state duration."""
        from modules.hmm.core.swings.swing_utils import average_swing_distance

        if isinstance(swing_highs_info.index, pd.DatetimeIndex) and isinstance(swing_lows_info.index, pd.DatetimeIndex):
            average_distance = average_swing_distance(swing_highs_info, swing_lows_info) or 3600
        else:
            log_warn("Non-datetime index detected. Using default swing distance.")
            average_distance = 3600  # Default to 1 hour in seconds

        # Convert time units
        if interval_str.startswith("h"):
            converted_distance = average_distance / 3600  # to hours
        elif interval_str.startswith("m"):
            converted_distance = average_distance / 60  # to minutes
        else:
            converted_distance = average_distance

        return int(converted_distance)

    def analyze(
        self,
        df: pd.DataFrame,
        eval_mode: bool = True,
    ) -> HMM_SWINGS:
        """
        Main analysis pipeline: detect swings, convert to states, train model, and predict.

        Args:
            df: DataFrame containing price data
            eval_mode: If True, evaluates model performance on test set

        Returns:
            HMM_SWINGS: Prediction result
        """
        from modules.hmm.core.swings.prediction import evaluate_model_accuracy

        # Validate input
        if not self._validate_dataframe(df):
            return HMM_SWINGS(
                next_state_with_high_order_hmm=NEUTRAL, next_state_duration=1, next_state_probability=0.33
            )

        # Determine interval
        interval_str = self._determine_interval(df)

        # Detect swings
        swing_highs_info, swing_lows_info = self.detect_swings(df)
        if swing_highs_info.empty or swing_lows_info.empty:
            return HMM_SWINGS(
                next_state_with_high_order_hmm=NEUTRAL, next_state_duration=1, next_state_probability=0.33
            )

        self.swing_highs_info = swing_highs_info
        self.swing_lows_info = swing_lows_info

        # Convert to states
        states = self.convert_to_states(swing_highs_info, swing_lows_info)
        if not states:
            log_warn("No states detected from swing points")
            return HMM_SWINGS(
                next_state_with_high_order_hmm=NEUTRAL, next_state_duration=1, next_state_probability=0.33
            )

        self.states = states

        # Split data
        train_size = int(len(states) * self.train_ratio)
        if train_size < 2:
            train_states, test_states = states, []
        else:
            train_states, test_states = states[:train_size], states[train_size:]

        self.train_states = train_states
        self.test_states = test_states

        # Build and train model
        model = self.optimize_and_create_model(train_states)
        self.model = model

        # Evaluate model if requested
        accuracy = evaluate_model_accuracy(model, train_states, test_states) if eval_mode and test_states else 0.0

        # Calculate duration
        duration = self._calculate_duration(swing_highs_info, swing_lows_info, interval_str)

        # Return NEUTRAL if accuracy is too low
        # Threshold increased from 0.3 to 0.33 to ensure higher model quality
        if accuracy <= 0.33:
            return HMM_SWINGS(
                next_state_with_high_order_hmm=NEUTRAL,
                next_state_duration=duration,
                next_state_probability=max(accuracy, 0.33),
            )

        # Predict next state
        max_index, max_value = self.predict_next_state(model, states)

        # Map index to signal
        signal_map = {0: BEARISH, 1: NEUTRAL, 2: BULLISH}
        signal = signal_map.get(max_index, NEUTRAL)

        return HMM_SWINGS(
            next_state_with_high_order_hmm=signal, next_state_duration=duration, next_state_probability=max_value
        )

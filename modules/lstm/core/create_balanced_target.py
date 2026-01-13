"""
This module provides functionality to generate balanced target labels
(-1 for sell, 0 for neutral, 1 for buy) for LSTM training based on
future returns computed from price data. The primary function
`create_balanced_target` computes classification targets given
customizable thresholds for strong market moves and a neutral zone.

Functions:
    create_balanced_target(df, threshold, neutral_zone):
        Computes and assigns balanced classification targets to
        the input DataFrame based on predicted future returns and
        user-defined thresholds.
"""

import numpy as np

from config.lstm import (
    FUTURE_RETURN_SHIFT,
    NEUTRAL_ZONE_LSTM,
    TARGET_THRESHOLD_LSTM,
)
from modules.common.ui.logging import log_debug, log_error, log_warn


def create_balanced_target(df, threshold=TARGET_THRESHOLD_LSTM, neutral_zone=NEUTRAL_ZONE_LSTM):
    """
    Create balanced target labels (-1, 0, 1) for LSTM training based on future returns.

    Args:
        df: DataFrame with price data (lowercase column names)
        threshold: Strong movement threshold for buy/sell signals
        neutral_zone: Threshold for neutral classification

    Returns:
        DataFrame with 'Target' column containing class labels:
        1 = Strong Buy, -1 = Strong Sell, 0 = Neutral
    """
    try:
        # Validate input data
        if df.empty or "close" not in df.columns:
            log_warn("Input DataFrame is empty or missing close column")
            return None

        if neutral_zone >= threshold:
            log_warn(f"neutral_zone ({neutral_zone}) must be less than threshold ({threshold})")
            # Explicitly signal failure to avoid returning a df without 'Target'
            return None

        if len(df) < abs(FUTURE_RETURN_SHIFT) + 1:
            log_warn("Not enough data points for future return calculation")
            return None

        # Calculate future price movement as percentage
        future_return = df["close"].shift(-abs(FUTURE_RETURN_SHIFT)) / df["close"] - 1

        # Create clear target labels for strong movements
        conditions = [
            future_return > threshold,  # Strong upward movement
            future_return < -threshold,  # Strong downward movement
            abs(future_return) <= neutral_zone,  # Minimal movement (neutral)
        ]
        choices = [1, -1, 0]  # Buy, Sell, Hold

        df["Target"] = np.select(conditions, choices, default=np.nan)

        # Handle intermediate cases with probabilistic assignment for balance
        # Use a fixed RNG seed for reproducibility in training/testing
        # (42 chosen arbitrarily, typical for examples and reproducibility).
        # To allow user/configuration, make seed configurable via argument
        # or environment variable if needed.
        rng = np.random.default_rng(
            42
        )  # Local generator with fixed seed ensures deterministic results for label assignment
        intermediate_mask = (abs(future_return) > neutral_zone) & (abs(future_return) <= threshold)

        # Vectorized assignment for intermediate cases with bias toward direction
        if intermediate_mask.any():
            # Vectorized assignment: 70% chance of following direction, 30% neutral
            positive_mask = (future_return > 0) & intermediate_mask
            negative_mask = (future_return < 0) & intermediate_mask

            # Generate random values separately for positive and negative cases
            if positive_mask.any():
                positive_random = rng.random(positive_mask.sum())
                df.loc[positive_mask, "Target"] = np.where(positive_random > 0.3, 1, 0)

            if negative_mask.any():
                negative_random = rng.random(negative_mask.sum())
                df.loc[negative_mask, "Target"] = np.where(negative_random > 0.3, -1, 0)

        # Remove rows with undefined targets
        df = df.dropna(subset=["Target"])

        # Log class distribution for monitoring
        if "Target" in df.columns and not df.empty:
            target_counts = df["Target"].value_counts().sort_index()
            log_debug(f"Target distribution: {dict(target_counts)}")

        return df

    except Exception as e:
        log_error(f"Error in create_balanced_target: {e}")
        return None

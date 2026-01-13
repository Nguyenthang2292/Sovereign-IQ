"""
Direction metrics calculation for quantitative analysis.

This module provides classification metrics calculation for directional prediction
evaluation. It can be used for any directional prediction task, not just pairs trading.

The main function `calculate_direction_metrics` evaluates how well z-score-based
signals predict future spread direction using standard classification metrics:
accuracy, precision, recall, and F1-score.

Key Features:
- Only evaluates active signals (when z-score exceeds threshold)
- Handles edge cases: NaN values, constant spreads, insufficient data
- Calculates macro-averaged metrics for Long and Short signals separately
- Validates all metrics are in valid range [0, 1]

Edge Cases Handled:
- None or invalid input types
- Insufficient data points
- Constant spread (std = 0)
- All actual directions are zero (no movement)
- Insufficient active signals
- Invalid parameters (zscore_lookback <= 0, classification_zscore <= 0)
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    from config import (
        PAIRS_TRADING_CLASSIFICATION_ZSCORE,
        PAIRS_TRADING_MIN_CLASSIFICATION_SAMPLES,
        PAIRS_TRADING_ZSCORE_LOOKBACK,
    )
except ImportError:
    PAIRS_TRADING_ZSCORE_LOOKBACK = 60
    PAIRS_TRADING_CLASSIFICATION_ZSCORE = 0.5
    PAIRS_TRADING_MIN_CLASSIFICATION_SAMPLES = 20


def calculate_direction_metrics(
    spread: pd.Series,
    zscore_lookback: int = PAIRS_TRADING_ZSCORE_LOOKBACK,
    classification_zscore: float = PAIRS_TRADING_CLASSIFICATION_ZSCORE,
) -> Dict[str, Optional[float]]:
    """
    Calculate classification metrics for spread direction prediction using z-score.

    Evaluates how well z-score predicts future spread direction for BOTH Long and Short signals.
    Metrics are calculated ONLY on active signals (when z-score exceeds threshold),
    ignoring neutral periods. This design focuses on evaluating signal quality when
    the model actually makes a prediction, rather than penalizing for neutral periods.

    Signal Logic:
    - **Long Signal**: zscore < -threshold → Predict UP (1).
      Correct if spread increases next period (actual = 1).
    - **Short Signal**: zscore > threshold → Predict DOWN (-1).
      Correct if spread decreases next period (actual = -1).
    - **Neutral**: -threshold <= zscore <= threshold → No prediction (0).
      Ignored in metrics calculation.

    Metrics Calculation:
    - **Precision**: Calculated separately for Long and Short, then macro-averaged.
      - Long Precision = TP / (TP + FP) = Correct Long predictions / All Long predictions
      - Short Precision = TP / (TP + FP) = Correct Short predictions / All Short predictions
      - Macro Precision = (Long Precision + Short Precision) / 2
      - If only one class exists, macro precision = that class's precision

    - **Recall**: Calculated separately for Long and Short, then macro-averaged.
      - Long Recall = TP / (TP + FN) = Correct Long predictions / All actual Long movements (in active set)
      - Short Recall = TP / (TP + FN) = Correct Short predictions / All actual Short movements (in active set)
      - Macro Recall = (Long Recall + Short Recall) / 2
      - Note: Recall is calculated only on active signals, not including neutral periods
      - If only one class exists, macro recall = that class's recall

    - **F1**: Harmonic mean of macro-averaged precision and recall.
      - F1 = 2 * (Precision * Recall) / (Precision + Recall)
      - Returns 0.0 if either precision or recall is 0

    - **Accuracy**: Overall accuracy = Correct predictions / Total active signals.
      - Note: actual_direction = 0 (unchanged) counts as wrong for both Long/Short

    Edge Cases:
    - Returns all None if: spread is None, not a Series, insufficient data,
      invalid parameters (zscore_lookback <= 0, classification_zscore <= 0),
      constant spread (std = 0), all actual directions are zero,
      or insufficient active signals (< PAIRS_TRADING_MIN_CLASSIFICATION_SAMPLES)

    Args:
        spread: Spread series (pd.Series). Typically price1 - hedge_ratio * price2.
               NaN values are automatically dropped before calculation.
        zscore_lookback: Rolling window size for z-score calculation. Must be > 0.
                        Default: 60 (from config or fallback).
        classification_zscore: Z-score threshold for signal generation. Must be > 0.
                              Default: 0.5 (from config or fallback).
                              Signals are generated when |zscore| > threshold.

    Returns:
        Dict[str, Optional[float]] with metrics (all in [0, 1] or None):
        - classification_accuracy: Overall accuracy of all active signals.
        - classification_precision: Macro-averaged precision (average of Long and Short precision).
        - classification_recall: Macro-averaged recall (average of Long and Short recall).
        - classification_f1: Harmonic mean of macro-averaged precision and recall.

        All values are guaranteed to be in [0, 1] if not None, or None if calculation
        cannot be performed due to insufficient data or edge cases.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> # Basic usage with mean-reverting spread
        >>> spread = pd.Series(np.sin(np.linspace(0, 20, 400)) + np.random.randn(400) * 0.1)
        >>> metrics = calculate_direction_metrics(spread)
        >>> print(f"Accuracy: {metrics['classification_accuracy']:.3f}")
        >>> print(f"F1 Score: {metrics['classification_f1']:.3f}")
        >>>
        >>> # Custom parameters
        >>> metrics = calculate_direction_metrics(
        ...     spread,
        ...     zscore_lookback=40,
        ...     classification_zscore=0.75
        ... )
        >>>
        >>> # Handle edge case: insufficient data
        >>> small_spread = pd.Series([0.1, -0.05, 0.15])
        >>> metrics = calculate_direction_metrics(small_spread)
        >>> assert metrics['classification_f1'] is None
    """
    result = {
        "classification_f1": None,
        "classification_precision": None,
        "classification_recall": None,
        "classification_accuracy": None,
    }

    if spread is None:
        return result

    if not isinstance(spread, pd.Series):
        return result

    if len(spread) < zscore_lookback:
        return result

    # Validate zscore_lookback
    if zscore_lookback <= 0:
        return result

    # Handle NaN values: drop NaN to ensure clean calculations
    # This ensures rolling window calculations are based on valid data only
    # Note: This may reduce the dataset size, so we re-check length requirements
    spread_clean = spread.dropna()

    # Check if we have enough valid data points after removing NaN
    # Need at least zscore_lookback points to calculate the first z-score
    if len(spread_clean) < zscore_lookback:
        return result

    # Validate classification_zscore
    if classification_zscore <= 0:
        return result

    # Calculate rolling z-score on clean data
    # Z-score = (value - rolling_mean) / rolling_std
    rolling_mean = spread_clean.rolling(zscore_lookback, min_periods=zscore_lookback).mean()
    rolling_std = spread_clean.rolling(zscore_lookback, min_periods=zscore_lookback).std().replace(0, np.nan)
    # Replace std=0 with NaN to avoid division by zero (constant spread case)
    # After division, drop NaN values (from std=0 or insufficient rolling window)
    zscore = ((spread_clean - rolling_mean) / rolling_std).dropna()

    # Validate zscore has enough valid values
    if zscore.empty or len(zscore) < 2:
        return result

    # Actual direction: Compare current spread to next period's spread
    # 1 if spread increases (mean-reversion: spread goes up after being low)
    # -1 if spread decreases (mean-reversion: spread goes down after being high)
    # 0 if spread unchanged
    # Note: shift(-1) looks forward, so we're predicting next period's direction
    future_return = spread_clean.shift(-1) - spread_clean
    actual_direction = np.sign(future_return).dropna()

    # Validate actual_direction has enough valid values
    if actual_direction.empty or len(actual_direction) < 2:
        return result

    # Edge case: If spread never changes (all future_return = 0), cannot calculate meaningful metrics
    if (actual_direction == 0).all():
        return result

    # Align indices: zscore and actual_direction may have different indices
    # due to different dropna() operations (zscore from std=0, actual_direction from shift)
    # We need common indices where both zscore and actual_direction are valid
    common_idx = zscore.index.intersection(actual_direction.index)

    # Validate we have enough common indices (need at least 2 for meaningful metrics)
    if len(common_idx) < 2:
        return result

    # Extract aligned data for calculation
    zscore = zscore.loc[common_idx]
    actual_direction = actual_direction.loc[common_idx]

    # Prediction logic (3-class system):
    # 1 (Long/Buy): zscore < -threshold → Spread is low, predict it will increase
    # -1 (Short/Sell): zscore > threshold → Spread is high, predict it will decrease
    # 0 (Neutral/Hold): -threshold <= zscore <= threshold → No clear signal
    threshold = classification_zscore
    predicted_signal = np.select([zscore < -threshold, zscore > threshold], [1.0, -1.0], default=0.0)

    # Filter only active signals (ignore Neutral 0s)
    # This design choice means we only evaluate when the model makes a prediction,
    # not penalizing for neutral periods where no signal is generated
    active_mask = predicted_signal != 0

    # Require minimum samples of ACTIVE SIGNALS for reliable metrics
    # If we have very few active signals (< MIN_CLASSIFICATION_SAMPLES),
    # metrics would be too noisy/unreliable, so return None
    if active_mask.sum() < PAIRS_TRADING_MIN_CLASSIFICATION_SAMPLES:
        return result

    # Get active predictions and actual outcomes
    active_pred = predicted_signal[active_mask]
    active_actual = actual_direction[active_mask]

    # Calculate Accuracy: Fraction of times Signal matches Direction
    # Note: actual_direction can be 0 (unchanged), which counts as wrong for both Long/Short
    correct_predictions = active_pred == active_actual
    accuracy = correct_predictions.mean()

    # Validate accuracy is finite and in [0, 1]
    if np.isnan(accuracy) or np.isinf(accuracy) or accuracy < 0 or accuracy > 1:
        accuracy = None

    try:
        # Calculate metrics separately for Long and Short signals
        # Long signals: predicted = 1, correct when actual = 1
        long_mask = active_pred == 1.0
        long_predicted = long_mask.sum()

        # Short signals: predicted = -1, correct when actual = -1
        short_mask = active_pred == -1.0
        short_predicted = short_mask.sum()

        # Calculate Long metrics
        long_precision = None
        long_recall = None

        if long_predicted > 0:
            # Long Precision: TP / (TP + FP) = correct Long predictions / all Long predictions
            long_correct = ((active_pred == 1.0) & (active_actual == 1.0)).sum()
            long_precision = long_correct / long_predicted if long_predicted > 0 else 0.0

            # Long Recall: TP / (TP + FN) = correct Long predictions / all actual Long movements
            # Note: We only consider active signals, so actual Long = actual_direction == 1 in active set
            long_actual = (active_actual == 1.0).sum()
            long_recall = long_correct / long_actual if long_actual > 0 else 0.0

            # Long F1: harmonic mean of precision and recall
            # if long_precision > 0 and long_recall > 0:
            #     long_f1 = 2 * (long_precision * long_recall) / (long_precision + long_recall)
            # else:
            #     long_f1 = 0.0

        # Calculate Short metrics
        short_precision = None
        short_recall = None

        if short_predicted > 0:
            # Short Precision: TP / (TP + FP) = correct Short predictions / all Short predictions
            short_correct = ((active_pred == -1.0) & (active_actual == -1.0)).sum()
            short_precision = short_correct / short_predicted if short_predicted > 0 else 0.0

            # Short Recall: TP / (TP + FN) = correct Short predictions / all actual Short movements
            short_actual = (active_actual == -1.0).sum()
            short_recall = short_correct / short_actual if short_actual > 0 else 0.0

            # Short F1: harmonic mean of precision and recall
            # if short_precision > 0 and short_recall > 0:
            #     short_f1 = 2 * (short_precision * short_recall) / (short_precision + short_recall)
            # else:
            #     short_f1 = 0.0

        # Calculate macro-averaged metrics (average of Long and Short)
        # Macro-averaging treats both classes equally, which is appropriate when
        # both Long and Short signals are equally important
        # If one class is missing (e.g., only Long signals exist),
        # macro metrics = that single class's metrics
        precision_values = [v for v in [long_precision, short_precision] if v is not None]
        recall_values = [v for v in [long_recall, short_recall] if v is not None]

        macro_precision = None
        macro_recall = None

        if precision_values:
            macro_precision_val = np.mean(precision_values)
            # Validate macro_precision is finite and in [0, 1]
            if (
                not np.isnan(macro_precision_val)
                and not np.isinf(macro_precision_val)
                and 0 <= macro_precision_val <= 1
            ):
                macro_precision = float(macro_precision_val)

        if recall_values:
            macro_recall_val = np.mean(recall_values)
            # Validate macro_recall is finite and in [0, 1]
            if not np.isnan(macro_recall_val) and not np.isinf(macro_recall_val) and 0 <= macro_recall_val <= 1:
                macro_recall = float(macro_recall_val)

        # Calculate macro F1 from macro precision and recall
        # Using macro-averaged precision/recall for F1 is more consistent than
        # averaging individual F1 scores, as it properly weights precision and recall
        if macro_precision is not None and macro_recall is not None:
            if macro_precision > 0 and macro_recall > 0:
                # Harmonic mean: F1 = 2 * (P * R) / (P + R)
                macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)
            else:
                # If either precision or recall is 0, F1 = 0
                macro_f1 = 0.0
        else:
            macro_f1 = None

        # Set results (use macro-averaged metrics)
        # Validate all metrics are in [0, 1] before setting
        if accuracy is not None and 0 <= accuracy <= 1:
            result["classification_accuracy"] = float(accuracy)
        else:
            result["classification_accuracy"] = None
        result["classification_precision"] = macro_precision
        result["classification_recall"] = macro_recall
        result["classification_f1"] = macro_f1

        # Final validation: ensure F1 is in [0, 1] if not None
        if result["classification_f1"] is not None:
            if result["classification_f1"] < 0 or result["classification_f1"] > 1:
                result["classification_f1"] = None

        return result
    except (ValueError, ZeroDivisionError, AttributeError, TypeError, IndexError, KeyError):
        # Catch all potential errors during calculation and return None values
        # This ensures the function never crashes, even with unexpected data
        #
        # ValueError: Invalid values in calculations (e.g., NaN in mean, invalid array shapes)
        # ZeroDivisionError: Division by zero (shouldn't happen due to checks, but safety)
        # AttributeError: Missing attributes on pandas Series/DataFrame (e.g., .loc on non-Series)
        # TypeError: Type conversion errors (e.g., float() on invalid types, wrong dtypes)
        # IndexError: Index access errors (e.g., empty Series, out-of-bounds)
        # KeyError: Dictionary key access errors (shouldn't happen, but safety)
        return {
            "classification_f1": None,
            "classification_precision": None,
            "classification_recall": None,
            "classification_accuracy": None,
        }

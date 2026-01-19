"""
Threshold Calculator for Random Forest Classification.

Calculates feature-specific thresholds for pattern matching.
Based on Pine Script threshold logic (lines 69-83).
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ThresholdCalculator:
    """
    Calculate thresholds for different feature types.

    Threshold rules from Pine Script:
    - Volume: stdev(volume, 14)
    - Z-Score: 0.05
    - Others: 0.5
    """

    volume_std_length: int = 14

    def calculate_threshold(self, feature_type: str, historical_values: Optional[List[float]] = None) -> float:
        """
        Calculate threshold for a feature type.

        Based on Pine Script logic:
        ```pinescript
        if select_x1 == "Volume"
            x1_threshold := ta.stdev(volume, 14)
        else if select_x1 == "Z-Score"
            x1_threshold := 0.05
        else
            x1_threshold := 0.5
        ```

        Args:
            feature_type: Type of feature ("Volume", "Z-Score", "Stochastic", "RSI", "MFI", "EMA", "SMA")
            historical_values: Historical values for calculating standard deviation (required for Volume)

        Returns:
            Threshold value for the feature type

        Raises:
            ValueError: If feature_type is Volume and no historical_values provided
        """
        feature_type = feature_type.strip().lower()

        if feature_type == "volume":
            if historical_values is None or len(historical_values) == 0:
                raise ValueError("Historical values required for Volume threshold calculation")
            return self._calculate_stdev(historical_values)
        elif feature_type == "z-score":
            return 0.05
        else:
            return 0.5

    def _calculate_stdev(self, values: List[float]) -> float:
        """
        Calculate standard deviation of values.

        Args:
            values: List of values

        Returns:
            Standard deviation (minimum epsilon to avoid division issues)
        """
        if len(values) == 0:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        stdev = variance**0.5

        # Return small epsilon if stdev is effectively zero
        # This prevents threshold matching issues when all values are identical
        return max(stdev, 1e-8)


__all__ = ["ThresholdCalculator"]

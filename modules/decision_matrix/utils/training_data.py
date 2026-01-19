"""
Training Data Storage for Random Forest Classification.

Implements circular buffer for storing historical [feature_value, label] pairs.
Based on Pine Script matrix storage (x1m, x2m).
"""

from collections import deque
from typing import Deque, Tuple


class TrainingDataStorage:
    """
    Circular buffer storage for training data.

    Stores historical data as [feature_value, label] pairs.
    Similar to Pine Script matrix structure: column 0 = feature, column 1 = label.

    Attributes:
        training_length: Maximum number of historical bars to store (default: 850)
        x1_data: Storage for feature 1 data [value, label]
        x2_data: Storage for feature 2 data [value, label]
    """

    def __init__(self, training_length: int = 850):
        """Initialize storage with training length."""
        self._training_length = training_length
        self.x1_data: Deque[Tuple[float, int]] = deque(maxlen=training_length)
        self.x2_data: Deque[Tuple[float, int]] = deque(maxlen=training_length)

    @property
    def training_length(self) -> int:
        """Get training length."""
        return self._training_length

    @training_length.setter
    def training_length(self, value: int) -> None:
        """Set training length and recreate deques with new maxlen."""
        if value <= 0:
            raise ValueError(f"training_length must be positive, got {value}")

        # Check if x1_data and x2_data have been initialized (via __post_init__)
        if not hasattr(self, "x1_data") or not hasattr(self, "x2_data"):
            # Setter called before __post_init__, just store the value
            # __post_init__ will initialize the deques with this value
            self._training_length = value
            return

        if value != self._training_length:
            # Store current data before recreating deques
            old_x1 = list(self.x1_data)[-value:]
            old_x2 = list(self.x2_data)[-value:]

            # Update training length
            self._training_length = value

            # Recreate deques with new maxlen, preserving as much data as possible
            self.x1_data = deque(old_x1, maxlen=value)
            self.x2_data = deque(old_x2, maxlen=value)

    def add_sample(self, x1: float, x2: float, y: int) -> None:
        """
        Add new training sample.

        Args:
            x1: Feature 1 value
            x2: Feature 2 value
            y: Label (0 or 1)
        """
        if y not in (0, 1):
            raise ValueError(f"Label must be 0 or 1, got {y}")

        self.x1_data.append((x1, y))
        self.x2_data.append((x2, y))

    def get_x1_matrix(self) -> list:
        """
        Get x1 data as matrix-like structure.

        Returns:
            List of [value, label] tuples for feature 1
        """
        return list(self.x1_data)

    def get_x2_matrix(self) -> list:
        """
        Get x2 data as matrix-like structure.

        Returns:
            List of [value, label] tuples for feature 2
        """
        return list(self.x2_data)

    def get_size(self) -> int:
        """
        Get current number of stored samples.

        Returns:
            Number of samples in storage
        """
        return len(self.x1_data)

    def clear(self) -> None:
        """Clear all stored data."""
        self.x1_data.clear()
        self.x2_data.clear()

    def is_full(self) -> bool:
        """
        Check if storage is at capacity.

        Returns:
            True if storage contains training_length samples
        """
        return len(self.x1_data) >= self.training_length


__all__ = ["TrainingDataStorage"]

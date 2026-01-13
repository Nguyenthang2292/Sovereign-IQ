"""
This module defines the `Pivot` class for representing significant price levels (pivots)
in financial time series data, such as in smart money concept strategies. Pivots can be
of different types (support, resistance, swing highs/lows) and are characterized by their
level, associated time, type, strength, and pip size for accurate calculations. Utility functions
may be added here for technical analysis and pivot management.

Classes:
    Pivot: Data structure representing a price pivot with validation logic.

Typical usage example:
    pivot = Pivot(level=1.2345, bar_time=datetime.utcnow(), pivot_type="support", strength=3)
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

from modules.common.ui.logging import log_warn


@dataclass
class Pivot:
    """
    Represents a significant price level used in technical analysis.

    Attributes:
        level (float): The pivot level value.
        bar_time (datetime): The timestamp associated with this pivot, representing the bar's time.
        pivot_type (str): The type of pivot ('support', 'resistance', or other custom type).
        strength (int): A numerical representation of pivot strength (higher is stronger).
        pip_size (float): Size of one pip for distance calculations (default 0.0001 for forex).
    """

    level: float = 0.0
    bar_time: Optional[datetime] = None
    pivot_type: Literal["support", "resistance", "swing_high", "swing_low", "other"] = "other"
    strength: int = 1
    pip_size: float = 0.0001  # Default for forex, can be overridden

    def __post_init__(self):
        """Validate attributes after initialization"""
        # Fix: Separate validation logic
        if not isinstance(self.level, (int, float)):
            log_warn(f"Invalid pivot level type: {type(self.level)}. Setting to 0.0")
            self.level = 0.0
        elif self.level < 0:
            log_warn(f"Negative pivot level: {self.level}. Setting to 0.0")
            self.level = 0.0

        if self.bar_time is None:
            # Use UTC timezone for consistency
            self.bar_time = datetime.now(timezone.utc)
        elif self.bar_time.tzinfo is None:
            # Assume UTC if no timezone provided
            self.bar_time = self.bar_time.replace(tzinfo=timezone.utc)

        if self.strength < 1:
            self.strength = 1

        if self.pip_size <= 0:
            log_warn(f"Invalid pip_size: {self.pip_size}. Using default 0.0001")
            self.pip_size = 0.0001

    def __str__(self) -> str:
        """String representation of the pivot"""
        time_str = self.bar_time.strftime("%Y-%m-%d %H:%M") if self.bar_time else "None"
        return f"Pivot({self.pivot_type.capitalize()}, {self.level:.5f}, {time_str}, Strength: {self.strength})"

    def __repr__(self) -> str:
        """Developer-friendly representation"""
        return (
            f"Pivot(level={self.level}, bar_time={self.bar_time}, "
            f"pivot_type='{self.pivot_type}', strength={self.strength}, pip_size={self.pip_size})"
        )

    def distance_to(self, price: float) -> float:
        """
        Calculate the distance between this pivot and a given price.

        Args:
            price (float): The price to calculate distance to

        Returns:
            float: Absolute distance between pivot level and price
        """
        return abs(self.level - price)

    def percent_distance_to(self, price: float) -> float:
        """
        Calculate the percentage distance between this pivot and a given price.

        Args:
            price (float): The price to calculate distance to

        Returns:
            float: Percentage distance between pivot level and price
        """
        if self.level == 0:
            return float("inf") if price != 0 else 0.0
        return abs((price - self.level) / self.level * 100)

    def is_above(self, price: float, buffer_pips: float = 0) -> bool:
        """
        Check if pivot is above a given price (with optional buffer).

        Args:
            price (float): The price to compare with
            buffer_pips (float): Buffer in pips

        Returns:
            bool: True if pivot is above price + buffer
        """
        return self.level > price + (buffer_pips * self.pip_size)

    def is_below(self, price: float, buffer_pips: float = 0) -> bool:
        """
        Check if pivot is below a given price (with optional buffer).

        Args:
            price (float): The price to compare with
            buffer_pips (float): Buffer in pips

        Returns:
            bool: True if pivot is below price - buffer
        """
        return self.level < price - (buffer_pips * self.pip_size)

    def is_near(self, price: float, pips: float = 10, use_percentage: bool = False) -> bool:
        """
        Check if the pivot is within specified distance of a price.

        Args:
            price (float): The price to check against
            pips (float): Number of pips (if use_percentage=False) or percentage (if use_percentage=True)
            use_percentage (bool): If True, use percentage instead of pips (better for crypto)

        Returns:
            bool: True if the pivot is within the specified range
        """
        if use_percentage:
            return self.percent_distance_to(price) <= pips
        return self.distance_to(price) <= pips * self.pip_size

    def is_recent(self, max_age: timedelta = timedelta(days=30)) -> bool:
        """
        Check if the pivot is recent based on the specified maximum age.

        Args:
            max_age (timedelta): Maximum age to be considered recent

        Returns:
            bool: True if the pivot is within the specified age
        """
        if self.bar_time is None:
            return False
        now = datetime.now(timezone.utc) if self.bar_time.tzinfo else datetime.now()
        return (now - self.bar_time) <= max_age

    def merge_with(self, other: "Pivot", strengthen: bool = True) -> "Pivot":
        """
        Create a new pivot by merging this pivot with another one.

        If pivots are close together, they may represent the same support/resistance.
        This method creates a new pivot using the average price and newer timestamp.

        Args:
            other (Pivot): Another pivot to merge with
            strengthen (bool): Whether to increase strength when merging

        Returns:
            Pivot: A new pivot representing the merged result

        Raises:
            TypeError: If other is not a Pivot instance
        """
        if not isinstance(other, Pivot):
            raise TypeError(f"Expected Pivot instance, got {type(other)}")

        new_level = (self.level + other.level) / 2
        new_time = max(
            self.bar_time or datetime.min.replace(tzinfo=timezone.utc),
            other.bar_time or datetime.min.replace(tzinfo=timezone.utc),
        )
        new_type = self.pivot_type if self.strength >= other.strength else other.pivot_type
        new_strength = self.strength + other.strength if strengthen else max(self.strength, other.strength)
        # Use average pip_size or the larger one
        new_pip_size = (self.pip_size + other.pip_size) / 2

        return Pivot(
            level=new_level, bar_time=new_time, pivot_type=new_type, strength=new_strength, pip_size=new_pip_size
        )

    def __eq__(self, other: object) -> bool:
        """
        Check equality based on level and type.

        Args:
            other: Object to compare with

        Returns:
            bool: True if pivots have same level and type
        """
        if not isinstance(other, Pivot):
            return False
        return self.level == other.level and self.pivot_type == other.pivot_type

    def __lt__(self, other: "Pivot") -> bool:
        """
        Compare pivots by level (for sorting).

        Args:
            other: Pivot to compare with

        Returns:
            bool: True if this pivot's level is less than other's
        """
        if not isinstance(other, Pivot):
            return NotImplemented
        return self.level < other.level

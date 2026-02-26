"""
Swing High / Swing Low detector using Pivot Zigzag algorithm.

Identifies local pivot highs and lows in OHLCV data using a rolling
lookback window. Returns the most significant swing points for use
in Gann Square analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import pandas as pd


@dataclass
class SwingPoint:
    """Represents a single swing high or swing low pivot point."""

    index: int  # positional integer index in DataFrame
    timestamp: pd.Timestamp
    price: float  # high price for swing high, low price for swing low
    kind: Literal["high", "low"]

    def __repr__(self) -> str:
        direction = "▼" if self.kind == "high" else "▲"
        return f"SwingPoint({direction} {self.kind} @ {self.price:.4f} [{self.timestamp}])"


class SwingDetector:
    """
    Detects Swing High and Swing Low pivot points using a rolling window.

    A Swing High at index i occurs when:
        high[i] == max(high[i-N : i+N+1])

    A Swing Low at index i occurs when:
        low[i] == min(low[i-N : i+N+1])

    where N is the lookback window (default: 5 candles on each side).
    """

    def __init__(self, lookback: int = 5) -> None:
        """
        Initialize SwingDetector.

        Args:
            lookback: Number of candles on each side to confirm a pivot.
                      Higher values = fewer but more significant swings.
                      Typical values: 3 (fast), 5 (default), 10 (slow).
        """
        if lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {lookback}")
        self.lookback = lookback

    def detect(self, df: pd.DataFrame) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        """
        Detect all swing highs and lows in the DataFrame.

        Args:
            df: OHLCV DataFrame with DatetimeIndex and columns [open, high, low, close].

        Returns:
            Tuple of (swing_highs, swing_lows) — lists of SwingPoint objects,
            sorted chronologically (ascending index).

        Raises:
            ValueError: If DataFrame is empty or missing required columns.
        """
        self._validate(df)

        highs: List[SwingPoint] = []
        lows: List[SwingPoint] = []
        n = self.lookback
        window = 2 * n + 1
        length = len(df)

        high_vals = df["high"].to_numpy()
        low_vals = df["low"].to_numpy()
        timestamps = df.index

        rolling_high = df["high"].rolling(window=window, center=True).max().to_numpy()
        rolling_low = df["low"].rolling(window=window, center=True).min().to_numpy()

        high_mask = high_vals == rolling_high
        low_mask = low_vals == rolling_low

        for i in range(n, length - n):
            # Swing High: current high is the maximum in the window
            if high_mask[i]:
                highs.append(
                    SwingPoint(
                        index=i,
                        timestamp=timestamps[i],
                        price=float(high_vals[i]),
                        kind="high",
                    )
                )

            # Swing Low: current low is the minimum in the window
            if low_mask[i]:
                lows.append(
                    SwingPoint(
                        index=i,
                        timestamp=timestamps[i],
                        price=float(low_vals[i]),
                        kind="low",
                    )
                )

        return highs, lows

    def get_significant_swings(self, df: pd.DataFrame) -> Tuple[Optional[SwingPoint], Optional[SwingPoint]]:
        """
        Return the single most significant Swing High and Swing Low.

        The most significant Swing High = highest price among all pivot highs.
        The most significant Swing Low  = lowest price among all pivot lows.

        Args:
            df: OHLCV DataFrame.

        Returns:
            Tuple of (highest_swing_high, lowest_swing_low).
            Either can be None if no pivots are detected (too little data).
        """
        swing_highs, swing_lows = self.detect(df)

        highest: Optional[SwingPoint] = None
        if swing_highs:
            highest = max(swing_highs, key=lambda sp: sp.price)

        lowest: Optional[SwingPoint] = None
        if swing_lows:
            lowest = min(swing_lows, key=lambda sp: sp.price)

        return highest, lowest

    def _validate(self, df: pd.DataFrame) -> None:
        """Validate DataFrame has the required structure."""
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None.")

        required = ["high", "low"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        min_length = 2 * self.lookback + 1
        if len(df) < min_length:
            raise ValueError(f"DataFrame has {len(df)} rows, need at least {min_length} for lookback={self.lookback}.")

"""True O(1) incremental moving average implementations.

This module provides constant-time update implementations for WMA, HMA, LSMA, and KAMA.
All classes maintain state that allows O(1) updates without iterating over the window.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict

import numpy as np


class TrueO1WMA:
    """Weighted Moving Average with true O(1) incremental updates.

    Maintains running weighted sum that can be updated in constant time
    by subtracting the contribution of the outgoing price and adding
    the contribution of the incoming price.

    WMA(n) = sum(prices[i] * weight[i]) / sum(weights)
    where weight[i] = i + 1 for i = 0..n-1 (oldest has weight 1, newest has weight n)

    Mathematical identity for O(1) update:
    When we shift the window by 1 (remove oldest, add newest):
    - Each existing price loses weight 1 (shifts left)
    - New price gets weight n
    - Oldest price is removed

    weighted_sum_new = weighted_sum_old - sum_prices_old + new_price * n
    where sum_prices_old = sum of all prices in window BEFORE adding new
    """

    def __init__(self, length: int):
        """Initialize O(1) WMA.

        Args:
            length: Window length for the WMA
        """
        if length <= 0:
            raise ValueError(f"length must be > 0, got {length}")

        self.length = length
        self.denominator = length * (length + 1) / 2.0

        # State for O(1) updates
        self.price_window: Deque[float] = deque(maxlen=length)
        self.weighted_sum = 0.0
        self.sum_prices = 0.0  # Running sum of all prices in window for O(1) update
        self.current_value = 0.0
        self.is_initialized = False

    def update(self, price: float) -> float:
        """Update WMA with new price in O(1) time.

        Args:
            price: New price value

        Returns:
            Current WMA value
        """
        if not self.is_initialized:
            # Initialization phase - just accumulate
            self.price_window.append(price)
            self.weighted_sum += price * len(self.price_window)
            self.sum_prices += price

            if len(self.price_window) == self.length:
                self.current_value = self.weighted_sum / self.denominator
                self.is_initialized = True
            elif len(self.price_window) > 0:
                # Use simple average during warmup
                self.current_value = self.sum_prices / len(self.price_window)

            return self.current_value

        # O(1) update:
        # When window shifts: each price's weight decreases by 1, new price gets weight n
        # weighted_sum_new = weighted_sum_old - sum_prices_old + new_price * n
        # sum_prices_new = sum_prices_old - oldest + new_price
        oldest_price = self.price_window[0]

        # Update weighted_sum: subtract sum of current prices (weights shift down by 1)
        # then add new price with full weight n
        self.weighted_sum -= self.sum_prices
        self.weighted_sum += price * self.length

        # Update sum_prices: remove oldest, add new
        self.sum_prices -= oldest_price
        self.sum_prices += price

        # Add to window (this automatically removes oldest due to maxlen)
        self.price_window.append(price)

        self.current_value = self.weighted_sum / self.denominator
        return self.current_value

    def reset(self):
        """Reset the WMA state."""
        self.price_window.clear()
        self.weighted_sum = 0.0
        self.sum_prices = 0.0
        self.current_value = 0.0
        self.is_initialized = False

    def get_state(self) -> Dict[str, Any]:
        """Get serializable state."""
        return {
            "price_window": list(self.price_window),
            "weighted_sum": self.weighted_sum,
            "sum_prices": self.sum_prices,
            "current_value": self.current_value,
            "is_initialized": self.is_initialized,
        }

    def set_state(self, state: Dict[str, Any]):
        """Restore state."""
        self.price_window = deque(state["price_window"], maxlen=self.length)
        self.weighted_sum = state["weighted_sum"]
        self.sum_prices = state.get("sum_prices", sum(self.price_window))  # Backward compat
        self.current_value = state["current_value"]
        self.is_initialized = state["is_initialized"]


class TrueO1HMA:
    """Hull Moving Average with true O(1) incremental updates.

    HMA formula:
    HMA(n) = WMA(sqrt(n), 2*WMA(n/2) - WMA(n))

    Uses three nested TrueO1WMA states for constant-time updates:
    - wma_half: WMA of length n/2
    - wma_full: WMA of length n
    - wma_final: WMA of length sqrt(n) on the series (2*wma_half - wma_full)

    DESIGN NOTE - Window Management:
    This class maintains TWO separate data structures:
    1. Three TrueO1WMA instances (wma_half, wma_full, wma_final) - each with own window
    2. intermediate_series deque - stores intermediate values (2*wma_half - wma_full)

    This is INTENTIONAL, not a bug. The HMA formula requires:
    - First compute intermediate series: 2*WMA(n/2) - WMA(n)
    - Then apply WMA(sqrt(n)) on that intermediate series
    - intermediate_series deque is necessary to feed wma_final.update()

    Memory overhead is minimal: O(sqrt(n)) for intermediate_series vs O(n) for full price window.
    This is significantly more efficient than O(n) full recalculation.
    """

    def __init__(self, length: int):
        """Initialize O(1) HMA.

        Args:
            length: Window length for the HMA
        """
        if length <= 0:
            raise ValueError(f"length must be > 0, got {length}")

        self.length = length
        self.half_len = max(1, length // 2)
        self.sqrt_len = max(1, int(np.sqrt(length)))

        # Three nested WMA states
        self.wma_half = TrueO1WMA(self.half_len)
        self.wma_full = TrueO1WMA(length)
        self.wma_final = TrueO1WMA(self.sqrt_len)

        # Track intermediate values
        self.intermediate_series: Deque[float] = deque(maxlen=self.sqrt_len)
        self.current_value = 0.0
        self.is_initialized = False

    def update(self, price: float) -> float:
        """Update HMA with new price in O(1) time.

        Args:
            price: New price value

        Returns:
            Current HMA value
        """
        # Update nested WMAs
        half_val = self.wma_half.update(price)
        full_val = self.wma_full.update(price)

        # Calculate intermediate value: 2 * WMA(n/2) - WMA(n)
        intermediate = 2.0 * half_val - full_val
        self.intermediate_series.append(intermediate)

        # Final WMA on intermediate series
        final_val = self.wma_final.update(intermediate)

        if self.wma_final.is_initialized:
            self.current_value = final_val
            self.is_initialized = True
        elif self.intermediate_series:
            self.current_value = sum(self.intermediate_series) / len(self.intermediate_series)

        return self.current_value

    def reset(self):
        """Reset the HMA state."""
        self.wma_half.reset()
        self.wma_full.reset()
        self.wma_final.reset()
        self.intermediate_series.clear()
        self.current_value = 0.0
        self.is_initialized = False

    def get_state(self) -> Dict[str, Any]:
        """Get serializable state."""
        return {
            "wma_half": self.wma_half.get_state(),
            "wma_full": self.wma_full.get_state(),
            "wma_final": self.wma_final.get_state(),
            "intermediate_series": list(self.intermediate_series),
            "current_value": self.current_value,
            "is_initialized": self.is_initialized,
        }

    def set_state(self, state: Dict[str, Any]):
        """Restore state."""
        self.wma_half.set_state(state["wma_half"])
        self.wma_full.set_state(state["wma_full"])
        self.wma_final.set_state(state["wma_final"])
        self.intermediate_series = deque(state["intermediate_series"], maxlen=self.sqrt_len)
        self.current_value = state["current_value"]
        self.is_initialized = state["is_initialized"]


class TrueO1LSMA:
    """Least Squares Moving Average with true O(1) incremental updates.

    LSMA fits a linear regression line to the last n price points and
    returns the value at the end of the line.

    Maintains running sums for O(1) updates:
    - sum_x: sum of x values (constant: n*(n-1)/2)
    - sum_y: sum of y values (prices)
    - sum_xy: sum of x*y products
    - sum_x2: sum of x² values (constant)
    """

    def __init__(self, length: int):
        """Initialize O(1) LSMA.

        Args:
            length: Window length for the LSMA
        """
        if length <= 0:
            raise ValueError(f"length must be > 0, got {length}")

        self.length = length
        self.x_values = np.arange(length, dtype=np.float64)

        # Pre-compute constant values
        self.sum_x = np.sum(self.x_values)
        self.sum_x2 = np.sum(self.x_values**2)
        self.denom = self.length * self.sum_x2 - self.sum_x**2

        # State for O(1) updates
        self.price_window: Deque[float] = deque(maxlen=length)
        self.sum_y = 0.0
        self.sum_xy = 0.0
        self.current_value = 0.0
        self.is_initialized = False

    def update(self, price: float) -> float:
        """Update LSMA with new price in O(1) time.

        Args:
            price: New price value

        Returns:
            Current LSMA value
        """
        if not self.is_initialized and len(self.price_window) < self.length:
            # Initialization phase
            idx = len(self.price_window)
            self.price_window.append(price)
            self.sum_y += price
            self.sum_xy += price * self.x_values[idx]

            if len(self.price_window) == self.length:
                self._compute_lsma()
                self.is_initialized = True
            else:
                # Use current price during warmup
                self.current_value = price

            return self.current_value

        # O(1) update: subtract oldest contribution, add newest contribution
        oldest_price = self.price_window[0]

        self.price_window.append(price)
        # sum_y: subtract oldest, add new
        self.sum_y -= oldest_price
        self.sum_y += price
        # sum_xy: all x-values shift left by 1, so subtract sum_y(excluding oldest) and add new * (n-1)
        self.sum_xy -= self.sum_y - price
        self.sum_xy += price * self.x_values[-1]

        self._compute_lsma()
        return self.current_value

    def _compute_lsma(self):
        """Compute LSMA from maintained sums."""
        if self.denom == 0:
            self.current_value = sum(self.price_window) / len(self.price_window) if self.price_window else 0.0
            return

        slope = (self.length * self.sum_xy - self.sum_x * self.sum_y) / self.denom
        intercept = (self.sum_y - slope * self.sum_x) / self.length
        self.current_value = intercept + slope * self.x_values[-1]

    def reset(self):
        """Reset the LSMA state."""
        self.price_window.clear()
        self.sum_y = 0.0
        self.sum_xy = 0.0
        self.current_value = 0.0
        self.is_initialized = False

    def get_state(self) -> Dict[str, Any]:
        """Get serializable state."""
        return {
            "price_window": list(self.price_window),
            "sum_y": self.sum_y,
            "sum_xy": self.sum_xy,
            "current_value": self.current_value,
            "is_initialized": self.is_initialized,
        }

    def set_state(self, state: Dict[str, Any]):
        """Restore state."""
        self.price_window = deque(state["price_window"], maxlen=self.length)
        self.sum_y = state["sum_y"]
        self.sum_xy = state["sum_xy"]
        self.current_value = state["current_value"]
        self.is_initialized = state["is_initialized"]


class TrueO1KAMA:
    """Kaufman Adaptive Moving Average with true O(1) incremental updates.

    KAMA adapts based on the efficiency ratio (ER):
    ER = change / volatility
    where change = abs(price - price_n_periods_ago)
    and volatility = sum of absolute price changes over the period

    Maintains rolling sum of absolute changes for O(1) updates.
    """

    def __init__(self, length: int, fast_period: int = 2, slow_period: int = 30):
        """Initialize O(1) KAMA.

        Args:
            length: Window length for efficiency ratio calculation
            fast_period: Fast EMA period for smoothing constant
            slow_period: Slow EMA period for smoothing constant
        """
        if length <= 0:
            raise ValueError(f"length must be > 0, got {length}")
        if fast_period <= 0 or slow_period <= 0:
            raise ValueError("fast_period and slow_period must be > 0")

        self.length = length
        self.fast_sc = 2.0 / (fast_period + 1.0)
        self.slow_sc = 2.0 / (slow_period + 1.0)

        # State for O(1) efficiency ratio
        self.price_window: Deque[float] = deque(maxlen=length + 1)
        self.volatility_sum = 0.0
        self.current_value = 0.0
        self.is_initialized = False

    def update(self, price: float) -> float:
        """Update KAMA with new price in O(1) time.

        Args:
            price: New price value

        Returns:
            Current KAMA value
        """
        if not self.is_initialized:
            # Initialization phase
            if len(self.price_window) > 0:
                prev_price = self.price_window[-1]
                self.volatility_sum += abs(price - prev_price)

            self.price_window.append(price)

            if len(self.price_window) == self.length + 1:
                self.is_initialized = True
                self._update_kama(price)
            else:
                self.current_value = price

            return self.current_value

        # O(1) update for volatility
        oldest_price = self.price_window[0]
        second_oldest = self.price_window[1] if len(self.price_window) > 1 else oldest_price
        prev_price = self.price_window[-1]

        # Remove contribution of oldest pair
        self.volatility_sum -= abs(second_oldest - oldest_price)

        # Add contribution of new pair
        self.volatility_sum += abs(price - prev_price)

        self.price_window.append(price)
        self._update_kama(price)
        return self.current_value

    def _update_kama(self, price: float):
        """Compute KAMA value from efficiency ratio."""
        # Direction: change from n periods ago
        change = abs(price - self.price_window[0])

        # Volatility: sum of absolute changes
        volatility = self.volatility_sum

        # Efficiency ratio
        er = change / volatility if volatility != 0 else 0.0

        # Smoothing constant
        sc = (er * (self.fast_sc - self.slow_sc) + self.slow_sc) ** 2

        # EMA-style update
        if not self.is_initialized:
            self.current_value = price
        else:
            self.current_value = self.current_value + sc * (price - self.current_value)

    def reset(self):
        """Reset the KAMA state."""
        self.price_window.clear()
        self.volatility_sum = 0.0
        self.current_value = 0.0
        self.is_initialized = False

    def get_state(self) -> Dict[str, Any]:
        """Get serializable state."""
        return {
            "price_window": list(self.price_window),
            "volatility_sum": self.volatility_sum,
            "current_value": self.current_value,
            "is_initialized": self.is_initialized,
        }

    def set_state(self, state: Dict[str, Any]):
        """Restore state."""
        self.price_window = deque(state["price_window"], maxlen=self.length + 1)
        self.volatility_sum = state["volatility_sum"]
        self.current_value = state["current_value"]
        self.is_initialized = state["is_initialized"]


__all__ = ["TrueO1WMA", "TrueO1HMA", "TrueO1LSMA", "TrueO1KAMA"]

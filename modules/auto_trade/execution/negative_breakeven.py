"""
Negative Breakeven Logic Module
================================

Pure logic functions for negative breakeven calculations.
No side effects - all inputs are explicit parameters.

Created: 2026-02-06
"""

from typing import Optional


def calculate_profit_pct(entry_price: float, mark_price: float, side: str) -> float:
    """
    Calculate profit percentage based on entry price and mark price.

    Args:
        entry_price: Entry price of the position
        mark_price: Current mark price
        side: "LONG" or "SHORT"

    Returns:
        Profit percentage (can be negative for losses)

    Examples:
        >>> calculate_profit_pct(100.0, 103.0, "LONG")
        3.0
        >>> calculate_profit_pct(100.0, 97.0, "LONG")
        -3.0
        >>> calculate_profit_pct(100.0, 103.0, "SHORT")
        -3.0
        >>> calculate_profit_pct(100.0, 97.0, "SHORT")
        3.0
    """
    if entry_price <= 0:
        return 0.0

    if side.upper() == "LONG":
        return ((mark_price - entry_price) / entry_price) * 100
    elif side.upper() == "SHORT":
        return ((entry_price - mark_price) / entry_price) * 100
    else:
        return 0.0


def has_hit_stop_loss(mark_price: float, stop_loss: float, side: str) -> bool:
    """
    Check if the mark price has hit or exceeded the stop loss level.

    Args:
        mark_price: Current mark price
        stop_loss: Stop loss price level
        side: "LONG" or "SHORT"

    Returns:
        True if mark price has hit stop loss, False otherwise

    Examples:
        >>> has_hit_stop_loss(95.0, 98.0, "LONG")  # mark < stop_loss
        True
        >>> has_hit_stop_loss(100.0, 98.0, "LONG")  # mark > stop_loss
        False
        >>> has_hit_stop_loss(105.0, 102.0, "SHORT")  # mark > stop_loss
        True
        >>> has_hit_stop_loss(100.0, 102.0, "SHORT")  # mark < stop_loss
        False
    """
    if side.upper() == "LONG":
        # For LONG: hit SL when mark <= stop_loss
        return mark_price <= stop_loss
    elif side.upper() == "SHORT":
        # For SHORT: hit SL when mark >= stop_loss
        return mark_price >= stop_loss
    else:
        return False


def should_trigger_negative_be(
    profit_pct: float,
    threshold_pct: float,
    mark_price: float,
    stop_loss: float,
    side: str,
    be_moved: bool,
) -> bool:
    """
    Determine if negative breakeven should be triggered.

    Triggers when:
    1. Profit % <= -threshold (position is losing by threshold amount)
    2. Mark price hasn't hit stop loss yet
    3. Breakeven hasn't been moved yet (be_moved is False)
    4. Threshold is positive (> 0)

    Args:
        profit_pct: Current profit percentage
        threshold_pct: Negative breakeven threshold percentage (e.g., 2.0)
        mark_price: Current mark price
        stop_loss: Stop loss price level
        side: "LONG" or "SHORT"
        be_moved: Whether breakeven has already been moved

    Returns:
        True if negative breakeven should trigger, False otherwise

    Examples:
        >>> should_trigger_negative_be(-3.0, 2.0, 97.0, 95.0, "LONG", False)
        True  # Loss >= threshold, hasn't hit SL, be_moved=False
        >>> should_trigger_negative_be(-1.0, 2.0, 99.0, 95.0, "LONG", False)
        False  # Loss < threshold
        >>> should_trigger_negative_be(-3.0, 2.0, 94.0, 95.0, "LONG", False)
        False  # Already hit stop loss
        >>> should_trigger_negative_be(-3.0, 2.0, 97.0, 95.0, "LONG", True)
        False  # Already moved breakeven
    """
    # Don't trigger if already moved
    if be_moved:
        return False

    # Don't trigger if threshold is not positive
    if threshold_pct <= 0:
        return False

    # Check if loss exceeds threshold (profit_pct is negative for losses)
    if profit_pct > -threshold_pct:
        return False

    # Check if stop loss has been hit
    if has_hit_stop_loss(mark_price, stop_loss, side):
        return False

    return True


def calculate_take_profit_for_be(
    entry_price: float,
    side: str,
) -> float:
    """
    Calculate the new take profit price for break-even.

    For negative breakeven, this is simply the entry price.

    Args:
        entry_price: Entry price of the position
        side: "LONG" or "SHORT" (for API consistency, not used in calculation)

    Returns:
        New take profit price (entry price)

    Examples:
        >>> calculate_take_profit_for_be(100.0, "LONG")
        100.0
        >>> calculate_take_profit_for_be(100.0, "SHORT")
        100.0
    """
    return entry_price


class NegativeBreakevenLogic:
    """
    Encapsulated negative breakeven logic for use in jobs and handlers.
    """

    @staticmethod
    def calculate_profit_pct(entry_price: float, mark_price: float, side: str) -> float:
        """Static wrapper for calculate_profit_pct."""
        return calculate_profit_pct(entry_price, mark_price, side)

    @staticmethod
    def should_trigger(
        entry_price: float,
        mark_price: float,
        stop_loss: float,
        side: str,
        threshold_pct: float,
        be_moved: bool,
    ) -> bool:
        """
        Check if negative breakeven should trigger.

        Args:
            entry_price: Entry price
            mark_price: Current mark price
            stop_loss: Stop loss price
            side: "LONG" or "SHORT"
            threshold_pct: Threshold percentage
            be_moved: Whether breakeven already moved

        Returns:
            True if should trigger
        """
        profit_pct = calculate_profit_pct(entry_price, mark_price, side)
        return should_trigger_negative_be(
            profit_pct=profit_pct,
            threshold_pct=threshold_pct,
            mark_price=mark_price,
            stop_loss=stop_loss,
            side=side,
            be_moved=be_moved,
        )

    @staticmethod
    def get_new_take_profit(entry_price: float) -> float:
        """Get new take profit price (entry price for breakeven)."""
        return entry_price

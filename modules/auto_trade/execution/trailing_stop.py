"""
Trailing Stop Step Management Module
=====================================

Implements step-based trailing stop logic:
- Step 0 (BE): Move SL to entry price when profit >= 0
- Step 1: Move SL to entry + step% when profit >= step%
- Step 2: Move SL to entry + 2*step% when profit >= 2*step%
- And so on...

Created: 2026-02-06
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TrailingStopResult:
    """Result of trailing stop calculation."""

    should_step: bool
    new_sl_price: Optional[float]
    next_step_index: int
    threshold_price: float
    message: str


def calculate_trailing_stop(
    entry_price: float,
    current_price: float,
    side: str,
    step_index: int,
    step_pct: float,
    current_sl: Optional[float] = None,
    limit_steps: bool = False,
    max_steps: int = 5,
) -> TrailingStopResult:
    """
    Calculate whether to step the trailing stop and what the new SL should be.

    Args:
        entry_price: Order entry price
        current_price: Current market price (mark price)
        side: 'LONG' or 'SHORT'
        step_index: Current trailing step index (0 = initial, 1 = first step, etc.)
        step_pct: Percentage for each step (e.g., 2.0 for 2%)
        current_sl: Current stop loss price (optional)
        limit_steps: Whether to limit the number of steps
        max_steps: Maximum number of steps allowed (if limit_steps=True)

    Returns:
        TrailingStopResult with should_step flag, new SL price, and next step index

    Examples:
        >>> # LONG position, entry=100, step=2%, current price=104
        >>> result = calculate_trailing_stop(
        ...     entry_price=100.0,
        ...     current_price=104.0,
        ...     side='LONG',
        ...     step_index=0,
        ...     step_pct=2.0,
        ... )
        >>> result.should_step
        True
        >>> result.new_sl_price
        100.0  # BE (entry price)

        >>> # Step 1: profit >= 2%, move SL to entry + 2%
        >>> result = calculate_trailing_stop(
        ...     entry_price=100.0,
        ...     current_price=102.5,
        ...     side='LONG',
        ...     step_index=1,
        ...     step_pct=2.0,
        ...     current_sl=100.0,
        ... )
        >>> result.should_step
        True
        >>> result.new_sl_price
        102.0  # entry + 2%
    """
    if side not in ("LONG", "SHORT"):
        return TrailingStopResult(
            should_step=False,
            new_sl_price=None,
            next_step_index=step_index,
            threshold_price=0.0,
            message=f"Invalid side: {side}. Must be 'LONG' or 'SHORT'",
        )

    if step_pct <= 0:
        return TrailingStopResult(
            should_step=False,
            new_sl_price=None,
            next_step_index=step_index,
            threshold_price=0.0,
            message="Step percentage must be positive",
        )

    # Check if we've reached max steps
    if limit_steps and step_index >= max_steps:
        return TrailingStopResult(
            should_step=False,
            new_sl_price=None,
            next_step_index=step_index,
            threshold_price=0.0,
            message=f"Maximum steps ({max_steps}) reached",
        )

    # Calculate profit percentage
    if side == "LONG":
        profit_pct = ((current_price - entry_price) / entry_price) * 100
    else:  # SHORT
        profit_pct = ((entry_price - current_price) / entry_price) * 100

    # Step 0 (BE): Move SL to entry when profit >= 0
    if step_index == 0:
        threshold_pct = 0.0
        new_sl = entry_price
    else:
        # Step N: Move SL to entry + N*step% when profit >= N*step%
        threshold_pct = step_index * step_pct
        step_multiplier = step_index if side == "LONG" else -step_index
        new_sl = entry_price * (1 + (step_multiplier * step_pct / 100))

    # Calculate threshold price for reference
    if side == "LONG":
        threshold_price = entry_price * (1 + threshold_pct / 100)
    else:
        threshold_price = entry_price * (1 - threshold_pct / 100)

    # Check if we should step
    if profit_pct >= threshold_pct:
        # Verify new SL is better than current SL
        if current_sl is not None:
            if side == "LONG" and new_sl <= current_sl:
                return TrailingStopResult(
                    should_step=False,
                    new_sl_price=None,
                    next_step_index=step_index,
                    threshold_price=threshold_price,
                    message=f"New SL ({new_sl:.4f}) is not better than current SL ({current_sl:.4f})",
                )
            elif side == "SHORT" and new_sl >= current_sl:
                return TrailingStopResult(
                    should_step=False,
                    new_sl_price=None,
                    next_step_index=step_index,
                    threshold_price=threshold_price,
                    message=f"New SL ({new_sl:.4f}) is not better than current SL ({current_sl:.4f})",
                )

        return TrailingStopResult(
            should_step=True,
            new_sl_price=new_sl,
            next_step_index=step_index + 1,
            threshold_price=threshold_price,
            message=f"Step {step_index} triggered at {profit_pct:.2f}% profit (threshold: {threshold_pct:.2f}%)",
        )
    else:
        return TrailingStopResult(
            should_step=False,
            new_sl_price=None,
            next_step_index=step_index,
            threshold_price=threshold_price,
            message=f"Profit {profit_pct:.2f}% below threshold {threshold_pct:.2f}% for step {step_index}",
        )


def calculate_next_threshold(
    entry_price: float,
    side: str,
    step_index: int,
    step_pct: float,
) -> Tuple[float, float]:
    """
    Calculate the next threshold price and percentage.

    Args:
        entry_price: Order entry price
        side: 'LONG' or 'SHORT'
        step_index: Current step index
        step_pct: Step percentage

    Returns:
        Tuple of (threshold_price, threshold_percentage)

    Example:
        >>> # LONG, step_index=1, step_pct=2%
        >>> price, pct = calculate_next_threshold(100.0, 'LONG', 1, 2.0)
        >>> price
        102.0  # Need price >= 102 to trigger step 1
        >>> pct
        2.0
    """
    if step_index == 0:
        threshold_pct = 0.0
    else:
        threshold_pct = step_index * step_pct

    if side == "LONG":
        threshold_price = entry_price * (1 + threshold_pct / 100)
    else:  # SHORT
        threshold_price = entry_price * (1 - threshold_pct / 100)

    return threshold_price, threshold_pct


def get_trailing_stop_info(
    entry_price: float,
    current_price: float,
    side: str,
    step_index: int,
    step_pct: float,
    limit_steps: bool = False,
    max_steps: int = 5,
) -> dict:
    """
    Get comprehensive information about trailing stop status.

    Args:
        entry_price: Order entry price
        current_price: Current market price
        side: 'LONG' or 'SHORT'
        step_index: Current step index
        step_pct: Step percentage
        limit_steps: Whether steps are limited
        max_steps: Maximum steps allowed

    Returns:
        Dictionary with trailing stop status information
    """
    # Calculate current profit
    if side == "LONG":
        profit_pct = ((current_price - entry_price) / entry_price) * 100
    else:
        profit_pct = ((entry_price - current_price) / entry_price) * 100

    # Calculate next threshold
    next_threshold_price, next_threshold_pct = calculate_next_threshold(entry_price, side, step_index, step_pct)

    # Calculate distance to next step
    if side == "LONG":
        distance_to_step = next_threshold_price - current_price
    else:
        distance_to_step = current_price - next_threshold_price

    # Calculate SL at next step
    if step_index == 0:
        sl_at_next_step = entry_price
    else:
        step_multiplier = step_index if side == "LONG" else -step_index
        sl_at_next_step = entry_price * (1 + (step_multiplier * step_pct / 100))

    return {
        "current_step": step_index,
        "current_profit_pct": round(profit_pct, 2),
        "next_step": step_index if profit_pct >= next_threshold_pct else step_index,
        "next_threshold_price": round(next_threshold_price, 4),
        "next_threshold_pct": round(next_threshold_pct, 2),
        "distance_to_next_step": round(distance_to_step, 4),
        "sl_at_next_step": round(sl_at_next_step, 4),
        "steps_remaining": max(0, max_steps - step_index) if limit_steps else None,
        "is_max_steps_reached": limit_steps and step_index >= max_steps,
    }

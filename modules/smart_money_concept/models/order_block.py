"""
OrderBlock dataclass for Smart Money Concept module.

An OrderBlock represents a price zone where institutional orders are likely placed.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

BULLISH = 1
NEUTRAL = 0
BEARISH = -1


@dataclass
class OrderBlock:
    """
    Represents an order block zone in price action analysis.

    Attributes:
        start: The starting timestamp of the order block
        end: The ending timestamp of the order block
        level_y0: The lower price level of the block (typically Low)
        level_y1: The upper price level of the block (typically High)
        bias: Direction bias - BULLISH (1), BEARISH (-1), or NEUTRAL (0)
    """

    start: Optional[datetime] = None
    end: Optional[datetime] = None
    level_y0: float = 0.0
    level_y1: float = 0.0
    bias: int = NEUTRAL

    def __post_init__(self):
        if self.level_y0 < 0:
            self.level_y0 = 0.0
        if self.level_y1 < 0:
            self.level_y1 = 0.0
        if self.bias not in (BULLISH, BEARISH, NEUTRAL):
            self.bias = NEUTRAL

    def __str__(self) -> str:
        bias_str = {BULLISH: "BULLISH", BEARISH: "BEARISH", NEUTRAL: "NEUTRAL"}.get(self.bias, "UNKNOWN")
        start_str = self.start.strftime("%Y-%m-%d %H:%M") if self.start else "None"
        end_str = self.end.strftime("%Y-%m-%d %H:%M") if self.end else "None"
        return f"OrderBlock({bias_str}, [{self.level_y0:.5f} - {self.level_y1:.5f}], {start_str} -> {end_str})"

    def __repr__(self) -> str:
        return (
            f"OrderBlock(start={self.start}, end={self.end}, "
            f"level_y0={self.level_y0}, level_y1={self.level_y1}, bias={self.bias})"
        )

    @property
    def is_bullish(self) -> bool:
        return self.bias == BULLISH

    @property
    def is_bearish(self) -> bool:
        return self.bias == BEARISH

    @property
    def is_neutral(self) -> bool:
        return self.bias == NEUTRAL

    @property
    def height(self) -> float:
        return abs(self.level_y1 - self.level_y0)

    @property
    def mid_price(self) -> float:
        return (self.level_y0 + self.level_y1) / 2

    def contains_price(self, price: float) -> bool:
        return self.level_y0 <= price <= self.level_y1

    def is_price_below(self, price: float) -> bool:
        return price < self.level_y0

    def is_price_above(self, price: float) -> bool:
        return price > self.level_y1

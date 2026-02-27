"""
BacktestConfig – configuration dataclass for the static backtester.

Supports both percentage-based and ATR-based TP/SL with optional trailing stop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class BacktestConfig:
    """
    Configuration for StaticBacktester.

    Attributes:
        mode:            ``"pct"`` — TP/SL as % of entry price.
                         ``"atr"`` — TP/SL as multiples of ATR.
        tp:              Take-profit level (percent or ATR multiple, default 2.0).
        sl:              Stop-loss level  (percent or ATR multiple, default 1.0).
        trailing_stop:   Trailing-stop level (same unit as ``sl``).
                         ``None`` disables trailing stop.
        atr_period:      ATR smoothing period (only used when ``mode="atr"``).
        initial_capital: Starting capital for equity-curve calculation.
        max_hold_bars:   Force-close a position after this many bars if not yet exited.
    """

    mode: Literal["pct", "atr"] = "pct"
    tp: float = 2.0
    sl: float = 1.0
    trailing_stop: Optional[float] = None
    atr_period: int = 14
    initial_capital: float = 10_000.0
    max_hold_bars: int = 100

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if self.mode not in ("pct", "atr"):
            raise ValueError(f"mode must be 'pct' or 'atr', got '{self.mode}'")
        if self.tp <= 0:
            raise ValueError(f"tp must be > 0, got {self.tp}")
        if self.sl <= 0:
            raise ValueError(f"sl must be > 0, got {self.sl}")
        if self.trailing_stop is not None and self.trailing_stop <= 0:
            raise ValueError(f"trailing_stop must be > 0 or None, got {self.trailing_stop}")
        if self.atr_period < 2:
            raise ValueError(f"atr_period must be >= 2, got {self.atr_period}")
        if self.max_hold_bars < 1:
            raise ValueError(f"max_hold_bars must be >= 1, got {self.max_hold_bars}")

    def summary(self) -> str:
        trailing = f" | Trail {self.trailing_stop}" if self.trailing_stop else ""
        return (
            f"mode={self.mode.upper()} | TP={self.tp} | SL={self.sl}{trailing}"
            f" | capital={self.initial_capital:,.0f} | max_hold={self.max_hold_bars}"
        )

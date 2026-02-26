"""
Gann Square Calculator.

Builds Gann Fan zones from Swing High/Low pivot points,
determines trend direction, identifies which zone the current
price is in, and provides a preliminary signal (before Gemini).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

from .swing_detector import SwingPoint

TrendType = Literal["UP", "DOWN"]
SignalCode = Literal["LONG", "SHORT", "SKIP"]


@dataclass
class GannZone:
    """One of 4 Gann Fan zones - diagonal lines radiating from pivot point."""

    zone_number: int  # 1 to 4
    pivot_index: int  # candle index of pivot (swing high/low)
    pivot_price: float  # price at pivot point
    slope: float  # price change per candle (negative for downtrend fan)
    label: str  # e.g. "Zone 1 (SHORT)"
    is_tradeable: bool  # True for Zone 1 & 2, False for Zone 3 & 4
    signal: SignalCode  # LONG / SHORT / SKIP
    _upper_slope: float = 0.0  # slope of upper boundary (for zone 1-3)
    _lower_slope: float = 0.0  # slope of lower boundary (for zone 2-4)

    def __post_init__(self):
        if self._upper_slope == 0.0:
            self._upper_slope = self.slope
        if self._lower_slope == 0.0:
            self._lower_slope = self.slope

    def price_at(self, candle_index: int) -> float:
        """Calculate price at given candle index using linear slope."""
        delta_index = candle_index - self.pivot_index
        return self.pivot_price + (self.slope * delta_index)

    def upper_price_at(self, candle_index: int) -> float:
        """Get upper boundary price at candle index."""
        delta_index = candle_index - self.pivot_index
        return self.pivot_price + (self._upper_slope * delta_index)

    def lower_price_at(self, candle_index: int) -> float:
        """Get lower boundary price at candle index."""
        delta_index = candle_index - self.pivot_index
        return self.pivot_price + (self._lower_slope * delta_index)

    def contains_at(self, price: float, candle_index: int) -> bool:
        """Return True if price falls within this fan zone at the given candle index.

        Uses half-open interval (lower_price, upper_price] so that a price
        sitting exactly on a shared boundary belongs to exactly one zone.
        """
        zone_upper = self.upper_price_at(candle_index)
        zone_lower = self.lower_price_at(candle_index)
        return zone_lower < price <= zone_upper

    def midpoint_at(self, candle_index: int) -> float:
        """Midpoint of zone at given candle index."""
        return (self.upper_price_at(candle_index) + self.lower_price_at(candle_index)) / 2


@dataclass
class GannSquareResult:
    """Full result of a Gann Square calculation."""

    trend: TrendType
    swing_high: SwingPoint
    swing_low: SwingPoint
    price_range: float  # swing_high.price - swing_low.price
    zones: List[GannZone]  # 4 zones, index 0 = zone 1
    current_zone: int  # 1-4 (0 if undetermined)
    signal_code: SignalCode  # code-calculated signal before Gemini
    current_index: int = 0  # current candle index for zone evaluation

    @property
    def preliminary_signal(self) -> SignalCode:
        """Backward-compatible alias for signal_code."""
        return self.signal_code

    @property
    def active_zone(self) -> Optional[GannZone]:
        """Return the GannZone object for the current zone."""
        if 1 <= self.current_zone <= 4:
            return self.zones[self.current_zone - 1]
        return None

    def summary(self) -> str:
        """Human-readable one-line summary."""
        return (
            f"Trend={self.trend} | "
            f"H={self.swing_high.price:.4f} L={self.swing_low.price:.4f} | "
            f"Range={self.price_range:.4f} | "
            f"Zone={self.current_zone} | "
            f"Signal={self.signal_code}"
        )


class GannCalculator:
    """
    Calculates Gann Fan zones and determines trading signal.

    Zone Mapping (Fan Lines):
    ─────────────────────────────────────────────────────────────
    Fan boundary slopes = [0, 0.5, 1.0, 1.5, 2.0] × price_per_candle
    The 1×1 line (1.0 × ppc) passes exactly through swing_low/high.

    DOWN Trend (fan radiates downward from swing_high):
        Zone 4 (shallowest):  0 → −0.5×ppc          → SHORT
        Zone 3:               −0.5×ppc → −1.0×ppc    → SHORT
        Zone 2:               −1.0×ppc → −1.5×ppc    → SKIP
        Zone 1 (steepest):    −1.5×ppc → −2.0×ppc    → SKIP

    UP Trend (fan radiates upward from swing_low):
        Zone 1 (steepest):    +1.5×ppc → +2.0×ppc    → SKIP
        Zone 2:               +1.0×ppc → +1.5×ppc    → SKIP
        Zone 3:               +0.5×ppc → +1.0×ppc    → LONG
        Zone 4 (shallowest):  0 → +0.5×ppc            → LONG
    ─────────────────────────────────────────────────────────────
    """

    def calculate(
        self,
        swing_high: SwingPoint,
        swing_low: SwingPoint,
        current_price: float,
        current_index: int = 0,
    ) -> GannSquareResult:
        """
        Build Gann Fan zones and identify the current zone.

        Args:
            swing_high: The most significant Swing High pivot point.
            swing_low:  The most significant Swing Low pivot point.
            current_price: Most recent close price.
            current_index: Candle index for zone evaluation.

        Returns:
            GannSquareResult with full zone breakdown and preliminary signal.

        Raises:
            ValueError: If swing_high.price <= swing_low.price.
        """
        if swing_high.price <= swing_low.price:
            raise ValueError(
                f"swing_high.price ({swing_high.price}) must be greater than swing_low.price ({swing_low.price})."
            )

        # Detect trend: if swing high appeared BEFORE swing low → price fell → DOWN
        trend: TrendType = "DOWN" if swing_high.index < swing_low.index else "UP"

        price_range = swing_high.price - swing_low.price

        # Build 4 fan zones
        zones = self._build_zones(trend, swing_high, swing_low, price_range, current_index)

        # Find which zone current price is in
        current_zone = self._find_zone(zones, current_price, current_index)

        # Code-calculated signal from zone
        signal_code: SignalCode = "SKIP"
        if current_zone > 0:
            signal_code = zones[current_zone - 1].signal

        return GannSquareResult(
            trend=trend,
            swing_high=swing_high,
            swing_low=swing_low,
            price_range=price_range,
            zones=zones,
            current_zone=current_zone,
            signal_code=signal_code,
            current_index=current_index,
        )

    def _build_zones(
        self,
        trend: TrendType,
        swing_high: SwingPoint,
        swing_low: SwingPoint,
        price_range: float,
        current_index: int,
    ) -> List[GannZone]:
        """Build 4 Gann Fan zones based on trend direction.

        Five boundary slopes [0, 0.5, 1.0, 1.5, 2.0] × ppc create four zones.
        The 1×1 line (1.0 × ppc) is the Zone 2/Zone 3 boundary and passes
        exactly through swing_low (DOWN) or swing_high (UP).
        """
        # Calculate price per candle (always positive)
        t_high = swing_high.index
        t_low = swing_low.index
        ppc = price_range / abs(t_low - t_high)

        # Five boundary magnitudes creating four zones
        b = [0.0, 0.5 * ppc, 1.0 * ppc, 1.5 * ppc, 2.0 * ppc]

        if trend == "DOWN":
            # Fan radiates DOWNWARD from swing_high
            pivot_idx = swing_high.index
            pivot_pr = swing_high.price
            # Zone definitions: (zone_num, upper_slope, lower_slope, signal, tradeable)
            zone_defs: list[tuple[int, float, float, SignalCode, bool]] = [
                (1, -b[3], -b[4], "SKIP",  False),   # steepest
                (2, -b[2], -b[3], "SKIP",  False),
                (3, -b[1], -b[2], "SHORT", True),
                (4,  b[0], -b[1], "SHORT", True),    # shallowest
            ]
        else:
            # UP trend: Fan radiates UPWARD from swing_low
            pivot_idx = swing_low.index
            pivot_pr = swing_low.price
            zone_defs = [
                (1,  b[4],  b[3], "SKIP", False),    # steepest (top)
                (2,  b[3],  b[2], "SKIP", False),
                (3,  b[2],  b[1], "LONG", True),
                (4,  b[1],  b[0], "LONG", True),     # shallowest (bottom)
            ]

        zones: List[GannZone] = []
        for zone_num, upper_slope, lower_slope, signal, tradeable in zone_defs:
            slope = (upper_slope + lower_slope) / 2   # midpoint of boundaries
            label = f"Zone {zone_num} ({'SKIP' if not tradeable else signal})"
            zones.append(
                GannZone(
                    zone_number=zone_num,
                    pivot_index=pivot_idx,
                    pivot_price=pivot_pr,
                    slope=slope,
                    _upper_slope=upper_slope,
                    _lower_slope=lower_slope,
                    label=label,
                    is_tradeable=tradeable,
                    signal=signal,
                )
            )

        return zones

    def _find_zone(self, zones: List[GannZone], current_price: float, current_index: int) -> int:
        """
        Find which zone contains the current price at the given candle index.

        Returns zone number 1-4, or 0 if outside all zones.
        """
        for zone in zones:
            if zone.contains_at(current_price, current_index):
                return zone.zone_number

        # Price outside all zones — assign to nearest boundary zone
        if zones:
            highest = max(zones, key=lambda z: z.upper_price_at(current_index))
            if current_price > highest.upper_price_at(current_index):
                return highest.zone_number

            lowest = min(zones, key=lambda z: z.lower_price_at(current_index))
            if current_price <= lowest.lower_price_at(current_index):
                return 0

        return 0

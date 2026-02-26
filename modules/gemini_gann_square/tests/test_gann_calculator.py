"""
Unit tests for GannCalculator (zone calculation and trend detection).
"""

from __future__ import annotations

import pandas as pd
import pytest

from modules.gemini_gann_square.core.gann_calculator import (
    GannCalculator,
    GannSquareResult,
    GannZone,
)
from modules.gemini_gann_square.core.swing_detector import SwingPoint


def make_swing(index: int, price: float, kind: str) -> SwingPoint:
    """Helper: create a SwingPoint at a given index."""
    ts = pd.Timestamp("2024-01-01") + pd.Timedelta(hours=index)
    return SwingPoint(index=index, timestamp=ts, price=price, kind=kind)  # type: ignore[arg-type]


# ──────────────────────────────────────────────
# Trend detection tests
# ──────────────────────────────────────────────


class TestTrendDetection:
    def test_down_trend_when_high_before_low(self):
        """Swing High at index 10, Swing Low at index 50 → DOWN trend."""
        high = make_swing(index=10, price=100.0, kind="high")
        low = make_swing(index=50, price=60.0, kind="low")
        calc = GannCalculator()
        result = calc.calculate(high, low, current_price=80.0, current_index=30)
        assert result.trend == "DOWN"

    def test_up_trend_when_low_before_high(self):
        """Swing Low at index 10, Swing High at index 50 → UP trend."""
        high = make_swing(index=50, price=100.0, kind="high")
        low = make_swing(index=10, price=60.0, kind="low")
        calc = GannCalculator()
        result = calc.calculate(high, low, current_price=80.0, current_index=30)
        assert result.trend == "UP"


# ──────────────────────────────────────────────
# Zone boundary tests (DOWN trend)
# ──────────────────────────────────────────────


class TestGannZoneBoundariesDown:
    """
    DOWN trend: High=100 at index 5, Low=60 at index 50, Range=40
    ppc = 40 / abs(50 - 5) = 0.8889

    Boundary slopes: [0, -0.5, -1.0, -1.5, -2.0] × ppc
      Zone 1 (steepest, SKIP):    upper=-1.5×ppc  lower=-2.0×ppc
      Zone 2 (SKIP):              upper=-1.0×ppc  lower=-1.5×ppc
      Zone 3 (SHORT):             upper=-0.5×ppc  lower=-1.0×ppc
      Zone 4 (shallowest, SHORT): upper=0         lower=-0.5×ppc
    """

    @pytest.fixture
    def down_result(self) -> GannSquareResult:
        high = make_swing(index=5, price=100.0, kind="high")
        low = make_swing(index=50, price=60.0, kind="low")
        return GannCalculator().calculate(high, low, current_price=95.0, current_index=10)

    def test_has_exactly_four_zones(self, down_result):
        assert len(down_result.zones) == 4

    def test_zone_1_slope(self, down_result):
        z1 = down_result.zones[0]
        price_at_10 = z1.price_at(10)
        assert price_at_10 < 100.0  # Zone 1 trends down from pivot

    def test_zone_1_slope_steepest(self, down_result):
        # Zone 1 should have the steepest slope (most negative)
        slopes = [z.slope for z in down_result.zones]
        assert slopes[0] < slopes[3]  # zone 1 slope is more negative than zone 4

    def test_zone_3_4_are_short(self, down_result):
        assert down_result.zones[2].signal == "SHORT"
        assert down_result.zones[3].signal == "SHORT"

    def test_zone_1_2_are_skip(self, down_result):
        assert down_result.zones[0].signal == "SKIP"
        assert down_result.zones[1].signal == "SKIP"

    def test_zone_3_4_tradeable(self, down_result):
        assert down_result.zones[2].is_tradeable is True
        assert down_result.zones[3].is_tradeable is True

    def test_zone_1_2_not_tradeable(self, down_result):
        assert down_result.zones[0].is_tradeable is False
        assert down_result.zones[1].is_tradeable is False


# ──────────────────────────────────────────────
# Zone boundary tests (UP trend)
# ──────────────────────────────────────────────


class TestGannZoneBoundariesUp:
    """
    UP trend: High=100 at index 50, Low=60 at index 5, Range=40
    ppc = 40 / abs(5 - 50) = 0.8889

    Boundary slopes: [0, 0.5, 1.0, 1.5, 2.0] × ppc
      Zone 1 (steepest, SKIP): upper=+2.0×ppc  lower=+1.5×ppc
      Zone 2 (SKIP):           upper=+1.5×ppc  lower=+1.0×ppc
      Zone 3 (LONG):           upper=+1.0×ppc  lower=+0.5×ppc
      Zone 4 (shallowest, LONG): upper=+0.5×ppc lower=0
    """

    @pytest.fixture
    def up_result(self) -> GannSquareResult:
        high = make_swing(index=50, price=100.0, kind="high")
        low = make_swing(index=5, price=60.0, kind="low")
        return GannCalculator().calculate(high, low, current_price=75.0, current_index=30)

    def test_trend_is_up(self, up_result):
        assert up_result.trend == "UP"

    def test_zone_3_4_are_long(self, up_result):
        assert up_result.zones[2].signal == "LONG"
        assert up_result.zones[3].signal == "LONG"

    def test_zone_1_2_are_skip(self, up_result):
        assert up_result.zones[0].signal == "SKIP"
        assert up_result.zones[1].signal == "SKIP"


# ──────────────────────────────────────────────
# Current zone identification
# ──────────────────────────────────────────────


class TestCurrentZone:
    """
    DOWN trend: High=100@5, Low=60@50, ppc=0.8889
    At index 10 (delta=5 from pivot):
      Zone 4: (97.78, 100.0]   Zone 3: (95.56, 97.78]
      Zone 2: (93.33, 95.56]   Zone 1: (91.11, 93.33]
    """

    @pytest.fixture
    def down_calc(self):
        high = make_swing(index=5, price=100.0, kind="high")
        low = make_swing(index=50, price=60.0, kind="low")
        return GannCalculator(), high, low

    def test_price_in_zone_4(self, down_calc):
        calc, high, low = down_calc
        # Zone 4 (shallowest, SHORT): (97.78, 100.0] at index 10
        result = calc.calculate(high, low, current_price=99.0, current_index=10)
        assert result.current_zone == 4
        assert result.signal_code == "SHORT"

    def test_price_in_zone_3(self, down_calc):
        calc, high, low = down_calc
        # Zone 3 (SHORT): (95.56, 97.78] at index 10
        result = calc.calculate(high, low, current_price=97.0, current_index=10)
        assert result.current_zone == 3
        assert result.signal_code == "SHORT"

    def test_price_in_zone_2(self, down_calc):
        calc, high, low = down_calc
        # Zone 2 (SKIP): (93.33, 95.56] at index 10
        result = calc.calculate(high, low, current_price=94.5, current_index=10)
        assert result.current_zone == 2
        assert result.signal_code == "SKIP"

    def test_price_in_zone_1(self, down_calc):
        calc, high, low = down_calc
        # Zone 1 (steepest, SKIP): (91.11, 93.33] at index 10
        result = calc.calculate(high, low, current_price=92.0, current_index=10)
        assert result.current_zone == 1
        assert result.signal_code == "SKIP"

    def test_price_exactly_at_boundary(self, down_calc):
        """Price near a boundary falls into the zone below it (half-open interval)."""
        calc, high, low = down_calc
        # At index 20 (delta=15): Zone 3/4 boundary ≈ 93.333
        # Price 93.33 < 93.333 → belongs to Zone 3
        result = calc.calculate(high, low, current_price=93.33, current_index=20)
        assert result.current_zone == 3

    def test_price_above_all_zones_defaults_to_nearest(self, down_calc):
        calc, high, low = down_calc
        # At index 10: Zone 4 upper = 100. Price 110 > 100 → Zone 4
        result = calc.calculate(high, low, current_price=110.0, current_index=10)
        assert result.current_zone == 4

    def test_price_below_all_zones_out_of_range(self, down_calc):
        """Price below all fan zones returns zone 0 (out of range)."""
        calc, high, low = down_calc
        # At index 55 (delta=50): Zone 1 lower ≈ 11.11
        # Price 5.0 < 11.11 → out of range
        result = calc.calculate(high, low, current_price=5.0, current_index=55)
        assert result.current_zone == 0
        assert result.signal_code == "SKIP"


# ──────────────────────────────────────────────
# Validation tests
# ──────────────────────────────────────────────


class TestGannCalculatorValidation:
    def test_raises_when_high_equals_low(self):
        high = make_swing(10, 100.0, "high")
        low = make_swing(50, 100.0, "low")
        with pytest.raises(ValueError, match="greater than"):
            GannCalculator().calculate(high, low, current_price=100.0, current_index=30)

    def test_raises_when_high_below_low(self):
        high = make_swing(10, 50.0, "high")
        low = make_swing(50, 80.0, "low")
        with pytest.raises(ValueError, match="greater than"):
            GannCalculator().calculate(high, low, current_price=60.0, current_index=30)


# ──────────────────────────────────────────────
# GannZone.contains_at() tests
# ──────────────────────────────────────────────


class TestGannZoneContainsAt:
    def make_zone(self, pivot_idx: int, pivot_price: float, slope: float) -> GannZone:
        return GannZone(
            zone_number=1,
            pivot_index=pivot_idx,
            pivot_price=pivot_price,
            slope=slope,
            label="Test",
            is_tradeable=True,
            signal="SHORT",
            _upper_slope=slope,
            _lower_slope=slope * 2,  # zone 2's lower slope
        )

    def test_price_inside_zone(self):
        z = self.make_zone(pivot_idx=0, pivot_price=100.0, slope=-1.0)
        # At index 5, zone upper = 95, zone lower = 90
        assert z.contains_at(92.5, 5) is True

    def test_price_at_pivot_index_equals_pivot_price(self):
        z = self.make_zone(pivot_idx=0, pivot_price=100.0, slope=-1.0)
        assert z.price_at(0) == pytest.approx(100.0)

    def test_price_at_upper_bound(self):
        z = self.make_zone(pivot_idx=0, pivot_price=100.0, slope=-1.0)
        assert z.contains_at(95.0, 5) is True

    def test_price_at_lower_bound(self):
        z = self.make_zone(pivot_idx=0, pivot_price=100.0, slope=-1.0)
        assert z.contains_at(90.0, 5) is False

    def test_price_just_above_lower_bound(self):
        z = self.make_zone(pivot_idx=0, pivot_price=100.0, slope=-1.0)
        assert z.contains_at(90.001, 5) is True

    def test_price_above_zone(self):
        z = self.make_zone(pivot_idx=0, pivot_price=100.0, slope=-1.0)
        assert z.contains_at(96.0, 5) is False

    def test_price_below_zone(self):
        z = self.make_zone(pivot_idx=0, pivot_price=100.0, slope=-1.0)
        assert z.contains_at(89.0, 5) is False

    def test_midpoint_calculation(self):
        z = self.make_zone(pivot_idx=0, pivot_price=100.0, slope=-1.0)
        mid = z.midpoint_at(5)
        assert mid == pytest.approx(92.5)


class TestGannSquareResultHelpers:
    def test_preliminary_signal_alias(self):
        high = make_swing(index=5, price=100.0, kind="high")
        low = make_swing(index=50, price=60.0, kind="low")
        result = GannCalculator().calculate(high, low, current_price=95.0, current_index=10)
        assert result.preliminary_signal == result.signal_code

    def test_summary_contains_core_fields(self):
        high = make_swing(index=5, price=100.0, kind="high")
        low = make_swing(index=50, price=60.0, kind="low")
        result = GannCalculator().calculate(high, low, current_price=95.0, current_index=10)
        summary = result.summary()
        assert "Trend=" in summary
        assert "Range=" in summary
        assert "Zone=" in summary
        assert "Signal=" in summary

"""
Smoke tests for GannChartGenerator.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from modules.gemini_gann_square.core.gann_calculator import GannSquareResult, GannZone
from modules.gemini_gann_square.core.gann_chart_generator import GannChartGenerator
from modules.gemini_gann_square.core.swing_detector import SwingPoint


def _make_df(n: int = 30) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    base = pd.Series(range(n), index=idx, dtype=float) + 100.0
    return pd.DataFrame(
        {
            "open": base,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base + 0.5,
            "volume": 1000.0,
        },
        index=idx,
    )


def _make_gann_result() -> GannSquareResult:
    # Create fan zones with pivot at swing_high (index 5, price 100)
    # price_range=40, candle_count=20, ppc=2.0
    # Boundary slopes for DOWN: [0, -1.0, -2.0, -3.0, -4.0]
    zone1 = GannZone(
        zone_number=1,
        pivot_index=5,
        pivot_price=100.0,
        slope=-3.5,            # midpoint of -3.0 and -4.0
        _upper_slope=-3.0,
        _lower_slope=-4.0,
        label="Zone 1 (SKIP)",
        is_tradeable=False,
        signal="SKIP",
    )
    zone2 = GannZone(
        zone_number=2,
        pivot_index=5,
        pivot_price=100.0,
        slope=-2.5,            # midpoint of -2.0 and -3.0
        _upper_slope=-2.0,
        _lower_slope=-3.0,
        label="Zone 2 (SKIP)",
        is_tradeable=False,
        signal="SKIP",
    )
    zone3 = GannZone(
        zone_number=3,
        pivot_index=5,
        pivot_price=100.0,
        slope=-1.5,            # midpoint of -1.0 and -2.0
        _upper_slope=-1.0,
        _lower_slope=-2.0,
        label="Zone 3 (SHORT)",
        is_tradeable=True,
        signal="SHORT",
    )
    zone4 = GannZone(
        zone_number=4,
        pivot_index=5,
        pivot_price=100.0,
        slope=-0.5,            # midpoint of 0.0 and -1.0
        _upper_slope=0.0,
        _lower_slope=-1.0,
        label="Zone 4 (SHORT)",
        is_tradeable=True,
        signal="SHORT",
    )

    swing_high = SwingPoint(index=5, timestamp=pd.Timestamp("2024-01-01 05:00:00"), price=100.0, kind="high")  # type: ignore[arg-type]
    swing_low = SwingPoint(index=25, timestamp=pd.Timestamp("2024-01-02 01:00:00"), price=60.0, kind="low")  # type: ignore[arg-type]

    return GannSquareResult(
        trend="DOWN",
        swing_high=swing_high,
        swing_low=swing_low,
        price_range=40.0,
        zones=[zone1, zone2, zone3, zone4],
        current_zone=3,                    # Zone 3 is a tradeable SHORT zone
        signal_code="SHORT",
        current_index=15,
    )


def test_create_chart_smoke(tmp_path: Path):
    df = _make_df()
    gann = _make_gann_result()
    out_path = tmp_path / "chart.png"

    generator = GannChartGenerator(output_dir=str(tmp_path))
    result_path = generator.create_chart(
        df=df,
        gann_result=gann,
        symbol="BTC/USDT",
        timeframe="4h",
        output_path=str(out_path),
    )

    assert result_path == str(out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_chart_has_fan_lines_not_horizontal_bands(tmp_path: Path):
    """Verify chart uses diagonal fan lines instead of horizontal bands."""
    from matplotlib.axes import Axes

    df = _make_df()
    gann = _make_gann_result()
    out_path = tmp_path / "chart_fan.png"

    calls = {"fill_between": 0, "plot": 0, "axhspan": 0}

    original_fill_between = Axes.fill_between
    original_plot = Axes.plot
    original_axhspan = Axes.axhspan

    def tracked_fill_between(self, *args, **kwargs):
        calls["fill_between"] += 1
        return original_fill_between(self, *args, **kwargs)

    def tracked_plot(self, *args, **kwargs):
        calls["plot"] += 1
        return original_plot(self, *args, **kwargs)

    def tracked_axhspan(self, *args, **kwargs):
        calls["axhspan"] += 1
        return original_axhspan(self, *args, **kwargs)

    Axes.fill_between = tracked_fill_between
    Axes.plot = tracked_plot
    Axes.axhspan = tracked_axhspan

    try:
        generator = GannChartGenerator(output_dir=str(tmp_path))
        generator.create_chart(
            df=df,
            gann_result=gann,
            symbol="BTC/USDT",
            timeframe="4h",
            output_path=str(out_path),
        )
    finally:
        Axes.fill_between = original_fill_between
        Axes.plot = original_plot
        Axes.axhspan = original_axhspan

    # Verify file was created
    assert out_path.exists()
    assert out_path.stat().st_size > 0
    # Verify fan-zone rendering was used and horizontal spans were not
    assert calls["fill_between"] >= 4
    assert calls["plot"] >= 4
    assert calls["axhspan"] == 0

"""
Tests for OrderBuilder with ROI-mode TP/SL calculation.

All tp_pct / sl_pct values are ROI% on capital.
price_move_pct = roi_pct / leverage
"""

import pytest

from modules.auto_trade.core.signal_selector import FinalSignal
from modules.auto_trade.execution.order_builder import OrderBuilder


def test_order_builder_long_roi_mode():
    """
    LONG, leverage=10x, tp_roi=5%, sl_roi=2.5%
    price_move: tp = 5/10 = 0.5%, sl = 2.5/10 = 0.25%
    tp_price = 10000 * 1.005 = 10050.0
    sl_price = 10000 * 0.9975 = 9975.0
    """
    builder = OrderBuilder(default_tp_pct=5.0, default_sl_pct=2.5, default_leverage=10)
    sig = FinalSignal(
        symbol="BTC/USDT",
        signal_type="LONG",
        entry_price=10000.0,
        stop_loss=9975.0,
        take_profit=10050.0,
        leverage=10,
        score=80.0,
    )
    ticket = builder.build_order(sig, position_size=500.0)

    assert ticket.amount == 500.0
    assert ticket.leverage == 10

    tp, sl = builder.calculate_tp_sl_prices(ticket, entry_price=10000.0)
    assert tp == pytest.approx(10050.0), f"Expected tp=10050.0 (0.5% price move), got {tp}"
    assert sl == pytest.approx(9975.0), f"Expected sl=9975.0 (0.25% price move), got {sl}"


def test_order_builder_short_roi_mode():
    """
    SHORT, leverage=10x, tp_roi=5%, sl_roi=2.5%
    price_move: tp = 5/10 = 0.5%, sl = 2.5/10 = 0.25%
    tp_price = 10000 * (1 - 0.005) = 9950.0
    sl_price = 10000 * (1 + 0.0025) = 10025.0
    """
    builder = OrderBuilder(default_tp_pct=5.0, default_sl_pct=2.5, default_leverage=10)
    sig = FinalSignal(
        symbol="BTC/USDT",
        signal_type="SHORT",
        entry_price=10000.0,
        stop_loss=10025.0,
        take_profit=9950.0,
        leverage=10,
        score=80.0,
    )
    ticket = builder.build_order(sig, position_size=500.0)

    tp, sl = builder.calculate_tp_sl_prices(ticket, entry_price=10000.0)
    assert tp == pytest.approx(9950.0), f"Expected tp=9950.0 (0.5% price move), got {tp}"
    assert sl == pytest.approx(10025.0), f"Expected sl=10025.0 (0.25% price move), got {sl}"


def test_order_builder_leverage_1x_equals_price_move():
    """
    With leverage=1x, ROI% == price-move% (no conversion needed).
    LONG, leverage=1x, tp_roi=5%, sl_roi=2.5%
    tp_price = 10000 * 1.05 = 10500.0
    sl_price = 10000 * 0.975 = 9750.0
    """
    builder = OrderBuilder(default_tp_pct=5.0, default_sl_pct=2.5, default_leverage=1)
    sig = FinalSignal(
        symbol="BTC/USDT",
        signal_type="LONG",
        entry_price=10000.0,
        stop_loss=9750.0,
        take_profit=10500.0,
        leverage=1,
        score=80.0,
    )
    ticket = builder.build_order(sig, position_size=500.0)

    tp, sl = builder.calculate_tp_sl_prices(ticket, entry_price=10000.0)
    assert tp == pytest.approx(10500.0), f"Expected tp=10500.0 (5% price move at 1x), got {tp}"
    assert sl == pytest.approx(9750.0), f"Expected sl=9750.0 (2.5% price move at 1x), got {sl}"


def test_roi_scale_formula():
    """
    Verify ROI% / leverage = price-move% at various leverage values.
    Entry=100, roi_tp=10%, roi_sl=5%
    """
    for lev, expected_tp_price, expected_sl_price in [
        (1, 110.0, 95.0),  # 10/1=10%, 5/1=5%
        (5, 102.0, 99.0),  # 10/5=2%,  5/5=1%
        (10, 101.0, 99.5),  # 10/10=1%, 5/10=0.5%
    ]:
        builder = OrderBuilder(default_tp_pct=10.0, default_sl_pct=5.0, default_leverage=lev)
        sig = FinalSignal(
            symbol="ETH/USDT",
            signal_type="LONG",
            entry_price=100.0,
            stop_loss=expected_sl_price,
            take_profit=expected_tp_price,
            leverage=lev,
            score=70.0,
        )
        ticket = builder.build_order(sig, position_size=100.0)
        tp, sl = builder.calculate_tp_sl_prices(ticket, entry_price=100.0)
        assert tp == pytest.approx(expected_tp_price), f"lev={lev}: tp={tp} != {expected_tp_price}"
        assert sl == pytest.approx(expected_sl_price), f"lev={lev}: sl={sl} != {expected_sl_price}"

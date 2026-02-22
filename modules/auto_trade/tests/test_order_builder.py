from modules.auto_trade.core.signal_selector import FinalSignal
from modules.auto_trade.execution.order_builder import OrderBuilder


def test_order_builder_rounding_and_tp_sl():
    builder = OrderBuilder(default_tp_pct=5.0, default_sl_pct=2.5, default_leverage=10)
    sig = FinalSignal(
        symbol="BTC/USDT",
        signal_type="LONG",
        entry_price=10000.0,
        stop_loss=9750.0,
        take_profit=10500.0,
        leverage=10,
        score=80.0,
    )
    ticket = builder.build_order(sig, position_size=500.0)

    assert ticket.amount == 500.0
    assert ticket.leverage == 10

    tp, sl = builder.calculate_tp_sl_prices(ticket, entry_price=10000.0)
    assert tp == 10500.0  # 10000 * 1.05
    assert sl == 9750.0  # 10000 * 0.975


def test_order_builder_short():
    builder = OrderBuilder(default_tp_pct=5.0, default_sl_pct=2.5, default_leverage=10)
    sig = FinalSignal(
        symbol="BTC/USDT",
        signal_type="SHORT",
        entry_price=10000.0,
        stop_loss=10250.0,
        take_profit=9500.0,
        leverage=10,
        score=80.0,
    )
    ticket = builder.build_order(sig, position_size=500.0)

    tp, sl = builder.calculate_tp_sl_prices(ticket, entry_price=10000.0)
    assert tp == 9500.0  # 10000 * 0.95
    assert sl == 10250.0  # 10000 * 1.025

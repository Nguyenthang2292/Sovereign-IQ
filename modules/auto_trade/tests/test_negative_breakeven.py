from modules.auto_trade.execution.negative_breakeven import should_trigger_negative_be


def test_should_trigger_negative_be_long():
    trigger = should_trigger_negative_be(
        profit_pct=-3.0, threshold_pct=2.0, mark_price=97.0, stop_loss=95.0, side="LONG", be_moved=False
    )
    assert trigger is True


def test_should_not_trigger_be_moved():
    trigger = should_trigger_negative_be(
        profit_pct=-3.0, threshold_pct=2.0, mark_price=97.0, stop_loss=95.0, side="LONG", be_moved=True
    )
    assert trigger is False


def test_should_not_trigger_hit_sl_short():
    trigger = should_trigger_negative_be(
        profit_pct=-5.0, threshold_pct=2.0, mark_price=105.0, stop_loss=104.0, side="SHORT", be_moved=False
    )
    # Stop loss is hit for short when mark >= stop_loss
    assert trigger is False

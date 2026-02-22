from modules.auto_trade.execution.trailing_stop import calculate_trailing_stop


def test_long_step_0_be_trigger():
    result = calculate_trailing_stop(entry_price=100.0, current_price=104.0, side="LONG", step_index=0, step_pct=2.0)
    assert result.should_step is True
    assert result.new_sl_price == 100.0  # BE
    assert result.next_step_index == 1


def test_short_step_1_trigger():
    result = calculate_trailing_stop(
        entry_price=100.0, current_price=97.0, side="SHORT", step_index=1, step_pct=2.0, current_sl=100.0
    )
    assert result.should_step is True
    assert result.new_sl_price == 98.0
    assert result.next_step_index == 2


def test_no_trigger_below_threshold():
    result = calculate_trailing_stop(
        entry_price=100.0,
        current_price=101.0,  # 1% < 2% threshold
        side="LONG",
        step_index=1,
        step_pct=2.0,
        current_sl=100.0,
    )
    assert result.should_step is False

from modules.auto_trade.strategies.martingale import MartingaleStrategy


def test_martingale_progression():
    strategy = MartingaleStrategy(initial_leverage=2, max_steps=3, max_leverage=16)

    # Step 0: next leverage is 2
    assert strategy.get_next_leverage() == 2

    # Step 1: record loss
    strategy.record_loss(10.0, leverage=2)
    assert strategy.current_step == 1
    assert strategy.get_next_leverage() == 4

    # Step 2: record loss
    strategy.record_loss(20.0, leverage=4)
    assert strategy.current_step == 2
    assert strategy.get_next_leverage() == 8

    # Step 3: record loss (should cap at max_steps)
    strategy.record_loss(40.0, leverage=8)
    assert strategy.current_step == 3
    assert strategy.should_stop() is True

    # Leverage caps at 16
    assert strategy.get_next_leverage() == 16

    # Recover profit
    strategy.record_profit(80.0)
    assert strategy.is_active is False
    assert strategy.current_step == 0

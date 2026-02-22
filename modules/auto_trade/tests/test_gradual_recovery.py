from modules.auto_trade.strategies.gradual_recovery import GradualRecoveryStrategy, RecoveryConfig


def test_gradual_recovery_basics():
    config: RecoveryConfig = {
        "target_profit_per_trade": 5.0,
        "margin_scaling_mode": "fixed",
        "leverage_scaling_mode": "progressive",
        "min_leverage": 2,
        "max_leverage": 10,
    }

    # Init with $100 loss
    strategy = GradualRecoveryStrategy(initial_loss=100.0, config=config)

    state = strategy.get_state()
    assert state.remaining_loss == 100.0
    assert state.is_complete is False

    # Record profit $50
    strategy.record_profit(50.0)
    state = strategy.get_state()
    assert state.remaining_loss == 50.0
    assert strategy.recovery_percentage == 50.0
    assert strategy.is_active is True

    # Progress leverage -> (10 - 2) * 0.5 + 2 = 6
    next_lev = strategy.calculate_next_leverage()
    assert next_lev == 6

    # Record profit $50 -> Complete
    strategy.record_profit(50.0)
    state = strategy.get_state()
    assert state.remaining_loss == 0.0
    assert state.is_complete is True
    assert strategy.is_active is False

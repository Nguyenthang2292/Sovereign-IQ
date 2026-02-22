from datetime import datetime, timezone
from unittest.mock import MagicMock

from modules.auto_trade.monitoring.event_system import Event, EventType
from modules.auto_trade.strategies.recovery_manager import RecoveryManager


def test_recovery_manager_activation():
    # Setup mock event bus
    bus = MagicMock()

    # Initialize manager
    manager = RecoveryManager(
        event_bus=bus,
        config={"target_profit_per_trade": 5.0, "min_leverage": 2, "max_leverage": 10},
        enabled=True,
        database=None,  # No DB for testing
    )

    manager.start()
    bus.subscribe.assert_called_with(EventType.POSITION_CLOSED, manager._on_position_closed)

    # Simulate a position close (loss)
    loss_event = Event(
        type=EventType.POSITION_CLOSED,
        timestamp=datetime.now(timezone.utc),
        source="test",
        data={"symbol": "BTC", "pnl": -100.0, "is_programmatic": True},
    )
    manager._on_position_closed(loss_event)

    # Recovery should now be active
    assert manager.is_active is True
    assert manager.get_recovery_parameters()["active"] is True
    assert manager.get_state().remaining_loss == 100.0

    # Simulate a position close (profit)
    profit_event = Event(
        type=EventType.POSITION_CLOSED,
        timestamp=datetime.now(timezone.utc),
        source="test",
        data={"symbol": "BTC", "pnl": 50.0, "is_programmatic": True},
    )
    manager._on_position_closed(profit_event)

    assert manager.get_state().remaining_loss == 50.0

    # Simulate second profit, completing recovery
    manager._on_position_closed(profit_event)

    assert manager.is_active is False


def test_recovery_manager_disabled():
    manager = RecoveryManager(event_bus=MagicMock(), config={}, enabled=False, database=None)

    # Simulate a position close (loss)
    loss_event = Event(
        type=EventType.POSITION_CLOSED,
        timestamp=datetime.now(timezone.utc),
        source="test",
        data={"symbol": "BTC", "pnl": -100.0, "is_programmatic": True},
    )
    manager._on_position_closed(loss_event)

    # Recovery should NOT be active since it's disabled
    assert manager.is_active is False

"""
Tests for Event Tracking System.

Tests event types, event creation, event bus subscriptions, publishing, and thread safety.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from modules.auto_trade.monitoring.event_system import Event, EventBus, EventType


class TestEventType:
    """Test EventType enum."""

    def test_all_event_types_exist(self):
        """Test that all expected event types are defined."""
        expected_types = [
            # Pipeline Events
            "PIPELINE_START", "PIPELINE_COMPLETE", "PIPELINE_ERROR",
            # Signal Events
            "SIGNAL_GENERATED", "SIGNAL_REJECTED",
            # Execution Events
            "ORDER_CREATED", "ORDER_FILLED", "ORDER_FAILED",
            # Position Events
            "POSITION_OPENED", "POSITION_CLOSED", "BREAK_EVEN_MOVED",
            # System Events
            "CIRCUIT_OPEN", "HEALTH_CHECK_FAILED"
        ]

        for event_type in expected_types:
            assert hasattr(EventType, event_type)
            assert getattr(EventType, event_type).value == event_type.lower()

    def test_event_type_values_are_strings(self):
        """Test that EventType values are strings."""
        assert isinstance(EventType.PIPELINE_START.value, str)
        assert isinstance(EventType.ORDER_CREATED.value, str)
        assert isinstance(EventType.HEALTH_CHECK_FAILED.value, str)


class TestEvent:
    """Test Event dataclass."""

    def test_event_creation_with_defaults(self):
        """Test creating an event with default values."""
        event = Event(type=EventType.PIPELINE_START)

        assert event.type == EventType.PIPELINE_START
        assert event.data == {}
        assert isinstance(event.timestamp, float)
        assert event.source == "system"

    def test_event_creation_with_custom_data(self):
        """Test creating an event with custom data."""
        data = {"symbol": "BTC/USDT", "price": 50000}
        event = Event(type=EventType.ORDER_CREATED, data=data)

        assert event.type == EventType.ORDER_CREATED
        assert event.data == data
        assert event.data["symbol"] == "BTC/USDT"
        assert event.data["price"] == 50000

    def test_event_timestamp_is_generated(self):
        """Test that timestamp is automatically generated."""
        before = time.time()
        event = Event(type=EventType.SIGNAL_GENERATED)
        after = time.time()

        assert before <= event.timestamp <= after

    def test_event_custom_source(self):
        """Test creating an event with custom source."""
        event = Event(
            type=EventType.POSITION_OPENED,
            source="trading_bot"
        )

        assert event.source == "trading_bot"

    def test_event_with_all_fields(self):
        """Test creating an event with all fields specified."""
        data = {"action": "buy", "quantity": 1.5}
        timestamp = time.time()

        event = Event(
            type=EventType.ORDER_FILLED,
            data=data,
            timestamp=timestamp,
            source="manual"
        )

        assert event.type == EventType.ORDER_FILLED
        assert event.data == data
        assert event.timestamp == timestamp
        assert event.source == "manual"


class TestEventBusSubscription:
    """Test EventBus subscription functionality."""

    def test_subscribe_to_specific_event(self):
        """Test subscribing to a specific event type."""
        bus = EventBus()
        callback = MagicMock()

        bus.subscribe(EventType.ORDER_CREATED, callback)

        assert bus.get_subscriber_count(EventType.ORDER_CREATED) == 1

    def test_subscribe_multiple_callbacks_to_same_event(self):
        """Test subscribing multiple callbacks to the same event."""
        bus = EventBus()
        callback1 = MagicMock()
        callback2 = MagicMock()

        bus.subscribe(EventType.ORDER_CREATED, callback1)
        bus.subscribe(EventType.ORDER_CREATED, callback2)

        assert bus.get_subscriber_count(EventType.ORDER_CREATED) == 2

    def test_subscribe_to_multiple_events(self):
        """Test subscribing to multiple different events."""
        bus = EventBus()
        callback1 = MagicMock()
        callback2 = MagicMock()

        bus.subscribe(EventType.ORDER_CREATED, callback1)
        bus.subscribe(EventType.POSITION_OPENED, callback2)

        assert bus.get_subscriber_count(EventType.ORDER_CREATED) == 1
        assert bus.get_subscriber_count(EventType.POSITION_OPENED) == 1

    def test_subscribe_all(self):
        """Test subscribing to all events."""
        bus = EventBus()
        callback = MagicMock()

        bus.subscribe_all(callback)

        # Global subscribers are counted in total
        assert bus.get_subscriber_count() == 1

    def test_subscribe_duplicate_callback_logs_error(self):
        """Test that subscribing same callback twice logs error."""
        bus = EventBus()
        callback = MagicMock()

        with patch('modules.auto_trade.monitoring.event_system.log_error') as mock_log:
            bus.subscribe(EventType.ORDER_CREATED, callback)
            bus.subscribe(EventType.ORDER_CREATED, callback)

            # Should only have one subscription
            assert bus.get_subscriber_count(EventType.ORDER_CREATED) == 1
            # Should log error on duplicate
            mock_log.assert_called_once()

    def test_subscribe_with_non_callable_raises_error(self):
        """Test that subscribing with non-callable raises TypeError."""
        bus = EventBus()

        with pytest.raises(TypeError, match="callback must be callable"):
            bus.subscribe(EventType.ORDER_CREATED, "not a function")  # type: ignore[arg-type]

    def test_subscribe_all_with_non_callable_raises_error(self):
        """Test that subscribe_all with non-callable raises TypeError."""
        bus = EventBus()

        with pytest.raises(TypeError, match="callback must be callable"):
            bus.subscribe_all(123)  # type: ignore[arg-type]


class TestEventBusUnsubscription:
    """Test EventBus unsubscription functionality."""

    def test_unsubscribe_removes_callback(self):
        """Test that unsubscribe removes the callback."""
        bus = EventBus()
        callback = MagicMock()

        bus.subscribe(EventType.ORDER_CREATED, callback)
        assert bus.get_subscriber_count(EventType.ORDER_CREATED) == 1

        result = bus.unsubscribe(EventType.ORDER_CREATED, callback)

        assert result is True
        assert bus.get_subscriber_count(EventType.ORDER_CREATED) == 0

    def test_unsubscribe_non_existent_callback_returns_false(self):
        """Test that unsubscribing non-existent callback returns False."""
        bus = EventBus()
        callback = MagicMock()

        result = bus.unsubscribe(EventType.ORDER_CREATED, callback)

        assert result is False

    def test_unsubscribe_from_non_existent_event_type_returns_false(self):
        """Test unsubscribing from event type with no subscribers."""
        bus = EventBus()
        callback = MagicMock()

        result = bus.unsubscribe(EventType.PIPELINE_ERROR, callback)

        assert result is False

    def test_unsubscribe_one_of_multiple_callbacks(self):
        """Test unsubscribing one callback when multiple are subscribed."""
        bus = EventBus()
        callback1 = MagicMock()
        callback2 = MagicMock()

        bus.subscribe(EventType.ORDER_CREATED, callback1)
        bus.subscribe(EventType.ORDER_CREATED, callback2)

        bus.unsubscribe(EventType.ORDER_CREATED, callback1)

        assert bus.get_subscriber_count(EventType.ORDER_CREATED) == 1

    def test_unsubscribe_all_removes_global_callback(self):
        """Test that unsubscribe_all removes global callback."""
        bus = EventBus()
        callback = MagicMock()

        bus.subscribe_all(callback)
        assert bus.get_subscriber_count() == 1

        result = bus.unsubscribe_all(callback)

        assert result is True
        assert bus.get_subscriber_count() == 0

    def test_unsubscribe_all_non_existent_returns_false(self):
        """Test unsubscribe_all with non-existent callback returns False."""
        bus = EventBus()
        callback = MagicMock()

        result = bus.unsubscribe_all(callback)

        assert result is False


class TestEventBusPublishing:
    """Test EventBus publishing functionality."""

    def test_publish_to_specific_subscribers(self):
        """Test publishing event to specific subscribers."""
        bus = EventBus()
        callback = MagicMock()

        bus.subscribe(EventType.ORDER_CREATED, callback)

        event = Event(type=EventType.ORDER_CREATED, data={"order_id": "123"})
        bus.publish(event)

        callback.assert_called_once_with(event)

    def test_publish_to_multiple_subscribers(self):
        """Test publishing event to multiple subscribers."""
        bus = EventBus()
        callback1 = MagicMock()
        callback2 = MagicMock()

        bus.subscribe(EventType.ORDER_CREATED, callback1)
        bus.subscribe(EventType.ORDER_CREATED, callback2)

        event = Event(type=EventType.ORDER_CREATED)
        bus.publish(event)

        callback1.assert_called_once_with(event)
        callback2.assert_called_once_with(event)

    def test_publish_to_global_subscribers(self):
        """Test publishing event to global subscribers."""
        bus = EventBus()
        callback = MagicMock()

        bus.subscribe_all(callback)

        event = Event(type=EventType.ORDER_CREATED)
        bus.publish(event)

        callback.assert_called_once_with(event)

    def test_publish_to_both_specific_and_global_subscribers(self):
        """Test event reaches both specific and global subscribers."""
        bus = EventBus()
        specific_callback = MagicMock()
        global_callback = MagicMock()

        bus.subscribe(EventType.ORDER_CREATED, specific_callback)
        bus.subscribe_all(global_callback)

        event = Event(type=EventType.ORDER_CREATED)
        bus.publish(event)

        specific_callback.assert_called_once_with(event)
        global_callback.assert_called_once_with(event)

    def test_publish_with_no_subscribers(self):
        """Test publishing event when no subscribers exist."""
        bus = EventBus()

        event = Event(type=EventType.ORDER_CREATED)
        # Should not raise exception
        bus.publish(event)

    def test_publish_only_notifies_matching_event_type(self):
        """Test that only subscribers to matching event type are notified."""
        bus = EventBus()
        callback1 = MagicMock()
        callback2 = MagicMock()

        bus.subscribe(EventType.ORDER_CREATED, callback1)
        bus.subscribe(EventType.POSITION_OPENED, callback2)

        event = Event(type=EventType.ORDER_CREATED)
        bus.publish(event)

        callback1.assert_called_once_with(event)
        callback2.assert_not_called()

    def test_exception_in_subscriber_does_not_affect_others(self):
        """Test that exception in one subscriber doesn't stop others."""
        bus = EventBus()
        failing_callback = MagicMock(side_effect=Exception("Test error"))
        working_callback = MagicMock()

        bus.subscribe(EventType.ORDER_CREATED, failing_callback)
        bus.subscribe(EventType.ORDER_CREATED, working_callback)

        with patch('modules.auto_trade.monitoring.event_system.log_error'):
            event = Event(type=EventType.ORDER_CREATED)
            bus.publish(event)

        # Both callbacks should be called
        failing_callback.assert_called_once()
        working_callback.assert_called_once()

    def test_exception_in_subscriber_is_logged(self):
        """Test that exceptions in subscribers are logged."""
        bus = EventBus()
        callback = MagicMock(side_effect=Exception("Test error"))

        bus.subscribe(EventType.ORDER_CREATED, callback)

        with patch('modules.auto_trade.monitoring.event_system.log_error') as mock_log:
            event = Event(type=EventType.ORDER_CREATED)
            bus.publish(event)

            mock_log.assert_called_once()
            assert "Error in event subscriber" in str(mock_log.call_args)

    def test_exception_in_global_subscriber_is_logged(self):
        """Test that exceptions in global subscribers are logged."""
        bus = EventBus()
        callback = MagicMock(side_effect=Exception("Test error"))

        bus.subscribe_all(callback)

        with patch('modules.auto_trade.monitoring.event_system.log_error') as mock_log:
            event = Event(type=EventType.ORDER_CREATED)
            bus.publish(event)

            mock_log.assert_called_once()
            assert "Error in global event subscriber" in str(mock_log.call_args)


class TestEventBusCleanup:
    """Test EventBus cleanup functionality."""

    def test_clear_subscribers_removes_all_specific_subscribers(self):
        """Test that clear_subscribers removes all specific subscribers."""
        bus = EventBus()
        callback1 = MagicMock()
        callback2 = MagicMock()

        bus.subscribe(EventType.ORDER_CREATED, callback1)
        bus.subscribe(EventType.POSITION_OPENED, callback2)

        bus.clear_subscribers()

        assert bus.get_subscriber_count() == 0
        assert bus.get_subscriber_count(EventType.ORDER_CREATED) == 0
        assert bus.get_subscriber_count(EventType.POSITION_OPENED) == 0

    def test_clear_subscribers_removes_global_subscribers(self):
        """Test that clear_subscribers removes global subscribers."""
        bus = EventBus()
        callback = MagicMock()

        bus.subscribe_all(callback)

        bus.clear_subscribers()

        assert bus.get_subscriber_count() == 0

    def test_clear_subscribers_removes_all_types(self):
        """Test clearing both specific and global subscribers."""
        bus = EventBus()
        specific = MagicMock()
        global_cb = MagicMock()

        bus.subscribe(EventType.ORDER_CREATED, specific)
        bus.subscribe_all(global_cb)

        bus.clear_subscribers()

        assert bus.get_subscriber_count() == 0


class TestEventBusSubscriberCount:
    """Test EventBus subscriber counting."""

    def test_get_subscriber_count_for_specific_event(self):
        """Test getting subscriber count for specific event type."""
        bus = EventBus()
        callback1 = MagicMock()
        callback2 = MagicMock()

        bus.subscribe(EventType.ORDER_CREATED, callback1)
        bus.subscribe(EventType.ORDER_CREATED, callback2)

        assert bus.get_subscriber_count(EventType.ORDER_CREATED) == 2

    def test_get_subscriber_count_for_event_with_no_subscribers(self):
        """Test getting count for event type with no subscribers."""
        bus = EventBus()

        assert bus.get_subscriber_count(EventType.ORDER_CREATED) == 0

    def test_get_total_subscriber_count(self):
        """Test getting total subscriber count across all events."""
        bus = EventBus()
        callback1 = MagicMock()
        callback2 = MagicMock()
        callback3 = MagicMock()

        bus.subscribe(EventType.ORDER_CREATED, callback1)
        bus.subscribe(EventType.POSITION_OPENED, callback2)
        bus.subscribe_all(callback3)

        # 2 specific + 1 global = 3 total
        assert bus.get_subscriber_count() == 3

    def test_get_subscriber_count_after_unsubscribe(self):
        """Test subscriber count decreases after unsubscribe."""
        bus = EventBus()
        callback = MagicMock()

        bus.subscribe(EventType.ORDER_CREATED, callback)
        assert bus.get_subscriber_count(EventType.ORDER_CREATED) == 1

        bus.unsubscribe(EventType.ORDER_CREATED, callback)
        assert bus.get_subscriber_count(EventType.ORDER_CREATED) == 0


class TestThreadSafety:
    """Test EventBus thread safety."""

    def test_concurrent_subscriptions(self):
        """Test that concurrent subscriptions are thread-safe."""
        bus = EventBus()
        callbacks = [MagicMock() for _ in range(10)]

        def subscribe_callback(cb):
            bus.subscribe(EventType.ORDER_CREATED, cb)

        threads = [threading.Thread(target=subscribe_callback, args=(cb,)) for cb in callbacks]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All callbacks should be subscribed
        assert bus.get_subscriber_count(EventType.ORDER_CREATED) == 10

    def test_concurrent_publishing(self):
        """Test that concurrent publishing is thread-safe."""
        bus = EventBus()
        callback = MagicMock()
        bus.subscribe(EventType.ORDER_CREATED, callback)

        event_count = 10

        def publish_event():
            event = Event(type=EventType.ORDER_CREATED)
            bus.publish(event)

        threads = [threading.Thread(target=publish_event) for _ in range(event_count)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Callback should be called for each published event
        assert callback.call_count == event_count

    def test_concurrent_subscribe_and_publish(self):
        """Test concurrent subscribing and publishing."""
        bus = EventBus()
        results = []

        def subscriber():
            callback = MagicMock()
            bus.subscribe(EventType.ORDER_CREATED, callback)
            results.append(callback)

        def publisher():
            event = Event(type=EventType.ORDER_CREATED)
            bus.publish(event)

        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=subscriber))
            threads.append(threading.Thread(target=publisher))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have 5 subscribers
        assert bus.get_subscriber_count(EventType.ORDER_CREATED) == 5


class TestIntegration:
    """Integration tests for EventBus."""

    def test_complete_event_flow(self):
        """Test complete event flow from subscription to publishing."""
        bus = EventBus()
        events_received = []

        def handler(event: Event):
            events_received.append(event)

        # Subscribe to multiple event types
        bus.subscribe(EventType.ORDER_CREATED, handler)
        bus.subscribe(EventType.ORDER_FILLED, handler)
        bus.subscribe(EventType.POSITION_OPENED, handler)

        # Publish events
        event1 = Event(type=EventType.ORDER_CREATED, data={"order_id": "1"})
        event2 = Event(type=EventType.ORDER_FILLED, data={"order_id": "1"})
        event3 = Event(type=EventType.POSITION_OPENED, data={"position_id": "A"})

        bus.publish(event1)
        bus.publish(event2)
        bus.publish(event3)

        # All events should be received
        assert len(events_received) == 3
        assert events_received[0].type == EventType.ORDER_CREATED
        assert events_received[1].type == EventType.ORDER_FILLED
        assert events_received[2].type == EventType.POSITION_OPENED

    def test_unsubscribe_stops_receiving_events(self):
        """Test that unsubscribing stops receiving events."""
        bus = EventBus()
        callback = MagicMock()

        bus.subscribe(EventType.ORDER_CREATED, callback)

        # Publish first event
        event1 = Event(type=EventType.ORDER_CREATED, data={"order_id": "1"})
        bus.publish(event1)

        # Unsubscribe
        bus.unsubscribe(EventType.ORDER_CREATED, callback)

        # Publish second event
        event2 = Event(type=EventType.ORDER_CREATED, data={"order_id": "2"})
        bus.publish(event2)

        # Callback should only be called once (for first event)
        callback.assert_called_once()

    def test_global_subscriber_receives_all_events(self):
        """Test that global subscriber receives all event types."""
        bus = EventBus()
        events_received = []

        def global_handler(event: Event):
            events_received.append(event.type)

        bus.subscribe_all(global_handler)

        # Publish different event types
        bus.publish(Event(type=EventType.ORDER_CREATED))
        bus.publish(Event(type=EventType.POSITION_OPENED))
        bus.publish(Event(type=EventType.HEALTH_CHECK_FAILED))

        # Should receive all 3 events
        assert len(events_received) == 3
        assert EventType.ORDER_CREATED in events_received
        assert EventType.POSITION_OPENED in events_received
        assert EventType.HEALTH_CHECK_FAILED in events_received

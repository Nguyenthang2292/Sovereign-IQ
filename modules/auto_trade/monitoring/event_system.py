"""
Event System Module

Publish-subscribe event system for position lifecycle and trading events.
Allows components to subscribe to events without tight coupling.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import RLock
from typing import Any, Callable, Dict, List, Optional, Union

from modules.common.ui.logging import log_error


class EventType(Enum):
    """Types of events in the auto-trading system."""

    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_UPDATE = "position_update"
    BE_MOVED = "breakeven_moved"
    BREAK_EVEN_MOVED = "break_even_moved"
    MARTINGALE_TRIGGERED = "martingale_triggered"
    SIGNAL_GENERATED = "signal_generated"
    SIGNAL_REJECTED = "signal_rejected"
    PIPELINE_START = "pipeline_start"
    PIPELINE_COMPLETE = "pipeline_complete"
    PIPELINE_ERROR = "pipeline_error"
    ORDER_CREATED = "order_created"
    ORDER_FILLED = "order_filled"
    ORDER_EXECUTED = "order_executed"
    ORDER_FAILED = "order_failed"
    CIRCUIT_OPEN = "circuit_open"
    HEALTH_CHECK_FAILED = "health_check_failed"
    SETTINGS_SAVED = "settings_saved"
    ERROR = "error"
    SCAN_COMPLETED = "scan_completed"


@dataclass
class Event:
    """Event data structure."""

    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str = "system"

    def to_dict(self) -> dict:
        """Convert event to dictionary."""
        return {
            "type": self.type.value,
            "timestamp": datetime.fromtimestamp(self.timestamp).isoformat(),
            "data": self.data,
            "source": self.source,
        }


class EventSystem:
    """
    Publish-subscribe event system.

    Example:
        >>> event_system = EventSystem()
        >>> event_system.subscribe(EventType.POSITION_OPENED, on_position_opened)
        >>> event_system.publish(EventType.POSITION_OPENED, {"symbol": "BTC/USDT"})
    """

    def __init__(self, max_history: int = 1000):
        """
        Initialize EventSystem.

        Args:
            max_history: Maximum number of events to keep in history
        """
        self._subscribers: Dict[EventType, List[Callable[[Event], None]]] = {}
        self._all_subscribers: List[Callable[[Event], None]] = []
        self._event_history: deque[Event] = deque(maxlen=max_history)
        self._max_history = max_history
        self._lock = RLock()

        # Initialize subscriber lists for all event types
        for event_type in EventType:
            self._subscribers[event_type] = []

    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """
        Subscribe to an event type.

        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when event is published
        """
        if not callable(callback):
            raise TypeError(f"callback must be callable, got {type(callback)}")

        with self._lock:
            if callback in self._subscribers[event_type]:
                log_error(f"Callback already subscribed to {event_type}")
                return
            self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> bool:
        """
        Unsubscribe from an event type.

        Args:
            event_type: Type of event to unsubscribe from
            callback: Callback function to remove
        """
        with self._lock:
            if callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)
                return True
            return False

    def subscribe_all(self, callback: Callable[[Event], None]) -> None:
        """Subscribe to all published events."""
        if not callable(callback):
            raise TypeError(f"callback must be callable, got {type(callback)}")

        with self._lock:
            if callback in self._all_subscribers:
                log_error("Callback already subscribed to all events")
                return
            self._all_subscribers.append(callback)

    def unsubscribe_all(self, callback: Callable[[Event], None]) -> bool:
        """Unsubscribe a callback from all events."""
        with self._lock:
            if callback in self._all_subscribers:
                self._all_subscribers.remove(callback)
                return True
            return False

    def publish(
        self,
        event_or_type: Union[Event, EventType],
        data: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
    ) -> None:
        """
        Publish an event to all subscribers.

        Args:
            event_or_type: Event object or event type
            data: Event data
            source: Optional source identifier
        """
        if isinstance(event_or_type, Event):
            event = event_or_type
        else:
            event = Event(type=event_or_type, data=data or {}, source=source or "system")

        with self._lock:
            self._event_history.append(event)
            callbacks = list(self._subscribers[event.type])
            all_callbacks = list(self._all_subscribers)

        # Notify subscribers outside lock
        for callback in callbacks:
            try:
                callback(event)
            except Exception as e:
                log_error(f"Error in event subscriber: {e}", exc_info=True)

        for callback in all_callbacks:
            try:
                callback(event)
            except Exception as e:
                log_error(f"Error in global event subscriber: {e}", exc_info=True)

    def get_history(self, event_type: Optional[EventType] = None, limit: int = 100) -> List[Event]:
        """
        Get event history.

        Args:
            event_type: Optional filter by event type
            limit: Maximum number of events to return

        Returns:
            List of events (most recent first)
        """
        with self._lock:
            history = list(self._event_history)

        if event_type:
            history = [event for event in history if event.type == event_type]

        return list(reversed(history[-limit:]))

    def clear_history(self) -> None:
        """Clear event history."""
        with self._lock:
            self._event_history.clear()

    def clear_subscribers(self) -> None:
        """Clear all specific and global subscribers."""
        with self._lock:
            for event_type in EventType:
                self._subscribers[event_type].clear()
            self._all_subscribers.clear()

    def get_subscriber_count(self, event_type: Optional[EventType] = None) -> int:
        """Get number of subscribers for an event type."""
        with self._lock:
            if event_type is not None:
                return len(self._subscribers[event_type])
            return sum(len(callbacks) for callbacks in self._subscribers.values()) + len(self._all_subscribers)

    def get_stats(self) -> dict:
        """
        Get event system statistics.

        Returns:
            Dict with total events, subscribers per type, etc.
        """
        with self._lock:
            return {
                "total_events": len(self._event_history),
                "subscribers": {event_type.value: len(callbacks) for event_type, callbacks in self._subscribers.items()},
                "global_subscribers": len(self._all_subscribers),
                "max_history": self._max_history,
            }


class EventBus(EventSystem):
    """Backward-compatible alias for EventSystem."""

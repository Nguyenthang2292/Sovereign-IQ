"""
Event System Module

Publish-subscribe event system for position lifecycle and trading events.
Allows components to subscribe to events without tight coupling.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class EventType(Enum):
    """Types of events in the auto-trading system."""

    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_UPDATE = "position_update"
    BE_MOVED = "breakeven_moved"
    MARTINGALE_TRIGGERED = "martingale_triggered"
    SIGNAL_GENERATED = "signal_generated"
    ORDER_EXECUTED = "order_executed"
    ORDER_FAILED = "order_failed"
    ERROR = "error"
    SCAN_COMPLETED = "scan_completed"


@dataclass
class Event:
    """Event data structure."""

    type: EventType
    timestamp: datetime
    data: Dict[str, Any]
    source: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert event to dictionary."""
        return {
            "type": self.type.value,
            "timestamp": self.timestamp.isoformat(),
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
        self._event_history: List[Event] = []
        self._max_history = max_history

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
        if callback not in self._subscribers[event_type]:
            self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """
        Unsubscribe from an event type.

        Args:
            event_type: Type of event to unsubscribe from
            callback: Callback function to remove
        """
        if callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)

    def publish(self, event_type: EventType, data: Dict[str, Any], source: Optional[str] = None) -> None:
        """
        Publish an event to all subscribers.

        Args:
            event_type: Type of event
            data: Event data
            source: Optional source identifier
        """
        event = Event(type=event_type, timestamp=datetime.now(), data=data, source=source)

        # Add to history
        self._event_history.append(event)

        # Trim history if too long
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history :]

        # Notify subscribers
        for callback in self._subscribers[event_type]:
            try:
                callback(event)
            except Exception as e:
                # Log error but don't let one subscriber break others
                from modules.common.ui.logging import log_error

                log_error(f"Error in event subscriber: {e}", exc_info=True)

    def get_history(self, event_type: Optional[EventType] = None, limit: int = 100) -> List[Event]:
        """
        Get event history.

        Args:
            event_type: Optional filter by event type
            limit: Maximum number of events to return

        Returns:
            List of events (most recent first)
        """
        history = self._event_history

        # Filter by type if specified
        if event_type:
            history = [e for e in history if e.type == event_type]

        # Return most recent events up to limit
        return list(reversed(history[-limit:]))

    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history = []

    def get_subscriber_count(self, event_type: EventType) -> int:
        """Get number of subscribers for an event type."""
        return len(self._subscribers[event_type])

    def get_stats(self) -> dict:
        """
        Get event system statistics.

        Returns:
            Dict with total events, subscribers per type, etc.
        """
        return {
            "total_events": len(self._event_history),
            "subscribers": {event_type.value: len(callbacks) for event_type, callbacks in self._subscribers.items()},
            "max_history": self._max_history,
        }

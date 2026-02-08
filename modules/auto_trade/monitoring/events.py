"""
Event Tracking System.

Implements a Publish-Subscribe pattern for decoupling system components.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any, Callable, Dict, List, Optional

from modules.common.ui.logging import log_error


class EventType(Enum):
    """Enumeration of system event types."""

    # Pipeline Events
    PIPELINE_START = "PIPELINE_START"
    PIPELINE_COMPLETE = "PIPELINE_COMPLETE"
    PIPELINE_ERROR = "PIPELINE_ERROR"

    # Signal Events
    SIGNAL_GENERATED = "SIGNAL_GENERATED"
    SIGNAL_REJECTED = "SIGNAL_REJECTED"

    # Execution Events
    ORDER_CREATED = "ORDER_CREATED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_FAILED = "ORDER_FAILED"

    # Position Events
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_CLOSED = "POSITION_CLOSED"
    BREAK_EVEN_MOVED = "BREAK_EVEN_MOVED"

    # System Events
    CIRCUIT_OPEN = "CIRCUIT_OPEN"
    HEALTH_CHECK_FAILED = "HEALTH_CHECK_FAILED"


@dataclass
class Event:
    """System Event."""

    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str = "system"


class EventBus:
    """
    Central event bus for the application.
    Thread-safe implementation.
    """

    def __init__(self) -> None:
        self._subscribers: Dict[EventType, List[Callable[[Event], None]]] = {}
        self._all_subscribers: List[Callable[[Event], None]] = []
        self._lock = RLock()

    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """
        Subscribe to a specific event type.

        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when event is published

        Raises:
            TypeError: If callback is not callable
        """
        if not callable(callback):
            raise TypeError(f"callback must be callable, got {type(callback)}")

        with self._lock:
            # Prevent duplicate subscriptions
            if event_type in self._subscribers and callback in self._subscribers[event_type]:
                log_error(f"Callback already subscribed to {event_type}")
                return

            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> bool:
        """
        Unsubscribe a callback from a specific event type.

        Args:
            event_type: Type of event to unsubscribe from
            callback: Function to remove from subscribers

        Returns:
            True if callback was found and removed, False otherwise
        """
        with self._lock:
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(callback)
                    # Clean up empty subscriber lists
                    if not self._subscribers[event_type]:
                        del self._subscribers[event_type]
                    return True
                except ValueError:
                    return False
            return False

    def subscribe_all(self, callback: Callable[[Event], None]) -> None:
        """
        Subscribe to all events.

        Args:
            callback: Function to call for all events

        Raises:
            TypeError: If callback is not callable
        """
        if not callable(callback):
            raise TypeError(f"callback must be callable, got {type(callback)}")

        with self._lock:
            # Prevent duplicate subscriptions
            if callback in self._all_subscribers:
                log_error(f"Callback already subscribed to all events")
                return

            self._all_subscribers.append(callback)

    def unsubscribe_all(self, callback: Callable[[Event], None]) -> bool:
        """
        Unsubscribe a callback from all events.

        Args:
            callback: Function to remove from global subscribers

        Returns:
            True if callback was found and removed, False otherwise
        """
        with self._lock:
            try:
                self._all_subscribers.remove(callback)
                return True
            except ValueError:
                return False

    def publish(self, event: Event) -> None:
        """Publish an event to all subscribers."""
        with self._lock:
            # Notify specific subscribers
            if event.type in self._subscribers:
                for callback in self._subscribers[event.type]:
                    try:
                        callback(event)
                    except Exception as e:
                        log_error(f"Error in event subscriber for {event.type}: {e}")

            # Notify global subscribers
            for callback in self._all_subscribers:
                try:
                    callback(event)
                except Exception as e:
                    log_error(f"Error in global event subscriber: {e}")

    def clear_subscribers(self) -> None:
        """
        Clear all subscribers from the event bus.

        Useful for cleanup or testing. Call this method when shutting down
        to ensure proper cleanup and prevent memory leaks.
        """
        with self._lock:
            self._subscribers.clear()
            self._all_subscribers.clear()

    def get_subscriber_count(self, event_type: Optional[EventType] = None) -> int:
        """
        Get the number of subscribers.

        Args:
            event_type: Specific event type to count, or None for total count

        Returns:
            Number of subscribers for the event type, or total if event_type is None
        """
        with self._lock:
            if event_type is not None:
                return len(self._subscribers.get(event_type, []))
            else:
                # Total unique subscribers across all event types + global subscribers
                return sum(len(subs) for subs in self._subscribers.values()) + len(self._all_subscribers)

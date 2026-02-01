"""
Event Tracking System.

Implements a Publish-Subscribe pattern for decoupling system components.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any, Callable, Dict, List

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
        """Subscribe to a specific event type."""
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)

    def subscribe_all(self, callback: Callable[[Event], None]) -> None:
        """Subscribe to all events."""
        with self._lock:
            self._all_subscribers.append(callback)

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

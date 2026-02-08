"""
Alert Management System.

Subscribes to system events and dispatches notifications for critical conditions.
"""

from enum import Enum
from typing import Any, Callable, Dict

from modules.auto_trade.monitoring.events import Event, EventBus, EventType
from modules.common.ui.logging import log_error, log_info, log_warn


class AlertLevel(str, Enum):
    """Alert severity levels."""
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


class AlertManager:
    """
    Manages system alerts and notifications.
    """

    def __init__(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self._setup_subscriptions()

    def _setup_subscriptions(self) -> None:
        """Subscribe to critical events."""
        self.event_bus.subscribe(EventType.PIPELINE_ERROR, self._handle_error)
        self.event_bus.subscribe(EventType.HEALTH_CHECK_FAILED, self._handle_health_failure)
        self.event_bus.subscribe(EventType.CIRCUIT_OPEN, self._handle_circuit_open)
        self.event_bus.subscribe(EventType.SIGNAL_GENERATED, self._handle_signal)

    def _handle_error(self, event: Event) -> None:
        """Handle pipeline error events with proper error handling."""
        try:
            error_msg = self._extract_error_message(event.data)
            self._send_alert(AlertLevel.CRITICAL, f"Pipeline Error: {error_msg}")
        except Exception as e:
            log_error(f"Failed to handle error event: {e}")

    def _handle_health_failure(self, event: Event) -> None:
        """Handle health check failure events with proper error handling."""
        try:
            health_data = self._format_event_data(event.data)
            self._send_alert(AlertLevel.WARNING, f"Health Check Failed: {health_data}")
        except Exception as e:
            log_error(f"Failed to handle health failure event: {e}")

    def _handle_circuit_open(self, event: Event) -> None:
        """Handle circuit breaker open events with proper error handling."""
        try:
            circuit_data = self._format_event_data(event.data)
            self._send_alert(AlertLevel.WARNING, f"Circuit Breaker Opened: {circuit_data}")
        except Exception as e:
            log_error(f"Failed to handle circuit open event: {e}")

    def _handle_signal(self, event: Event) -> None:
        """Handle signal generation events (info notification)."""
        try:
            symbol = self._get_event_field(event.data, "symbol", "UNKNOWN")
            signal_type = self._get_event_field(event.data, "type", "UNKNOWN")
            log_info(f"ALERT [INFO]: Signal Generated - {symbol} {signal_type}")
        except Exception as e:
            log_error(f"Failed to handle signal event: {e}")

    def _extract_error_message(self, data: Any) -> str:
        """Extract error message from event data safely."""
        if isinstance(data, dict):
            return str(data.get("error", "Unknown error"))
        return str(data) if data else "No error details provided"

    def _format_event_data(self, data: Any) -> str:
        """Format event data for display safely."""
        if data is None:
            return "No data provided"
        if isinstance(data, dict):
            return ", ".join(f"{k}={v}" for k, v in data.items()) or "Empty data"
        return str(data)

    def _get_event_field(self, data: Any, field: str, default: str = "UNKNOWN") -> str:
        """Safely extract a field from event data."""
        if isinstance(data, dict):
            value = data.get(field, default)
            return str(value) if value is not None else default
        return default

    def _send_alert(self, level: AlertLevel, message: str) -> None:
        """
        Dispatch alert to configured channels.
        Currently logs to console/file, extensible to Email/Telegram.
        """
        alert_msg = f"ALERT [{level.value}]: {message}"

        log_methods: Dict[AlertLevel, Callable[[str], None]] = {
            AlertLevel.CRITICAL: log_error,
            AlertLevel.WARNING: log_warn,
            AlertLevel.INFO: log_info,
        }

        log_method = log_methods.get(level, log_info)
        log_method(alert_msg)
        # Future: send_email(subject="Critical Alert", body=message) for CRITICAL

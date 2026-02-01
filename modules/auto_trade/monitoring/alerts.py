"""
Alert Management System.

Subscribes to system events and dispatches notifications for critical conditions.
"""

from modules.auto_trade.monitoring.events import Event, EventBus, EventType
from modules.common.ui.logging import log_error, log_info, log_warn


class AlertManager:
    """
    Manages system alerts and notifications.
    """

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._setup_subscriptions()

    def _setup_subscriptions(self) -> None:
        """Subscribe to critical events."""
        self.event_bus.subscribe(EventType.PIPELINE_ERROR, self._handle_error)
        self.event_bus.subscribe(EventType.HEALTH_CHECK_FAILED, self._handle_health_failure)
        self.event_bus.subscribe(EventType.CIRCUIT_OPEN, self._handle_circuit_open)
        self.event_bus.subscribe(
            EventType.SIGNAL_GENERATED, self._handle_signal
        )  # Notification, not alert, but good to know

    def _handle_error(self, event: Event) -> None:
        self._send_alert("CRITICAL", f"Pipeline Error: {event.data.get('error')}")

    def _handle_health_failure(self, event: Event) -> None:
        self._send_alert("WARNING", f"Health Check Failed: {event.data}")

    def _handle_circuit_open(self, event: Event) -> None:
        self._send_alert("WARNING", f"Circuit Breaker Opened: {event.data}")

    def _handle_signal(self, event: Event) -> None:
        # Info notification
        log_info(f"ALERT [INFO]: Signal Generated - {event.data.get('symbol')} {event.data.get('type')}")

    def _send_alert(self, level: str, message: str) -> None:
        """
        Dispatch alert to configured channels.
        Currently logs to console/file, extensible to Email/Telegram.
        """
        alert_msg = f"ALERT [{level}]: {message}"

        if level == "CRITICAL":
            log_error(alert_msg)
            # Future: send_email(subject="Critical Alert", body=message)
        elif level == "WARNING":
            log_warn(alert_msg)
        else:
            log_info(alert_msg)

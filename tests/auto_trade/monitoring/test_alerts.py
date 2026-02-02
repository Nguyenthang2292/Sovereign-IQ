"""
Tests for Alert Management System.

Tests alert handling, event processing, and notification dispatching.
"""

from unittest.mock import MagicMock, call, patch

import pytest

from modules.auto_trade.monitoring.alerts import AlertLevel, AlertManager
from modules.auto_trade.monitoring.events import Event, EventBus, EventType


@pytest.fixture
def mock_event_bus():
    """Create a mock event bus for testing."""
    return MagicMock(spec=EventBus)


@pytest.fixture
def alert_manager(mock_event_bus):
    """Create an AlertManager instance with mocked event bus."""
    return AlertManager(mock_event_bus)


class TestAlertManagerInitialization:
    """Test AlertManager initialization and setup."""

    def test_init_subscribes_to_events(self, mock_event_bus, alert_manager):
        """Test that AlertManager subscribes to all required events."""
        assert mock_event_bus.subscribe.call_count == 4

        # Verify subscriptions
        calls = mock_event_bus.subscribe.call_args_list
        event_types = [call[0][0] for call in calls]

        assert EventType.PIPELINE_ERROR in event_types
        assert EventType.HEALTH_CHECK_FAILED in event_types
        assert EventType.CIRCUIT_OPEN in event_types
        assert EventType.SIGNAL_GENERATED in event_types


class TestAlertLevelEnum:
    """Test AlertLevel enum."""

    def test_alert_levels_exist(self):
        """Test that all alert levels are defined."""
        assert AlertLevel.CRITICAL.value == "CRITICAL"
        assert AlertLevel.WARNING.value == "WARNING"
        assert AlertLevel.INFO.value == "INFO"

    def test_alert_level_is_string(self):
        """Test that AlertLevel values are strings."""
        assert isinstance(AlertLevel.CRITICAL.value, str)
        assert isinstance(AlertLevel.WARNING.value, str)
        assert isinstance(AlertLevel.INFO.value, str)


class TestErrorEventHandling:
    """Test handling of error events."""

    @patch('modules.auto_trade.monitoring.alerts.log_error')
    def test_handle_error_with_dict_data(self, mock_log, alert_manager):
        """Test handling error event with dictionary data."""
        event = Event(
            type=EventType.PIPELINE_ERROR,
            data={'error': 'Database connection failed'}
        )

        alert_manager._handle_error(event)

        mock_log.assert_called_once()
        call_args = mock_log.call_args[0][0]
        assert "ALERT [CRITICAL]" in call_args
        assert "Pipeline Error: Database connection failed" in call_args

    @patch('modules.auto_trade.monitoring.alerts.log_error')
    def test_handle_error_with_missing_error_field(self, mock_log, alert_manager):
        """Test handling error event with missing error field."""
        event = Event(
            type=EventType.PIPELINE_ERROR,
            data={'status': 'failed'}
        )

        alert_manager._handle_error(event)

        mock_log.assert_called_once()
        call_args = mock_log.call_args[0][0]
        assert "Unknown error" in call_args

    @patch('modules.auto_trade.monitoring.alerts.log_error')
    def test_handle_error_with_none_data(self, mock_log, alert_manager):
        """Test handling error event with None data."""
        event = Event(
            type=EventType.PIPELINE_ERROR,
            data=None
        )

        alert_manager._handle_error(event)

        mock_log.assert_called_once()
        call_args = mock_log.call_args[0][0]
        assert "No error details provided" in call_args

    @patch('modules.auto_trade.monitoring.alerts.log_error')
    def test_handle_error_with_exception(self, mock_log, alert_manager):
        """Test that exceptions in error handler are caught."""
        # Mock the helper method to raise an exception
        with patch.object(alert_manager, '_extract_error_message', side_effect=Exception("Test exception")):
            event = Event(
                type=EventType.PIPELINE_ERROR,
                data={'error': 'test'}
            )

            # Should not raise exception
            alert_manager._handle_error(event)

        # Should log the handler failure
        assert mock_log.call_count >= 1
        handler_error_logged = any(
            "Failed to handle error event" in str(call[0][0])
            for call in mock_log.call_args_list
        )
        assert handler_error_logged


class TestHealthCheckEventHandling:
    """Test handling of health check failure events."""

    @patch('modules.auto_trade.monitoring.alerts.log_warn')
    def test_handle_health_failure_with_dict_data(self, mock_log, alert_manager):
        """Test handling health check failure with dictionary data."""
        event = Event(
            type=EventType.HEALTH_CHECK_FAILED,
            data={'service': 'database', 'status': 'unhealthy'}
        )

        alert_manager._handle_health_failure(event)

        mock_log.assert_called_once()
        call_args = mock_log.call_args[0][0]
        assert "ALERT [WARNING]" in call_args
        assert "Health Check Failed" in call_args
        assert "service=database" in call_args

    @patch('modules.auto_trade.monitoring.alerts.log_warn')
    def test_handle_health_failure_with_none_data(self, mock_log, alert_manager):
        """Test handling health check failure with None data."""
        event = Event(
            type=EventType.HEALTH_CHECK_FAILED,
            data=None
        )

        alert_manager._handle_health_failure(event)

        mock_log.assert_called_once()
        call_args = mock_log.call_args[0][0]
        assert "No data provided" in call_args

    @patch('modules.auto_trade.monitoring.alerts.log_error')
    def test_handle_health_failure_with_exception(self, mock_log, alert_manager):
        """Test that exceptions in health failure handler are caught."""
        # Mock the helper method to raise an exception
        with patch.object(alert_manager, '_format_event_data', side_effect=Exception("Test exception")):
            event = Event(
                type=EventType.HEALTH_CHECK_FAILED,
                data={'service': 'test'}
            )

            alert_manager._handle_health_failure(event)

        # Should log the handler failure
        handler_error_logged = any(
            "Failed to handle health failure event" in str(call[0][0])
            for call in mock_log.call_args_list
        )
        assert handler_error_logged


class TestCircuitBreakerEventHandling:
    """Test handling of circuit breaker open events."""

    @patch('modules.auto_trade.monitoring.alerts.log_warn')
    def test_handle_circuit_open_with_dict_data(self, mock_log, alert_manager):
        """Test handling circuit breaker open with dictionary data."""
        event = Event(
            type=EventType.CIRCUIT_OPEN,
            data={'circuit': 'payment_api', 'reason': 'too_many_failures'}
        )

        alert_manager._handle_circuit_open(event)

        mock_log.assert_called_once()
        call_args = mock_log.call_args[0][0]
        assert "ALERT [WARNING]" in call_args
        assert "Circuit Breaker Opened" in call_args
        assert "circuit=payment_api" in call_args

    @patch('modules.auto_trade.monitoring.alerts.log_error')
    def test_handle_circuit_open_with_exception(self, mock_log, alert_manager):
        """Test that exceptions in circuit open handler are caught."""
        # Mock the helper method to raise an exception
        with patch.object(alert_manager, '_format_event_data', side_effect=Exception("Test exception")):
            event = Event(
                type=EventType.CIRCUIT_OPEN,
                data={'circuit': 'test'}
            )

            alert_manager._handle_circuit_open(event)

        # Should log the handler failure
        handler_error_logged = any(
            "Failed to handle circuit open event" in str(call[0][0])
            for call in mock_log.call_args_list
        )
        assert handler_error_logged


class TestSignalEventHandling:
    """Test handling of signal generation events."""

    @patch('modules.auto_trade.monitoring.alerts.log_info')
    def test_handle_signal_with_complete_data(self, mock_log, alert_manager):
        """Test handling signal event with complete data."""
        event = Event(
            type=EventType.SIGNAL_GENERATED,
            data={'symbol': 'BTC/USDT', 'type': 'BUY'}
        )

        alert_manager._handle_signal(event)

        mock_log.assert_called_once()
        call_args = mock_log.call_args[0][0]
        assert "ALERT [INFO]" in call_args
        assert "Signal Generated" in call_args
        assert "BTC/USDT" in call_args
        assert "BUY" in call_args

    @patch('modules.auto_trade.monitoring.alerts.log_info')
    def test_handle_signal_with_missing_fields(self, mock_log, alert_manager):
        """Test handling signal event with missing fields."""
        event = Event(
            type=EventType.SIGNAL_GENERATED,
            data={'symbol': 'BTC/USDT'}
        )

        alert_manager._handle_signal(event)

        mock_log.assert_called_once()
        call_args = mock_log.call_args[0][0]
        assert "UNKNOWN" in call_args

    @patch('modules.auto_trade.monitoring.alerts.log_error')
    def test_handle_signal_with_exception(self, mock_log, alert_manager):
        """Test that exceptions in signal handler are caught."""
        # Mock the helper method to raise an exception
        with patch.object(alert_manager, '_get_event_field', side_effect=Exception("Test exception")):
            event = Event(
                type=EventType.SIGNAL_GENERATED,
                data={'symbol': 'BTC/USDT'}
            )

            alert_manager._handle_signal(event)

        # Should log the handler failure
        handler_error_logged = any(
            "Failed to handle signal event" in str(call[0][0])
            for call in mock_log.call_args_list
        )
        assert handler_error_logged


class TestDataExtractionHelpers:
    """Test helper methods for data extraction."""

    def test_extract_error_message_with_dict(self, alert_manager):
        """Test extracting error message from dictionary."""
        result = alert_manager._extract_error_message({'error': 'Connection lost'})
        assert result == 'Connection lost'

    def test_extract_error_message_with_missing_key(self, alert_manager):
        """Test extracting error message with missing key."""
        result = alert_manager._extract_error_message({'status': 'failed'})
        assert result == 'Unknown error'

    def test_extract_error_message_with_none(self, alert_manager):
        """Test extracting error message from None."""
        result = alert_manager._extract_error_message(None)
        assert result == 'No error details provided'

    def test_extract_error_message_with_string(self, alert_manager):
        """Test extracting error message from string."""
        result = alert_manager._extract_error_message('Direct error message')
        assert result == 'Direct error message'

    def test_format_event_data_with_dict(self, alert_manager):
        """Test formatting dictionary event data."""
        result = alert_manager._format_event_data({'key1': 'value1', 'key2': 'value2'})
        assert 'key1=value1' in result
        assert 'key2=value2' in result

    def test_format_event_data_with_empty_dict(self, alert_manager):
        """Test formatting empty dictionary."""
        result = alert_manager._format_event_data({})
        assert result == 'Empty data'

    def test_format_event_data_with_none(self, alert_manager):
        """Test formatting None data."""
        result = alert_manager._format_event_data(None)
        assert result == 'No data provided'

    def test_get_event_field_with_existing_field(self, alert_manager):
        """Test getting existing field from event data."""
        result = alert_manager._get_event_field({'symbol': 'BTC/USDT'}, 'symbol')
        assert result == 'BTC/USDT'

    def test_get_event_field_with_missing_field(self, alert_manager):
        """Test getting missing field from event data."""
        result = alert_manager._get_event_field({'other': 'value'}, 'symbol')
        assert result == 'UNKNOWN'

    def test_get_event_field_with_custom_default(self, alert_manager):
        """Test getting missing field with custom default."""
        result = alert_manager._get_event_field({}, 'symbol', 'N/A')
        assert result == 'N/A'

    def test_get_event_field_with_none_value(self, alert_manager):
        """Test getting field with None value."""
        result = alert_manager._get_event_field({'symbol': None}, 'symbol')
        assert result == 'UNKNOWN'


class TestSendAlert:
    """Test alert dispatching."""

    @patch('modules.auto_trade.monitoring.alerts.log_error')
    def test_send_alert_critical(self, mock_log, alert_manager):
        """Test sending critical alert."""
        alert_manager._send_alert(AlertLevel.CRITICAL, "Critical error")

        mock_log.assert_called_once()
        call_args = mock_log.call_args[0][0]
        assert "ALERT [CRITICAL]" in call_args
        assert "Critical error" in call_args

    @patch('modules.auto_trade.monitoring.alerts.log_warn')
    def test_send_alert_warning(self, mock_log, alert_manager):
        """Test sending warning alert."""
        alert_manager._send_alert(AlertLevel.WARNING, "Warning message")

        mock_log.assert_called_once()
        call_args = mock_log.call_args[0][0]
        assert "ALERT [WARNING]" in call_args
        assert "Warning message" in call_args

    @patch('modules.auto_trade.monitoring.alerts.log_info')
    def test_send_alert_info(self, mock_log, alert_manager):
        """Test sending info alert."""
        alert_manager._send_alert(AlertLevel.INFO, "Info message")

        mock_log.assert_called_once()
        call_args = mock_log.call_args[0][0]
        assert "ALERT [INFO]" in call_args
        assert "Info message" in call_args


class TestIntegration:
    """Integration tests for AlertManager."""

    @patch('modules.auto_trade.monitoring.alerts.log_error')
    @patch('modules.auto_trade.monitoring.alerts.log_warn')
    @patch('modules.auto_trade.monitoring.alerts.log_info')
    def test_multiple_events_handling(self, mock_info, mock_warn, mock_error, alert_manager):
        """Test handling multiple events in sequence."""
        # Error event
        error_event = Event(
            type=EventType.PIPELINE_ERROR,
            data={'error': 'Test error'}
        )
        alert_manager._handle_error(error_event)

        # Health check event
        health_event = Event(
            type=EventType.HEALTH_CHECK_FAILED,
            data={'service': 'db'}
        )
        alert_manager._handle_health_failure(health_event)

        # Signal event
        signal_event = Event(
            type=EventType.SIGNAL_GENERATED,
            data={'symbol': 'BTC/USDT', 'type': 'BUY'}
        )
        alert_manager._handle_signal(signal_event)

        # Verify all handlers were called
        assert mock_error.call_count >= 1
        assert mock_warn.call_count >= 1
        assert mock_info.call_count >= 1

    def test_alert_manager_with_real_event_bus(self):
        """Test AlertManager with real EventBus instance."""
        event_bus = EventBus()
        manager = AlertManager(event_bus)

        # Verify manager is properly initialized
        assert manager.event_bus is event_bus
        assert hasattr(manager, '_handle_error')
        assert hasattr(manager, '_handle_health_failure')
        assert hasattr(manager, '_handle_circuit_open')
        assert hasattr(manager, '_handle_signal')

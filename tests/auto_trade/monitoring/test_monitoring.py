"""Tests for Logging & Monitoring modules."""

import json
import logging
from unittest.mock import patch

from modules.auto_trade.monitoring.alerts import AlertManager
from modules.auto_trade.monitoring.audit import AuditLogger
from modules.auto_trade.monitoring.events import Event, EventBus, EventType
from modules.auto_trade.monitoring.logger import JSONFormatter, get_logger, setup_logging
from modules.auto_trade.monitoring.metrics import MetricsCollector


class TestLogger:
    def test_json_formatter(self):
        formatter = JSONFormatter()
        record = logging.LogRecord("test", logging.INFO, "path", 10, "message", (), None)
        json_str = formatter.format(record)
        data = json.loads(json_str)
        assert data["message"] == "message"
        assert data["level"] == "INFO"
        assert "timestamp" in data

    def test_setup_logging(self, tmp_path):
        log_dir = tmp_path / "logs"
        setup_logging(log_dir=str(log_dir))

        logger = get_logger("test_logger")
        logger.info("test log entry")

        log_file = log_dir / "system.log"
        assert log_file.exists()

        # Read file content
        with open(log_file, "r") as f:
            content = f.read()
            assert "test log entry" in content
            assert '"level": "INFO"' in content


class TestEventBus:
    def test_publish_subscribe(self):
        bus = EventBus()
        received_events = []

        def callback(event):
            received_events.append(event)

        bus.subscribe(EventType.PIPELINE_START, callback)
        bus.publish(Event(EventType.PIPELINE_START))
        bus.publish(Event(EventType.PIPELINE_COMPLETE))  # Should not be received

        assert len(received_events) == 1
        assert received_events[0].type == EventType.PIPELINE_START

    def test_subscribe_all(self):
        bus = EventBus()
        received_events = []

        bus.subscribe_all(lambda e: received_events.append(e))

        bus.publish(Event(EventType.PIPELINE_START))
        bus.publish(Event(EventType.PIPELINE_COMPLETE))

        assert len(received_events) == 2


class TestMetricsCollector:
    def test_counters(self):
        metrics = MetricsCollector()
        metrics.increment("test_counter")
        metrics.increment("test_counter", 2)

        data = metrics.get_metrics()
        assert data["counters"]["test_counter"] == 3

    def test_gauges(self):
        metrics = MetricsCollector()
        metrics.gauge("test_gauge", 42.5)

        data = metrics.get_metrics()
        assert data["gauges"]["test_gauge"] == 42.5

    def test_histograms(self):
        metrics = MetricsCollector()
        metrics.histogram("test_hist", 1.0)
        metrics.histogram("test_hist", 2.0)

        data = metrics.get_metrics()
        assert len(data["histograms"]["test_hist"]) == 2
        assert sum(data["histograms"]["test_hist"]) == 3.0


class TestAuditLogger:
    def test_audit_log(self, tmp_path):
        log_dir = tmp_path / "audit_logs"
        audit = AuditLogger(log_dir=str(log_dir))

        audit.log_event("TEST_EVENT", {"key": "value"})

        log_file = log_dir / "audit.log"
        assert log_file.exists()

        with open(log_file, "r") as f:
            line = f.readline()
            data = json.loads(line)
            assert data["event_type"] == "TEST_EVENT"
            assert data["details"]["key"] == "value"


class TestAlertManager:
    @patch("modules.auto_trade.monitoring.alerts.log_error")
    def test_critical_alert(self, mock_log):
        bus = EventBus()
        _ = AlertManager(bus)

        bus.publish(Event(EventType.PIPELINE_ERROR, {"error": "Keep Calm"}))

        mock_log.assert_called()
        args, _ = mock_log.call_args
        assert "CRITICAL" in args[0]
        assert "Keep Calm" in args[0]

    @patch("modules.auto_trade.monitoring.alerts.log_warn")
    def test_warning_alert(self, mock_log):
        bus = EventBus()
        _ = AlertManager(bus)

        bus.publish(Event(EventType.CIRCUIT_OPEN, {"service": "Gemini"}))

        mock_log.assert_called()
        args, _ = mock_log.call_args
        assert "WARNING" in args[0]

"""
Unit tests for scanner overlap guard: prevent concurrent scan cycles and log duration.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _make_scanner_parent(*, mode="DRY_RUN"):
    """Create a mock parent for ScannerManager."""
    parent = SimpleNamespace()
    parent.mode = mode
    parent.settings_manager = MagicMock()
    parent.settings_manager.get.return_value = None
    parent.data_service = MagicMock()
    parent.data_service.get_signals.return_value = []
    parent._update_queue = MagicMock()
    return parent


def test_overlap_guard_skips_second_thread():
    """When _scanner_cycle is called while another is running, second gets skipped."""
    from modules.auto_trade.gui.main_window.scanner import ScannerManager

    parent = _make_scanner_parent(mode="DRY_RUN")
    manager = ScannerManager(parent)

    # Simulate first scan is running by setting flag
    manager._scan_running = True

    with patch("modules.auto_trade.gui.main_window.scanner.logger") as mock_logger:
        # Second call should be skipped since _scan_running is True
        manager._scanner_cycle()

        # Check skip warning was logged
        skip_logs = [call for call in mock_logger.warning.call_args_list if "already in progress" in str(call)]
        assert len(skip_logs) == 1, "Skip message should be logged when scan already running"

    # _scan_running should remain True (wasn't cleared since we skipped)
    assert manager._scan_running, "_scan_running should remain True when skipped"


def test_overlap_guard_logs_duration():
    """Completed scan cycle logs duration."""
    from modules.auto_trade.gui.main_window.scanner import ScannerManager

    parent = _make_scanner_parent(mode="DRY_RUN")
    manager = ScannerManager(parent)

    with patch("modules.auto_trade.gui.main_window.scanner.logger") as mock_logger:
        manager._scanner_cycle()

        # Check that duration was logged
        duration_logs = [call for call in mock_logger.info.call_args_list if "Scanner cycle completed in" in str(call)]
        assert len(duration_logs) == 1, "Duration should be logged exactly once"
        # Verify format contains seconds
        log_message = str(duration_logs[0])
        assert "s" in log_message, "Duration should be in seconds"


def test_overlap_guard_logs_skip_message():
    """When scan is skipped due to overlap, 'skipped' message is logged."""
    from modules.auto_trade.gui.main_window.scanner import ScannerManager

    parent = _make_scanner_parent(mode="DRY_RUN")
    manager = ScannerManager(parent)

    # Set _scan_running to True to simulate an ongoing scan
    manager._scan_running = True

    with patch("modules.auto_trade.gui.main_window.scanner.logger") as mock_logger:
        manager._scanner_cycle()

        # Check that skip warning was logged
        skip_logs = [
            call
            for call in mock_logger.warning.call_args_list
            if "already in progress" in str(call) or "skipping" in str(call).lower()
        ]
        assert len(skip_logs) == 1, "Skip message should be logged"

    # _scan_running should remain True (not cleared since we didn't actually run)
    assert manager._scan_running, "_scan_running should remain True when skipped"


def test_scan_lock_is_cleared_after_exception():
    """Even if scan raises exception, _scan_running is cleared."""
    from modules.auto_trade.gui.main_window.scanner import ScannerManager

    parent = _make_scanner_parent(mode="PRODUCTION")
    manager = ScannerManager(parent)

    # Make data_service raise an exception
    parent.data_service.get_signals.side_effect = RuntimeError("DB error")

    # Should not raise, but should clear _scan_running
    manager._scanner_cycle()

    # Verify flag is cleared even after exception
    assert not manager._scan_running, "_scan_running should be False after exception"

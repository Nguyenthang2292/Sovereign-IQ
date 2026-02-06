"""
Unit tests for scanner gate: skip full scan (Gemini) when open positions >= max.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _make_scanner_parent(*, mode="PRODUCTION", max_open_positions=1):
    parent = SimpleNamespace()
    parent.mode = mode
    parent.settings_manager = MagicMock()
    parent.settings_manager.get.side_effect = lambda key, default=None: (
        max_open_positions if key == "risk.max_open_positions" else default
    )
    parent.data_service = MagicMock()
    parent.data_service.get_signals.return_value = []
    parent._update_queue = MagicMock()
    return parent


def test_scanner_skips_when_open_count_ge_max():
    """When DB has open_count >= max_open_positions, _run_signal_scan is not called."""
    from modules.auto_trade.gui.main_window.scanner import ScannerManager

    parent = _make_scanner_parent(mode="PRODUCTION", max_open_positions=1)
    manager = ScannerManager(parent)

    # One open order in DB -> gate should skip full scan
    mock_open_orders = [MagicMock()]

    with patch("modules.auto_trade.database.get_open_positions") as mock_get:
        with patch("modules.auto_trade.database.session_scope") as mock_scope:
            mock_get.return_value = mock_open_orders
            ctx = MagicMock()
            ctx.__enter__ = MagicMock(return_value=MagicMock())
            ctx.__exit__ = MagicMock(return_value=False)
            mock_scope.return_value = ctx

            with patch.object(manager, "_run_signal_scan") as mock_run:
                manager._scanner_cycle()

                mock_run.assert_not_called()

    # scanner_done should be put with skipped=True and count=1
    put_calls = parent._update_queue.put.call_args_list
    payloads = [c[0][0] if isinstance(c[0][0], tuple) else c[0] for c in put_calls]
    scanner_done_payloads = [p for p in payloads if isinstance(p, tuple) and p[0] == "scanner_done"]
    assert len(scanner_done_payloads) == 1
    assert scanner_done_payloads[0][1] == {"skipped": True, "count": 1}


def test_scanner_runs_when_open_count_lt_max():
    """When DB has open_count < max_open_positions, _run_signal_scan is called."""
    from modules.auto_trade.gui.main_window.scanner import ScannerManager

    parent = _make_scanner_parent(mode="PRODUCTION", max_open_positions=1)
    manager = ScannerManager(parent)

    # Zero open orders -> full scan should run
    with patch("modules.auto_trade.database.get_open_positions") as mock_get:
        with patch("modules.auto_trade.database.session_scope") as mock_scope:
            mock_get.return_value = []
            ctx = MagicMock()
            ctx.__enter__ = MagicMock(return_value=MagicMock())
            ctx.__exit__ = MagicMock(return_value=False)
            mock_scope.return_value = ctx

            with patch.object(manager, "_run_signal_scan", return_value=None) as mock_run:
                manager._scanner_cycle()

                mock_run.assert_called_once()

    # scanner_done should be put with None (full run)
    put_calls = parent._update_queue.put.call_args_list
    payloads = [c[0][0] if isinstance(c[0][0], tuple) else c[0] for c in put_calls]
    scanner_done_payloads = [p for p in payloads if isinstance(p, tuple) and p[0] == "scanner_done"]
    assert len(scanner_done_payloads) == 1
    assert scanner_done_payloads[0][1] is None


def test_scanner_skips_on_db_error():
    """When get_open_positions/session_scope raises, scan is skipped (no Gemini)."""
    from modules.auto_trade.gui.main_window.scanner import ScannerManager

    parent = _make_scanner_parent(mode="PRODUCTION", max_open_positions=1)
    manager = ScannerManager(parent)

    with patch("modules.auto_trade.database.session_scope") as mock_scope:
        mock_scope.side_effect = RuntimeError("DB unavailable")

        with patch.object(manager, "_run_signal_scan") as mock_run:
            manager._scanner_cycle()

            mock_run.assert_not_called()

    put_calls = parent._update_queue.put.call_args_list
    payloads = [c[0][0] if isinstance(c[0][0], tuple) else c[0] for c in put_calls]
    scanner_done_payloads = [p for p in payloads if isinstance(p, tuple) and p[0] == "scanner_done"]
    assert len(scanner_done_payloads) == 1
    assert scanner_done_payloads[0][1] == {"skipped": True, "count": 0}

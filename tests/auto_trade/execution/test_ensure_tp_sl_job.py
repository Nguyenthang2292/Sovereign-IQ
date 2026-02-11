"""
Tests for Ensure TP/SL Job
===========================

Unit tests for _tp_sl_prices_from_pct and EnsureTPSLJob (with mocked DB/client).
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent
auto_trade_root = project_root / "modules" / "auto_trade"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(auto_trade_root) not in sys.path:
    sys.path.insert(0, str(auto_trade_root))

from unittest.mock import MagicMock, patch

from modules.auto_trade.execution.ensure_tp_sl_job import (
    EnsureTPSLJob,
    _tp_sl_prices_from_pct,
    create_ensure_tp_sl_job,
)


class TestTpSlPricesFromPct:
    """Test _tp_sl_prices_from_pct helper."""

    def test_long_tp_above_sl_below(self):
        """LONG: TP above entry, SL below entry."""
        tp, sl = _tp_sl_prices_from_pct(100.0, "LONG", 5.0, 2.5)
        assert tp == 105.0
        assert sl == 97.5

    def test_short_tp_below_sl_above(self):
        """SHORT: TP below entry, SL above entry."""
        tp, sl = _tp_sl_prices_from_pct(100.0, "SHORT", 5.0, 2.5)
        assert tp == 95.0
        assert sl == 102.5

    def test_long_case_insensitive(self):
        """Side is case insensitive."""
        tp1, sl1 = _tp_sl_prices_from_pct(100.0, "long", 5.0, 2.5)
        tp2, sl2 = _tp_sl_prices_from_pct(100.0, "LONG", 5.0, 2.5)
        assert tp1 == tp2 == 105.0
        assert sl1 == sl2 == 97.5

    def test_zero_entry_returns_zeros(self):
        """Zero or invalid entry returns (0, 0)."""
        assert _tp_sl_prices_from_pct(0.0, "LONG", 5.0, 2.5) == (0.0, 0.0)
        assert _tp_sl_prices_from_pct(-1.0, "LONG", 5.0, 2.5) == (0.0, 0.0)


class TestEnsureTPSLJob:
    """Test EnsureTPSLJob with mocked session and client."""

    def test_run_returns_structure(self):
        """run() returns dict with orders_checked, tp_added, sl_added, errors, updates."""
        settings = MagicMock()
        settings.get.return_value = {"default_tp": 5.0, "default_sl": 2.5}
        session_mock = MagicMock()
        session_mock.commit = MagicMock()
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=session_mock)
        ctx.__exit__ = MagicMock(return_value=None)
        scope_mock = MagicMock(return_value=ctx)
        job = EnsureTPSLJob(
            settings_manager=settings,
            db_session_scope=scope_mock,
            binance_client=None,
        )
        with patch("modules.auto_trade.execution.ensure_tp_sl_job.get_open_positions", return_value=[]):
            result = job.run()
        assert "orders_checked" in result
        assert "tp_added" in result
        assert "sl_added" in result
        assert "errors" in result
        assert "updates" in result
        assert result["orders_checked"] == 0
        assert result["tp_added"] == 0
        assert result["sl_added"] == 0

    def test_create_ensure_tp_sl_job_factory(self):
        """create_ensure_tp_sl_job returns EnsureTPSLJob instance."""
        settings = MagicMock()
        scope = MagicMock()
        job = create_ensure_tp_sl_job(settings, scope, binance_client=None)
        assert isinstance(job, EnsureTPSLJob)
        assert job.settings_manager is settings
        assert job.db_session_scope is scope
        assert job.binance_client is None

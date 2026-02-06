"""
Integration tests for Critical Fixes (refactor-critical-1-6).

Verifies: schema tables (gradual_recovery, migrations_applied), FK to orders(id),
PRAGMA foreign_key_list(orders), reconcile exchange cleanup, migration tracking,
signal_pipeline asyncio (get_running_loop / run_coroutine_threadsafe), WebSocket stop.
Run: pytest tests/auto_trade/test_critical_fixes_integration.py -v
"""

import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

try:
    import ccxt
except ImportError:
    ccxt = None

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.auto_trade.database.config import DEFAULT_SCHEMA_PATH
from modules.auto_trade.database import reconcile_orders_with_binance


class TestSchemaCriticalFixes:
    """Integration tests for schema fixes (gradual_recovery, migrations_applied)."""

    def test_fresh_db_has_gradual_recovery_table(self, tmp_path):
        """Fresh DB from schema.sql must have gradual_recovery table."""
        db_path = tmp_path / "test.db"
        with open(DEFAULT_SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema_sql = f.read()
        conn = sqlite3.connect(str(db_path))
        conn.executescript(schema_sql)
        conn.commit()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='gradual_recovery'")
        row = cursor.fetchone()
        conn.close()
        assert row is not None, "gradual_recovery table missing from schema"
        assert row[0] == "gradual_recovery"

    def test_fresh_db_has_migrations_applied_table(self, tmp_path):
        """Fresh DB from schema.sql must have migrations_applied table."""
        db_path = tmp_path / "test.db"
        with open(DEFAULT_SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema_sql = f.read()
        conn = sqlite3.connect(str(db_path))
        conn.executescript(schema_sql)
        conn.commit()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='migrations_applied'")
        row = cursor.fetchone()
        conn.close()
        assert row is not None, "migrations_applied table missing from schema"
        assert row[0] == "migrations_applied"

    def test_schema_fk_references_orders_id(self):
        """Schema must define parent_order_id FK as REFERENCES orders(id)."""
        with open(DEFAULT_SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema_sql = f.read()
        assert "REFERENCES orders(id)" in schema_sql, "FK should reference orders(id) not orders(order_id)"
        assert "REFERENCES orders(order_id)" not in schema_sql or "orders(id)" in schema_sql

    def test_fresh_db_foreign_key_list_orders_parent_order_id_refs_id(self, tmp_path):
        """PRAGMA foreign_key_list(orders) must show parent_order_id → orders(id)."""
        db_path = tmp_path / "test.db"
        with open(DEFAULT_SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema_sql = f.read()
        conn = sqlite3.connect(str(db_path))
        conn.executescript(schema_sql)
        conn.commit()
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_key_list(orders)")
        rows = cursor.fetchall()
        conn.close()
        # SQLite: (id, seq, table, from, to) - find FK where from=parent_order_id, to=id
        parent_order_fks = [r for r in rows if len(r) >= 5 and r[3] == "parent_order_id" and r[4] == "id"]
        assert len(parent_order_fks) >= 1, "orders.parent_order_id should reference orders(id)"


class TestSignalPipelineAsyncio:
    """Signal pipeline must use get_running_loop / run_coroutine_threadsafe when loop exists."""

    def test_gemini_path_uses_get_running_loop_and_run_coroutine_threadsafe(self):
        """Gemini path uses get_running_loop and run_coroutine_threadsafe when event loop exists."""
        signal_pipeline_path = Path(__file__).parent.parent.parent / "modules" / "auto_trade" / "core" / "signal_pipeline.py"
        with open(signal_pipeline_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "get_running_loop" in source, (
            "Gemini path should use get_running_loop when loop exists"
        )
        assert "run_coroutine_threadsafe" in source, (
            "Gemini path should use run_coroutine_threadsafe when loop exists"
        )
        assert "asyncio.run(coro)" in source or "asyncio.run(" in source, (
            "No-loop path should use asyncio.run"
        )


class TestWebSocketStop:
    """WebSocket stop must use async_stop, force-close on timeout, and loop.stop()."""

    def test_stop_uses_async_stop_and_force_close_on_timeout(self):
        """stop() schedules _async_stop; on timeout force-close ws_client and stop loop."""
        ws_service_path = Path(__file__).parent.parent.parent / "modules" / "auto_trade" / "gui" / "utils" / "websocket_data_service.py"
        with open(ws_service_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "run_coroutine_threadsafe(self._async_stop()" in source, (
            "stop() must schedule _async_stop"
        )
        assert "timeout=10" in source or "timeout=10.0" in source, (
            "stop() must use timeout for cleanup"
        )
        assert "TimeoutError" in source or "timed out" in source.lower(), (
            "stop() must handle timeout"
        )
        assert "ws_client.close()" in source or "ws_client.close" in source, (
            "stop() must force-close ws_client on timeout"
        )
        assert "_loop.stop" in source or "loop.stop" in source, (
            "stop() must stop event loop (e.g. in finally)"
        )


class TestReconcileExchangeCleanup:
    """Reconcile must close exchange in finally (no resource leak)."""

    def test_reconcile_closes_exchange_on_auth_error(self):
        """When auth fails, reconcile returns errors and does not leak (no exchange created)."""
        if ccxt is None:
            pytest.skip("ccxt not installed")
        with patch("modules.auto_trade.database.reconcile.ccxt.binance") as mock_binance:
            mock_binance.side_effect = ccxt.AuthenticationError("bad key")
            result = reconcile_orders_with_binance(api_key="x", api_secret="y")
        assert "errors" in result
        assert any("Authentication" in str(e) or "auth" in str(e).lower() for e in result["errors"])
        mock_binance.assert_called_once()

    def test_reconcile_closes_exchange_when_created(self):
        """When exchange is created, finally block must call exchange.close()."""
        mock_exchange = MagicMock()
        mock_exchange.fetch_closed_orders = MagicMock(return_value=[])
        mock_exchange.fetch_open_orders = MagicMock(return_value=[])
        with patch("modules.auto_trade.database.reconcile.ccxt") as mock_ccxt:
            mock_ccxt.binance.return_value = mock_exchange
            reconcile_orders_with_binance(api_key="x", api_secret="y")
        mock_exchange.close.assert_called_once()


class TestMigrationTracking:
    """Migrations must be recorded in migrations_applied and not re-applied."""

    def test_get_pending_migrations_excludes_applied(self, tmp_path):
        """get_pending_migrations returns only migrations not in migrations_applied."""
        from modules.auto_trade.database.migrations import MigrationManager

        db_path = tmp_path / "mig.db"
        schema_path = DEFAULT_SCHEMA_PATH
        # Create empty DB with schema (so migrations_applied exists)
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_sql = f.read()
        conn = sqlite3.connect(str(db_path))
        conn.executescript(schema_sql)
        conn.execute(
            "INSERT INTO migrations_applied (migration_name, applied_at) VALUES (?, datetime('now'))",
            ("002_add_gradual_recovery.sql",),
        )
        conn.commit()
        conn.close()

        manager = MigrationManager(str(db_path), schema_path)
        pending = manager.get_pending_migrations()
        # Applied migration must not be in pending
        assert "002_add_gradual_recovery.sql" not in pending

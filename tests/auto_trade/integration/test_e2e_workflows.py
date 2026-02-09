"""
End-to-end workflow tests for auto_trade module.

Tests complete workflows: signal pipeline, database init/migrate/query,
reconcile with mocked exchange. Run: pytest tests/auto_trade/integration/test_e2e_workflows.py -v
"""

import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from modules.auto_trade.database.config import DEFAULT_SCHEMA_PATH


def _load_schema_sql() -> str:
    """Load schema SQL for tests."""
    schema_path = Path(DEFAULT_SCHEMA_PATH)
    if schema_path.exists():
        return schema_path.read_text(encoding="utf-8")
    fallback = Path(__file__).parent.parent.parent.parent / "modules" / "auto_trade" / "database" / "schema.sql"
    return fallback.read_text(encoding="utf-8")


class TestDatabaseWorkflowE2E:
    """E2E: Database init, migrate, insert, query."""

    def test_database_full_workflow_init_migrate_insert_query(self, tmp_path):
        """Full workflow: init DB, run migrations, insert order, query stats."""
        import modules.auto_trade.database as db_module
        from modules.auto_trade.database import (
            create_order,
            get_db_manager,
            get_overall_stats,
            initialize_database,
        )

        db_module._db_manager_instance = None
        db_path = str(tmp_path / "e2e_workflow.db")

        # 1. Initialize database
        initialize_database(db_path)
        db_manager = get_db_manager(db_path)

        # 2. Insert orders via create_order
        with db_manager.session_scope() as session:
            create_order(
                session,
                {
                    "order_id": "E2E_ORDER_001",
                    "symbol": "BTCUSDT",
                    "side": "LONG",
                    "entry_price": 50000.0,
                    "amount": 0.01,
                    "status": "CLOSED",
                    "order_source": "PROGRAMMATIC",
                    "client_order_id": "AT_E2E_001_CLIENT",
                },
            )
            session.flush()
            create_order(
                session,
                {
                    "order_id": "E2E_ORDER_002",
                    "symbol": "ETHUSDT",
                    "side": "SHORT",
                    "entry_price": 3000.0,
                    "amount": 0.1,
                    "status": "CLOSED",
                    "order_source": "PROGRAMMATIC",
                    "client_order_id": "AT_E2E_002_CLIENT",
                },
            )

        # 3. Query stats
        with db_manager.session_scope() as session:
            stats = get_overall_stats(session)

        assert stats["total_trades"] >= 2
        assert "win_rate" in stats
        assert "total_pnl" in stats

        db_module._db_manager_instance = None


class TestSignalPipelineWorkflowE2E:
    """E2E: Signal pipeline with mocked components."""

    def test_signal_pipeline_workflow_with_mocked_components(self):
        """Full pipeline: symbol refresh -> ATC scan -> XGBoost -> selector -> final signal."""
        from modules.auto_trade.core.signal_pipeline import SignalPipeline
        from modules.auto_trade.core.signal_selector import FinalSignal, SignalSelector

        # Mock symbol manager
        mock_symbol_manager = MagicMock()
        mock_symbol_manager.get_symbols.return_value = ["BTC/USDT", "ETH/USDT"]
        mock_symbol_manager.sample_percentage = 100.0
        mock_symbol_manager.sampling_strategy = "random"

        # Mock ATC scanner - returns one signal (pipeline calls scan_symbols(symbols))
        mock_atc = MagicMock()
        from modules.auto_trade.core.atc_scanner import SignalResult

        mock_atc.scan_symbols.return_value = [
            SignalResult("BTC/USDT", 0.85, "LONG", {"xgboost_conf": 0.8}, {"5m": 0.8, "15m": 0.7, "1h": 0.9}),
        ]

        # Mock XGBoost filter - passthrough
        mock_xgboost = MagicMock()
        mock_xgboost.filter_signals.side_effect = lambda sigs: sigs

        # Mock Gemini - must return async coroutine for analyze_candidates_batch_async
        mock_gemini = MagicMock()
        from modules.auto_trade.core.gemini_integration import GeminiSignal

        async def _analyze_batch(*args, **kwargs):
            return {"BTC/USDT": GeminiSignal("UP", "LONG", 0.9, 50000, 49000, 52000, "Test")}

        mock_gemini.analyze_candidates_batch_async = _analyze_batch
        mock_gemini.is_available.return_value = True

        # Real selector
        selector = SignalSelector(
            config={
                "weight_xgboost": 0.4,
                "weight_gemini": 0.6,
                "min_confidence_threshold": 0.5,
            }
        )

        pipeline = SignalPipeline(
            symbol_manager=mock_symbol_manager,
            atc_scanner=mock_atc,
            xgboost_filter=mock_xgboost,
            gemini_integration=mock_gemini,
            signal_selector=selector,
            config={"max_symbols_to_scan": 5, "pipeline_timeout": 30, "max_ai_candidates": 3},
        )

        result = pipeline.run_pipeline()

        # Pipeline should complete; result may be FinalSignal or None
        assert result is None or isinstance(result, FinalSignal)
        mock_atc.scan_symbols.assert_called()
        mock_xgboost.filter_signals.assert_called()


class TestReconcileWorkflowE2E:
    """E2E: Reconcile workflow with mocked Binance."""

    def test_reconcile_workflow_with_mocked_exchange(self):
        """Reconcile: create exchange, fetch orders, insert missing, return result."""
        try:
            import ccxt
        except ImportError:
            pytest.skip("ccxt not installed")

        from modules.auto_trade.database import reconcile_orders_with_binance

        mock_exchange = MagicMock()
        mock_exchange.fetch_closed_orders.return_value = []
        mock_exchange.fetch_open_orders.return_value = []

        with patch("modules.auto_trade.database.reconcile.ccxt") as mock_ccxt:
            mock_ccxt.binance.return_value = mock_exchange

            result = reconcile_orders_with_binance(
                api_key="test_key",
                api_secret="test_secret",
                symbols=["BTC/USDT"],
                since_hours=24,
            )

        assert "inserted" in result
        assert "skipped" in result
        assert "errors" in result
        assert isinstance(result["inserted"], int)
        mock_exchange.close.assert_called_once()


class TestBackupRestoreWorkflowE2E:
    """E2E: Backup and restore workflow."""

    def test_backup_workflow_create_and_verify(self, tmp_path):
        """Full workflow: create DB, insert data, backup, verify backup file."""
        from modules.auto_trade.database.backup import BackupManager

        import modules.auto_trade.database as db_module
        from modules.auto_trade.database import get_db_manager, initialize_database
        from modules.auto_trade.database.models import Order

        db_module._db_manager_instance = None
        db_path = str(tmp_path / "backup_test.db")
        backup_dir = str(tmp_path / "backups")

        initialize_database(db_path)
        db_manager = get_db_manager(db_path)

        # Insert initial data
        with db_manager.session_scope() as session:
            session.bulk_insert_mappings(
                Order,  # type: ignore[arg-type]
                [
                    {
                        "order_id": "BACKUP_001",
                        "symbol": "BTCUSDT",
                        "side": "LONG",
                        "entry_price": 50000.0,
                        "amount": 0.01,
                        "status": "CLOSED",
                        "order_source": "PROGRAMMATIC",
                    },
                ],
            )

        # Create backup
        manager = BackupManager(db_path, backup_dir=backup_dir, max_backups=5, compress=True)
        backup_path = manager.create_backup()
        assert backup_path is not None
        assert Path(backup_path).exists()
        assert manager.verify_backup(backup_path) is True

        db_module._db_manager_instance = None

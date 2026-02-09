"""Tests for Database Service Layer."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch, mock_open

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from modules.auto_trade.gui.services.database_service import (
    DatabaseService,
    ReconciliationService,
    DataViewerService,
)


class TestDatabaseService(unittest.TestCase):
    """Test cases for DatabaseService."""

    @patch("modules.auto_trade.gui.services.database_service.session_scope")
    def test_get_stats_success(self, mock_session_scope):
        """Test getting database stats successfully."""
        mock_session = MagicMock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        # Setup query counts
        mock_session.query.return_value.count.return_value = 10
        mock_session.query.return_value.filter.return_value.count.return_value = 5

        stats = DatabaseService.get_stats()

        self.assertEqual(stats["total_orders"], 10)
        self.assertEqual(stats["open_positions"], 5)
        self.assertEqual(stats["total_signals"], 10)
        self.assertEqual(stats["active_chains"], 5)
        self.assertEqual(stats["audit_logs"], 10)

    @patch("modules.auto_trade.gui.services.database_service.session_scope")
    def test_get_stats_failure(self, mock_session_scope):
        """Test getting database stats when exception occurs."""
        mock_session_scope.side_effect = Exception("DB Error")

        stats = DatabaseService.get_stats()

        self.assertEqual(stats, {})

    @patch("modules.auto_trade.gui.services.database_service.Path")
    def test_get_last_backup_time_with_backups(self, mock_path):
        """Test getting last backup time when backups exist."""
        mock_backup_file = MagicMock()
        mock_backup_file.stat.return_value.st_mtime = 1704067200  # 2024-01-01 00:00:00

        mock_backup_dir = MagicMock()
        mock_backup_dir.exists.return_value = True
        mock_backup_dir.glob.return_value = [mock_backup_file]

        mock_path.return_value = mock_backup_dir

        result = DatabaseService.get_last_backup_time()

        self.assertIsNotNone(result)
        self.assertIn("2024", result)  # type: ignore[arg-type]

    @patch("modules.auto_trade.gui.services.database_service.Path")
    def test_get_last_backup_time_no_backups(self, mock_path):
        """Test getting last backup time when no backups exist."""
        mock_backup_dir = MagicMock()
        mock_backup_dir.exists.return_value = True
        mock_backup_dir.glob.return_value = []

        mock_path.return_value = mock_backup_dir

        result = DatabaseService.get_last_backup_time()

        self.assertIsNone(result)

    @patch("modules.auto_trade.gui.services.database_service.create_database_backup")
    def test_create_backup_success(self, mock_create_backup):
        """Test creating backup successfully."""
        mock_create_backup.return_value = "/path/to/backup.db"

        result = DatabaseService.create_backup()

        self.assertEqual(result, "/path/to/backup.db")

    @patch("modules.auto_trade.gui.services.database_service.create_database_backup")
    def test_create_backup_failure(self, mock_create_backup):
        """Test creating backup when it fails."""
        mock_create_backup.side_effect = Exception("Backup failed")

        result = DatabaseService.create_backup()

        self.assertIsNone(result)

    @patch("modules.auto_trade.gui.services.database_service.get_migration_manager")
    def test_run_migrations_success(self, mock_get_manager):
        """Test running migrations successfully."""
        mock_get_manager.return_value = MagicMock()

        success, msg = DatabaseService.run_migrations()

        self.assertTrue(success)
        self.assertIn("Migration manager ready", msg)

    @patch("modules.auto_trade.gui.services.database_service.get_migration_manager")
    def test_run_migrations_failure(self, mock_get_manager):
        """Test running migrations when it fails."""
        mock_get_manager.side_effect = Exception("Migration error")

        success, msg = DatabaseService.run_migrations()

        self.assertFalse(success)

    @patch("modules.auto_trade.gui.services.database_service.session_scope")
    def test_cleanup_old_records_success(self, mock_session_scope):
        """Test cleanup old records successfully."""
        mock_session = MagicMock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        with patch("modules.auto_trade.database.utils.DatabaseCleaner") as mock_cleaner:
            mock_cleaner.cleanup_old_records.return_value = 5

            success, msg = DatabaseService.cleanup_old_records(days_to_keep=90)

            self.assertTrue(success)
            self.assertIn("Deleted", msg)
            self.assertIn("5", msg)

    @patch("modules.auto_trade.gui.services.database_service.get_db_manager")
    def test_check_integrity_success(self, mock_get_manager):
        """Test checking database integrity - OK."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = ("ok",)
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_get_manager.return_value.engine = mock_engine

        is_ok, status = DatabaseService.check_integrity()

        self.assertTrue(is_ok)
        self.assertEqual(status, "ok")

    @patch("modules.auto_trade.gui.services.database_service.get_db_manager")
    def test_check_integrity_failure(self, mock_get_manager):
        """Test checking database integrity - error."""
        mock_get_manager.side_effect = Exception("DB error")

        is_ok, status = DatabaseService.check_integrity()

        self.assertFalse(is_ok)


class TestReconciliationService(unittest.TestCase):
    """Test cases for ReconciliationService."""

    @patch("modules.auto_trade.gui.services.database_service.reconcile_orders_with_binance")
    def test_reconcile_success(self, mock_reconcile):
        """Test successful reconciliation."""
        mock_reconcile.return_value = {
            "inserted": 5,
            "skipped": 3,
            "closed_stale": 2,
            "errors": [],
        }

        result = ReconciliationService.reconcile_with_binance(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True,
        )

        self.assertEqual(result["inserted"], 5)
        self.assertEqual(result["skipped"], 3)

    @patch("modules.auto_trade.gui.services.database_service.reconcile_orders_with_binance")
    def test_reconcile_failure(self, mock_reconcile):
        """Test reconciliation failure."""
        mock_reconcile.side_effect = Exception("Reconcile error")

        result = ReconciliationService.reconcile_with_binance(
            api_key="test_key",
            api_secret="test_secret",
        )

        self.assertEqual(result["inserted"], 0)
        self.assertEqual(result["errors"], ["Reconcile error"])


class TestDataViewerService(unittest.TestCase):
    """Test cases for DataViewerService."""

    @patch("modules.auto_trade.gui.services.database_service.session_scope")
    def test_get_table_count_orders(self, mock_session_scope):
        """Test getting table count for Orders."""
        mock_session = MagicMock()
        mock_session.query.return_value.count.return_value = 100
        mock_session_scope.return_value.__enter__.return_value = mock_session

        count = DataViewerService.get_table_count("Orders")

        self.assertEqual(count, 100)

    @patch("modules.auto_trade.gui.services.database_service.session_scope")
    def test_get_table_count_signals(self, mock_session_scope):
        """Test getting table count for Signals."""
        mock_session = MagicMock()
        mock_session.query.return_value.count.return_value = 50
        mock_session_scope.return_value.__enter__.return_value = mock_session

        count = DataViewerService.get_table_count("Signals")

        self.assertEqual(count, 50)

    @patch("modules.auto_trade.gui.services.database_service.session_scope")
    def test_get_table_data(self, mock_session_scope):
        """Test getting table data."""
        mock_session = MagicMock()
        mock_data = [MagicMock(id=1), MagicMock(id=2)]
        mock_session_scope.return_value.__enter__.return_value = mock_session

        with patch("modules.auto_trade.database.get_orders_cursor") as mock_cursor:
            mock_cursor.return_value = mock_data

            result = DataViewerService.get_table_data("Orders", limit=20)

            self.assertEqual(len(result), 2)

    @patch("modules.auto_trade.gui.services.database_service.session_scope")
    def test_get_audit_logs(self, mock_session_scope):
        """Test getting audit logs."""
        mock_session = MagicMock()
        mock_logs = [MagicMock(), MagicMock()]
        mock_session_scope.return_value.__enter__.return_value = mock_session

        with patch("modules.auto_trade.gui.services.database_service.get_recent_audit_logs") as mock_get_logs:
            mock_get_logs.return_value = mock_logs

            result = DataViewerService.get_audit_logs(limit=50)

            self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()

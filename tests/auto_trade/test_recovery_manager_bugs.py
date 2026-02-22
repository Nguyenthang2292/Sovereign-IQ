"""
Regression tests for Recovery Manager bugs:
  Bug #2 - cancel() must persist CANCELLED status to DB before clearing _recovery_id
  Bug #3 - _mark_recovery_complete() must write status="COMPLETE" which triggers completed_at
"""

import unittest
from unittest.mock import MagicMock, call, patch


class TestRecoveryManagerCancelPersistence(unittest.TestCase):
    """Bug #2: cancel() must call _cancel_in_database() BEFORE nulling _recovery_id."""

    def _make_manager(self, recovery_id: str = "test-recovery-123") -> "RecoveryManager":  # type: ignore[name-defined]
        from modules.auto_trade.strategies.recovery_manager import RecoveryManager

        manager = RecoveryManager(
            event_bus=MagicMock(),
            config={"initial_loss": 500.0, "max_recovery_trades": 10},
            enabled=True,
            database=False,
        )
        # Inject a fake active strategy and recovery_id
        fake_strategy = MagicMock()
        fake_strategy.is_active = True
        manager._strategy = fake_strategy
        manager._recovery_id = recovery_id
        return manager

    def test_cancel_calls_db_before_clearing_id(self):
        manager = self._make_manager("rid-001")

        cancelled_with_id: list[str] = []

        def fake_cancel_in_db():
            # Capture the _recovery_id value at the moment DB cancel is called
            cancelled_with_id.append(manager._recovery_id)

        with patch.object(manager, "_cancel_in_database", side_effect=fake_cancel_in_db):
            manager.cancel()

        # DB cancel must have been called with a non-None id
        self.assertEqual(len(cancelled_with_id), 1, "cancel_in_database must be called exactly once")
        self.assertEqual(cancelled_with_id[0], "rid-001", "_recovery_id must still be set when cancel_in_database runs")

    def test_cancel_clears_state_after_db_call(self):
        manager = self._make_manager("rid-002")

        with patch.object(manager, "_cancel_in_database"):
            manager.cancel()

        self.assertIsNone(manager._recovery_id, "_recovery_id must be None after cancel()")
        self.assertIsNone(manager._strategy, "_strategy must be None after cancel()")

    def test_cancel_when_no_active_strategy_is_noop(self):
        from modules.auto_trade.strategies.recovery_manager import RecoveryManager

        manager = RecoveryManager(
            event_bus=MagicMock(),
            config={},
            enabled=True,
            database=False,
        )
        db_mock = MagicMock()
        with patch.object(manager, "_cancel_in_database", db_mock):
            manager.cancel()

        db_mock.assert_not_called()


class TestRecoveryManagerCompleteStatus(unittest.TestCase):
    """Bug #3: _mark_recovery_complete() must write status='COMPLETE' so completed_at is set in DynamoDB."""

    def _make_active_manager(self, recovery_id: str = "test-complete-456") -> "RecoveryManager":  # type: ignore[name-defined]
        from modules.auto_trade.strategies.recovery_manager import RecoveryManager

        manager = RecoveryManager(
            event_bus=MagicMock(),
            config={"initial_loss": 500.0},
            enabled=True,
            database=True,
        )
        fake_strategy = MagicMock()
        fake_strategy.is_active = True
        manager._strategy = fake_strategy
        manager._recovery_id = recovery_id
        return manager

    def test_mark_complete_writes_COMPLETE_not_COMPLETED(self):
        manager = self._make_active_manager("rid-complete-001")

        update_calls: list[dict] = []

        with patch(
            "modules.auto_trade.database.queries.update_gradual_recovery",
            side_effect=lambda rid, updates: update_calls.append({"rid": rid, "updates": updates}),
        ):
            manager._mark_recovery_complete()

        self.assertEqual(len(update_calls), 1, "_mark_recovery_complete must call update once")
        written_status = update_calls[0]["updates"].get("status")
        self.assertEqual(
            written_status,
            "COMPLETE",
            f"Status must be 'COMPLETE' (not '{written_status}') so DynamoDB sets completed_at",
        )


class TestDynamoDBGradualRecoveryCompletedAt(unittest.TestCase):
    """Bug #3: DynamoDB repo must set completed_at when status='COMPLETE' is written."""

    def _make_repo(self) -> "DynamoDBGradualRecoveryRepository":  # type: ignore[name-defined]
        from modules.auto_trade.database.repository.dynamodb.gradual_recovery import (
            DynamoDBGradualRecoveryRepository,
        )

        with patch("modules.auto_trade.database.repository.dynamodb.gradual_recovery.get_dynamodb_table", return_value=MagicMock()):
            repo = DynamoDBGradualRecoveryRepository()

        repo._table = MagicMock()
        repo._table.get_item.return_value = {
            "Item": {
                "pk": "RECOVERY#test-123",
                "sk": "RECOVERY#test-123",
                "created_at": "2026-01-01T00:00:00",
                "status": "ACTIVE",
            }
        }
        repo._table.update_item.return_value = {}
        return repo

    def test_completed_at_set_when_status_is_COMPLETE(self):
        repo = self._make_repo()

        update_expr_used: list[str] = []

        def capture_update(**kwargs):
            update_expr_used.append(kwargs.get("UpdateExpression", ""))
            return {}

        repo._table.update_item.side_effect = capture_update

        repo.update_gradual_recovery("test-123", {"status": "COMPLETE"})

        self.assertTrue(len(update_expr_used) > 0, "update_item must be called")
        self.assertIn(
            "completed_at",
            update_expr_used[0],
            "UpdateExpression must include completed_at when status='COMPLETE'",
        )

    def test_completed_at_not_set_for_other_statuses(self):
        repo = self._make_repo()

        update_expr_used: list[str] = []

        def capture_update(**kwargs):
            update_expr_used.append(kwargs.get("UpdateExpression", ""))
            return {}

        repo._table.update_item.side_effect = capture_update

        repo.update_gradual_recovery("test-123", {"status": "ACTIVE"})

        self.assertTrue(len(update_expr_used) > 0)
        self.assertNotIn(
            "completed_at",
            update_expr_used[0],
            "completed_at must NOT appear for non-COMPLETE statuses",
        )


if __name__ == "__main__":
    unittest.main()

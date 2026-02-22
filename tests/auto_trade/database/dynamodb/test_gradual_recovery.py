"""
Tests for DynamoDB Gradual Recovery Repository.

Created: 2026-02-20
"""

from modules.auto_trade.database.repository.dynamodb.gradual_recovery import DynamoDBGradualRecoveryRepository


class TestDynamoDBGradualRecoveryRepository:
    def test_create_get_cancel(self, setup_dynamodb_table):
        repo = DynamoDBGradualRecoveryRepository()

        created = repo.create_gradual_recovery(
            {
                "recovery_id": "rec_1",
                "initial_loss": 150.0,
                "config": {"step_size": 0.1},
                "symbol": "BTCUSDT",
            }
        )
        assert created["status"] == "ACTIVE"

        active = repo.get_active_gradual_recovery("BTCUSDT")
        assert active is not None
        assert active["recovery_id"] == "rec_1"

        cancelled = repo.cancel_gradual_recovery("rec_1")
        assert cancelled is True

        after_cancel = repo.get_gradual_recovery_by_id("rec_1")
        assert after_cancel is not None
        assert after_cancel["status"] == "CANCELLED"

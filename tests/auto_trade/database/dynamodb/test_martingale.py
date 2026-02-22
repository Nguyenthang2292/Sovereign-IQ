"""
Tests for DynamoDB Martingale Repository.

Created: 2026-02-20
"""

from modules.auto_trade.database.repository.dynamodb.martingale import DynamoDBMartingaleRepository


class TestDynamoDBMartingaleRepository:
    def test_find_or_create_idempotent(self, setup_dynamodb_table):
        repo = DynamoDBMartingaleRepository()

        first = repo.find_or_create_martingale_chain(symbol="BTCUSDT", initial_order_id="ord_1", loss=100.0)
        second = repo.find_or_create_martingale_chain(symbol="BTCUSDT", initial_order_id="ord_2", loss=200.0)

        assert first["chain_id"] == second["chain_id"]
        assert second["status"] == "ACTIVE"

    def test_update_and_get_active(self, setup_dynamodb_table):
        repo = DynamoDBMartingaleRepository()

        chain = repo.find_or_create_martingale_chain(symbol="ETHUSDT", initial_order_id="ord_a", loss=50.0)
        chain_id = chain["chain_id"]

        updated = repo.update_martingale_chain(
            chain_id,
            {
                "current_step": 2,
                "latest_order_id": "ord_b",
                "total_loss": 90.0,
            },
        )
        assert updated is True

        active = repo.get_martingale_state("ETHUSDT")
        assert active is not None
        assert active["current_step"] == 2

        active_chains = repo.get_active_martingale_chains()
        assert any(item["chain_id"] == chain_id for item in active_chains)

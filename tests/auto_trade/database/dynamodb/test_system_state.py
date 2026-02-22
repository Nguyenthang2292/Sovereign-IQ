"""
Tests for DynamoDB System State Repository.

Created: 2026-02-20
"""

from modules.auto_trade.database.repository.dynamodb.system_state import DynamoDBSystemStateRepository


class TestDynamoDBSystemStateRepository:
    def test_get_set_all_value_types(self, setup_dynamodb_table):
        repo = DynamoDBSystemStateRepository()

        assert repo.set_system_state("trading.mode", "AUTO", value_type="string") is True
        assert repo.get_system_state("trading.mode") == "AUTO"

        assert repo.set_system_state("risk.max_positions", 5, value_type="integer") is True
        assert repo.get_system_state("risk.max_positions") == 5

        assert repo.set_system_state("risk.max_drawdown", 0.35, value_type="float") is True
        assert repo.get_system_state("risk.max_drawdown") == 0.35

        assert repo.set_system_state("trading.enabled", True, value_type="boolean") is True
        assert repo.get_system_state("trading.enabled") is True

        payload = {"symbols": ["BTCUSDT", "ETHUSDT"], "leverage": 3}
        assert repo.set_system_state("portfolio.config", payload, value_type="json") is True
        assert repo.get_system_state("portfolio.config") == payload

    def test_get_with_default(self, setup_dynamodb_table):
        repo = DynamoDBSystemStateRepository()
        assert repo.get_system_state("non.existent.key", default="fallback") == "fallback"

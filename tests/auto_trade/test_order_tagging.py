"""
Unit Tests for Order Tagging System
====================================

Tests client order ID generation, metadata creation, and validation.

Run: pytest tests/auto_trade/test_order_tagging.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.auto_trade.execution.order_tagging import (
    CLIENT_ORDER_ID_PREFIX,
    EXECUTION_MODE_AUTO,
    ORDER_SOURCE_PROGRAMMATIC,
    OrderTagger,
    extract_order_info,
    generate_order_id,
    get_order_tag_stats,
    is_auto_trade_order,
    tag_multiple_orders,
    tag_programmatic_order,
    validate_order_metadata,
)


class TestClientOrderIDGeneration:
    """Test client order ID generation."""

    def test_generate_basic_id(self):
        """Test basic ID generation."""
        order_id = OrderTagger.generate_client_order_id("BTCUSDT")

        assert order_id.startswith(CLIENT_ORDER_ID_PREFIX)
        assert "BTCUSDT" in order_id

    def test_id_uniqueness(self):
        """Test that IDs are unique."""
        ids = [OrderTagger.generate_client_order_id("BTCUSDT") for _ in range(100)]
        unique_ids = set(ids)

        assert len(unique_ids) == 100

    def test_id_with_milliseconds(self):
        """Test ID generation with milliseconds."""
        id_ms = OrderTagger.generate_client_order_id("ETHUSDT", use_milliseconds=True)

        assert id_ms.startswith(CLIENT_ORDER_ID_PREFIX)
        assert "ETHUSDT" in id_ms

    def test_convenience_function(self):
        """Test generate_order_id convenience function."""
        order_id = generate_order_id("BTCUSDT")

        assert order_id.startswith(CLIENT_ORDER_ID_PREFIX)


class TestOrderIDParsing:
    """Test client order ID parsing."""

    def test_parse_valid_id(self):
        """Test parsing a valid client order ID."""
        order_id = OrderTagger.generate_client_order_id("BTCUSDT")
        info = OrderTagger.parse_client_order_id(order_id)

        assert info is not None
        assert info["symbol"] == "BTCUSDT"
        assert info["is_programmatic"] is True
        assert "timestamp" in info
        assert "random_suffix" in info
        assert "datetime" in info

    def test_parse_invalid_id(self):
        """Test parsing an invalid ID."""
        info = OrderTagger.parse_client_order_id("MANUAL_ORDER_123")

        assert info is None

    def test_extract_order_info(self):
        """Test extract_order_info convenience function."""
        order_id = generate_order_id("ETHUSDT")
        info = extract_order_info(order_id)

        assert info["symbol"] == "ETHUSDT"


class TestOrderIdentification:
    """Test order identification."""

    def test_identify_programmatic(self):
        """Test identifying programmatic order."""
        prog_id = generate_order_id("BTCUSDT")

        assert OrderTagger.is_programmatic_order_id(prog_id) is True
        assert is_auto_trade_order(prog_id) is True

    def test_identify_manual(self):
        """Test identifying manual order."""
        manual_id = "MANUAL_ORDER_12345"

        assert OrderTagger.is_programmatic_order_id(manual_id) is False
        assert is_auto_trade_order(manual_id) is False

    def test_identify_empty(self):
        """Test handling empty/None IDs."""
        assert is_auto_trade_order("") is False
        assert is_auto_trade_order(None) is False


class TestMetadataCreation:
    """Test order metadata creation."""

    def test_create_basic_metadata(self):
        """Test creating basic order metadata."""
        client_id = generate_order_id("BTCUSDT")
        metadata = OrderTagger.create_order_metadata(client_id)

        assert metadata["client_order_id"] == client_id
        assert metadata["order_source"] == ORDER_SOURCE_PROGRAMMATIC
        assert metadata["execution_mode"] == EXECUTION_MODE_AUTO
        assert metadata["is_programmatic"] is True
        assert "created_at" in metadata

    def test_create_metadata_with_signal(self):
        """Test metadata with signal correlation ID."""
        client_id = generate_order_id("BTCUSDT")
        signal_id = "SIGNAL_001"

        metadata = OrderTagger.create_order_metadata(client_id, signal_correlation_id=signal_id)

        assert metadata["signal_correlation_id"] == signal_id

    def test_create_metadata_with_martingale(self):
        """Test metadata with Martingale chain ID."""
        client_id = generate_order_id("BTCUSDT")
        chain_id = "CHAIN_001"

        metadata = OrderTagger.create_order_metadata(client_id, martingale_chain_id=chain_id)

        assert metadata["martingale_chain_id"] == chain_id

    def test_tag_programmatic_order(self):
        """Test tag_programmatic_order convenience function."""
        metadata = tag_programmatic_order("BTCUSDT", signal_id="SIGNAL_001")

        assert "client_order_id" in metadata
        assert metadata["order_source"] == ORDER_SOURCE_PROGRAMMATIC
        assert metadata["signal_correlation_id"] == "SIGNAL_001"


class TestMetadataValidation:
    """Test metadata validation."""

    def test_validate_valid_metadata(self):
        """Test validating valid metadata."""
        metadata = tag_programmatic_order("BTCUSDT")
        is_valid, error = validate_order_metadata(metadata)

        assert is_valid is True
        assert error is None

    def test_validate_missing_field(self):
        """Test validation fails for missing fields."""
        metadata = {"client_order_id": "AT_123_BTCUSDT_abc"}
        is_valid, error = validate_order_metadata(metadata)

        assert is_valid is False
        assert "Missing required field" in error

    def test_validate_wrong_prefix(self):
        """Test validation fails for wrong prefix."""
        metadata = {
            "client_order_id": "WRONG_PREFIX_123",
            "order_source": ORDER_SOURCE_PROGRAMMATIC,
            "execution_mode": EXECUTION_MODE_AUTO,
        }
        is_valid, error = validate_order_metadata(metadata)

        assert is_valid is False
        assert "Invalid client_order_id format" in error


class TestBatchOperations:
    """Test batch tagging operations."""

    def test_tag_multiple_orders(self):
        """Test tagging multiple orders."""
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        batch = tag_multiple_orders(symbols)

        assert len(batch) == 3

        # All should have unique IDs
        ids = [m["client_order_id"] for m in batch]
        assert len(set(ids)) == 3

        # All should be programmatic
        for metadata in batch:
            assert metadata["order_source"] == ORDER_SOURCE_PROGRAMMATIC


class TestIDGenerators:
    """Test Martingale and signal ID generators."""

    def test_generate_martingale_chain_id(self):
        """Test Martingale chain ID generation."""
        chain_id = OrderTagger.generate_martingale_chain_id("BTCUSDT", "ORDER_123")

        assert chain_id.startswith("CHAIN_")
        assert "BTCUSDT" in chain_id

    def test_generate_signal_correlation_id(self):
        """Test signal correlation ID generation."""
        signal_id = OrderTagger.generate_signal_correlation_id("ETHUSDT", "LONG")

        assert signal_id.startswith("SIGNAL_")
        assert "ETHUSDT" in signal_id
        assert "LONG" in signal_id


class TestStatistics:
    """Test order tag statistics."""

    def test_stats_all_programmatic(self):
        """Test stats with all programmatic orders."""
        ids = [generate_order_id("BTCUSDT") for _ in range(5)]
        stats = get_order_tag_stats(ids)

        assert stats["total_orders"] == 5
        assert stats["programmatic_orders"] == 5
        assert stats["manual_orders"] == 0
        assert stats["programmatic_percentage"] == 100.0

    def test_stats_mixed(self):
        """Test stats with mixed orders."""
        prog_ids = [generate_order_id("BTCUSDT") for _ in range(7)]
        manual_ids = [f"MANUAL_{i}" for i in range(3)]
        all_ids = prog_ids + manual_ids

        stats = get_order_tag_stats(all_ids)

        assert stats["total_orders"] == 10
        assert stats["programmatic_orders"] == 7
        assert stats["manual_orders"] == 3
        assert stats["programmatic_percentage"] == 70.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

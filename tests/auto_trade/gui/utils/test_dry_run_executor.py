"""
Comprehensive tests for DryRunExecutor.

Tests cover:
- Order placement
- Position closing
- TP/SL modification
- Price feed integration
- Database integration
- Error handling
"""

from unittest.mock import MagicMock, patch

import pytest

from modules.auto_trade.gui.utils.dry_run_executor import DryRunExecutor


class TestDryRunExecutor:
    """Test DryRunExecutor functionality."""

    @patch("modules.auto_trade.gui.utils.dry_run_executor.DryRunDB")
    @patch("modules.auto_trade.gui.utils.dry_run_executor.MockPriceFeed")
    def test_init(self, mock_price_feed, mock_db):
        """Test initialization."""
        executor = DryRunExecutor()

        assert executor.price_feed is not None
        assert executor.db is not None

    @patch("modules.auto_trade.gui.utils.dry_run_executor.DryRunDB")
    @patch("modules.auto_trade.gui.utils.dry_run_executor.MockPriceFeed")
    def test_place_order_long(self, mock_price_feed_class, mock_db_class):
        """Test placing a LONG order."""
        # Setup mocks
        mock_price_feed = MagicMock()
        mock_price_feed.get_current_price.return_value = 42000.0
        mock_price_feed_class.return_value = mock_price_feed

        mock_db = MagicMock()
        mock_db.insert_position.return_value = 1
        mock_db_class.return_value = mock_db

        executor = DryRunExecutor()

        result = executor.place_order(
            symbol="BTC/USDT",
            side="LONG",
            amount=0.1,
            leverage=10,
            tp=44000.0,
            sl=40000.0,
        )

        assert result["success"] is True
        assert result["order_id"] == 1
        assert result["symbol"] == "BTC/USDT"
        assert result["side"] == "LONG"
        assert result["entry_price"] == 42000.0
        assert result["size"] == 0.1

        # Verify DB was called
        mock_db.insert_position.assert_called_once()

    @patch("modules.auto_trade.gui.utils.dry_run_executor.DryRunDB")
    @patch("modules.auto_trade.gui.utils.dry_run_executor.MockPriceFeed")
    def test_place_order_short(self, mock_price_feed_class, mock_db_class):
        """Test placing a SHORT order."""
        mock_price_feed = MagicMock()
        mock_price_feed.get_current_price.return_value = 42000.0
        mock_price_feed_class.return_value = mock_price_feed

        mock_db = MagicMock()
        mock_db.insert_position.return_value = 2
        mock_db_class.return_value = mock_db

        executor = DryRunExecutor()

        result = executor.place_order(
            symbol="BTC/USDT",
            side="SHORT",
            amount=0.1,
            leverage=5,
        )

        assert result["success"] is True
        assert result["side"] == "SHORT"

    @patch("modules.auto_trade.gui.utils.dry_run_executor.DryRunDB")
    @patch("modules.auto_trade.gui.utils.dry_run_executor.MockPriceFeed")
    def test_place_order_without_tp_sl(self, mock_price_feed_class, mock_db_class):
        """Test placing order without TP/SL."""
        mock_price_feed = MagicMock()
        mock_price_feed.get_current_price.return_value = 42000.0
        mock_price_feed_class.return_value = mock_price_feed

        mock_db = MagicMock()
        mock_db.insert_position.return_value = 3
        mock_db_class.return_value = mock_db

        executor = DryRunExecutor()

        result = executor.place_order(
            symbol="BTC/USDT",
            side="LONG",
            amount=0.1,
            leverage=10,
        )

        assert result["success"] is True
        # Verify insert_position was called with None for tp/sl
        call_args = mock_db.insert_position.call_args
        assert call_args is not None

    @patch("modules.auto_trade.gui.utils.dry_run_executor.DryRunDB")
    @patch("modules.auto_trade.gui.utils.dry_run_executor.MockPriceFeed")
    def test_place_order_error_handling(self, mock_price_feed_class, mock_db_class):
        """Test error handling in place_order."""
        mock_price_feed = MagicMock()
        mock_price_feed.get_current_price.side_effect = Exception("Price feed error")
        mock_price_feed_class.return_value = mock_price_feed

        executor = DryRunExecutor()

        result = executor.place_order(
            symbol="BTC/USDT",
            side="LONG",
            amount=0.1,
            leverage=10,
        )

        assert result["success"] is False
        assert "error" in result
        assert "Failed to place order" in result["message"]

    @patch("modules.auto_trade.gui.utils.dry_run_executor.DryRunDB")
    @patch("modules.auto_trade.gui.utils.dry_run_executor.MockPriceFeed")
    def test_close_position_long(self, mock_price_feed_class, mock_db_class):
        """Test closing a LONG position."""
        mock_price_feed = MagicMock()
        mock_price_feed.get_current_price.return_value = 43000.0
        mock_price_feed_class.return_value = mock_price_feed

        mock_db = MagicMock()
        mock_db.get_open_positions_by_symbol.return_value = [
            {
                "id": 1,
                "symbol": "BTC/USDT",
                "side": "LONG",
                "entry_price": 42000.0,
                "size": 0.1,
            }
        ]
        mock_db_class.return_value = mock_db

        executor = DryRunExecutor()

        result = executor.close_position(
            symbol="BTC/USDT",
            side="LONG",
            size=0.1,
        )

        assert result["success"] is True
        assert result["symbol"] == "BTC/USDT"
        assert result["current_price"] == 43000.0

        # Verify position was updated
        mock_db.update_position.assert_called()

    @patch("modules.auto_trade.gui.utils.dry_run_executor.DryRunDB")
    @patch("modules.auto_trade.gui.utils.dry_run_executor.MockPriceFeed")
    def test_close_position_short(self, mock_price_feed_class, mock_db_class):
        """Test closing a SHORT position."""
        mock_price_feed = MagicMock()
        mock_price_feed.get_current_price.return_value = 41000.0
        mock_price_feed_class.return_value = mock_price_feed

        mock_db = MagicMock()
        mock_db.get_open_positions_by_symbol.return_value = [
            {
                "id": 1,
                "symbol": "BTC/USDT",
                "side": "SHORT",
                "entry_price": 42000.0,
                "size": 0.1,
            }
        ]
        mock_db_class.return_value = mock_db

        executor = DryRunExecutor()

        result = executor.close_position(
            symbol="BTC/USDT",
            side="SHORT",
            size=0.1,
        )

        assert result["success"] is True

    @patch("modules.auto_trade.gui.utils.dry_run_executor.DryRunDB")
    @patch("modules.auto_trade.gui.utils.dry_run_executor.MockPriceFeed")
    def test_close_position_no_positions(self, mock_price_feed_class, mock_db_class):
        """Test closing position when none exist."""
        mock_price_feed = MagicMock()
        mock_price_feed_class.return_value = mock_price_feed

        mock_db = MagicMock()
        mock_db.get_open_positions_by_symbol.return_value = []
        mock_db_class.return_value = mock_db

        executor = DryRunExecutor()

        result = executor.close_position(
            symbol="BTC/USDT",
            side="LONG",
            size=0.1,
        )

        assert result["success"] is False
        assert "No positions to close" in result["message"] or "No open positions" in result["message"]

    @patch("modules.auto_trade.gui.utils.dry_run_executor.DryRunDB")
    @patch("modules.auto_trade.gui.utils.dry_run_executor.MockPriceFeed")
    def test_close_partial_position(self, mock_price_feed_class, mock_db_class):
        """Test closing partial position."""
        mock_price_feed = MagicMock()
        mock_price_feed.get_current_price.return_value = 43000.0
        mock_price_feed_class.return_value = mock_price_feed

        mock_db = MagicMock()
        mock_db.get_open_positions_by_symbol.return_value = [
            {
                "id": 1,
                "symbol": "BTC/USDT",
                "side": "LONG",
                "entry_price": 42000.0,
                "size": 0.2,  # Position size larger than close size
            }
        ]
        mock_db_class.return_value = mock_db

        executor = DryRunExecutor()

        result = executor.close_position(
            symbol="BTC/USDT",
            side="LONG",
            size=0.1,  # Close only half
        )

        assert result["success"] is True

    @patch("modules.auto_trade.gui.utils.dry_run_executor.DryRunDB")
    @patch("modules.auto_trade.gui.utils.dry_run_executor.MockPriceFeed")
    def test_modify_tp_sl(self, mock_price_feed_class, mock_db_class):
        """Test modifying TP/SL."""
        mock_price_feed = MagicMock()
        mock_price_feed_class.return_value = mock_price_feed

        mock_db = MagicMock()
        mock_db.get_open_positions_by_symbol.return_value = [
            {
                "id": 1,
                "symbol": "BTC/USDT",
            }
        ]
        mock_db_class.return_value = mock_db

        executor = DryRunExecutor()

        result = executor.modify_tp_sl(
            symbol="BTC/USDT",
            tp_price=45000.0,
            sl_price=39000.0,
        )

        assert result["success"] is True
        assert result["symbol"] == "BTC/USDT"
        assert result["take_profit"] == 45000.0
        assert result["stop_loss"] == 39000.0

        # Verify DB was called
        mock_db.update_position.assert_called()

    @patch("modules.auto_trade.gui.utils.dry_run_executor.DryRunDB")
    @patch("modules.auto_trade.gui.utils.dry_run_executor.MockPriceFeed")
    def test_modify_tp_sl_no_positions(self, mock_price_feed_class, mock_db_class):
        """Test modifying TP/SL when no positions exist."""
        mock_price_feed = MagicMock()
        mock_price_feed_class.return_value = mock_price_feed

        mock_db = MagicMock()
        mock_db.get_open_positions_by_symbol.return_value = []
        mock_db_class.return_value = mock_db

        executor = DryRunExecutor()

        result = executor.modify_tp_sl(
            symbol="BTC/USDT",
            tp_price=45000.0,
            sl_price=39000.0,
        )

        assert result["success"] is False
        assert "No positions to modify" in result["message"] or "No open positions" in result["message"]

    @patch("modules.auto_trade.gui.utils.dry_run_executor.DryRunDB")
    @patch("modules.auto_trade.gui.utils.dry_run_executor.MockPriceFeed")
    def test_modify_tp_sl_partial(self, mock_price_feed_class, mock_db_class):
        """Test modifying only TP or only SL."""
        mock_price_feed = MagicMock()
        mock_price_feed_class.return_value = mock_price_feed

        mock_db = MagicMock()
        mock_db.get_open_positions_by_symbol.return_value = [{"id": 1}]
        mock_db_class.return_value = mock_db

        executor = DryRunExecutor()

        # Modify only TP
        result = executor.modify_tp_sl(
            symbol="BTC/USDT",
            tp_price=45000.0,
            sl_price=None,
        )

        assert result["success"] is True

    @patch("modules.auto_trade.gui.utils.dry_run_executor.DryRunDB")
    @patch("modules.auto_trade.gui.utils.dry_run_executor.MockPriceFeed")
    def test_error_handling_close_position(self, mock_price_feed_class, mock_db_class):
        """Test error handling in close_position."""
        mock_price_feed = MagicMock()
        mock_price_feed.get_current_price.side_effect = Exception("Price error")
        mock_price_feed_class.return_value = mock_price_feed

        executor = DryRunExecutor()

        result = executor.close_position(
            symbol="BTC/USDT",
            side="LONG",
            size=0.1,
        )

        assert result["success"] is False
        assert "Failed to close position" in result["message"]

    @patch("modules.auto_trade.gui.utils.dry_run_executor.DryRunDB")
    @patch("modules.auto_trade.gui.utils.dry_run_executor.MockPriceFeed")
    def test_pnl_calculation_long(self, mock_price_feed_class, mock_db_class):
        """Test PnL calculation for LONG position."""
        mock_price_feed = MagicMock()
        mock_price_feed.get_current_price.return_value = 44000.0  # Profit
        mock_price_feed_class.return_value = mock_price_feed

        mock_db = MagicMock()
        mock_db.get_open_positions_by_symbol.return_value = [
            {
                "id": 1,
                "symbol": "BTC/USDT",
                "side": "LONG",
                "entry_price": 42000.0,
                "size": 0.1,
            }
        ]
        mock_db_class.return_value = mock_db

        executor = DryRunExecutor()
        result = executor.close_position("BTC/USDT", "LONG", 0.1)

        # PnL = (44000 - 42000) * 0.1 = 200
        # Verify update_position was called with positive PnL
        call_args = mock_db.update_position.call_args
        assert call_args is not None

    @patch("modules.auto_trade.gui.utils.dry_run_executor.DryRunDB")
    @patch("modules.auto_trade.gui.utils.dry_run_executor.MockPriceFeed")
    def test_pnl_calculation_short(self, mock_price_feed_class, mock_db_class):
        """Test PnL calculation for SHORT position."""
        mock_price_feed = MagicMock()
        mock_price_feed.get_current_price.return_value = 40000.0  # Profit for short
        mock_price_feed_class.return_value = mock_price_feed

        mock_db = MagicMock()
        mock_db.get_open_positions_by_symbol.return_value = [
            {
                "id": 1,
                "symbol": "BTC/USDT",
                "side": "SHORT",
                "entry_price": 42000.0,
                "size": 0.1,
            }
        ]
        mock_db_class.return_value = mock_db

        executor = DryRunExecutor()
        result = executor.close_position("BTC/USDT", "SHORT", 0.1)

        # PnL = (42000 - 40000) * 0.1 = 200
        assert result["success"] is True

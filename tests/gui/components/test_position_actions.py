"""
Unit tests for PositionActions validation logic
"""

import pytest
from unittest.mock import MagicMock, patch
from modules.auto_trade.gui.components.position_actions import PositionActions


class TestPositionActionsValidation:
    """Test cases for position actions validation logic"""

    @pytest.fixture
    def sample_long_position(self):
        """Create a sample LONG position"""
        return {
            "symbol": "BTC/USDT",
            "side": "LONG",
            "size": 0.1,
            "entry_price": 50000,
            "current_price": 51000,
            "take_profit": 52000,
            "stop_loss": 48000,
            "leverage": 10,
            "margin_used": 500,
            "unrealized_pnl": 100,
        }

    @pytest.fixture
    def sample_short_position(self):
        """Create a sample SHORT position"""
        return {
            "symbol": "BTC/USDT",
            "side": "SHORT",
            "size": 0.1,
            "entry_price": 50000,
            "current_price": 49000,
            "take_profit": 48000,
            "stop_loss": 52000,
            "leverage": 10,
            "margin_used": 500,
            "unrealized_pnl": 100,
        }

    @pytest.fixture
    def mock_parent(self):
        """Create a mock parent widget"""
        parent = MagicMock()
        parent.winfo_exists.return_value = True
        return parent

    def test_validate_tp_sl_long_position_valid(self, sample_long_position, mock_parent):
        """Test valid TP/SL for LONG position"""
        with patch("modules.auto_trade.gui.components.position_actions.messagebox"):
            actions = PositionActions(mock_parent, sample_long_position)

            # Valid: TP above entry, SL below entry
            result = actions._validate_tp_sl(tp=52000, sl=48000)
            assert result is True

    def test_validate_tp_sl_long_position_invalid_tp(self, sample_long_position, mock_parent):
        """Test invalid TP for LONG position (below entry)"""
        with patch("modules.auto_trade.gui.components.position_actions.messagebox") as mock_msg:
            actions = PositionActions(mock_parent, sample_long_position)

            # Invalid: TP below entry for LONG
            result = actions._validate_tp_sl(tp=49000, sl=48000)
            assert result is False
            # Should show error message
            mock_msg.showerror.assert_called_once()

    def test_validate_tp_sl_long_position_invalid_sl(self, sample_long_position, mock_parent):
        """Test invalid SL for LONG position (above entry)"""
        with patch("modules.auto_trade.gui.components.position_actions.messagebox") as mock_msg:
            actions = PositionActions(mock_parent, sample_long_position)

            # Invalid: SL above entry for LONG
            result = actions._validate_tp_sl(tp=52000, sl=51000)
            assert result is False
            mock_msg.showerror.assert_called_once()

    def test_validate_tp_sl_short_position_valid(self, sample_short_position, mock_parent):
        """Test valid TP/SL for SHORT position"""
        with patch("modules.auto_trade.gui.components.position_actions.messagebox"):
            actions = PositionActions(mock_parent, sample_short_position)

            # Valid: TP below entry, SL above entry
            result = actions._validate_tp_sl(tp=48000, sl=52000)
            assert result is True

    def test_validate_tp_sl_short_position_invalid_tp(self, sample_short_position, mock_parent):
        """Test invalid TP for SHORT position (above entry)"""
        with patch("modules.auto_trade.gui.components.position_actions.messagebox") as mock_msg:
            actions = PositionActions(mock_parent, sample_short_position)

            # Invalid: TP above entry for SHORT
            result = actions._validate_tp_sl(tp=51000, sl=52000)
            assert result is False
            mock_msg.showerror.assert_called_once()

    def test_validate_tp_sl_short_position_invalid_sl(self, sample_short_position, mock_parent):
        """Test invalid SL for SHORT position (below entry)"""
        with patch("modules.auto_trade.gui.components.position_actions.messagebox") as mock_msg:
            actions = PositionActions(mock_parent, sample_short_position)

            # Invalid: SL below entry for SHORT
            result = actions._validate_tp_sl(tp=48000, sl=49000)
            assert result is False
            mock_msg.showerror.assert_called_once()

    def test_validate_tp_sl_zero_values(self, sample_long_position, mock_parent):
        """Test validation with zero TP/SL (meaning not set)"""
        with patch("modules.auto_trade.gui.components.position_actions.messagebox"):
            actions = PositionActions(mock_parent, sample_long_position)

            # Zero values should be allowed (means not set)
            result = actions._validate_tp_sl(tp=0, sl=48000)
            assert result is True

            result = actions._validate_tp_sl(tp=52000, sl=0)
            assert result is True

    def test_validate_tp_sl_sl_too_close_to_current(self, sample_long_position, mock_parent):
        """Test warning when SL is too close to current price"""
        with patch("modules.auto_trade.gui.components.position_actions.messagebox") as mock_msg:
            actions = PositionActions(mock_parent, sample_long_position)

            # SL very close to current price (51000 * 0.98 = 49980)
            result = actions._validate_tp_sl(tp=52000, sl=50000)
            assert result is False  # Should warn
            mock_msg.showerror.assert_called_once()

    def test_format_pnl_positive(self, sample_long_position, mock_parent):
        """Test P&L formatting for positive values"""
        with patch("modules.auto_trade.gui.components.position_actions.messagebox"):
            actions = PositionActions(mock_parent, sample_long_position)

            formatted = actions._format_pnl(100.50)
            assert formatted == "+$100.50"

    def test_format_pnl_negative(self, sample_long_position, mock_parent):
        """Test P&L formatting for negative values"""
        with patch("modules.auto_trade.gui.components.position_actions.messagebox"):
            actions = PositionActions(mock_parent, sample_long_position)

            formatted = actions._format_pnl(-50.25)
            assert formatted == "-$50.25"

    def test_format_pnl_zero(self, sample_long_position, mock_parent):
        """Test P&L formatting for zero"""
        with patch("modules.auto_trade.gui.components.position_actions.messagebox"):
            actions = PositionActions(mock_parent, sample_long_position)

            formatted = actions._format_pnl(0)
            assert formatted == "+$0.00"


class TestPositionActionsRetryLogic:
    """Test retry logic integration in PositionActions"""

    @pytest.fixture
    def sample_position(self):
        return {
            "symbol": "BTC/USDT",
            "side": "LONG",
            "size": 0.1,
            "entry_price": 50000,
            "id": "12345",
        }

    @pytest.fixture
    def mock_parent(self):
        parent = MagicMock()
        parent.winfo_exists.return_value = True
        return parent

    def test_execute_with_retry_success(self, sample_position, mock_parent):
        """Test successful execution with retry logic"""
        callback = MagicMock(return_value={"success": True})

        with patch("modules.auto_trade.gui.components.position_actions.messagebox"):
            actions = PositionActions(mock_parent, sample_position, callback)

            result = actions._execute_with_retry({"action": "test"})

            assert result["success"] is True
            callback.assert_called_once()

    def test_execute_with_retry_network_error_then_success(self, sample_position, mock_parent):
        """Test retry on network error then success"""
        import ccxt

        callback = MagicMock()
        callback.side_effect = [ccxt.NetworkError("Temporary error"), {"success": True}]

        with patch("modules.auto_trade.gui.components.position_actions.messagebox"):
            actions = PositionActions(mock_parent, sample_position, callback)

            result = actions._execute_with_retry({"action": "test"})

            assert result["success"] is True
            assert callback.call_count == 2  # First failed, second succeeded

    def test_execute_with_retry_all_attempts_fail(self, sample_position, mock_parent):
        """Test all retry attempts exhausted"""
        import ccxt

        callback = MagicMock(side_effect=ccxt.NetworkError("Persistent error"))

        with patch("modules.auto_trade.gui.components.position_actions.messagebox"):
            actions = PositionActions(mock_parent, sample_position, callback)

            result = actions._execute_with_retry({"action": "test"})

            assert result["success"] is False
            assert "Network error after retries" in result["error"]
            # Should try max_retries + 1 times (3 + 1 = 4)
            assert callback.call_count == 4

    def test_execute_with_retry_no_callback(self, sample_position, mock_parent):
        """Test execution with no callback configured"""
        with patch("modules.auto_trade.gui.components.position_actions.messagebox"):
            actions = PositionActions(mock_parent, sample_position, None)

            result = actions._execute_with_retry({"action": "test"})

            assert result["success"] is False
            assert "No callback configured" in result["error"]

"""
Tests for WebSocket Data Service initialization, lifecycle, callbacks.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from modules.auto_trade.gui.utils.websocket_data_service import WebSocketDataService
from modules.auto_trade.monitoring.account_monitor import BalanceSnapshot, OrderSnapshot
from modules.auto_trade.monitoring.position_monitor import PositionSnapshot


class TestWebSocketServiceInit:
    """Test WebSocket service initialization and credential loading."""

    @pytest.fixture
    def mock_service(self):
        """Create WebSocket service with mocked dependencies."""
        with patch("modules.auto_trade.gui.utils.websocket_data_service.CredentialManager") as mock_cm:
            mock_cm_instance = MagicMock()
            mock_cm_instance.load_credentials.return_value = {"api_key": "test_key", "api_secret": "test_secret"}
            mock_cm.return_value = mock_cm_instance

            service = WebSocketDataService(mode="DEMO")
            yield service

    def test_websocket_service_init_loads_credentials(self, mock_service):
        """Test that credentials are loaded during initialization."""
        assert mock_service.api_key == "test_key"
        assert mock_service.api_secret == "test_secret"

    def test_websocket_service_init_dry_run_mode(self):
        """Test initialization in DRY_RUN mode."""
        with patch("modules.auto_trade.gui.utils.websocket_data_service.CredentialManager"):
            service = WebSocketDataService(mode="DRY_RUN")

            assert service.mode == "DRY_RUN"
            assert service._running is False


class TestWebSocketServiceCallbacks:
    """Test callback registration and invocation."""

    @pytest.fixture
    def mock_service(self):
        """Create WebSocket service with mocked dependencies."""
        with (
            patch("modules.auto_trade.gui.utils.websocket_data_service.CredentialManager"),
            patch("modules.auto_trade.gui.utils.websocket_data_service.BinanceWebSocketClient"),
        ):
            service = WebSocketDataService(mode="DRY_RUN")
            yield service

    def test_register_position_callback(self, mock_service):
        """Test position callback registration and invocation."""
        callback_called = False
        received_position = None

        def callback(position):
            nonlocal callback_called, received_position
            callback_called = True
            received_position = position

        mock_service.on_position_update(callback)

        # Simulate position update
        test_position = PositionSnapshot(
            symbol="BTC/USDT",
            side="long",
            position_amt=0.1,
            entry_price=50000.0,
            mark_price=51000.0,
            liquidation_price=None,
            unrealized_pnl=100.0,
            unrealized_pnl_percent=2.0,
            margin_type="cross",
            leverage=10,
            timestamp=datetime.now(),
        )
        mock_service._handle_position_update(test_position)

        assert callback_called
        assert received_position == test_position

    def test_register_balance_callback(self, mock_service):
        """Test balance callback registration and invocation."""
        callback_called = False
        received_balance = None

        def callback(balance):
            nonlocal callback_called, received_balance
            callback_called = True
            received_balance = balance

        mock_service.on_balance_update(callback)

        # Simulate balance update
        test_balance = BalanceSnapshot(
            currency="USDT", total=10000.0, free=9500.0, used=500.0, timestamp=datetime.now()
        )
        mock_service._handle_balance_update(test_balance)

        assert callback_called
        assert received_balance == test_balance

    def test_register_order_callback(self, mock_service):
        """Test order callback registration and invocation."""
        callback_called = False
        received_order = None

        def callback(order):
            nonlocal callback_called, received_order
            callback_called = True
            received_order = order

        mock_service.on_order_update(callback)

        # Simulate order update
        test_order = OrderSnapshot(
            order_id="12345",
            client_order_id="AT_12345",
            symbol="BTC/USDT",
            side="BUY",
            type="LIMIT",
            status="filled",
            price=50000.0,
            amount=0.1,
            filled=0.1,
            remaining=0.0,
            timestamp=datetime.now(),
            last_update_timestamp=datetime.now(),
        )
        mock_service._handle_order_update(test_order)

        assert callback_called
        assert received_order == test_order


class TestWebSocketServiceDryRun:
    """Test WebSocket service in DRY_RUN mode."""

    @pytest.fixture
    def dry_run_service(self):
        """Create WebSocket service in DRY_RUN mode."""
        with patch("modules.auto_trade.gui.utils.websocket_data_service.CredentialManager"):
            service = WebSocketDataService(mode="DRY_RUN")
            yield service

    def test_dry_run_get_current_price(self, dry_run_service):
        """Test getting current price in DRY_RUN mode."""
        price = dry_run_service.get_current_price("BTC/USDT")

        assert isinstance(price, float)
        assert price > 0

    def test_dry_run_get_positions(self, dry_run_service):
        """Test getting positions in DRY_RUN mode."""
        positions = dry_run_service.get_positions()

        assert positions == []

    def test_dry_run_get_orders(self, dry_run_service):
        """Test getting orders in DRY_RUN mode."""
        orders = dry_run_service.get_orders()

        assert orders == []

    def test_dry_run_get_balance(self, dry_run_service):
        """Test getting balance in DRY_RUN mode."""
        balance = dry_run_service.get_balance()

        assert balance is not None
        assert isinstance(balance, BalanceSnapshot)
        assert balance.currency == "USDT"

    def test_dry_run_is_connected(self, dry_run_service):
        """Test that DRY_RUN mode always reports as connected."""
        assert dry_run_service.is_connected is True

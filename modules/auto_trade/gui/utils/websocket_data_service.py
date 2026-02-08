"""
WebSocket Data Service for GUI

Real-time data service using WebSocket for instant GUI updates.
Replaces REST polling with event-driven WebSocket streams.

Features:
- Real-time price updates
- Live position tracking
- Instant balance updates
- Order status notifications
"""

import asyncio
import logging
import os
import threading
from typing import Any, Callable, Dict, List, Optional

from modules.auto_trade.gui.utils.credential_manager import CredentialManager
from modules.auto_trade.gui.utils.mock_price_feed import MockPriceFeed
from modules.auto_trade.monitoring.account_monitor import BalanceMonitor, BalanceSnapshot, OrderMonitor, OrderSnapshot
from modules.auto_trade.monitoring.position_monitor import PositionMonitor, PositionSnapshot
from modules.auto_trade.websocket.client import BinanceWebSocketClient

logger = logging.getLogger(__name__)


class WebSocketDataService:
    """
    WebSocket-based data service for GUI with real-time updates.

    Replaces REST polling with WebSocket streams for:
    - Position updates (real-time P&L)
    - Balance updates (instant changes)
    - Order updates (live status)
    - Price updates (continuous)

    Example:
        >>> service = WebSocketDataService(mode="DEMO")
        >>> await service.start()
        >>> service.on_position_update(my_callback)
        >>> service.on_balance_update(my_callback)
    """

    def __init__(self, mode: str = "DRY_RUN", settings_manager: Optional[Any] = None) -> None:
        """
        Initialize WebSocket data service.

        Args:
            mode: Operating mode ("DRY_RUN", "DEMO", or "PRODUCTION")
            settings_manager: SettingsManager instance for loading API credentials
        """
        self.mode: str = mode
        self.ws_client: Optional[BinanceWebSocketClient] = None
        self.position_monitor: Optional[PositionMonitor] = None
        self.balance_monitor: Optional[BalanceMonitor] = None
        self.order_monitor: Optional[OrderMonitor] = None

        # Mock data for DRY_RUN mode
        self.mock_price_feed: MockPriceFeed = MockPriceFeed()

        # Event loop for async operations
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._running: bool = False

        # GUI callbacks
        self._position_callbacks: List[Callable[[PositionSnapshot], None]] = []
        self._balance_callbacks: List[Callable[[BalanceSnapshot], None]] = []
        self._order_callbacks: List[Callable[[OrderSnapshot], None]] = []
        self._price_callbacks: Dict[str, List[Callable[[float], None]]] = {}

        # Load API credentials from CredentialManager (reads from .env file)
        # Note: settings_manager doesn't store API keys for security reasons
        credential_manager: CredentialManager = CredentialManager()

        # Get exchange from settings (default to binance)
        exchange: str = "binance"
        if settings_manager:
            exchange = settings_manager.get("api.exchange", "binance").lower()
            # Map exchange names
            if exchange == "demo":
                exchange = "binance"  # Demo uses binance testnet
            self.testnet: bool = bool(settings_manager.get("api.testnet", False))
        else:
            self.testnet = os.getenv("BINANCE_TESTNET", "false").lower() == "true"

        # Load credentials from .env via CredentialManager
        creds: Dict[str, Optional[str]] = credential_manager.load_credentials(exchange)
        self.api_key: str = creds.get("api_key") or ""
        self.api_secret: str = creds.get("api_secret") or ""

        # Log credential status (without exposing actual keys)
        if self.api_key and self.api_secret:
            logger.info(f"Credentials loaded for {exchange} (key length: {len(self.api_key)})")
        else:
            logger.warning(f"No credentials found for {exchange} - WebSocket will fail in PRODUCTION mode")

        logger.info(f"WebSocketDataService initialized (mode={mode})")

    def start(self) -> None:
        """
        Start WebSocket service in background thread.

        This starts an asyncio event loop in a separate thread to handle
        WebSocket connections without blocking the GUI thread.
        """
        if self._running:
            logger.warning("WebSocket service already running")
            return

        if self.mode == "DRY_RUN":
            logger.info("DRY_RUN mode - WebSocket not started (using mock data)")
            return

        self._running = True

        # Start event loop in background thread
        self._loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._loop_thread.start()

        logger.info("✅ WebSocket service started in background")

    def _run_event_loop(self) -> None:
        """Run asyncio event loop in background thread."""
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            # Run async initialization
            self._loop.run_until_complete(self._async_start())

            # Keep loop running
            self._loop.run_forever()

        except Exception as e:
            logger.error(f"Error in WebSocket event loop: {e}", exc_info=True)
        finally:
            if self._loop:
                self._loop.close()

    async def _async_start(self) -> None:
        """Initialize WebSocket client and monitors (async)."""
        try:
            # Initialize WebSocket client
            self.ws_client = BinanceWebSocketClient(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet,
            )

            await self.ws_client.connect()
            logger.info("WebSocket client connected")

            # Initialize monitors
            self.position_monitor = PositionMonitor(self.ws_client, max_positions=5)
            self.balance_monitor = BalanceMonitor(self.ws_client)
            self.order_monitor = OrderMonitor(self.ws_client)

            # Register internal callbacks
            self.position_monitor.add_callback(self._handle_position_update)
            self.balance_monitor.add_callback(self._handle_balance_update)
            self.order_monitor.add_callback(self._handle_order_update)

            # Start monitors
            await self.position_monitor.start()
            await self.balance_monitor.start()
            await self.order_monitor.start()

            # Start watching WebSocket streams
            await self.ws_client.start_watching_all()

            logger.info("✅ All WebSocket monitors started")

        except Exception as e:
            logger.error(f"Failed to start WebSocket monitors: {e}", exc_info=True)
            self._running = False

    def stop(self) -> None:
        """Stop WebSocket service."""
        if not self._running:
            return

        self._running = False

        if self._loop and self._loop.is_running():
            # Schedule async cleanup and wait for it to finish
            try:
                future = asyncio.run_coroutine_threadsafe(self._async_stop(), self._loop)
                future.result(timeout=10.0)  # Increased timeout for cleanup
            except TimeoutError:
                logger.warning("WebSocket cleanup timed out after 10s, forcing close")
                # Force close ws_client if it exists
                if self.ws_client:
                    try:
                        # Try to close synchronously by scheduling and waiting briefly
                        future = asyncio.run_coroutine_threadsafe(self.ws_client.close(), self._loop)
                        future.result(timeout=2.0)
                    except Exception:
                        pass  # Already tried our best
            except Exception as e:
                logger.error(f"Error waiting for WebSocket cleanup: {e}")
            finally:
                # Ensure event loop is stopped
                if self._loop and self._loop.is_running():
                    self._loop.call_soon_threadsafe(self._loop.stop)

        # Join the thread
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=2.0)

        logger.info("WebSocket service stopped")

    async def _async_stop(self) -> None:
        """Stop WebSocket monitors and close connection (async)."""
        try:
            if self.position_monitor:
                await self.position_monitor.stop()
            if self.balance_monitor:
                await self.balance_monitor.stop()
            if self.order_monitor:
                await self.order_monitor.stop()
            if self.ws_client:
                await self.ws_client.close()

            logger.info("WebSocket monitors stopped")

        except Exception as e:
            logger.error(f"Error stopping WebSocket monitors: {e}")

    # ==================== GUI Callback Registration ====================

    def on_position_update(self, callback: Callable[[PositionSnapshot], None]) -> None:
        """
        Register callback for position updates.

        Args:
            callback: Function(PositionSnapshot) called on position update
        """
        self._position_callbacks.append(callback)
        logger.debug(f"Registered position callback: {callback.__name__}")

    def on_balance_update(self, callback: Callable[[BalanceSnapshot], None]) -> None:
        """
        Register callback for balance updates.

        Args:
            callback: Function(BalanceSnapshot) called on balance update
        """
        self._balance_callbacks.append(callback)
        logger.debug(f"Registered balance callback: {callback.__name__}")

    def on_order_update(self, callback: Callable[[OrderSnapshot], None]) -> None:
        """
        Register callback for order updates.

        Args:
            callback: Function(OrderSnapshot) called on order update
        """
        self._order_callbacks.append(callback)
        logger.debug(f"Registered order callback: {callback.__name__}")

    def on_price_update(self, symbol: str, callback: Callable[[float], None]) -> None:
        """
        Register callback for price updates.

        Args:
            symbol: Trading symbol
            callback: Function(price) called on price update
        """
        if symbol not in self._price_callbacks:
            self._price_callbacks[symbol] = []

        self._price_callbacks[symbol].append(callback)
        logger.debug(f"Registered price callback for {symbol}: {callback.__name__}")

    # ==================== Internal Callbacks (WebSocket -> GUI) ====================

    def _handle_position_update(self, position: PositionSnapshot) -> None:
        """
        Handle position update from WebSocket.

        Args:
            position: Position snapshot
        """
        # Trigger GUI callbacks (in GUI thread)
        for callback in self._position_callbacks:
            try:
                callback(position)
            except Exception as e:
                logger.error(f"Error in GUI position callback: {e}")

    def _handle_balance_update(self, balance: BalanceSnapshot) -> None:
        """
        Handle balance update from WebSocket.

        Args:
            balance: Balance snapshot
        """
        # Trigger GUI callbacks (in GUI thread)
        for callback in self._balance_callbacks:
            try:
                callback(balance)
            except Exception as e:
                logger.error(f"Error in GUI balance callback: {e}")

    def _handle_order_update(self, order: OrderSnapshot) -> None:
        """
        Handle order update from WebSocket.

        Args:
            order: Order snapshot
        """
        # Trigger GUI callbacks (in GUI thread)
        for callback in self._order_callbacks:
            try:
                callback(order)
            except Exception as e:
                logger.error(f"Error in GUI order callback: {e}")

        # Sync to DB if order is closed/canceled/rejected
        if order.status in ("closed", "canceled", "rejected"):
            try:
                from modules.auto_trade.execution.order_tagging import OrderTagger

                if OrderTagger.is_programmatic_order_id(order.client_order_id):
                    from modules.auto_trade.database import session_scope, update_order_status_by_client_id

                    # Map WebSocket status to DB status
                    status_map: Dict[str, str] = {"closed": "CLOSED", "canceled": "CANCELLED", "rejected": "FAILED"}
                    db_status: str = status_map.get(order.status, "CLOSED")

                    with session_scope() as session:
                        updated: bool = update_order_status_by_client_id(
                            session=session,
                            client_order_id=order.client_order_id,
                            status=db_status,
                            closed_at=order.last_update_timestamp,
                            pnl=None,  # May be updated later from snapshot if available
                        )
                        if updated:
                            logger.info(f"WS sync: updated order {order.client_order_id} to {db_status}")
            except Exception as e:
                logger.error(f"WS sync error for order {order.client_order_id}: {e}")

    # ==================== Synchronous API for GUI ====================

    def get_current_price(self, symbol: str) -> float:
        """
        Get current price for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Current price
        """
        if self.mode == "DRY_RUN":
            return self.mock_price_feed.get_current_price(symbol)

        # For WebSocket mode, get from position monitor's mark prices
        if self.position_monitor:
            position = self.position_monitor.get_position(symbol)
            if position:
                return position.mark_price

        # Fallback to mock prices if position not found
        return self.mock_price_feed.get_current_price(symbol)

    def get_positions(self) -> List[PositionSnapshot]:
        """
        Get current open positions.

        Returns:
            List of position snapshots
        """
        if self.mode == "DRY_RUN":
            return []  # No positions in dry run

        if self.position_monitor:
            return self.position_monitor.get_open_positions()

        return []

    def get_balance(self) -> Optional[BalanceSnapshot]:
        """
        Get current account balance.

        Returns:
            Balance snapshot or None
        """
        if self.mode == "DRY_RUN":
            # Return mock balance
            from datetime import datetime

            return BalanceSnapshot(currency="USDT", total=10000.0, free=9500.0, used=500.0, timestamp=datetime.now())

        if self.balance_monitor:
            return self.balance_monitor.get_balance()

        return None

    def get_orders(self) -> List[OrderSnapshot]:
        """
        Get current open orders.

        Returns:
            List of order snapshots
        """
        if self.mode == "DRY_RUN":
            return []  # No orders in dry run

        if self.order_monitor:
            return self.order_monitor.get_open_orders()

        return []

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        if self.mode == "DRY_RUN":
            return True  # Always "connected" in dry run

        return self._running and self.ws_client is not None

    @property
    def is_running(self) -> bool:
        """Check if service is running."""
        return self._running

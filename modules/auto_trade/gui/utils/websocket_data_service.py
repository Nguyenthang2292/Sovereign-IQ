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
import os
import threading
from typing import Any, Callable, Dict, List, Optional, TypeVar

_T = TypeVar("_T")

from modules.auto_trade.gui.utils.credential_manager import CredentialManager
from modules.auto_trade.gui.utils.mock_price_feed import MockPriceFeed
from modules.auto_trade.monitoring.account_monitor import BalanceMonitor, BalanceSnapshot, OrderMonitor, OrderSnapshot
from modules.auto_trade.monitoring.position_monitor import PositionMonitor, PositionSnapshot
from modules.auto_trade.websocket.client import BinanceWebSocketClient
from modules.common.ui.logging import log_debug, log_error, log_info, log_warn


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

    def __init__(
        self,
        mode: str = "DRY_RUN",
        settings_manager: Optional[Any] = None,
        event_bus: Optional[Any] = None,
        tk_root: Optional[Any] = None,
    ) -> None:
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

        # tkinter root for thread-safe GUI callback dispatch
        self._tk_root: Optional[Any] = tk_root

        # Event loop for async operations
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._running: bool = False
        self.event_bus: Optional[Any] = event_bus
        self._published_closed_events: set[str] = set()

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
            log_info(f"Credentials loaded for {exchange} (key length: {len(self.api_key)})")
        else:
            log_warn(f"No credentials found for {exchange} - WebSocket will fail in PRODUCTION mode")

        log_info(f"WebSocketDataService initialized (mode={mode})")

    def start(self) -> None:
        """
        Start WebSocket service in background thread.

        This starts an asyncio event loop in a separate thread to handle
        WebSocket connections without blocking the GUI thread.
        """
        if self._running:
            log_warn("WebSocket service already running")
            return

        if self.mode == "DRY_RUN":
            log_info("DRY_RUN mode - WebSocket not started (using mock data)")
            return

        self._running = True

        # Start event loop in background thread
        self._loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._loop_thread.start()

        log_info("✅ WebSocket service started in background")

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
            log_error(f"Error in WebSocket event loop: {e}", exc_info=True)
        finally:
            if self._loop:
                self._loop.close()

    async def _async_start(self) -> None:
        """Initialize WebSocket client and start monitors (async).

        connect() with binanceusdm is instant (no REST pre-flight) — the
        listen key is created lazily on the first watch_* call.  Monitor
        startup failures (e.g., auth errors) are caught and logged.
        """
        try:
            self.ws_client = BinanceWebSocketClient(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet,
            )

            await self.ws_client.connect()  # instant — no REST call
            log_info("WebSocket client ready")

            # Initialize monitors
            self.position_monitor = PositionMonitor(self.ws_client, max_positions=5)
            self.balance_monitor = BalanceMonitor(self.ws_client)
            self.order_monitor = OrderMonitor(self.ws_client)

            # Register internal callbacks
            self.position_monitor.add_callback(self._handle_position_update)
            self.balance_monitor.add_callback(self._handle_balance_update)
            self.order_monitor.add_callback(self._handle_order_update)

            # Start monitors (these start watch_* tasks which make the first
            # REST call to create the listen key on Binance)
            await self.position_monitor.start()
            await self.balance_monitor.start()
            await self.order_monitor.start()

            # Start watching WebSocket streams
            await self.ws_client.start_watching_all()

            log_info("✅ All WebSocket monitors started")

        except Exception as exc:
            log_error(f"Failed to start WebSocket service: {exc}")
            self._running = False

    def stop(self) -> None:
        """Stop WebSocket service.

        Designed to be called from a background daemon thread (see on_closing).
        Uses a short timeout so the calling thread (and thus the process) can
        exit cleanly without hanging if the network is slow.
        """
        if not self._running:
            return

        self._running = False

        if self._loop and self._loop.is_running():
            try:
                future = asyncio.run_coroutine_threadsafe(self._async_stop(), self._loop)
                # 3 s is plenty in normal conditions; if Binance REST is slow
                # the daemon thread (and the process) will be killed anyway.
                future.result(timeout=3.0)
            except TimeoutError:
                log_warn("WebSocket cleanup timed out after 3s, forcing loop stop")
            except Exception as e:
                log_error(f"Error waiting for WebSocket cleanup: {e}")
            finally:
                # Ensure event loop is stopped regardless
                if self._loop and self._loop.is_running():
                    self._loop.call_soon_threadsafe(self._loop.stop)

        # Join the thread with a short timeout; daemon=True means the OS will
        # clean it up when the process exits even if join() hits the limit.
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=1.0)

        log_info("WebSocket service stopped")

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

            log_info("WebSocket monitors stopped")

        except Exception as e:
            log_error(f"Error stopping WebSocket monitors: {e}")

    # ==================== GUI Callback Registration ====================

    def on_position_update(self, callback: Callable[[PositionSnapshot], None]) -> None:
        """
        Register callback for position updates.

        Args:
            callback: Function(PositionSnapshot) called on position update
        """
        self._position_callbacks.append(callback)
        log_debug(f"Registered position callback: {callback.__name__}")

    def on_balance_update(self, callback: Callable[[BalanceSnapshot], None]) -> None:
        """
        Register callback for balance updates.

        Args:
            callback: Function(BalanceSnapshot) called on balance update
        """
        self._balance_callbacks.append(callback)
        log_debug(f"Registered balance callback: {callback.__name__}")

    def on_order_update(self, callback: Callable[[OrderSnapshot], None]) -> None:
        """
        Register callback for order updates.

        Args:
            callback: Function(OrderSnapshot) called on order update
        """
        self._order_callbacks.append(callback)
        log_debug(f"Registered order callback: {callback.__name__}")

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
        log_debug(f"Registered price callback for {symbol}: {callback.__name__}")

    # ==================== Internal Callbacks (WebSocket -> GUI) ====================

    def _dispatch_to_main(self, callback: Callable, data: Any) -> None:
        """Call *callback(data)* on the tkinter main thread.

        When invoked from a background thread (WebSocket asyncio loop) and a
        tk_root is available, we use ``root.after(0, ...)`` which is the
        correct tkinter thread-safe mechanism.  If already on the main thread
        (or no tk_root), the callback is called directly.
        """
        is_background = threading.current_thread() is not threading.main_thread()
        if is_background and self._tk_root is not None:
            try:
                self._tk_root.after(0, callback, data)
            except Exception as e:
                log_error(f"Error scheduling GUI callback via after(): {e}")
        else:
            # Already on main thread or no root available (e.g. tests)
            callback(data)

    def _handle_position_update(self, position: PositionSnapshot) -> None:
        """
        Handle position update from WebSocket.

        Args:
            position: Position snapshot
        """
        # ── 1. If position size is 0, it means the position just closed ──
        # This fires for BOTH manual close AND TP/SL fill.
        # _handle_order_update already handles TP/SL fill (with real PnL).
        # This block acts as a SAFETY NET for manual closes where no TP/SL fill event arrives.
        if position.position_amt == 0:
            symbol_normalized = position.symbol.replace("/", "").split(":")[0]
            log_info(f"[WS Data] Position {symbol_normalized} closed (size→0). Checking DB...")
            try:
                from modules.auto_trade.database.repository.context import RepositoryContext
                from modules.auto_trade.monitoring.event_system import EventType

                ctx = RepositoryContext.from_env()
                db_orders = ctx.orders.get_open_positions(symbol=symbol_normalized)

                # Filter to orders that have NOT been closed yet by _handle_order_update
                pending_orders = [o for o in db_orders if o.get("status", "").upper() == "OPEN"]

                if not pending_orders:
                    # Already handled by _handle_order_update (TP/SL fill path)
                    log_info(f"[WS Data] {symbol_normalized}: already CLOSED in DB — skip duplicate cleanup.")
                else:
                    # Manual close — no TP/SL conditional order fired → we must cleanup
                    log_info(
                        f"[WS Data] {symbol_normalized}: manual close detected "
                        f"({len(pending_orders)} OPEN DB record(s)). Running cleanup..."
                    )

                    # Cancel any orphaned conditional orders still on Binance
                    if self.api_key and self.mode != "DRY_RUN":
                        from modules.auto_trade.execution.binance_client import BinanceClient
                        _client = BinanceClient(
                            api_key=self.api_key,
                            api_secret=self.api_secret,
                            testnet=self.testnet,
                            dry_run=False,
                        )
                        try:
                            cancel_res = _client.cancel_open_orders(symbol_normalized)
                            log_info(f"[WS Data] Cancelled orphaned orders for {symbol_normalized}: {cancel_res}")
                        except Exception as _ce:
                            log_warn(f"[WS Data] cancel_open_orders({symbol_normalized}) non-fatal: {_ce}")

                    # Mark DB CLOSED and publish event
                    for db_order in pending_orders:
                        order_id = db_order.get("order_id")
                        client_order_id = db_order.get("client_order_id")
                        if not order_id:
                            continue

                        ctx.orders.update_order_status(order_id, "CLOSED")
                        log_info(
                            f"[WS Data] DB closed (manual) for {symbol_normalized} (order={order_id})"
                        )

                        if self.event_bus and client_order_id and client_order_id not in self._published_closed_events:
                            # PnL for manual close: DB value (0.0 — no realized PnL available from WS)
                            pnl_value = float(db_order.get("pnl", 0.0) or 0.0)
                            self.event_bus.publish(
                                EventType.POSITION_CLOSED,
                                {
                                    "symbol": symbol_normalized,
                                    "pnl": pnl_value,
                                    "is_profit": pnl_value >= 0,
                                    "exit_price": position.mark_price,
                                    "entry_price": position.entry_price or float(db_order.get("entry_price", 0)),
                                    "leverage": position.leverage or int(db_order.get("leverage", 1) or 1),
                                    "duration_seconds": 0,
                                    "is_programmatic": True,
                                },
                                source="WebSocketDataService (manual close)",
                            )
                            self._published_closed_events.add(client_order_id)
                            log_info(
                                f"[WS Data] POSITION_CLOSED event (manual) for {symbol_normalized} "
                                f"(pnl={pnl_value:+.2f})"
                            )

            except Exception as e:
                log_error(
                    f"[WS Data] Error in position close cleanup for {position.symbol}: {e}",
                    exc_info=True,
                )

        # ── 2. Pass to GUI callbacks (always, including zero-size snapshots) ──
        for callback in self._position_callbacks:
            try:
                self._dispatch_to_main(callback, position)
            except Exception as e:
                log_error(f"Error in GUI position callback: {e}")

    def _handle_balance_update(self, balance: BalanceSnapshot) -> None:
        """
        Handle balance update from WebSocket.

        Args:
            balance: Balance snapshot
        """
        for callback in self._balance_callbacks:
            try:
                self._dispatch_to_main(callback, balance)
            except Exception as e:
                log_error(f"Error in GUI balance callback: {e}")

    def _handle_order_update(self, order: OrderSnapshot) -> None:
        """
        Handle order update from WebSocket.

        Args:
            order: Order snapshot
        """
        for callback in self._order_callbacks:
            try:
                self._dispatch_to_main(callback, order)
            except Exception as e:
                log_error(f"Error in GUI order callback: {e}")

        # ── Order DB sync logic ──────────────────────────────────────────────
        try:
            from modules.auto_trade.execution.order_tagging import OrderTagger

            order_type_raw = order.type.lower()  # 'take_profit_market', 'stop_market', 'market', 'limit', ...
            # Detect if this is a TP/SL conditional order that just filled
            is_tp_sl_fill = (
                order.status == "closed"
                and any(t in order_type_raw for t in ("take_profit", "stop_market", "stop_loss"))
            )

            if is_tp_sl_fill:
                # ── TP or SL order was filled → position is now closed ───────────────
                symbol_normalized = order.symbol.replace("/", "").split(":")[0]
                pnl_from_ws: Optional[float] = order.realized_pnl   # Real PnL from Binance WS event

                log_info(
                    f"[WS Data] TP/SL fill detected for {symbol_normalized} "
                    f"(type={order.type}, pnl={pnl_from_ws})"
                )

                # 1. Cancel any remaining sibling conditional orders (the paired TP or SL)
                if self.api_key and self.mode != "DRY_RUN":
                    from modules.auto_trade.execution.binance_client import BinanceClient
                    _client = BinanceClient(
                        api_key=self.api_key,
                        api_secret=self.api_secret,
                        testnet=self.testnet,
                        dry_run=False,
                    )
                    try:
                        cancel_res = _client.cancel_open_orders(symbol_normalized)
                        log_info(f"[WS Data] Cancelled sibling orders for {symbol_normalized}: {cancel_res}")
                    except Exception as _ce:
                        log_warn(f"[WS Data] cancel_open_orders({symbol_normalized}) non-fatal: {_ce}")

                # 2. Update DB to CLOSED with actual PnL + emit recovery event
                try:
                    from modules.auto_trade.database.repository.context import RepositoryContext
                    from modules.auto_trade.monitoring.event_system import EventType

                    ctx = RepositoryContext.from_env()
                    db_orders = ctx.orders.get_open_positions(symbol=symbol_normalized)

                    for db_order in db_orders:
                        order_id = db_order.get("order_id")
                        client_order_id = db_order.get("client_order_id")
                        if not order_id:
                            continue

                        # Compute effective PnL: prefer live WS value, fallback to DB stored value
                        if pnl_from_ws is not None:
                            effective_pnl = pnl_from_ws
                        else:
                            try:
                                effective_pnl = float(db_order.get("pnl", 0.0) or 0.0)
                            except (TypeError, ValueError):
                                effective_pnl = 0.0

                        ctx.orders.update_order_status(order_id, "CLOSED")
                        log_info(
                            f"[WS Data] DB closed for {symbol_normalized} "
                            f"(order_id={order_id}, pnl={effective_pnl:+.2f})"
                        )

                        if self.event_bus and client_order_id and client_order_id not in self._published_closed_events:
                            self.event_bus.publish(
                                EventType.POSITION_CLOSED,
                                {
                                    "symbol": symbol_normalized,
                                    "pnl": effective_pnl,
                                    "is_profit": effective_pnl >= 0,
                                    "exit_price": order.price,
                                    "entry_price": float(db_order.get("entry_price", 0) or 0),
                                    "leverage": int(db_order.get("leverage", 1) or 1),
                                    "duration_seconds": 0,
                                    "is_programmatic": True,
                                },
                                source="WebSocketDataService (TP/SL fill)",
                            )
                            self._published_closed_events.add(client_order_id)
                            log_info(
                                f"[WS Data] POSITION_CLOSED event published for {symbol_normalized} "
                                f"(pnl={effective_pnl:+.2f})"
                            )
                except Exception as _db_err:
                    log_error(f"[WS Data] DB/event error after TP/SL fill for {order.symbol}: {_db_err}")

            elif order.status in ("canceled", "rejected"):
                # ── Entry / other orders canceled or rejected → mark DB only ─────────
                if OrderTagger.is_programmatic_order_id(order.client_order_id):
                    from modules.auto_trade.database import update_order_status_by_client_id
                    status_map: Dict[str, str] = {"canceled": "CANCELLED", "rejected": "FAILED"}
                    db_status: str = status_map.get(order.status, "FAILED")
                    updated: bool = update_order_status_by_client_id(
                        client_order_id=order.client_order_id,
                        status=db_status,
                    )
                    if updated:
                        log_info(f"[WS Data] order {order.client_order_id} → {db_status}")

        except Exception as e:
            log_error(f"[WS Data] _handle_order_update sync error: {e}", exc_info=True)

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

"""
Balance and Order Monitor Module (WebSocket Version)

Monitors account balance and open orders in real-time using WebSocket.
Provides instant updates for:
- Account balance changes (deposits, withdrawals, realized P&L, funding)
- Order status changes (fills, cancellations, rejections)
- Order lifecycle tracking

Key improvements over REST polling:
- Real-time balance updates (<100ms vs polling)
- Instant order fill notifications
- Immediate order rejection detection
- Lower API rate limit usage
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional

from modules.auto_trade.websocket.client import BinanceWebSocketClient

logger = logging.getLogger(__name__)


@dataclass
class BalanceSnapshot:
    """Snapshot of account balance."""

    currency: str  # e.g., 'USDT'
    total: float  # Total balance
    free: float  # Available balance
    used: float  # Balance in orders
    timestamp: datetime


class BalanceMonitor:
    """
    Monitors account balance in real-time using WebSocket.

    Provides real-time balance updates via Binance User Data Stream.

    Example:
        >>> monitor = BalanceMonitor(ws_client)
        >>> monitor.add_callback(on_balance_update)
        >>> await monitor.start()
    """

    def __init__(self, ws_client: BinanceWebSocketClient):
        """
        Initialize BalanceMonitor.

        Args:
            ws_client: WebSocket client instance
        """
        self.ws_client = ws_client
        self._running = False
        self._callbacks: List[Callable[[BalanceSnapshot], None]] = []
        self._last_balance: Optional[BalanceSnapshot] = None

        logger.info("BalanceMonitor initialized (WebSocket mode)")

    def add_callback(self, callback: Callable[[BalanceSnapshot], None]) -> None:
        """
        Add callback for balance updates.

        Args:
            callback: Function(BalanceSnapshot) called on balance update
        """
        self._callbacks.append(callback)
        logger.info(f"Added balance callback: {callback.__name__}")

    async def start(self) -> None:
        """Start monitoring balance via WebSocket."""
        if self._running:
            logger.warning("BalanceMonitor is already running")
            return

        self._running = True

        # Fetch initial balance via REST
        initial_balance = await self.ws_client.get_initial_balance()

        if initial_balance:
            logger.info("Loaded initial balance")
            self._process_balance_update(initial_balance)

        # Register WebSocket callback
        self.ws_client.on_balance_update(self._handle_ws_balance_update)

        logger.info("✅ BalanceMonitor started (WebSocket mode)")

    async def stop(self) -> None:
        """Stop monitoring balance."""
        if not self._running:
            return

        self._running = False
        logger.info("⏹️  BalanceMonitor stopped")

    def _handle_ws_balance_update(self, balance: dict) -> None:
        """
        Handle WebSocket balance update.

        Args:
            balance: Balance dict from ccxt.pro
        """
        if not self._running:
            return

        try:
            self._process_balance_update(balance)
        except Exception as e:
            logger.error(f"Error handling WebSocket balance update: {e}", exc_info=True)

    def _process_balance_update(self, balance: dict) -> None:
        """
        Process balance update.

        Args:
            balance: Balance dict from ccxt.pro format:
                {
                    'USDT': {
                        'free': 1000.0,
                        'used': 100.0,
                        'total': 1100.0
                    },
                    ...
                }
        """
        # Extract USDT balance (primary currency for futures)
        usdt_balance = balance.get("USDT", {})

        if not usdt_balance:
            logger.warning("No USDT balance found in update")
            return

        snapshot = BalanceSnapshot(
            currency="USDT",
            total=float(usdt_balance.get("total") or 0),
            free=float(usdt_balance.get("free") or 0),
            used=float(usdt_balance.get("used") or 0),
            timestamp=datetime.now(),
        )

        # Log if changed significantly
        if self._has_significant_change(snapshot):
            logger.info(f"💰 Balance update: ${snapshot.total:.2f} USDT (free: ${snapshot.free:.2f})")

        # Update last balance
        self._last_balance = snapshot

        # Trigger callbacks
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(snapshot))
                else:
                    callback(snapshot)
            except Exception as e:
                logger.error(f"Error in balance callback {callback.__name__}: {e}")

    def _has_significant_change(self, snapshot: BalanceSnapshot) -> bool:
        """
        Check if balance changed significantly.

        Args:
            snapshot: Current balance snapshot

        Returns:
            True if significant change
        """
        if not self._last_balance:
            return True

        # Consider change significant if more than $1 change
        total_diff = abs(snapshot.total - self._last_balance.total)
        return total_diff > 1.0

    def get_balance(self) -> Optional[BalanceSnapshot]:
        """Get current balance snapshot."""
        return self._last_balance

    @property
    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._running


@dataclass
class OrderSnapshot:
    """Snapshot of an order."""

    order_id: str
    client_order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    type: str  # 'market', 'limit', 'stop_market', 'take_profit_market'
    status: str  # 'open', 'closed', 'canceled', 'rejected'
    price: float
    amount: float
    filled: float
    remaining: float
    timestamp: datetime
    last_update_timestamp: datetime


class OrderMonitor:
    """
    Monitors open orders in real-time using WebSocket.

    Provides real-time order updates via Binance User Data Stream.

    Example:
        >>> monitor = OrderMonitor(ws_client)
        >>> monitor.add_callback(on_order_update)
        >>> await monitor.start()
    """

    def __init__(self, ws_client: BinanceWebSocketClient) -> None:
        """
        Initialize OrderMonitor.

        Args:
            ws_client: WebSocket client instance
        """
        self.ws_client = ws_client
        self._running = False
        self._callbacks: List[Callable[[OrderSnapshot], None]] = []
        self._open_orders: Dict[str, OrderSnapshot] = {}  # order_id -> snapshot

        logger.info("OrderMonitor initialized (WebSocket mode)")

    def add_callback(self, callback: Callable[[OrderSnapshot], None]) -> None:
        """
        Add callback for order updates.

        Args:
            callback: Function(OrderSnapshot) called on order update
        """
        self._callbacks.append(callback)
        logger.info(f"Added order callback: {callback.__name__}")

    async def start(self) -> None:
        """Start monitoring orders via WebSocket."""
        if self._running:
            logger.warning("OrderMonitor is already running")
            return

        self._running = True

        # Fetch initial orders via REST
        initial_orders = await self.ws_client.get_initial_orders()

        if initial_orders:
            logger.info(f"Loaded {len(initial_orders)} initial orders")
            for order in initial_orders:
                self._process_order_update([order])

        # Register WebSocket callback
        self.ws_client.on_order_update(self._handle_ws_order_update)

        logger.info("✅ OrderMonitor started (WebSocket mode)")

    async def stop(self) -> None:
        """Stop monitoring orders."""
        if not self._running:
            return

        self._running = False
        logger.info("⏹️  OrderMonitor stopped")

    def _handle_ws_order_update(self, orders: List[dict]) -> None:
        """
        Handle WebSocket order update.

        Args:
            orders: List of order dicts from ccxt.pro
        """
        if not self._running:
            return

        try:
            self._process_order_update(orders)
        except Exception as e:
            logger.error(f"Error handling WebSocket order update: {e}", exc_info=True)

    def _process_order_update(self, orders: List[dict]) -> None:
        """
        Process order updates.

        Args:
            orders: List of order dicts from ccxt.pro
        """
        for order_data in orders:
            try:
                snapshot = self._parse_order(order_data)

                # Log order status
                self._log_order_status(snapshot)

                # Update open orders dict
                if snapshot.status == "open":
                    self._open_orders[snapshot.order_id] = snapshot
                elif snapshot.order_id in self._open_orders:
                    # Order no longer open (filled, canceled, rejected)
                    del self._open_orders[snapshot.order_id]

                # Trigger callbacks
                for callback in self._callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            asyncio.create_task(callback(snapshot))
                        else:
                            callback(snapshot)
                    except Exception as e:
                        logger.error(f"Error in order callback {callback.__name__}: {e}")

            except Exception as e:
                logger.error(f"Error processing order: {e}", exc_info=True)

    def _parse_order(self, data: dict) -> OrderSnapshot:
        """
        Parse order data from ccxt.pro.

        ccxt.pro normalizes order data:
        {
            'id': '123456',
            'clientOrderId': 'abc',
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'type': 'limit',
            'status': 'open',
            'price': 35000,
            'amount': 0.001,
            'filled': 0.0005,
            'remaining': 0.0005,
            'timestamp': 1623456789000,
            'lastUpdateTimestamp': 1623456790000,
            'info': {...}
        }
        """
        return OrderSnapshot(
            order_id=str(data.get("id", "")),
            client_order_id=str(data.get("clientOrderId", "")),
            symbol=data.get("symbol", ""),
            side=data.get("side", "").lower(),
            type=data.get("type", "").lower(),
            status=data.get("status", "").lower(),
            price=float(data.get("price", 0)),
            amount=float(data.get("amount", 0)),
            filled=float(data.get("filled", 0)),
            remaining=float(data.get("remaining", 0)),
            timestamp=datetime.fromtimestamp(data.get("timestamp", 0) / 1000)
            if data.get("timestamp")
            else datetime.now(),
            last_update_timestamp=datetime.fromtimestamp(data.get("lastUpdateTimestamp", 0) / 1000)
            if data.get("lastUpdateTimestamp")
            else datetime.now(),
        )

    def _log_order_status(self, snapshot: OrderSnapshot) -> None:
        """
        Log order status changes.

        Args:
            snapshot: Order snapshot
        """
        if snapshot.status == "open":
            logger.info(
                f"📝 Order {snapshot.order_id[:8]}: "
                f"{snapshot.side.upper()} {snapshot.amount} {snapshot.symbol} @ ${snapshot.price} ({snapshot.type})"
            )
        elif snapshot.status == "closed":
            logger.info(
                f"✅ Order filled {snapshot.order_id[:8]}: "
                f"{snapshot.side.upper()} {snapshot.filled}/{snapshot.amount} {snapshot.symbol}"
            )
        elif snapshot.status == "canceled":
            logger.info(f"❌ Order canceled {snapshot.order_id[:8]}: {snapshot.symbol}")
        elif snapshot.status == "rejected":
            logger.error(f"⛔ Order rejected {snapshot.order_id[:8]}: {snapshot.symbol}")

    def get_open_orders(self) -> List[OrderSnapshot]:
        """Get all open orders."""
        return list(self._open_orders.values())

    def get_order(self, order_id: str) -> Optional[OrderSnapshot]:
        """
        Get order by ID.

        Args:
            order_id: Order ID

        Returns:
            Order snapshot or None
        """
        return self._open_orders.get(order_id)

    def get_orders_by_symbol(self, symbol: str) -> List[OrderSnapshot]:
        """
        Get orders for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            List of order snapshots
        """
        return [order for order in self._open_orders.values() if order.symbol == symbol]

    @property
    def order_count(self) -> int:
        """Get number of open orders."""
        return len(self._open_orders)

    @property
    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._running

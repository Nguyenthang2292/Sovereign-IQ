"""
Binance WebSocket Client for Auto Trade System

Wraps ccxt.pro Binance WebSocket functionality to provide real-time updates:
- Positions (watchPositions)
- Balance (watchBalance)
- Orders (watchOrders)
- Mark Prices (watchMarkPrice)

Uses ccxt.pro which handles:
- Automatic reconnection
- Listen key management and renewal
- Message parsing and normalization
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

import ccxt.pro as ccxtpro

logger = logging.getLogger(__name__)


class BinanceWebSocketClient:
    """
    WebSocket client for real-time Binance Futures data.

    This class uses ccxt.pro which provides:
    - Automatic WebSocket connection management
    - Listen key creation and renewal
    - Reconnection on disconnect
    - Unified data format across exchanges

    Example:
        >>> ws_client = BinanceWebSocketClient(api_key, api_secret, testnet=True)
        >>> await ws_client.connect()
        >>> await ws_client.watch_positions(callback=on_position_update)
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
        options: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize WebSocket client.

        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Use testnet/demo environment if True
            options: Additional ccxt options
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet

        # Initialize ccxt.pro exchange
        exchange_options = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",  # USDT-M Futures
                "adjustForTimeDifference": True,
                "recvWindow": 60000,
            },
        }

        # Add testnet URLs if needed
        if testnet:
            exchange_options["urls"] = {
                "api": {
                    "fapiPublic": "https://testnet.binancefuture.com/fapi/v1",
                    "fapiPrivate": "https://testnet.binancefuture.com/fapi/v1",
                    "fapiPrivateV2": "https://testnet.binancefuture.com/fapi/v2",
                },
                "ws": {
                    "future": "wss://fstream.binancefuture.com/ws",
                },
            }

        # Merge custom options
        if options:
            exchange_options = {**exchange_options, **options}

        self.exchange = ccxtpro.binance(exchange_options)

        # Callback storage
        self.position_callbacks: List[Callable] = []
        self.balance_callbacks: List[Callable] = []
        self.order_callbacks: List[Callable] = []
        self.mark_price_callbacks: Dict[str, List[Callable]] = {}

        # Task management
        self.watch_tasks: List[asyncio.Task] = []
        self.running = False

        logger.info(f"BinanceWebSocketClient initialized (testnet={testnet})")

    async def connect(self):
        """
        Connect to WebSocket and start watching streams.

        This loads markets and prepares the exchange for WebSocket operations.
        """
        try:
            await self.exchange.load_markets()
            self.running = True
            logger.info("✅ WebSocket client connected")
        except Exception as e:
            logger.error(f"Failed to connect WebSocket client: {e}")
            raise

    async def close(self):
        """
        Close WebSocket connection and cleanup.
        """
        self.running = False

        # Cancel all watch tasks
        for task in self.watch_tasks:
            task.cancel()

        # Close exchange connection
        await self.exchange.close()

        logger.info("WebSocket client closed")

    # ==================== Position Monitoring ====================

    def on_position_update(self, callback: Callable):
        """
        Register callback for position updates.

        Args:
            callback: Function(positions: List[dict]) called on position update
        """
        self.position_callbacks.append(callback)
        logger.info(f"Registered position callback: {callback.__name__}")

    async def watch_positions(self):
        """
        Watch position updates in real-time.

        Uses ccxt.pro watchPositions() which provides real-time position updates
        via Binance User Data Stream.

        Positions are automatically updated when:
        - New position opened
        - Position size changed
        - P&L changed
        - Position closed
        """
        try:
            logger.info("Started watching positions...")

            while self.running:
                try:
                    # Watch positions (blocks until update received)
                    positions = await self.exchange.watch_positions()

                    # Filter non-zero positions
                    open_positions = [p for p in positions if float(p.get("contracts", 0)) != 0]

                    if open_positions:
                        logger.debug(f"Position update received: {len(open_positions)} open positions")

                        # Trigger callbacks
                        for callback in self.position_callbacks:
                            try:
                                if asyncio.iscoroutinefunction(callback):
                                    await callback(open_positions)
                                else:
                                    callback(open_positions)
                            except Exception as e:
                                logger.error(f"Error in position callback {callback.__name__}: {e}")

                except Exception as e:
                    if self.running:
                        logger.error(f"Error watching positions: {e}")
                        await asyncio.sleep(5)  # Wait before retry

        except asyncio.CancelledError:
            logger.info("Position watching cancelled")
        except Exception as e:
            logger.error(f"Fatal error in watch_positions: {e}")

    # ==================== Balance Monitoring ====================

    def on_balance_update(self, callback: Callable):
        """
        Register callback for balance updates.

        Args:
            callback: Function(balance: dict) called on balance update
        """
        self.balance_callbacks.append(callback)
        logger.info(f"Registered balance callback: {callback.__name__}")

    async def watch_balance(self):
        """
        Watch balance updates in real-time.

        Uses ccxt.pro watchBalance() which provides real-time balance updates
        via Binance User Data Stream.

        Balance is automatically updated when:
        - Orders filled
        - Deposits/withdrawals
        - Funding payments
        - Realized P&L changes
        """
        try:
            logger.info("Started watching balance...")

            while self.running:
                try:
                    # Watch balance (blocks until update received)
                    balance = await self.exchange.watch_balance()

                    logger.debug(f"Balance update received: {balance.get('USDT', {}).get('total', 0)} USDT")

                    # Trigger callbacks
                    for callback in self.balance_callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(balance)
                            else:
                                callback(balance)
                        except Exception as e:
                            logger.error(f"Error in balance callback {callback.__name__}: {e}")

                except Exception as e:
                    if self.running:
                        logger.error(f"Error watching balance: {e}")
                        await asyncio.sleep(5)  # Wait before retry

        except asyncio.CancelledError:
            logger.info("Balance watching cancelled")
        except Exception as e:
            logger.error(f"Fatal error in watch_balance: {e}")

    # ==================== Order Monitoring ====================

    def on_order_update(self, callback: Callable):
        """
        Register callback for order updates.

        Args:
            callback: Function(orders: List[dict]) called on order update
        """
        self.order_callbacks.append(callback)
        logger.info(f"Registered order callback: {callback.__name__}")

    async def watch_orders(self, symbol: Optional[str] = None):
        """
        Watch order updates in real-time.

        Uses ccxt.pro watchOrders() which provides real-time order updates
        via Binance User Data Stream.

        Orders are automatically updated when:
        - New order created
        - Order filled (partial or complete)
        - Order cancelled
        - Order rejected

        Args:
            symbol: Optional symbol to watch (None = all symbols)
        """
        try:
            logger.info(f"Started watching orders (symbol={symbol or 'ALL'})...")

            while self.running:
                try:
                    # Watch orders (blocks until update received)
                    orders = await self.exchange.watch_orders(symbol)

                    if orders:
                        logger.debug(f"Order update received: {len(orders)} orders")

                        # Trigger callbacks
                        for callback in self.order_callbacks:
                            try:
                                if asyncio.iscoroutinefunction(callback):
                                    await callback(orders)
                                else:
                                    callback(orders)
                            except Exception as e:
                                logger.error(f"Error in order callback {callback.__name__}: {e}")

                except Exception as e:
                    if self.running:
                        logger.error(f"Error watching orders: {e}")
                        await asyncio.sleep(5)  # Wait before retry

        except asyncio.CancelledError:
            logger.info("Order watching cancelled")
        except Exception as e:
            logger.error(f"Fatal error in watch_orders: {e}")

    # ==================== Mark Price Monitoring ====================

    def on_mark_price_update(self, symbol: str, callback: Callable):
        """
        Register callback for mark price updates.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            callback: Function(mark_price: dict) called on price update
        """
        if symbol not in self.mark_price_callbacks:
            self.mark_price_callbacks[symbol] = []

        self.mark_price_callbacks[symbol].append(callback)
        logger.info(f"Registered mark price callback for {symbol}: {callback.__name__}")

    async def watch_mark_price(self, symbol: str):
        """
        Watch mark price updates for a symbol.

        Uses ccxt.pro watchMarkPrice() which provides real-time mark price updates.
        Mark price updates every ~3 seconds on Binance Futures.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
        """
        try:
            logger.info(f"Started watching mark price for {symbol}...")

            while self.running:
                try:
                    # Watch mark price (blocks until update received)
                    mark_price = await self.exchange.watch_mark_price(symbol)

                    if mark_price:
                        logger.debug(f"Mark price update for {symbol}: {mark_price.get('markPrice')}")

                        # Trigger callbacks for this symbol
                        for callback in self.mark_price_callbacks.get(symbol, []):
                            try:
                                if asyncio.iscoroutinefunction(callback):
                                    await callback(mark_price)
                                else:
                                    callback(mark_price)
                            except Exception as e:
                                logger.error(f"Error in mark price callback {callback.__name__}: {e}")

                except Exception as e:
                    if self.running:
                        logger.error(f"Error watching mark price for {symbol}: {e}")
                        await asyncio.sleep(5)  # Wait before retry

        except asyncio.CancelledError:
            logger.info(f"Mark price watching cancelled for {symbol}")
        except Exception as e:
            logger.error(f"Fatal error in watch_mark_price for {symbol}: {e}")

    # ==================== Convenience Methods ====================

    async def start_watching_all(self, symbols_for_mark_price: Optional[List[str]] = None):
        """
        Start watching all streams (positions, balance, orders, mark prices).

        This is a convenience method to start all watchers at once.

        Args:
            symbols_for_mark_price: Optional list of symbols to watch mark prices for
        """
        logger.info("Starting all WebSocket watchers...")

        # Start position watching
        if self.position_callbacks:
            task = asyncio.create_task(self.watch_positions())
            self.watch_tasks.append(task)

        # Start balance watching
        if self.balance_callbacks:
            task = asyncio.create_task(self.watch_balance())
            self.watch_tasks.append(task)

        # Start order watching
        if self.order_callbacks:
            task = asyncio.create_task(self.watch_orders())
            self.watch_tasks.append(task)

        # Start mark price watching for each symbol
        if symbols_for_mark_price:
            for symbol in symbols_for_mark_price:
                if symbol in self.mark_price_callbacks:
                    task = asyncio.create_task(self.watch_mark_price(symbol))
                    self.watch_tasks.append(task)

        logger.info(f"✅ Started {len(self.watch_tasks)} WebSocket watchers")

    async def get_initial_positions(self) -> List[dict]:
        """
        Fetch initial positions via REST API (for bootstrapping).

        Returns:
            List of position dicts
        """
        try:
            positions = await self.exchange.fetch_positions()
            # Filter non-zero positions
            open_positions = [p for p in positions if float(p.get("contracts", 0)) != 0]
            logger.info(f"Fetched initial positions: {len(open_positions)} open")
            return open_positions
        except Exception as e:
            logger.error(f"Failed to fetch initial positions: {e}")
            return []

    async def get_initial_balance(self) -> dict:
        """
        Fetch initial balance via REST API (for bootstrapping).

        Returns:
            Balance dict
        """
        try:
            balance = await self.exchange.fetch_balance()
            logger.info(f"Fetched initial balance: {balance.get('USDT', {}).get('total', 0)} USDT")
            return balance
        except Exception as e:
            logger.error(f"Failed to fetch initial balance: {e}")
            return {}

    async def get_initial_orders(self, symbol: Optional[str] = None) -> List[dict]:
        """
        Fetch initial open orders via REST API (for bootstrapping).

        Args:
            symbol: Optional symbol to filter (None = all symbols)

        Returns:
            List of order dicts
        """
        try:
            orders = await self.exchange.fetch_open_orders(symbol)
            logger.info(f"Fetched initial orders: {len(orders)} open")
            return orders
        except Exception as e:
            logger.error(f"Failed to fetch initial orders: {e}")
            return []

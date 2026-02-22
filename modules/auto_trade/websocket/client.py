"""
Binance Futures WebSocket Client – Direct Implementation
=========================================================

Connects directly to the Binance USDT-M Futures User Data Stream without
going through ccxt.pro.  ccxt internals call /sapi/v1/capital/config/getall
(Spot Wallet endpoint) on every authenticated REST request, which is
restricted on Futures-only accounts and causes every connection attempt to
fail.

This implementation follows the official Binance Futures API spec:
  POST   https://fapi.binance.com/fapi/v1/listenKey
         → requires only X-MBX-APIKEY header (NO signature!)
  WSS    wss://fstream.binance.com/ws/<listenKey>
  PUT    https://fapi.binance.com/fapi/v1/listenKey   (keepalive every 30 min)
  DELETE https://fapi.binance.com/fapi/v1/listenKey   (close on exit)

User Data Stream event types used:
  ACCOUNT_UPDATE     → balance + position changes
  ORDER_TRADE_UPDATE → order fills, TP/SL triggers
"""

import asyncio
import json
import time
from typing import Any, Callable, Dict, List, Optional

import aiohttp
import websockets
from websockets.exceptions import ConnectionClosed

from modules.common.ui.logging import log_debug, log_error, log_info, log_warn

# ── Binance Futures REST / WebSocket endpoints ────────────────────────────────
_FAPI_REST = "https://fapi.binance.com"
_FAPI_TESTNET_REST = "https://testnet.binancefuture.com"
_WSS_BASE = "wss://fstream.binance.com/ws"
_WSS_TESTNET_BASE = "wss://fstream.binancefuture.com/ws"

# Listen key must be renewed every 60 min; we renew every 30 min to be safe
_LISTEN_KEY_RENEW_INTERVAL = 30 * 60  # seconds


class BinanceWebSocketClient:
    """
    Real-time Binance Futures User Data Stream client.

    Provides callbacks for position, balance, and order updates using the
    official Binance Futures WebSocket API — no ccxt dependency.

    Example::

        client = BinanceWebSocketClient(api_key, api_secret, testnet=False)
        await client.connect()
        client.position_callbacks.append(my_pos_callback)
        await client.start_watching_all()
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
        options: Optional[Dict[str, Any]] = None,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet

        self._rest_base = _FAPI_TESTNET_REST if testnet else _FAPI_REST
        self._wss_base = _WSS_TESTNET_BASE if testnet else _WSS_BASE

        self._listen_key: Optional[str] = None
        self._ws: Optional[Any] = None  # websockets connection
        self._session: Optional[aiohttp.ClientSession] = None

        # Callbacks registered by monitors
        self.position_callbacks: List[Callable] = []
        self.balance_callbacks: List[Callable] = []
        self.order_callbacks: List[Callable] = []
        self.mark_price_callbacks: Dict[str, List[Callable]] = {}

        # Task management
        self.watch_tasks: List[asyncio.Task] = []
        self.running = False
        self._last_msg_time = time.time()
        self.staleness_timeout = 300  # 5 minutes

        log_info(f"BinanceWebSocketClient initialized (testnet={testnet})")

    # ── REST helpers ──────────────────────────────────────────────────────────

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session.

        Uses ThreadedResolver to bypass aiodns/c-ares which can fail to contact
        DNS servers on some Windows environments despite system DNS working fine.
        """
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(resolver=aiohttp.ThreadedResolver())
            self._session = aiohttp.ClientSession(
                headers={"X-MBX-APIKEY": self.api_key},
                connector=connector,
            )
        return self._session

    async def _create_listen_key(self) -> str:
        """
        Create a Binance Futures listen key.

        POST /fapi/v1/listenKey
        Requires only X-MBX-APIKEY header — no HMAC signature needed.
        """
        session = await self._get_session()
        url = f"{self._rest_base}/fapi/v1/listenKey"
        async with session.post(url) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Failed to create listen key: HTTP {resp.status} – {text}")
            data = await resp.json()
            key = data.get("listenKey")
            if not key:
                raise RuntimeError(f"No listenKey in response: {data}")
            log_info(f"Listen key created: {key[:16]}...")
            return key

    async def _renew_listen_key(self) -> None:
        """Renew (keepalive) the listen key. PUT /fapi/v1/listenKey."""
        if not self._listen_key:
            return
        try:
            session = await self._get_session()
            url = f"{self._rest_base}/fapi/v1/listenKey"
            async with session.put(url, params={"listenKey": self._listen_key}) as resp:
                if resp.status == 200:
                    log_debug("Listen key renewed")
                else:
                    text = await resp.text()
                    log_warn(f"Listen key renewal failed: HTTP {resp.status} – {text}")
        except Exception as e:
            log_warn(f"Listen key renewal error: {e}")

    async def _delete_listen_key(self) -> None:
        """Close the listen key on exit. DELETE /fapi/v1/listenKey."""
        if not self._listen_key:
            return
        try:
            session = await self._get_session()
            url = f"{self._rest_base}/fapi/v1/listenKey"
            async with session.delete(url, params={"listenKey": self._listen_key}) as resp:
                if resp.status == 200:
                    log_info("Listen key deleted")
                else:
                    log_warn(f"Listen key delete: HTTP {resp.status}")
        except Exception as e:
            log_warn(f"Listen key delete error: {e}")

    # ── Connect / Close ───────────────────────────────────────────────────────

    async def connect(self) -> None:
        """
        Mark client as ready.

        The listen key is created lazily when start_watching_all() is called,
        so connect() itself makes no network calls.
        """
        self.running = True
        log_info("✅ WebSocket client ready (direct Binance Futures API)")

    async def close(self) -> None:
        """Close WebSocket connection and cleanup."""
        self.running = False

        # Cancel watch tasks
        for task in self.watch_tasks:
            task.cancel()
        if self.watch_tasks:
            await asyncio.gather(*self.watch_tasks, return_exceptions=True)
        self.watch_tasks.clear()

        # Close WebSocket
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        # Delete listen key
        await self._delete_listen_key()
        self._listen_key = None

        # Close HTTP session
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

        log_info("WebSocket client closed")

    # ── Event-loop entry point ────────────────────────────────────────────────

    async def start_watching_all(self, symbols_for_mark_price: Optional[List[str]] = None) -> None:
        """
        Start the User Data Stream WebSocket and the listen-key keepalive.

        All position / balance / order events delivered by Binance via the
        stream are dispatched to the registered callbacks.
        """
        log_info("Starting all WebSocket watchers...")

        # Create listen key (single REST call to /fapi/v1/listenKey)
        self._listen_key = await self._create_listen_key()

        # Main stream reader
        stream_task = asyncio.create_task(self._stream_loop())
        self.watch_tasks.append(stream_task)

        # Keepalive task
        keepalive_task = asyncio.create_task(self._keepalive_loop())
        self.watch_tasks.append(keepalive_task)

        # Staleness monitor
        staleness_task = asyncio.create_task(self._monitor_staleness())
        self.watch_tasks.append(staleness_task)

        log_info(f"✅ Started {len(self.watch_tasks)} WebSocket watchers (including keepalive + staleness monitor)")

    # ── WebSocket stream loop ─────────────────────────────────────────────────

    async def _stream_loop(self) -> None:
        """Connect to User Data Stream and dispatch incoming events."""
        url = f"{self._wss_base}/{self._listen_key}"
        _RECONNECT_DELAY = 5  # seconds between reconnect attempts

        while self.running:
            try:
                log_info(f"Connecting to User Data Stream: {url[:60]}...")
                async with websockets.connect(url, ping_interval=180, ping_timeout=600) as ws:
                    self._ws = ws
                    log_info("✅ User Data Stream connected")

                    async for raw_msg in ws:
                        if not self.running:
                            break
                        try:
                            self._last_msg_time = time.time()
                            msg = json.loads(raw_msg)
                            await self._dispatch_event(msg)
                        except Exception as e:
                            log_error(f"Error processing WS message: {e}")

            except ConnectionClosed as e:
                if not self.running:
                    break
                log_warn(f"User Data Stream closed ({e}), reconnecting in {_RECONNECT_DELAY}s...")
                await asyncio.sleep(_RECONNECT_DELAY)

            except asyncio.CancelledError:
                break

            except Exception as e:
                if not self.running:
                    break
                log_error(f"Stream error: {e}, reconnecting in {_RECONNECT_DELAY}s...")
                await asyncio.sleep(_RECONNECT_DELAY)

        self._ws = None
        log_info("Stream loop exited")

    async def _dispatch_event(self, msg: Dict[str, Any]) -> None:
        """
        Dispatch a raw Binance User Data Stream event to registered callbacks.

        Binance Futures event types:
          ACCOUNT_UPDATE     – balance + unrealized PnL + position changes
          ORDER_TRADE_UPDATE – order lifecycle (new / fill / cancel / reject)
        """
        event_type = msg.get("e", "")

        if event_type == "ACCOUNT_UPDATE":
            await self._handle_account_update(msg)

        elif event_type == "ORDER_TRADE_UPDATE":
            await self._handle_order_update(msg)

        elif event_type == "listenKeyExpired":
            log_warn("Listen key expired — will reconnect with new key")
            if self._listen_key:
                self._listen_key = await self._create_listen_key()

        else:
            log_debug(f"Unhandled WS event: {event_type}")

    # ── Account update (balance + positions) ──────────────────────────────────

    async def _handle_account_update(self, msg: Dict[str, Any]) -> None:
        """
        Parse ACCOUNT_UPDATE and dispatch to balance + position callbacks.

        Payload structure::

            {
              "e": "ACCOUNT_UPDATE",
              "E": 1564745798939,
              "T": 1564745798938,
              "a": {
                "m": "ORDER",                    // update reason
                "B": [{"a": "USDT", "wb": "122624.12345678", "cw": "100.12345678", "bc": "50.12345678"}],
                "P": [
                  {
                    "s": "BTCUSDT", "pa": "0", "ep": "0.00000",
                    "cr": "200", "up": "0", "mt": "isolated",
                    "iw": "0.00000000", "ps": "BOTH",
                    "ma": "USDT"
                  }
                ]
              }
            }
        """
        account = msg.get("a", {})

        # ── Balances ─────────────────────────────────────────────────────────
        if self.balance_callbacks:
            balance_data = {}
            for b in account.get("B", []):
                asset = b.get("a", "")
                balance_data[asset] = {
                    "free": float(b.get("cw", 0)),  # cross wallet balance
                    "total": float(b.get("wb", 0)),  # wallet balance
                    "used": float(b.get("wb", 0)) - float(b.get("cw", 0)),
                }
            if balance_data:
                for cb in self.balance_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(cb):
                            await cb(balance_data)
                        else:
                            cb(balance_data)
                    except Exception as e:
                        log_error(f"Error in balance callback: {e}")

        # ── Positions ─────────────────────────────────────────────────────────
        if self.position_callbacks:
            positions = []
            for p in account.get("P", []):
                pos_amt = float(p.get("pa", 0))
                if pos_amt == 0:
                    continue
                symbol_raw = p.get("s", "")
                side = "long" if pos_amt > 0 else "short"
                positions.append(
                    {
                        "symbol": symbol_raw,
                        "contracts": abs(pos_amt),
                        "side": side,
                        "entryPrice": float(p.get("ep", 0)),
                        "unrealizedPnl": float(p.get("up", 0)),
                        "marginType": p.get("mt", "cross").lower(),
                        "notional": abs(pos_amt) * float(p.get("ep", 0) or 0),
                        "markPrice": 0.0,  # not in ACCOUNT_UPDATE; updated via REST if needed
                        "leverage": 1,
                        "liquidationPrice": None,
                        "collateral": float(p.get("iw", 0)),
                        "info": p,
                    }
                )
            if positions:
                for cb in self.position_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(cb):
                            await cb(positions)
                        else:
                            cb(positions)
                    except Exception as e:
                        log_error(f"Error in position callback: {e}")

    # ── Order update ──────────────────────────────────────────────────────────

    async def _handle_order_update(self, msg: Dict[str, Any]) -> None:
        """
        Parse ORDER_TRADE_UPDATE and dispatch to order callbacks.

        Payload structure::

            {
              "e": "ORDER_TRADE_UPDATE",
              "E": 1568879465651,
              "T": 1568879465650,
              "o": {
                "s": "BTCUSDT", "c": "abc123",
                "S": "SELL", "o": "TAKE_PROFIT_MARKET",
                "X": "FILLED",
                "i": 8886774,
                "q": "0.001", "p": "0", "ap": "9640.20",
                "rp": "0.00",   // realized profit
                ...
              }
            }
        """
        if not self.order_callbacks:
            return

        o = msg.get("o", {})
        raw_status = o.get("X", "").upper()
        status_map = {
            "NEW": "open",
            "PARTIALLY_FILLED": "open",
            "FILLED": "closed",
            "CANCELED": "canceled",
            "REJECTED": "rejected",
            "EXPIRED": "canceled",
        }
        order = {
            "id": str(o.get("i", "")),
            "clientOrderId": o.get("c", ""),
            "symbol": o.get("s", ""),
            "side": o.get("S", "").lower(),
            "type": o.get("o", "").lower(),
            "status": status_map.get(raw_status, raw_status.lower()),
            "price": float(o.get("ap") or o.get("p") or 0),
            "amount": float(o.get("q", 0)),
            "filled": float(o.get("z", 0)),
            "remaining": float(o.get("q", 0)) - float(o.get("z", 0)),
            "timestamp": msg.get("T"),
            "lastUpdateTimestamp": msg.get("E"),
            "info": o,
        }

        for cb in self.order_callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb([order])
                else:
                    cb([order])
            except Exception as e:
                log_error(f"Error in order callback: {e}")

    # ── Keepalive loop ────────────────────────────────────────────────────────

    async def _keepalive_loop(self) -> None:
        """Renew listen key every 30 minutes to prevent expiry."""
        try:
            while self.running:
                await asyncio.sleep(_LISTEN_KEY_RENEW_INTERVAL)
                if not self.running:
                    break
                await self._renew_listen_key()
        except asyncio.CancelledError:
            pass

    # ── Staleness monitor ─────────────────────────────────────────────────────

    async def _monitor_staleness(self) -> None:
        """Warn if no messages received for staleness_timeout seconds."""
        try:
            while self.running:
                await asyncio.sleep(60)
                elapsed = time.time() - self._last_msg_time
                if elapsed > self.staleness_timeout:
                    log_warn(f"WebSocket stream may be stale — no messages for {elapsed:.0f}s")
                    self._last_msg_time = time.time()  # reset to avoid spam
        except asyncio.CancelledError:
            pass

    # ── Initial snapshot via Futures REST (no Spot!) ─────────────────────────

    async def get_initial_positions(self) -> List[Dict[str, Any]]:
        """
        Fetch current open positions via GET /fapi/v2/positionRisk.

        Uses only Futures REST endpoint — no Spot wallet call.
        """
        try:
            import hashlib
            import hmac

            session = await self._get_session()
            timestamp = int(time.time() * 1000)
            params = f"timestamp={timestamp}&recvWindow=5000"
            signature = hmac.new(self.api_secret.encode(), params.encode(), hashlib.sha256).hexdigest()
            url = f"{self._rest_base}/fapi/v2/positionRisk?{params}&signature={signature}"
            async with session.get(url) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    log_warn(f"Failed to fetch positions: HTTP {resp.status} – {text}")
                    return []
                data = await resp.json()
                open_positions = [
                    {
                        "symbol": p["symbol"],
                        "contracts": abs(float(p.get("positionAmt", 0))),
                        "side": "long" if float(p.get("positionAmt", 0)) > 0 else "short",
                        "entryPrice": float(p.get("entryPrice", 0)),
                        "markPrice": float(p.get("markPrice", 0)),
                        "unrealizedPnl": float(p.get("unRealizedProfit", 0)),
                        "marginType": p.get("marginType", "cross").lower(),
                        "leverage": int(p.get("leverage", 1)),
                        "liquidationPrice": float(p.get("liquidationPrice", 0)) or None,
                        "notional": abs(float(p.get("notional", 0))),
                        "collateral": float(p.get("isolatedMargin", 0)),
                        "info": p,
                    }
                    for p in data
                    if float(p.get("positionAmt", 0)) != 0
                ]
                log_info(f"Fetched initial positions: {len(open_positions)} open")
                return open_positions
        except Exception as e:
            log_error(f"Failed to fetch initial positions: {e}")
            return []

    async def get_initial_balance(self) -> Dict[str, Any]:
        """
        Fetch account balance via GET /fapi/v2/balance.

        Uses only Futures REST endpoint — no Spot wallet call.
        """
        try:
            import hashlib
            import hmac

            session = await self._get_session()
            timestamp = int(time.time() * 1000)
            params = f"timestamp={timestamp}&recvWindow=5000"
            signature = hmac.new(self.api_secret.encode(), params.encode(), hashlib.sha256).hexdigest()
            url = f"{self._rest_base}/fapi/v2/balance?{params}&signature={signature}"
            async with session.get(url) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    log_warn(f"Failed to fetch balance: HTTP {resp.status} – {text}")
                    return {}
                data = await resp.json()
                # Normalise into ccxt-style dict for compatibility with BalanceMonitor
                balance: Dict[str, Any] = {}
                for entry in data:
                    asset = entry.get("asset", "")
                    total = float(entry.get("balance", 0))
                    free = float(entry.get("availableBalance", 0))
                    balance[asset] = {
                        "total": total,
                        "free": free,
                        "used": total - free,
                    }
                log_info(f"Fetched initial balance: {balance.get('USDT', {}).get('total', 0):.2f} USDT")
                return balance
        except Exception as e:
            log_error(f"Failed to fetch initial balance: {e}")
            return {}

    async def get_initial_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch open orders via GET /fapi/v1/openOrders.

        Uses only Futures REST endpoint — no Spot wallet call.
        """
        try:
            import hashlib
            import hmac

            session = await self._get_session()
            timestamp = int(time.time() * 1000)
            qs = f"timestamp={timestamp}&recvWindow=5000"
            if symbol:
                qs += f"&symbol={symbol}"
            signature = hmac.new(self.api_secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
            url = f"{self._rest_base}/fapi/v1/openOrders?{qs}&signature={signature}"
            async with session.get(url) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    log_warn(f"Failed to fetch orders: HTTP {resp.status} – {text}")
                    return []
                data = await resp.json()
                orders = [
                    {
                        "id": str(o["orderId"]),
                        "clientOrderId": o.get("clientOrderId", ""),
                        "symbol": o.get("symbol", ""),
                        "side": o.get("side", "").lower(),
                        "type": o.get("type", "").lower(),
                        "status": "open",
                        "price": float(o.get("price", 0)),
                        "amount": float(o.get("origQty", 0)),
                        "filled": float(o.get("executedQty", 0)),
                        "remaining": float(o.get("origQty", 0)) - float(o.get("executedQty", 0)),
                        "timestamp": o.get("time"),
                        "lastUpdateTimestamp": o.get("updateTime"),
                        "info": o,
                    }
                    for o in data
                ]
                log_info(f"Fetched initial orders: {len(orders)} open")
                return orders
        except Exception as e:
            log_error(f"Failed to fetch initial orders: {e}")
            return []

    # ── Legacy callback-registration shims ───────────────────────────────────

    def on_position_update(self, callback: Callable) -> None:
        """Register callback for position updates."""
        self.position_callbacks.append(callback)
        log_debug(f"Registered position callback: {callback.__name__}")

    def on_balance_update(self, callback: Callable) -> None:
        """Register callback for balance updates."""
        self.balance_callbacks.append(callback)
        log_debug(f"Registered balance callback: {callback.__name__}")

    def on_order_update(self, callback: Callable) -> None:
        """Register callback for order updates."""
        self.order_callbacks.append(callback)
        log_debug(f"Registered order callback: {callback.__name__}")

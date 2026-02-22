"""
Reconcile Binance orders with local DB.

Ensures every AT_* order that exists on Binance (open or recently closed)
has a corresponding row in the orders table.
"""

from modules.common.ui.logging import log_info, log_error, log_warn, log_debug, log_success, log_system
import threading
import time
from typing import Any, Dict, List, Optional, cast

import ccxt

from modules.auto_trade.execution.order_tagging import OrderTagger


# Lock for preventing concurrent DB writes during reconcile
_reconcile_lock = threading.Lock()

# Default symbols to reconcile if not provided
DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]


def _normalize_symbol(s: Optional[str]) -> str:
    """Ensure symbol is in CCXT form (e.g. BTC/USDT)."""
    s = (s or "").strip()
    if not s:
        return ""

    # Already in CCXT format
    if "/" in s:
        return s

    # Remove futures suffix if present
    s = s.replace("_PERP", "").replace("-PERP", "")

    # Convert from Binance format (BTCUSDT) to CCXT format (BTC/USDT)
    if s.endswith("USDT"):
        return s[:-4] + "/USDT"

    # Default: assume base currency, append /USDT
    return s + "/USDT" if s else ""


def reconcile_orders_with_binance(
    api_key: str,
    api_secret: str,
    testnet: bool = False,
    symbols: Optional[List[str]] = None,
    since_hours: int = 24,
    enable_profiling: bool = False,
) -> Dict[str, Any]:
    """
    Fetch recent closed/filled orders from Binance; for each AT_* order
    not present in DB, insert a row so DB stays in sync with Binance.

    Args:
        api_key: Binance API key
        api_secret: Binance API secret
        testnet: Use demo Binance if True
        symbols: List of symbols (e.g. ["BTC/USDT"]). If None, uses DEFAULT_SYMBOLS.
        since_hours: Look back period for closed orders (default 24).
        enable_profiling: If True, captures and returns timing information.

    Returns:
        Dict with keys: inserted (int), skipped (int), errors (list of str), timing (dict, optional)
    """
    from datetime import datetime

    from sqlalchemy import inspect

    from modules.auto_trade.database import get_order_by_client_id, session_scope
    from modules.auto_trade.database.models import Order

    # Timing tracking
    timing: Dict[str, Any] = {}
    total_start: float = time.perf_counter() if enable_profiling else 0.0

    result: Dict[str, Any] = {"inserted": 0, "skipped": 0, "errors": [], "closed_stale": 0}
    if enable_profiling:
        result["timing"] = timing

    # Batch collection for bulk insert
    orders_to_insert: List[Dict[str, Any]] = []
    failed_order_ids: List[str] = []
    if symbols is not None:
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.replace("\n", ",").split(",") if s.strip()]
        symbols = [x for x in (_normalize_symbol(s) for s in (symbols or [])) if x]
    symbols = symbols or DEFAULT_SYMBOLS
    since_ts = int((time.time() - since_hours * 3600) * 1000)

    # Status mapping from Binance to DB
    STATUS_MAP = {
        "FILLED": "CLOSED",
        "CANCELED": "CANCELLED",
        "CANCELLED": "CANCELLED",
        "EXPIRED": "CANCELLED",
        "REJECTED": "FAILED",
        "CLOSED": "CLOSED",
    }

    # Supported order types for futures
    SUPPORTED_ORDER_TYPES = {
        "MARKET",
        "LIMIT",
        "STOP_MARKET",
        "STOP_LIMIT",
        "TAKE_PROFIT_MARKET",
        "TAKE_PROFIT_LIMIT",
        "STOP",
    }

    exchange = None
    exchange_creation_error = None
    try:
        config = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "future", "adjustForTimeDifference": True},
        }
        if testnet:
            config["urls"] = {
                "api": {
                    "fapiPublic": "https://demo-fapi.binance.com/fapi/v1",
                    "fapiPublicV2": "https://demo-fapi.binance.com/fapi/v2",
                    "fapiPrivate": "https://demo-fapi.binance.com/fapi/v1",
                    "fapiPrivateV2": "https://demo-fapi.binance.com/fapi/v2",
                }
            }
        exchange = ccxt.binance(cast(Any, config))
    except ccxt.AuthenticationError as e:
        exchange_creation_error = f"Authentication failed: {e}"
        log_warn("Reconcile: authentication failed: %s", e)
    except ccxt.NetworkError as e:
        exchange_creation_error = f"Network error creating exchange: {e}"
        log_warn("Reconcile: network error: %s", e)
    except ccxt.ExchangeError as e:
        exchange_creation_error = f"Exchange error: {e}"
        log_warn("Reconcile: exchange error: %s", e)
    except Exception as e:
        exchange_creation_error = f"Failed to create exchange: {e}"
        log_warn("Reconcile: failed to create exchange: %s", e)

    if exchange_creation_error:
        result["errors"].append(exchange_creation_error)
    elif exchange is None:
        result["errors"].append("Failed to create exchange (no connection)")
    else:
        assert exchange is not None  # for type checker
        if enable_profiling:
            exchange_init_end = time.perf_counter()
            timing["exchange_init_seconds"] = round(exchange_init_end - total_start, 4)
            _fetch_time_total = 0.0
            _insert_time_total = 0.0
            _stale_time_total = 0.0

        try:
            if enable_profiling:
                _symbol_start_time = time.perf_counter()
            for symbol in symbols:
                try:
                    # Fetch closed orders with pagination (per-symbol cursor so since_ts is not shared)
                    all_orders = []
                    limit = 100
                    max_iterations = 10  # Safety limit to prevent infinite loops
                    symbol_since = since_ts

                    for iteration in range(max_iterations):
                        try:
                            batch = exchange.fetch_closed_orders(symbol, since=symbol_since, limit=limit)
                            if not batch:
                                break
                            all_orders.extend(batch)
                            if len(batch) < limit:
                                break
                            # Next page: move cursor past last order in batch (exchange-dependent order)
                            last_ts = batch[-1].get("timestamp")
                            if not isinstance(last_ts, (int, float)) or last_ts <= 0:
                                # Can't advance cursor safely; stop paginating for this symbol.
                                break
                            symbol_since = int(last_ts) + 1
                        except ccxt.NetworkError as e:
                            result["errors"].append(f"{symbol}: network error fetching orders: {e}")
                            log_warn("Reconcile: network error for %s: %s", symbol, e)
                            break
                        except ccxt.ExchangeError as e:
                            result["errors"].append(f"{symbol}: exchange error fetching orders: {e}")
                            log_warn("Reconcile: exchange error for %s: %s", symbol, e)
                            break
                        except Exception as e:
                            result["errors"].append(f"{symbol}: error fetching orders: {e}")
                            log_warn("Reconcile: fetch error for %s: %s", symbol, e)
                            break

                except Exception as e:
                    result["errors"].append(f"{symbol}: fetch_closed_orders failed: {e}")
                    log_warn("Reconcile: fetch_closed_orders %s failed: %s", symbol, e)
                    continue

                for o in all_orders:
                    client_order_id = (o.get("clientOrderId") or o.get("client_order_id") or "").strip()
                    if not OrderTagger.is_programmatic_order_id(client_order_id):
                        continue

                    with session_scope() as session:
                        existing = get_order_by_client_id(session, client_order_id)
                        if existing:
                            result["skipped"] += 1
                            continue

                        # Validate order type
                        order_type = (o.get("type") or "MARKET").upper()
                        if order_type not in SUPPORTED_ORDER_TYPES:
                            result["errors"].append(
                                f"{client_order_id}: unsupported order type '{order_type}' (skipped)"
                            )
                            log_warn("Reconcile: unsupported order type %s for %s", order_type, client_order_id)
                            continue

                        # Build order_data from Binance order
                        order_id_binance = str(o.get("id", ""))
                        if not order_id_binance:
                            result["errors"].append(f"{client_order_id}: missing order id")
                            continue

                        # Map Binance status to DB status
                        binance_status = (o.get("status") or "").upper()
                        order_status = STATUS_MAP.get(binance_status, "OPEN")

                        side_ccxt = (o.get("side") or "buy").upper()
                        side_db = "LONG" if side_ccxt == "BUY" else "SHORT"
                        symbol_db = (o.get("symbol") or symbol).replace("/", "")
                        entry_price = float(o.get("average") or o.get("price") or 0)
                        amount = float(o.get("filled") or o.get("amount") or 0)

                        if amount <= 0 or entry_price <= 0:
                            result["errors"].append(f"{client_order_id}: invalid amount/price")
                            continue

                        # Extract timestamps
                        created_timestamp = o.get("timestamp")
                        closed_timestamp = o.get("lastTradeTimestamp")
                        created_at = (
                            datetime.fromtimestamp(created_timestamp / 1000) if created_timestamp else datetime.now()
                        )
                        closed_at = (
                            datetime.fromtimestamp(closed_timestamp / 1000)
                            if closed_timestamp and order_status == "CLOSED"
                            else None
                        )

                        # Extract stop loss and take profit (may not be present in all responses)
                        stop_loss = None
                        take_profit = None
                        if "stopPrice" in o and o["stopPrice"]:
                            stop_loss = float(o["stopPrice"])
                        # Note: takeProfit may be in 'info' field depending on Binance response structure
                        if "info" in o and isinstance(o["info"], dict):
                            if "stopPrice" in o["info"] and o["info"]["stopPrice"]:
                                stop_loss = float(o["info"]["stopPrice"])
                            if "takeProfit" in o["info"] and o["info"]["takeProfit"]:
                                take_profit = float(o["info"]["takeProfit"])

                        # Extract PnL and fees for closed orders
                        pnl = None
                        if order_status == "CLOSED":
                            # Try to get realized PnL
                            pnl = o.get("info", {}).get("realizedPnl") if isinstance(o.get("info"), dict) else None
                            if pnl is not None:
                                pnl = float(pnl)

                        order_data = {
                            "order_id": order_id_binance,
                            "client_order_id": client_order_id,
                            "symbol": symbol_db,
                            "side": side_db,
                            "order_type": order_type,
                            "entry_price": entry_price,
                            "amount": amount,
                            "leverage": int(o.get("leverage") or 2),
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "status": order_status,
                            "order_source": "PROGRAMMATIC",
                            "execution_mode": "AUTO",
                            "pnl": pnl,
                            "created_at": created_at,
                            "closed_at": closed_at,
                        }

                        # Validate required fields
                        required_fields = ["order_id", "symbol", "side", "entry_price", "amount"]
                        missing_fields = [f for f in required_fields if not order_data.get(f)]
                        if missing_fields:
                            err_msg = f"{client_order_id}: missing required fields: {', '.join(missing_fields)}"
                            result["errors"].append(err_msg)
                            failed_order_ids.append(client_order_id)
                            log_warn("Reconcile: validation error for %s: %s", client_order_id, err_msg)
                            continue

                        # Validate side
                        if order_data.get("side") not in ("LONG", "SHORT"):
                            err_msg = f"{client_order_id}: invalid side '{order_data.get('side')}'"
                            result["errors"].append(err_msg)
                            failed_order_ids.append(client_order_id)
                            log_warn("Reconcile: validation error for %s: %s", client_order_id, err_msg)
                            continue

                        # Validate numeric fields
                        entry_price_raw = order_data.get("entry_price")
                        if not isinstance(entry_price_raw, (int, float)) or entry_price_raw <= 0:
                            err_msg = f"{client_order_id}: invalid entry_price {entry_price_raw}"
                            result["errors"].append(err_msg)
                            failed_order_ids.append(client_order_id)
                            log_warn("Reconcile: validation error for %s: %s", client_order_id, err_msg)
                            continue

                        amount_raw = order_data.get("amount")
                        if not isinstance(amount_raw, (int, float)) or amount_raw <= 0:
                            err_msg = f"{client_order_id}: invalid amount {amount_raw}"
                            result["errors"].append(err_msg)
                            failed_order_ids.append(client_order_id)
                            log_warn("Reconcile: validation error for %s: %s", client_order_id, err_msg)
                            continue

                        # Validate leverage
                        leverage = order_data.get("leverage", 2)
                        if not isinstance(leverage, int) or leverage < 1 or leverage > 125:
                            err_msg = f"{client_order_id}: invalid leverage {leverage}"
                            result["errors"].append(err_msg)
                            failed_order_ids.append(client_order_id)
                            log_warn("Reconcile: validation error for %s: %s", client_order_id, err_msg)
                            continue

                        orders_to_insert.append(order_data)

                # After processing all orders for this symbol, perform bulk insert
                if orders_to_insert:
                    # Acquire lock to prevent concurrent DB writes
                    lock_acquired = _reconcile_lock.acquire(timeout=30.0)
                    if not lock_acquired:
                        err_msg = f"{symbol}: could not acquire reconcile lock (timeout)"
                        result["errors"].append(err_msg)
                        log_error("Reconcile: %s", err_msg)
                        failed_order_ids.extend([o["client_order_id"] for o in orders_to_insert])
                        orders_to_insert = []
                    else:
                        try:
                            with session_scope() as session:
                                # Batch check which client_order_ids already exist
                                client_ids = [o["client_order_id"] for o in orders_to_insert]
                                existing_ids = set()
                                for cid in client_ids:
                                    if get_order_by_client_id(session, cid):
                                        existing_ids.add(cid)

                                # Filter out existing orders
                                new_orders = [o for o in orders_to_insert if o["client_order_id"] not in existing_ids]
                                result["skipped"] += len(existing_ids)

                                if new_orders:
                                    # Perform bulk insert using the mapper
                                    order_mapper = inspect(Order).mapper
                                    session.bulk_insert_mappings(order_mapper, new_orders)
                                    result["inserted"] += len(new_orders)
                                    log_info(
                                        "Reconcile: bulk inserted %d orders for symbol %s", len(new_orders), symbol
                                    )

                                # Clear the batch
                                orders_to_insert = []
                        except Exception as bulk_err:
                            err_msg = f"{symbol}: bulk insert failed: {bulk_err}"
                            result["errors"].append(err_msg)
                            log_error("Reconcile: %s", err_msg)
                            failed_order_ids.extend([o["client_order_id"] for o in orders_to_insert])
                            orders_to_insert = []
                        finally:
                            _reconcile_lock.release()

            # ============================================================================
            # Close stale OPEN orders not on Binance anymore
            # ============================================================================
            try:
                from modules.auto_trade.database import (
                    get_open_positions,
                    session_scope,
                    update_order_status_by_client_id,
                )

                # (1) Get OPEN programmatic orders (read-only, no lock needed)
                # Eagerly load all attributes we'll need to avoid detached instance errors
                with session_scope() as session:
                    open_orders_raw = get_open_positions(session)
                    # Convert to dict to avoid detached instance errors
                    open_orders = []
                    for order in open_orders_raw:
                        open_orders.append({
                            "client_order_id": order.client_order_id,
                            "order_id": order.order_id,
                            "symbol": order.symbol,
                            "created_at": order.created_at,
                        })

                if not open_orders:
                    log_info("Reconcile: no OPEN orders to check for staleness")
                else:
                    # Collect unique symbols and map DB symbol -> CCXT symbol
                    symbol_map: Dict[str, str] = {}  # DB symbol -> CCXT symbol
                    db_orders_by_symbol: Dict[str, List[Any]] = {}

                    for order_dict in open_orders:
                        db_symbol = str(order_dict.get("symbol", ""))  # e.g., BTCUSDT
                        ccxt_symbol = _normalize_symbol(db_symbol)  # e.g., BTC/USDT
                        symbol_map[db_symbol] = ccxt_symbol
                        if ccxt_symbol not in db_orders_by_symbol:
                            db_orders_by_symbol[ccxt_symbol] = []
                        db_orders_by_symbol[ccxt_symbol].append(order_dict)

                    # (3) For each symbol, fetch open orders from Binance
                    closed_stale_count = 0
                    stale_updates: List[Dict[str, Any]] = []  # Collect updates to batch process

                    for db_symbol, ccxt_symbol in symbol_map.items():
                        try:
                            # Build set of client_order_id from Binance open orders
                            binance_open_ids = set()
                            try:
                                open_orders_binance = exchange.fetch_open_orders(ccxt_symbol)
                                for o in open_orders_binance:
                                    cid = (o.get("clientOrderId") or o.get("client_order_id") or "").strip()
                                    if cid:
                                        binance_open_ids.add(cid)
                            except ccxt.NetworkError as e:
                                result["errors"].append(f"{ccxt_symbol}: network error fetching open orders: {e}")
                                log_warn(
                                    "Reconcile: network error fetching open orders for %s: %s", ccxt_symbol, e
                                )
                                continue
                            except ccxt.ExchangeError as e:
                                result["errors"].append(f"{ccxt_symbol}: exchange error fetching open orders: {e}")
                                log_warn(
                                    "Reconcile: exchange error fetching open orders for %s: %s", ccxt_symbol, e
                                )
                                continue

                            # (4) Find stale orders
                            stale_orders = []
                            for order in db_orders_by_symbol.get(ccxt_symbol, []):
                                if order["client_order_id"] not in binance_open_ids:
                                    stale_orders.append(order)

                            # (5) Batch fetch closed orders for this symbol to minimize API calls
                            closed_orders_map: Dict[str, Any] = {}
                            if stale_orders:
                                try:
                                    # Calculate since time (oldest stale order created_at - 1 hour buffer)
                                    oldest_created = min(
                                        (order["created_at"] for order in stale_orders if order.get("created_at")), default=None
                                    )
                                    stale_since_ts: Optional[int] = None
                                    if oldest_created:
                                        stale_since_ts = int(oldest_created.timestamp() * 1000) - 3600000  # -1 hour
                                    else:
                                        stale_since_ts = int((time.time() - since_hours * 3600) * 1000)

                                    # Fetch closed orders in batch
                                    closed_orders = exchange.fetch_closed_orders(
                                        ccxt_symbol, since=stale_since_ts, limit=1000
                                    )
                                    for o in closed_orders:
                                        cid = (o.get("clientOrderId") or o.get("client_order_id") or "").strip()
                                        if cid:
                                            closed_orders_map[cid] = o
                                    log_debug(
                                        "Reconcile: batch fetched %d closed orders for %s",
                                        len(closed_orders),
                                        ccxt_symbol,
                                    )
                                except Exception as batch_err:
                                    log_warn(
                                        "Reconcile: batch fetch failed for %s, falling back to per-order fetch: %s",
                                        ccxt_symbol,
                                        batch_err,
                                    )

                            # (6) Process each stale order
                            fetch_order_count = 0
                            for stale_order in stale_orders:
                                order_info = None

                                # Try to get from batch fetch first
                                if stale_order["client_order_id"] in closed_orders_map:
                                    order_info = closed_orders_map[stale_order["client_order_id"]]
                                    log_debug(
                                        "Reconcile: found stale order %s in batch fetch", stale_order["client_order_id"]
                                    )
                                else:
                                    # Fall back to individual fetch_order
                                    try:
                                        order_info = exchange.fetch_order(stale_order["order_id"], ccxt_symbol)
                                        fetch_order_count += 1
                                        log_debug(
                                            "Reconcile: fetched order %s individually (not in batch)",
                                            stale_order["client_order_id"],
                                        )
                                    except Exception as fetch_err:
                                        # If API fails, still mark as CLOSED with minimal info
                                        log_warn(
                                            "Reconcile: could not fetch order %s from Binance: %s",
                                            stale_order["client_order_id"],
                                            fetch_err,
                                        )

                                if order_info:
                                    # Map status
                                    binance_status = (order_info.get("status") or "").upper()
                                    status_map = {
                                        "FILLED": "CLOSED",
                                        "CANCELED": "CANCELLED",
                                        "CANCELLED": "CANCELLED",
                                        "EXPIRED": "CANCELLED",
                                        "REJECTED": "FAILED",
                                        "CLOSED": "CLOSED",
                                    }
                                    final_status = status_map.get(binance_status, "CLOSED")

                                    # Get closed_at and pnl if available
                                    closed_timestamp = order_info.get("lastTradeTimestamp")
                                    closed_at = (
                                        datetime.fromtimestamp(closed_timestamp / 1000) if closed_timestamp else None
                                    )
                                    pnl = None
                                    if final_status == "CLOSED" and isinstance(order_info.get("info"), dict):
                                        pnl_raw = order_info["info"].get("realizedPnl")
                                        if pnl_raw is not None:
                                            pnl = float(pnl_raw)
                                else:
                                    # API failed, mark as CLOSED with minimal info
                                    final_status = "CLOSED"
                                    closed_at = None
                                    pnl = None

                                # Collect update for batch processing
                                stale_updates.append(
                                    {
                                        "client_order_id": stale_order["client_order_id"],
                                        "status": final_status,
                                        "closed_at": closed_at,
                                        "pnl": pnl,
                                    }
                                )

                            if fetch_order_count > 0:
                                log_info(
                                    "Reconcile: %s required %d individual fetch_order calls (batch covered %d)",
                                    ccxt_symbol,
                                    fetch_order_count,
                                    len(stale_orders) - fetch_order_count,
                                )

                        except Exception as symbol_err:
                            result["errors"].append(f"{ccxt_symbol}: error processing stale orders: {symbol_err}")
                            log_warn(
                                "Reconcile: error processing stale orders for %s: %s", ccxt_symbol, symbol_err
                            )

                    # (6) Batch update stale orders with lock
                    if stale_updates:
                        lock_acquired = _reconcile_lock.acquire(timeout=30.0)
                        if not lock_acquired:
                            err_msg = "Could not acquire reconcile lock for stale order updates (timeout)"
                            result["errors"].append(err_msg)
                            log_error("Reconcile: %s", err_msg)
                        else:
                            try:
                                with session_scope() as session:
                                    for update_data in stale_updates:
                                        updated = update_order_status_by_client_id(
                                            session=session,
                                            client_order_id=update_data["client_order_id"],
                                            status=update_data["status"],
                                            closed_at=update_data["closed_at"],
                                            pnl=update_data["pnl"],
                                        )
                                        if updated:
                                            closed_stale_count += 1
                                            log_info(
                                                "Reconcile: closed stale order %s (status=%s)",
                                                update_data["client_order_id"],
                                                update_data["status"],
                                            )
                            finally:
                                _reconcile_lock.release()

                    result["closed_stale"] = closed_stale_count
                    log_info("Reconcile: closed %d stale orders", closed_stale_count)

            except Exception as stale_err:
                result["errors"].append(f"Failed to process stale orders: {stale_err}")
                log_error("Reconcile: failed to process stale orders: %s", stale_err)

        finally:
            # Clean up exchange connection
            if exchange:
                close_fn = getattr(exchange, "close", None)
                if callable(close_fn):
                    try:
                        close_fn()
                    except Exception as e:
                        log_warn("Reconcile: error closing exchange: %s", e)

            # Record timing if profiling enabled
            if enable_profiling:
                total_end = time.perf_counter()
                timing["total_seconds"] = round(total_end - total_start, 4)
                log_info("Reconcile timing: %s", timing)

    return result

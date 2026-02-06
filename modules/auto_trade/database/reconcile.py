"""
Reconcile Binance orders with local DB.

Ensures every AT_* order that exists on Binance (open or recently closed)
has a corresponding row in the orders table.
"""

import logging
import time
from typing import Any, Dict, List, Optional, cast

import ccxt

from modules.auto_trade.execution.order_tagging import OrderTagger

logger = logging.getLogger(__name__)

# Default symbols to reconcile if not provided
DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]


def _normalize_symbol(s: str) -> str:
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

    Returns:
        Dict with keys: inserted (int), skipped (int), errors (list of str)
    """
    from datetime import datetime

    from modules.auto_trade.database import create_order, get_order_by_client_id, session_scope

    result: Dict[str, Any] = {"inserted": 0, "skipped": 0, "errors": []}
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
        result["errors"].append(f"Authentication failed: {e}")
        logger.warning("Reconcile: authentication failed: %s", e)
        return result
    except ccxt.NetworkError as e:
        result["errors"].append(f"Network error creating exchange: {e}")
        logger.warning("Reconcile: network error: %s", e)
        return result
    except ccxt.ExchangeError as e:
        result["errors"].append(f"Exchange error: {e}")
        logger.warning("Reconcile: exchange error: %s", e)
        return result
    except Exception as e:
        result["errors"].append(f"Failed to create exchange: {e}")
        logger.warning("Reconcile: failed to create exchange: %s", e)
        return result

    try:
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
                        logger.warning("Reconcile: network error for %s: %s", symbol, e)
                        break
                    except ccxt.ExchangeError as e:
                        result["errors"].append(f"{symbol}: exchange error fetching orders: {e}")
                        logger.warning("Reconcile: exchange error for %s: %s", symbol, e)
                        break
                    except Exception as e:
                        result["errors"].append(f"{symbol}: error fetching orders: {e}")
                        logger.warning("Reconcile: fetch error for %s: %s", symbol, e)
                        break

            except Exception as e:
                result["errors"].append(f"{symbol}: fetch_closed_orders failed: {e}")
                logger.warning("Reconcile: fetch_closed_orders %s failed: %s", symbol, e)
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
                        logger.warning("Reconcile: unsupported order type %s for %s", order_type, client_order_id)
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

                    try:
                        create_order(session, order_data)
                        result["inserted"] += 1
                        logger.info(
                            "Reconcile: inserted order %s (client_order_id=%s, status=%s)",
                            order_id_binance,
                            client_order_id,
                            order_status,
                        )
                    except ValueError as ve:
                        result["errors"].append(f"{client_order_id}: {ve}")
                        logger.warning("Reconcile: validation error for %s: %s", client_order_id, ve)
                    except Exception as db_err:
                        result["errors"].append(f"{client_order_id}: {db_err}")
                        logger.warning("Reconcile: create_order failed for %s: %s", client_order_id, db_err)

    finally:
        # Clean up exchange connection
        if exchange:
            try:
                # Not all CCXT versions/types expose .close() in type stubs
                exchange.close()  # type: ignore[attr-defined]
            except Exception as e:
                logger.warning("Reconcile: error closing exchange: %s", e)

    return result

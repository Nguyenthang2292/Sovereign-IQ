import time
from typing import Any, Optional

import requests

from modules.common.domain.symbol_codec import SymbolCodec
from modules.common.ui.logging import log_warn

from .models import AggTrade, OrderBookSnapshot

_SYMBOL_CODEC = SymbolCodec()
_FAPI_REST = "https://fapi.binance.com"
_FAPI_TESTNET_REST = "https://testnet.binancefuture.com"
_DEFAULT_TIMEOUT_SECONDS = 10


def fetch_depth(
    symbol: str,
    limit: int = 100,
    testnet: bool = False,
) -> Optional[OrderBookSnapshot]:
    """
    Fetch Binance Futures depth snapshot.

    Fail-open policy:
    - Any exception/network/HTTP error returns None and logs warning.
    """
    rest_base = _FAPI_TESTNET_REST if testnet else _FAPI_REST
    endpoint = f"{rest_base}/fapi/v1/depth"
    symbol_key = _SYMBOL_CODEC.to_db(symbol)

    try:
        response = requests.get(
            endpoint,
            params={"symbol": symbol_key, "limit": int(limit)},
            timeout=_DEFAULT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()

        bids_raw = payload.get("bids")
        asks_raw = payload.get("asks")
        if not isinstance(bids_raw, list) or not isinstance(asks_raw, list):
            log_warn("[OrderBookFetcher] Invalid depth payload for %s", symbol_key)
            return None

        bids = _parse_price_qty_levels(bids_raw)
        asks = _parse_price_qty_levels(asks_raw)

        timestamp_ms = payload.get("E") or payload.get("T")
        timestamp = (
            float(timestamp_ms) / 1000.0
            if isinstance(timestamp_ms, (int, float, str)) and str(timestamp_ms).strip() != ""
            else time.time()
        )

        return OrderBookSnapshot(
            symbol=symbol_key,
            bids=bids,
            asks=asks,
            timestamp=timestamp,
        )
    except Exception as exc:
        log_warn("[OrderBookFetcher] fetch_depth failed for %s: %s", symbol_key, exc)
        return None


def fetch_agg_trades(
    symbol: str,
    window_minutes: int = 5,
    testnet: bool = False,
) -> Optional[list[AggTrade]]:
    """
    Fetch Binance Futures aggregate trades within a rolling time window.

    Fail-open policy:
    - Any exception/network/HTTP error returns None and logs warning.
    """
    rest_base = _FAPI_TESTNET_REST if testnet else _FAPI_REST
    endpoint = f"{rest_base}/fapi/v1/aggTrades"
    symbol_key = _SYMBOL_CODEC.to_db(symbol)

    end_time_ms = int(time.time() * 1000)
    start_time_ms = end_time_ms - int(max(window_minutes, 1) * 60 * 1000)

    try:
        response = requests.get(
            endpoint,
            params={
                "symbol": symbol_key,
                "startTime": start_time_ms,
                "endTime": end_time_ms,
            },
            timeout=_DEFAULT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload: Any = response.json()

        if not isinstance(payload, list):
            log_warn("[OrderBookFetcher] Invalid aggTrades payload for %s", symbol_key)
            return None

        trades: list[AggTrade] = []
        for item in payload:
            if not isinstance(item, dict):
                continue

            try:
                price = float(item["p"])
                quantity = float(item["q"])
                timestamp = float(item["T"]) / 1000.0
                is_buyer_maker = bool(item["m"])
            except (KeyError, TypeError, ValueError):
                continue

            trades.append(
                AggTrade(
                    price=price,
                    quantity=quantity,
                    timestamp=timestamp,
                    is_buyer_maker=is_buyer_maker,
                )
            )

        return trades
    except Exception as exc:
        log_warn("[OrderBookFetcher] fetch_agg_trades failed for %s: %s", symbol_key, exc)
        return None


def _parse_price_qty_levels(levels: list[Any]) -> list[tuple[float, float]]:
    parsed: list[tuple[float, float]] = []

    for level in levels:
        if not isinstance(level, (list, tuple)) or len(level) < 2:
            continue

        try:
            price = float(level[0])
            quantity = float(level[1])
        except (TypeError, ValueError):
            continue

        parsed.append((price, quantity))

    return parsed

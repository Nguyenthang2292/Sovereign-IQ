"""Utility functions for sampling strategies."""

from typing import Dict, List

from modules.common.ui.logging import log_error, log_success


def get_symbol_volumes(symbols: List[str], data_fetcher) -> Dict[str, float]:
    """
    Get volume data for symbols from exchange.

    Uses the same approach as list_binance_futures_symbols to extract volume.

    Args:
        symbols: List of symbols to get volumes for
        data_fetcher: DataFetcher instance

    Returns:
        Dictionary mapping symbol to volume (quote volume)
    """
    volumes = {}

    try:
        # Use public API - load_markets() doesn't require authentication
        exchange = data_fetcher.exchange_manager.public.connect_to_exchange_with_no_credentials("binance")
    except Exception as exc:
        log_error(f"Unable to connect to Binance for volume data: {exc}")
        return volumes

    try:
        # load_markets() is a public API call, no authentication needed
        markets = data_fetcher.exchange_manager.public.throttled_call(exchange.load_markets)
    except Exception as exc:
        log_error(f"Failed to load Binance markets for volume data: {exc}")
        return volumes

    # Build symbol set for faster lookup
    symbol_set = set(symbols)

    for market in markets.values():
        symbol = data_fetcher.exchange_manager.normalize_symbol(market.get("symbol", ""))

        if symbol not in symbol_set:
            continue

        info = market.get("info", {})
        volume_str = info.get("volume") or info.get("quoteVolume") or info.get("turnover")
        try:
            volume = float(volume_str)
        except (TypeError, ValueError):
            volume = 0.0

        volumes[symbol] = volume

    log_success(f"Retrieved volume data for {len(volumes)}/{len(symbols)} symbols")
    return volumes

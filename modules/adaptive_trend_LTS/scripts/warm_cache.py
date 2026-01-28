"""
Cache Warming Script for Adaptive Trend LTS

This script pre-calculates ATC signals for a set of symbols and configurations
 to populate the cache, improving performance for subsequent runs.

Usage:
    python -m modules.adaptive_trend_LTS.scripts.warm_cache --symbols BTC,ETH --bars 1000
"""

import argparse
import time

import numpy as np
import pandas as pd

from modules.adaptive_trend_LTS.utils.cache_manager import get_cache_manager
from modules.common.ui.logging import log_info, log_success


def generate_dummy_data(bars: int = 1000) -> pd.Series:
    """Generate dummy price data for warming."""
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(bars) * 0.5)
    return pd.Series(prices)


def main():
    parser = argparse.ArgumentParser(description="Warm ATC Cache")
    parser.add_argument("--symbols", type=str, default="BTC,ETH,SOL,BNB,XRP", help="Comma-separated symbols")
    parser.add_argument("--bars", type=int, default=1500, help="Number of bars per symbol")
    parser.add_argument("--configs", type=str, default="default", help="Presets for configurations (default, all)")

    args = parser.parse_args()
    symbol_list = args.symbols.split(",")

    log_info(f"Preparing to warm cache for {len(symbol_list)} symbols with {args.bars} bars each...")

    # Generate data for each symbol
    symbols_data = {symbol: generate_dummy_data(args.bars) for symbol in symbol_list}

    # Define configs to warm
    configs = []
    if args.configs == "default":
        configs = [
            {"ema_len": 28, "hull_len": 28, "wma_len": 28, "dema_len": 28, "lsma_len": 28, "kama_len": 28},
        ]
    elif args.configs == "all":
        configs = [
            {"ema_len": 14, "hull_len": 14, "wma_len": 14, "dema_len": 14, "lsma_len": 14, "kama_len": 14},
            {"ema_len": 28, "hull_len": 28, "wma_len": 28, "dema_len": 28, "lsma_len": 28, "kama_len": 28},
            {"ema_len": 50, "hull_len": 50, "wma_len": 50, "dema_len": 50, "lsma_len": 50, "kama_len": 50},
        ]

    cache_mgr = get_cache_manager()

    # Run warming
    start_time = time.time()
    cache_mgr.warm_cache(symbols_data, configs)
    duration = time.time() - start_time

    log_success(f"Successfully warmed cache in {duration:.2f} seconds.")
    cache_mgr.log_cache_effectiveness()


if __name__ == "__main__":
    main()

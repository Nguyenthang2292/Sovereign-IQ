#!/usr/bin/env python3
# ruff: noqa: I001
"""Example client for invoking ATC Serverless from Python."""

import copy
import json
import os
import sys
import time
from typing import Any

# Add project root to Python path.
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from modules.adaptive_trend_LTS_serverless.lambda_client import ATCLambdaClient, DEFAULT_ATC_CONFIG


def create_sample_symbols(num_symbols: int = 3) -> list[dict[str, Any]]:
    """Create sample OHLCV payloads that satisfy minimum data length checks."""
    symbols: list[dict[str, Any]] = []

    num_bars_1h = 50
    num_bars_4h = 50

    for i in range(num_symbols):
        base_price = 42000.0 + i * 100

        timestamps_1h = [1704067200 + j * 3600 for j in range(num_bars_1h)]
        closes_1h = [base_price + j * 0.5 + (j % 5) * 10 for j in range(num_bars_1h)]
        opens_1h = [closes_1h[j] - 50.0 for j in range(num_bars_1h)]
        highs_1h = [closes_1h[j] + 100.0 for j in range(num_bars_1h)]
        lows_1h = [closes_1h[j] - 100.0 for j in range(num_bars_1h)]
        volumes_1h = [100.0 + j * 10 for j in range(num_bars_1h)]

        timestamps_4h = [1704067200 + j * 14400 for j in range(num_bars_4h)]
        closes_4h = [base_price + j * 2.0 + (j % 3) * 20 for j in range(num_bars_4h)]
        opens_4h = [closes_4h[j] - 100.0 for j in range(num_bars_4h)]
        highs_4h = [closes_4h[j] + 200.0 for j in range(num_bars_4h)]
        lows_4h = [closes_4h[j] - 200.0 for j in range(num_bars_4h)]
        volumes_4h = [400.0 + j * 50 for j in range(num_bars_4h)]

        symbols.append(
            {
                "symbol": f"BTCUSDT_{i}",
                "timeframes": {
                    "1h": {
                        "timestamp": timestamps_1h,
                        "open": opens_1h,
                        "high": highs_1h,
                        "low": lows_1h,
                        "close": closes_1h,
                        "volume": volumes_1h,
                    },
                    "4h": {
                        "timestamp": timestamps_4h,
                        "open": opens_4h,
                        "high": highs_4h,
                        "low": lows_4h,
                        "close": closes_4h,
                        "volume": volumes_4h,
                    },
                },
            }
        )

    return symbols


def create_atc_config() -> dict[str, Any]:
    """Create a mutable config copy without altering defaults."""
    config = copy.deepcopy(DEFAULT_ATC_CONFIG)
    config["weights"] = {"1h": 0.6, "4h": 0.4}
    config["threshold"] = 0.3
    return config


def main() -> int:
    print("ATC Serverless Python Client Example")
    print("=" * 40)

    symbols = create_sample_symbols(num_symbols=3)
    config = create_atc_config()

    # Set mock_mode=False to run against AWS.
    client = ATCLambdaClient(mock_mode=True)

    start_time = time.time()
    result = client.invoke_batch(symbols, config)
    duration = time.time() - start_time

    print(f"Processed {len(symbols)} symbols in {duration:.2f}s")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

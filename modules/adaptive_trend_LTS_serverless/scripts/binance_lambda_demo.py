#!/usr/bin/env python3
# ruff: noqa: I001
"""
Binance Lambda Demo Script

Invokes the ATC Serverless Lambda with real Binance market data.

Architecture:
  this script --[boto3 Invoke RequestResponse]--> Lambda (atc-serverless) --> direct ScanResult JSON

This script:
  1. Fetches OHLCV data from Binance
  2. Invokes Lambda via boto3 (IAM-signed)
  3. Parses direct ScanResult response payload

Usage:
    python binance_lambda_demo.py --symbols 10
    python binance_lambda_demo.py --symbols 5 --timeframes 1h 4h --details
    python binance_lambda_demo.py --all-symbols
    python binance_lambda_demo.py --symbols 5 --mock
"""

import argparse
import copy
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# Add project root to path to import common modules
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.adaptive_trend_LTS_serverless.lambda_client import (
    ATCLambdaClient,
    DEFAULT_ATC_CONFIG,
    DEFAULT_FUNCTION_NAME,
    DEFAULT_REGION,
)
from modules.common.core.data_fetcher import DataFetcher, SymbolFetchError
from modules.common.core.exchange_manager import ExchangeManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

class BinanceDataLoader:
    """Loads Binance market data using the DataFetcher module."""

    def __init__(self):
        self.exchange_manager = ExchangeManager()
        self.data_fetcher = DataFetcher(self.exchange_manager)

    def get_usdt_symbols(self, limit: int | None = None) -> list[str]:
        """Get all USDT trading pairs from Binance."""
        try:
            logger.info("Fetching USDT symbols from Binance...")
            symbols = self.data_fetcher.get_spot_symbols(exchange_name="binance", quote_currency="USDT")
            logger.info(f"Found {len(symbols)} USDT trading pairs")
            if limit:
                symbols = symbols[:limit]
                logger.info(f"Limited to {len(symbols)} symbols")
            return symbols
        except SymbolFetchError as e:
            logger.error(f"Failed to fetch symbols: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching symbols: {e}")
            raise

    def get_ohlcv_data(
        self, symbols: list[str], timeframes: list[str], limit: int = 100
    ) -> dict[str, dict[str, dict[str, list[Any]]]]:
        """Fetch OHLCV data for symbols across multiple timeframes."""
        results = {}
        for symbol in symbols:
            logger.info(f"Fetching data for {symbol}...")
            symbol_data = {}
            for tf in timeframes:
                try:
                    df = self.data_fetcher.fetch_ohlcv(symbol=symbol, timeframe=tf, limit=limit, check_freshness=False)
                    if df is not None and not df.empty:
                        symbol_data[tf] = {
                            "timestamp": [int(ts.timestamp()) for ts in df.index],
                            "open": df["open"].tolist(),
                            "high": df["high"].tolist(),
                            "low": df["low"].tolist(),
                            "close": df["close"].tolist(),
                            "volume": df["volume"].tolist(),
                        }
                        logger.debug(f"Fetched {len(df)} candles for {symbol} {tf}")
                    else:
                        logger.warning(f"No data for {symbol} {tf}")
                except Exception as e:
                    logger.warning(f"Failed to fetch {symbol} {tf}: {e}")
            if symbol_data:
                results[symbol] = symbol_data
            time.sleep(0.05)
        return results


def mock_invoke(symbols_data: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Generate a mock result for testing without AWS."""
    import random

    batch_id = f"mock-{uuid.uuid4().hex[:8]}"
    results = []
    for sd in symbols_data:
        score = random.uniform(-1.0, 1.0)
        signal_type = "LONG" if score > 0.3 else ("SHORT" if score < -0.3 else "NEUTRAL")
        results.append(
            {
                "symbol": sd["symbol"],
                "score": score,
                "signal_type": signal_type,
                "details": {tf: "MOCK" for tf in sd["timeframes"].keys()},
                "strengths": {tf: abs(score) + random.uniform(-0.1, 0.1) for tf in sd["timeframes"].keys()},
            }
        )
    return {
        "batch_id": batch_id,
        "results": results,
        "errors": [],
        "success_count": len(results),
        "error_count": 0,
    }


def display_results(results: list[dict[str, Any]], show_details: bool = False):
    """Display signal results in a formatted table."""
    if not results:
        print("No results to display.")
        return

    print("\n" + "=" * 100)
    print("ATC SIGNAL RESULTS")
    print("=" * 100)

    sorted_results = sorted(results, key=lambda x: abs(x.get("score", 0)), reverse=True)

    print(f"{'#':<4} {'Symbol':<16} {'Signal':<10} {'Score':<10} {'Confidence':<12}")
    print("-" * 100)

    for idx, result in enumerate(sorted_results, 1):
        symbol = result.get("symbol", "?")
        signal = result.get("signal_type", "NEUTRAL")
        score = result.get("score", 0.0)
        confidence = abs(score)

        if signal == "LONG":
            signal_disp = f"\033[92m{signal}\033[0m"  # Green
        elif signal == "SHORT":
            signal_disp = f"\033[91m{signal}\033[0m"  # Red
        else:
            signal_disp = f"\033[93m{signal}\033[0m"  # Yellow

        print(f"{idx:<4} {symbol:<16} {signal_disp:<20} {score:<10.4f} {confidence:<12.2%}")

        if show_details and "strengths" in result:
            for tf, strength in result["strengths"].items():
                print(f"     └─ {tf}: {strength:.4f}")

    print("=" * 100)

    long_count = sum(1 for r in results if r.get("signal_type") == "LONG")
    short_count = sum(1 for r in results if r.get("signal_type") == "SHORT")
    neutral_count = sum(1 for r in results if r.get("signal_type") == "NEUTRAL")
    total = len(results)

    print("\nSummary:")
    print(f"  Total Signals: {total}")
    print(f"  LONG:    {long_count} ({long_count / total * 100:.1f}%)")
    print(f"  SHORT:   {short_count} ({short_count / total * 100:.1f}%)")
    print(f"  NEUTRAL: {neutral_count} ({neutral_count / total * 100:.1f}%)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Binance Lambda Demo - invokes ATC Serverless via boto3 RequestResponse."
    )
    parser.add_argument("--symbols", type=int, default=10, help="Number of symbols to process (default: 10)")
    parser.add_argument("--all-symbols", action="store_true", help="Process all available USDT symbols")
    parser.add_argument("--timeframes", nargs="+", default=["1h", "4h"], help="Timeframes to fetch (default: 1h 4h)")
    parser.add_argument("--limit", type=int, default=100, help="Candles per timeframe (default: 100)")
    parser.add_argument("--details", action="store_true", help="Show per-timeframe strength breakdown")
    parser.add_argument("--config", type=str, help="Path to custom ATC config JSON file")
    parser.add_argument("--mock", action="store_true", help="Use mock results (no AWS needed)")
    parser.add_argument(
        "--function-name",
        default=DEFAULT_FUNCTION_NAME,
        help=f"Lambda function name (default: {DEFAULT_FUNCTION_NAME})",
    )
    parser.add_argument("--region", default=DEFAULT_REGION, help=f"AWS region (default: {DEFAULT_REGION})")

    args = parser.parse_args()

    # Load config
    config = copy.deepcopy(DEFAULT_ATC_CONFIG)
    if args.config:
        try:
            with open(args.config) as f:
                config.update(json.load(f))
            logger.info(f"Loaded custom config from {args.config}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            sys.exit(1)

    # Adjust weights to match selected timeframes
    if args.timeframes:
        w = 1.0 / len(args.timeframes)
        config["weights"] = {tf: w for tf in args.timeframes}

    try:
        # Step 1: Fetch Binance symbols
        data_loader = BinanceDataLoader()
        limit = None if args.all_symbols else args.symbols
        symbols = data_loader.get_usdt_symbols(limit=limit)
        if not symbols:
            logger.error("No symbols found")
            sys.exit(1)

        preview = ", ".join(symbols[:10]) + ("..." if len(symbols) > 10 else "")
        logger.info(f"Processing {len(symbols)} symbols: {preview}")

        # Step 2: Fetch OHLCV data
        logger.info(f"Fetching OHLCV data for timeframes: {args.timeframes}")
        ohlcv_data = data_loader.get_ohlcv_data(symbols, args.timeframes, limit=args.limit)
        if not ohlcv_data:
            logger.error("No OHLCV data could be fetched")
            sys.exit(1)
        logger.info(f"Fetched data for {len(ohlcv_data)} symbols")

        symbols_data = [{"symbol": sym, "timeframes": tfs} for sym, tfs in ohlcv_data.items()]

        # Step 3: Invoke Lambda (or mock)
        t0 = time.time()
        if args.mock:
            logger.info("Using mock responses (--mock flag set)")
            response = mock_invoke(symbols_data, config)
        else:
            client = ATCLambdaClient(
                function_name=args.function_name,
                region=args.region,
            )
            response = client.invoke(symbols_data, config)

        duration = time.time() - t0

        # Step 4: Display results
        if "results" in response:
            display_results(response["results"], show_details=args.details)
            if response.get("errors"):
                print(f"\nErrors ({len(response['errors'])}):")
                for err in response["errors"]:
                    print(f"  - {err.get('symbol', '?')}: {err.get('error', '?')}")

        print("\nPerformance:")
        print(f"  Total Time:      {duration:.2f}s")
        print(f"  Symbols:         {len(symbols_data)}")
        if duration > 0:
            print(f"  Throughput:      {len(symbols_data) / duration:.2f} symbols/s")

        logger.info("Demo completed successfully!")

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

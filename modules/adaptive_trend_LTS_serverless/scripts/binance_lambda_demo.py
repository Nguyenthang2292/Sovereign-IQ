#!/usr/bin/env python3
"""
Binance Lambda Demo Script

Invokes the ATC Serverless Lambda with real Binance market data.

Architecture:
  Lambda (atc-serverless) --[process]--> SQS queue (atc-results) --> this script polls SQS

The Lambda function does NOT return JSON results directly. It processes the batch
and sends results to an SQS queue. This script:
  1. Fetches OHLCV data from Binance
  2. Invokes Lambda via boto3 (IAM-signed, no Function URL needed)
  3. Polls SQS for the results

Usage:
    python binance_lambda_demo.py --symbols 10
    python binance_lambda_demo.py --symbols 5 --timeframes 1h 4h --details
    python binance_lambda_demo.py --all-symbols
    python binance_lambda_demo.py --symbols 5 --mock
"""

import argparse
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3

# Add project root to path to import common modules
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.common.core.data_fetcher import DataFetcher, SymbolFetchError
from modules.common.core.exchange_manager import ExchangeManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_FUNCTION_NAME = "atc-serverless"
DEFAULT_SQS_QUEUE_NAME = "atc-results"
DEFAULT_REGION = "us-east-1"

# Default ATC configuration
DEFAULT_ATC_CONFIG = {
    "weights": {"1h": 0.6, "4h": 0.4},
    "threshold": 0.3,
    "min_signal": 0.0,
    "use_signal_strength": True,
    "lambda_param": 0.02,
    "decay": 0.03,
    "cutout": 0,
    "equity_floor": 0.25,
    "robustness": "Medium",
    "ma_configs": [
        {"ma_type": "EMA", "length": 12, "weight": 1.0},
        {"ma_type": "HMA", "length": 12, "weight": 1.0},
        {"ma_type": "WMA", "length": 12, "weight": 1.0},
        {"ma_type": "DEMA", "length": 12, "weight": 1.0},
        {"ma_type": "LSMA", "length": 12, "weight": 1.0},
        {"ma_type": "KAMA", "length": 12, "weight": 1.0},
    ],
}


class BinanceDataLoader:
    """Loads Binance market data using the DataFetcher module."""

    def __init__(self):
        self.exchange_manager = ExchangeManager()
        self.data_fetcher = DataFetcher(self.exchange_manager)

    def get_usdt_symbols(self, limit: int = None) -> List[str]:
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
        self, symbols: List[str], timeframes: List[str], limit: int = 100
    ) -> Dict[str, Dict[str, Dict[str, List]]]:
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


class ATCLambdaClient:
    """Invokes the ATC Serverless Lambda and polls SQS for results.

    Architecture (from Rust handler):
        boto3.invoke(Lambda) --> Lambda processes --> sends result to SQS
        This client then polls SQS to retrieve the result.

    The Lambda returns () / null directly, so we cannot read the result from
    the invoke response. Results come via SQS.
    """

    def __init__(
        self,
        function_name: str = DEFAULT_FUNCTION_NAME,
        sqs_queue_name: str = DEFAULT_SQS_QUEUE_NAME,
        region: str = DEFAULT_REGION,
        sqs_poll_timeout: int = 60,
        sqs_poll_interval: float = 2.0,
    ):
        self.function_name = function_name
        self.sqs_queue_name = sqs_queue_name
        self.region = region
        self.sqs_poll_timeout = sqs_poll_timeout
        self.sqs_poll_interval = sqs_poll_interval

        self._lambda = boto3.client("lambda", region_name=region)
        self._sqs = boto3.client("sqs", region_name=region)
        self._queue_url: Optional[str] = None

        logger.info(f"ATCLambdaClient: function='{function_name}', sqs='{sqs_queue_name}', region={region}")

    def _get_queue_url(self) -> str:
        if not self._queue_url:
            response = self._sqs.get_queue_url(QueueName=self.sqs_queue_name)
            self._queue_url = response["QueueUrl"]
        return self._queue_url

    def invoke_and_wait(
        self,
        symbols_data: List[Dict[str, Any]],
        config: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Invoke Lambda and poll SQS for the result.

        Args:
            symbols_data: List of {symbol, timeframes} dicts
            config: ATC config (uses DEFAULT_ATC_CONFIG if not provided)

        Returns:
            ScanResult dict: {batch_id, results, errors, success_count, error_count}
        """
        if config is None:
            config = DEFAULT_ATC_CONFIG.copy()

        batch_id = f"demo-{uuid.uuid4().hex[:8]}"
        payload = {
            "batch_id": batch_id,
            "symbols": symbols_data,
            "config": config,
        }

        logger.info(f"Invoking Lambda batch '{batch_id}' with {len(symbols_data)} symbols...")

        # Step 1: Invoke Lambda (fire-and-forget, result goes to SQS)
        response = self._lambda.invoke(
            FunctionName=self.function_name,
            InvocationType="RequestResponse",
            Payload=json.dumps(payload).encode("utf-8"),
        )

        status = response["StatusCode"]
        raw_body = response["Payload"].read().decode("utf-8")
        logger.debug(f"Lambda HTTP status: {status}, raw body: {raw_body!r}")

        if status != 200:
            raise RuntimeError(f"Lambda invocation failed (HTTP {status}): {raw_body}")

        # Lambda returns null on success (result is sent directly to SQS)
        # Check for Lambda runtime errors
        if response.get("FunctionError"):
            body = json.loads(raw_body) if raw_body and raw_body != "null" else {}
            raise RuntimeError(
                f"Lambda function error [{response['FunctionError']}]: {body.get('errorMessage', raw_body)}"
            )

        logger.info(f"Lambda invoked OK (batch_id={batch_id}). Polling SQS for results...")

        # Step 2: Poll SQS for the result message matching our batch_id
        return self._poll_sqs_for_batch(batch_id)

    def _poll_sqs_for_batch(self, batch_id: str) -> Dict[str, Any]:
        """Poll SQS until we find the message for our batch_id."""
        queue_url = self._get_queue_url()
        deadline = time.time() + self.sqs_poll_timeout
        attempts = 0

        while time.time() < deadline:
            attempts += 1
            messages = self._sqs.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=10,
                WaitTimeSeconds=5,  # long-polling: up to 5s
                VisibilityTimeout=30,
            ).get("Messages", [])

            for msg in messages:
                try:
                    body = json.loads(msg["Body"])
                    if body.get("batch_id") == batch_id:
                        # Found our result — delete from queue and return
                        self._sqs.delete_message(
                            QueueUrl=queue_url,
                            ReceiptHandle=msg["ReceiptHandle"],
                        )
                        logger.info(f"Got result from SQS (batch_id={batch_id}) after {attempts} poll(s)")
                        return body
                    else:
                        # Not our message — put it back (change visibility to 0)
                        self._sqs.change_message_visibility(
                            QueueUrl=queue_url,
                            ReceiptHandle=msg["ReceiptHandle"],
                            VisibilityTimeout=0,
                        )
                except json.JSONDecodeError:
                    pass  # Skip malformed messages

            remaining = int(deadline - time.time())
            logger.info(f"Waiting for SQS result... (attempt {attempts}, {remaining}s remaining)")
            time.sleep(self.sqs_poll_interval)

        raise TimeoutError(
            f"No SQS result received for batch_id='{batch_id}' "
            f"within {self.sqs_poll_timeout}s. "
            "Check CloudWatch logs for Lambda errors."
        )


def mock_invoke(symbols_data: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
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


def display_results(results: List[Dict[str, Any]], show_details: bool = False):
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

    print(f"\nSummary:")
    print(f"  Total Signals: {total}")
    print(f"  LONG:    {long_count} ({long_count / total * 100:.1f}%)")
    print(f"  SHORT:   {short_count} ({short_count / total * 100:.1f}%)")
    print(f"  NEUTRAL: {neutral_count} ({neutral_count / total * 100:.1f}%)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description=("Binance Lambda Demo — invokes ATC Serverless Lambda via boto3 and polls SQS for results.")
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
    parser.add_argument(
        "--sqs-queue",
        default=DEFAULT_SQS_QUEUE_NAME,
        help=f"SQS queue name for results (default: {DEFAULT_SQS_QUEUE_NAME})",
    )
    parser.add_argument("--region", default=DEFAULT_REGION, help=f"AWS region (default: {DEFAULT_REGION})")
    parser.add_argument("--sqs-timeout", type=int, default=60, help="Seconds to wait for SQS result (default: 60)")

    args = parser.parse_args()

    # Load config
    config = DEFAULT_ATC_CONFIG.copy()
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
                sqs_queue_name=args.sqs_queue,
                region=args.region,
                sqs_poll_timeout=args.sqs_timeout,
            )
            response = client.invoke_and_wait(symbols_data, config)

        duration = time.time() - t0

        # Step 4: Display results
        if "results" in response:
            display_results(response["results"], show_details=args.details)
            if response.get("errors"):
                print(f"\nErrors ({len(response['errors'])}):")
                for err in response["errors"]:
                    print(f"  - {err.get('symbol', '?')}: {err.get('error', '?')}")

        print(f"\nPerformance:")
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

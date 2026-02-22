#!/usr/bin/env python3
"""
Binance Lambda Demo Script for XGBoost Serverless

Fetches real market data from Binance and invokes the XGBoost Serverless Lambda.
Unlike the ATC module, the XGBoost Lambda returns the predictions directly in the
invocation response (though it can optionally also send to SQS).
This script uses the direct synchronous response.

Usage:
    python binance_lambda_demo.py --symbols 10
    python binance_lambda_demo.py --timeframes 15m 1h --symbols 5
    python binance_lambda_demo.py --all-symbols
    python binance_lambda_demo.py --mock
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

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
DEFAULT_FUNCTION_NAME = "xgboost-serverless-predict"
DEFAULT_REGION = "us-east-1"


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


class XGBoostLambdaClient:
    """Invokes the XGBoost Serverless Lambda via boto3."""

    def __init__(
        self,
        function_name: str = DEFAULT_FUNCTION_NAME,
        region: str = DEFAULT_REGION,
    ):
        self.function_name = function_name
        self.region = region
        self._lambda = boto3.client("lambda", region_name=region)

        logger.info(f"XGBoostLambdaClient: function='{function_name}', region={region}")

    def invoke(
        self,
        requests: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Invoke Lambda and return the response.

        Args:
            requests: List of prediction items

        Returns:
            XGBoostResponse dict
        """
        payload = {"version": "1.0", "mode": "batch", "requests": requests, "options": {"return_features": False}}

        logger.info(f"Invoking Lambda with {len(requests)} prediction requests...")

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

        if response.get("FunctionError"):
            body = json.loads(raw_body) if raw_body and raw_body != "null" else {}
            raise RuntimeError(
                f"Lambda function error [{response['FunctionError']}]: {body.get('errorMessage', raw_body)}"
            )

        return json.loads(raw_body)


def mock_invoke(requests: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a mock result for testing without AWS."""
    import random

    predictions = []
    for req in requests:
        score = random.uniform(0.0, 1.0)
        label = "UP" if score > 0.6 else ("DOWN" if score < 0.4 else "NEUTRAL")
        probs = [
            random.uniform(0, 0.4) if label != "DOWN" else random.uniform(0.6, 1.0),
            random.uniform(0, 0.4) if label != "NEUTRAL" else random.uniform(0.6, 1.0),
            random.uniform(0, 0.4) if label != "UP" else random.uniform(0.6, 1.0),
        ]
        # Normalize
        total = sum(probs)
        probs = [p / total for p in probs]
        confidence = max(probs)

        predictions.append(
            {
                "symbol": req["symbol"],
                "timeframe": req.get("timeframe", "15m"),
                "prediction": {"label": label, "probabilities": probs, "confidence": confidence},
                "metadata": {
                    "candles_processed": len(req["data"]["close"]),
                    "features_calculated": 92,
                    "inference_time_ms": random.randint(1, 5),
                },
            }
        )

    return {
        "success": True,
        "predictions": predictions,
        "timing": {
            "total_ms": random.randint(20, 100),
            "model_load_ms": random.randint(5, 50),
            "feature_calc_ms": random.randint(5, 20),
            "inference_ms": random.randint(1, 10),
        },
    }


def display_results(response: Dict[str, Any]):
    """Display prediction results in a formatted table."""
    predictions = response.get("predictions", [])
    if not predictions:
        print("No results to display.")
        return

    print("\n" + "=" * 100)
    print("XGBOOST SIGNAL RESULTS")
    print("=" * 100)

    # Sort by confidence descending
    sorted_results = sorted(predictions, key=lambda x: x.get("prediction", {}).get("confidence", 0), reverse=True)

    print(f"{'#':<4} {'Symbol':<16} {'TF':<6} {'Signal':<10} {'Confidence':<12} {'[Down, Neutral, Up] probabilities'}")
    print("-" * 100)

    for idx, result in enumerate(sorted_results, 1):
        symbol = result.get("symbol", "?")
        tf = result.get("timeframe", "?")

        pred = result.get("prediction", {})
        signal = pred.get("label", "NEUTRAL")
        confidence = pred.get("confidence", 0.0)
        probs = pred.get("probabilities", [0.0, 0.0, 0.0])

        if signal == "UP":
            signal_disp = f"\033[92m{signal}\033[0m"  # Green
        elif signal == "DOWN":
            signal_disp = f"\033[91m{signal}\033[0m"  # Red
        else:
            signal_disp = f"\033[93m{signal}\033[0m"  # Yellow

        probs_formatted = f"[{probs[0]:.2f}, {probs[1]:.2f}, {probs[2]:.2f}]"

        print(f"{idx:<4} {symbol:<16} {tf:<6} {signal_disp:<18} {confidence:<12.2%} {probs_formatted}")

    print("=" * 100)

    up_count = sum(1 for r in predictions if r.get("prediction", {}).get("label") == "UP")
    down_count = sum(1 for r in predictions if r.get("prediction", {}).get("label") == "DOWN")
    neutral_count = sum(1 for r in predictions if r.get("prediction", {}).get("label") == "NEUTRAL")
    total = len(predictions)

    print("\nSummary:")
    print(f"  Total Signals: {total}")
    print(f"  UP:      {up_count} ({up_count / total * 100:.1f}%)")
    print(f"  DOWN:    {down_count} ({down_count / total * 100:.1f}%)")
    print(f"  NEUTRAL: {neutral_count} ({neutral_count / total * 100:.1f}%)")

    if "timing" in response:
        t = response["timing"]
        print("\nServer Execution Timing:")
        print(f"  Total Lambda Time: {t.get('total_ms', 0)} ms")
        print(f"  Model Load Time:   {t.get('model_load_ms', 0)} ms")
        print(f"  Feature Calc Time: {t.get('feature_calc_ms', 0)} ms")
        print(f"  Inference Time:    {t.get('inference_ms', 0)} ms")
    print()


def main():
    parser = argparse.ArgumentParser(
        description=("Binance Lambda Demo for XGBoost — fetches real data and invokes Lambda.")
    )
    parser.add_argument("--symbols", type=int, default=10, help="Number of symbols to process (default: 10)")
    parser.add_argument(
        "--all-symbols",
        action="store_true",
        help="Process all available USDT symbols (Warning: might hit Lambda limitations)",
    )
    parser.add_argument("--timeframes", nargs="+", default=["15m", "1h"], help="Timeframes to fetch (default: 15m 1h)")
    parser.add_argument(
        "--limit", type=int, default=100, help="Candles per timeframe (minimum 50 required, default: 100)"
    )
    parser.add_argument("--mock", action="store_true", help="Use mock results (no AWS required)")
    parser.add_argument(
        "--function-name",
        default=DEFAULT_FUNCTION_NAME,
        help=f"Lambda function name (default: {DEFAULT_FUNCTION_NAME})",
    )
    parser.add_argument("--region", default=DEFAULT_REGION, help=f"AWS region (default: {DEFAULT_REGION})")

    args = parser.parse_args()

    if args.limit < 50:
        logger.error("XGBoost requires at least 50 candles to calculate features properly.")
        sys.exit(1)

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

        # Format payload elements
        # XGBoost lambda expects a flat list of prediction items
        requests = []
        for symbol, tf_data in ohlcv_data.items():
            for tf, data in tf_data.items():
                requests.append(
                    {
                        "symbol": symbol,
                        "timeframe": tf,
                        "model_version": "v1",
                        "timestamp": int(time.time() * 1000),
                        "data": data,
                        # "model_s3_key" is optional, lambda will use `{symbol}_{tf}_{model_version}.json` from S3 if missing in cache
                    }
                )

        logger.info(f"Prepared {len(requests)} prediction requests across {len(ohlcv_data)} symbols")

        # Due to max_batch_size=50 configured in lambda validate_request(), we chunk the requests
        BATCH_SIZE = 50
        all_responses = {
            "predictions": [],
            "success": True,
            "timing": {"total_ms": 0, "model_load_ms": 0, "feature_calc_ms": 0, "inference_ms": 0},
        }

        t0 = time.time()

        for i in range(0, len(requests), BATCH_SIZE):
            batch = requests[i : i + BATCH_SIZE]
            logger.info(
                f"Processing batch {i // BATCH_SIZE + 1}/{(len(requests) - 1) // BATCH_SIZE + 1} ({len(batch)} items)"
            )

            if args.mock:
                response = mock_invoke(batch)
            else:
                client = XGBoostLambdaClient(
                    function_name=args.function_name,
                    region=args.region,
                )
                response = client.invoke(batch)

            # Aggregate response
            if response.get("success"):
                all_responses["predictions"].extend(response.get("predictions", []))

                if "timing" in response:
                    for k in all_responses["timing"]:
                        all_responses["timing"][k] += response["timing"].get(k, 0)
            else:
                logger.error(f"Batch failed: {response}")

        duration = time.time() - t0

        # Step 4: Display results
        display_results(all_responses)

        print("\nOverall Client Performance:")
        print(f"  Total Time:      {duration:.2f}s")
        print(f"  Requests:        {len(requests)}")
        if duration > 0:
            print(f"  Throughput:      {len(requests) / duration:.2f} inferences/s")

        logger.info("Demo completed successfully!")

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

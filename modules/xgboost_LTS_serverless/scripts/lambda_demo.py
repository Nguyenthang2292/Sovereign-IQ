#!/usr/bin/env python3
"""
Binance Lambda Demo — XGBoost Serverless

Invokes the XGBoost Serverless Lambda with real Binance market data.

Architecture:
  This Lambda returns predictions DIRECTLY in the invoke response (synchronous),
  unlike ATC Serverless which publishes to SQS.

  1. Fetch OHLCV data from Binance
  2. Invoke Lambda via boto3 (IAM-signed)
  3. Parse and display prediction results

Usage:
    python scripts/lambda_demo.py
    python scripts/lambda_demo.py --symbols 5 --timeframe 15m --details
    python scripts/lambda_demo.py --symbols 5 --mock
    python scripts/lambda_demo.py --symbol BTCUSDT --timeframe 1h --model-version v1
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

# ── Project root so common modules are importable ─────────────────────────────
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.common.core.data_fetcher import DataFetcher, SymbolFetchError  # noqa: E402
from modules.common.core.exchange_manager import ExchangeManager  # noqa: E402

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_FUNCTION_NAME = "xgboost-serverless-predict"
DEFAULT_REGION = "us-east-1"
DEFAULT_TIMEFRAME = "15m"
DEFAULT_MODEL_VERSION = "v1"
DEFAULT_CANDLE_LIMIT = 200  # minimum 50 required by the Lambda validator


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  Binance data loader                                                        │
# └─────────────────────────────────────────────────────────────────────────────┘


class BinanceDataLoader:
    """Loads Binance OHLCV data using the shared DataFetcher module."""

    def __init__(self) -> None:
        self.exchange_manager = ExchangeManager()
        self.data_fetcher = DataFetcher(self.exchange_manager)

    def get_usdt_symbols(self, limit: int | None = None) -> list[str]:
        """Return a list of Binance USDT spot trading pairs."""
        try:
            logger.info("Fetching USDT symbols from Binance...")
            symbols = self.data_fetcher.get_spot_symbols(exchange_name="binance", quote_currency="USDT")
            logger.info(f"Found {len(symbols)} USDT trading pairs")
            if limit:
                symbols = symbols[:limit]
                logger.info(f"Limited to {limit} symbols")
            return symbols
        except SymbolFetchError as e:
            logger.error(f"Failed to fetch symbols: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching symbols: {e}")
            raise

    def get_ohlcv(self, symbol: str, timeframe: str, limit: int = DEFAULT_CANDLE_LIMIT) -> dict[str, list] | None:
        """Fetch OHLCV data for one symbol/timeframe.

        Returns a dict ready to embed in an XGBoostRequest, or None on failure.
        """
        try:
            df = self.data_fetcher.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit, check_freshness=False)
            if df is None or df.empty:
                logger.warning(f"No data returned for {symbol} {timeframe}")
                return None

            logger.debug(f"Fetched {len(df)} candles for {symbol} {timeframe}")
            return {
                "timestamp": [int(ts.timestamp() * 1000) for ts in df.index],
                "open": df["open"].tolist(),
                "high": df["high"].tolist(),
                "low": df["low"].tolist(),
                "close": df["close"].tolist(),
                "volume": df["volume"].tolist(),
            }
        except Exception as e:
            logger.warning(f"Failed to fetch {symbol} {timeframe}: {e}")
            return None


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  XGBoost Lambda client                                                      │
# └─────────────────────────────────────────────────────────────────────────────┘


class XGBoostLambdaClient:
    """Invokes the XGBoost Serverless Lambda via boto3 and returns results directly.

    The Lambda handler is SYNCHRONOUS — it returns the full prediction JSON in
    the invoke response body (unlike ATC which sends to SQS).
    """

    def __init__(
        self,
        function_name: str = DEFAULT_FUNCTION_NAME,
        region: str = DEFAULT_REGION,
    ) -> None:
        self.function_name = function_name
        self.region = region
        self._lambda = boto3.client("lambda", region_name=region)
        logger.info(f"XGBoostLambdaClient: function='{function_name}', region={region}")

    def predict(self, requests: list[dict[str, Any]]) -> dict[str, Any]:
        """Send a batch of prediction requests and return the parsed response.

        Parameters
        ----------
        requests : list of XGBoost prediction item dicts:
            - symbol        : str  (e.g. "BTCUSDT")
            - timeframe     : str  (e.g. "15m")
            - model_version : str  (e.g. "v1")
            - model_s3_key  : str  (e.g. "BTCUSDT_15m_v1.json")
            - data          : {timestamp, open, high, low, close, volume}  (lists)

        Returns
        -------
        Parsed response dict: {success, predictions, timing}
        """
        payload = {
            "version": "1.0",
            "mode": "batch" if len(requests) > 1 else "single",
            "requests": requests,
            "options": {"return_features": False},
        }

        logger.info(f"Invoking Lambda '{self.function_name}' with {len(requests)} request(s)...")

        try:
            response = self._lambda.invoke(
                FunctionName=self.function_name,
                InvocationType="RequestResponse",
                Payload=json.dumps(payload).encode("utf-8"),
            )
        except ClientError as e:
            raise RuntimeError(f"Lambda invoke failed: {e}") from e

        status = response["StatusCode"]
        raw_body = response["Payload"].read().decode("utf-8")

        if response.get("FunctionError"):
            detail = {}
            try:
                detail = json.loads(raw_body)
            except json.JSONDecodeError:
                pass
            raise RuntimeError(
                f"Lambda function error [{response['FunctionError']}]: {detail.get('errorMessage', raw_body)}"
            )

        if status != 200:
            raise RuntimeError(f"Lambda HTTP {status}: {raw_body}")

        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse Lambda response JSON: {e}\nRaw: {raw_body}") from e

        if not body.get("success"):
            raise RuntimeError(f"Lambda returned unsuccessful response: {body}")

        return body


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  Mock                                                                       │
# └─────────────────────────────────────────────────────────────────────────────┘


def mock_invoke(requests: list[dict]) -> dict[str, Any]:
    """Generate a realistic mock response without hitting AWS."""
    predictions = []
    for req in requests:
        label = random.choice(["UP", "DOWN", "NEUTRAL"])
        probs = [random.random() for _ in range(3)]
        total = sum(probs)
        probs = [p / total for p in probs]
        idx = probs.index(max(probs))
        label = ["DOWN", "NEUTRAL", "UP"][idx]
        predictions.append(
            {
                "symbol": req["symbol"],
                "timeframe": req.get("timeframe", DEFAULT_TIMEFRAME),
                "prediction": {
                    "label": label,
                    "probabilities": probs,
                    "confidence": max(probs),
                },
                "metadata": {
                    "candles_processed": len(req["data"]["close"]),
                    "features_calculated": 92,
                    "inference_time_ms": random.randint(1, 15),
                },
            }
        )

    return {
        "success": True,
        "predictions": predictions,
        "timing": {
            "total_ms": random.randint(50, 300),
            "model_load_ms": 0,
            "feature_calc_ms": random.randint(5, 30),
            "inference_ms": random.randint(1, 15),
        },
    }


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  Display                                                                    │
# └─────────────────────────────────────────────────────────────────────────────┘


def display_results(predictions: list[dict], timing: dict, show_details: bool = False) -> None:
    """Print prediction results in a formatted table."""
    if not predictions:
        print("No predictions to display.")
        return

    print("\n" + "=" * 110)
    print("XGBOOST PREDICTION RESULTS")
    print("=" * 110)

    # Sort: UP first (high confidence), then DOWN, NEUTRAL last
    def sort_key(p: dict) -> tuple:
        label = p["prediction"].get("label", "NEUTRAL")
        conf = p["prediction"].get("confidence", 0.0)
        order = {"UP": 0, "DOWN": 1, "NEUTRAL": 2}
        return (order.get(label, 2), -conf)

    sorted_preds = sorted(predictions, key=sort_key)

    print(
        f"{'#':<4} {'Symbol':<14} {'TF':<6} {'Signal':<10} "
        f"{'Confidence':<12} {'P(DOWN)':<10} {'P(NEUTRAL)':<12} {'P(UP)':<10}"
    )
    print("-" * 110)

    for idx, pred in enumerate(sorted_preds, 1):
        symbol = pred.get("symbol", "?")
        timeframe = pred.get("timeframe", "?")
        prediction = pred.get("prediction", {})
        label = prediction.get("label", "NEUTRAL")
        confidence = prediction.get("confidence", 0.0)
        probs = prediction.get("probabilities", [0.0, 0.0, 0.0])
        p_down = probs[0] if len(probs) > 0 else 0.0
        p_neutral = probs[1] if len(probs) > 1 else 0.0
        p_up = probs[2] if len(probs) > 2 else 0.0

        # ANSI colours
        if label == "UP":
            label_disp = f"\033[92m{label:<8}\033[0m"  # green
        elif label == "DOWN":
            label_disp = f"\033[91m{label:<8}\033[0m"  # red
        else:
            label_disp = f"\033[93m{label:<8}\033[0m"  # yellow

        print(
            f"{idx:<4} {symbol:<14} {timeframe:<6} {label_disp:<20} "
            f"{confidence:<12.2%} {p_down:<10.4f} {p_neutral:<12.4f} {p_up:<10.4f}"
        )

        if show_details:
            meta = pred.get("metadata", {})
            print(
                f"     └─ candles: {meta.get('candles_processed', '?')}  "
                f"features: {meta.get('features_calculated', '?')}  "
                f"inference: {meta.get('inference_time_ms', '?')}ms"
            )

    print("=" * 110)

    # Summary counts
    up_count = sum(1 for p in predictions if p["prediction"].get("label") == "UP")
    down_count = sum(1 for p in predictions if p["prediction"].get("label") == "DOWN")
    neutral_count = sum(1 for p in predictions if p["prediction"].get("label") == "NEUTRAL")
    total = len(predictions)
    avg_conf = sum(p["prediction"].get("confidence", 0) for p in predictions) / max(total, 1)

    print(f"\nSummary ({total} predictions):")
    print(f"  \033[92mUP\033[0m      : {up_count:>3}  ({up_count / total * 100:.1f}%)")
    print(f"  \033[91mDOWN\033[0m    : {down_count:>3}  ({down_count / total * 100:.1f}%)")
    print(f"  \033[93mNEUTRAL\033[0m : {neutral_count:>3}  ({neutral_count / total * 100:.1f}%)")
    print(f"  Avg Confidence: {avg_conf:.2%}")

    print(f"\nTiming:")
    print(f"  Total Lambda:   {timing.get('total_ms', '?')}ms")
    print(f"  Feature Calc:   {timing.get('feature_calc_ms', '?')}ms")
    print(f"  Inference:      {timing.get('inference_ms', '?')}ms")
    print(f"  Model Load:     {timing.get('model_load_ms', '?')}ms")
    print()


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  Entry-point                                                                │
# └─────────────────────────────────────────────────────────────────────────────┘


def main() -> None:
    parser = argparse.ArgumentParser(
        description="XGBoost Serverless Lambda demo — invokes the Lambda with real Binance data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single symbol
  python scripts/lambda_demo.py --symbol BTCUSDT --timeframe 15m

  # Batch: top-10 USDT symbols, 1h timeframe
  python scripts/lambda_demo.py --symbols 10 --timeframe 1h --details

  # Dry-run with mock results (no AWS needed)
  python scripts/lambda_demo.py --symbols 5 --mock

  # Custom model version
  python scripts/lambda_demo.py --symbol ETHUSDT --timeframe 4h --model-version v2

  # Force a specific S3 model key
  python scripts/lambda_demo.py --symbol BTCUSDT --timeframe 15m --model-s3-key BTCUSDT_15m_v1.json
""",
    )

    # ── Symbol selection ──────────────────────────────────────────────────────
    sym_group = parser.add_mutually_exclusive_group()
    sym_group.add_argument(
        "--symbol",
        default=None,
        help="Single symbol to predict (e.g. BTCUSDT). Overrides --symbols.",
    )
    sym_group.add_argument(
        "--symbols",
        type=int,
        default=5,
        help="Number of top Binance USDT symbols to process (default: 5)",
    )
    sym_group.add_argument(
        "--all-symbols",
        action="store_true",
        help="Process ALL Binance USDT symbols (may take a long time)",
    )

    # ── Data config ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--timeframe",
        default=DEFAULT_TIMEFRAME,
        help=f"Candle timeframe (default: {DEFAULT_TIMEFRAME})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_CANDLE_LIMIT,
        help=f"Number of candles to fetch per symbol (default: {DEFAULT_CANDLE_LIMIT}, min: 50)",
    )

    # ── Model config ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--model-version",
        default=DEFAULT_MODEL_VERSION,
        help=f"Model version tag (default: {DEFAULT_MODEL_VERSION})",
    )
    parser.add_argument(
        "--model-s3-key",
        default=None,
        help=(
            "Explicit S3 key for the model file (e.g. BTCUSDT_15m_v1.json). "
            "If not set, the Lambda will use its in-memory cache or fail if not pre-loaded."
        ),
    )

    # ── AWS config ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--function-name",
        default=DEFAULT_FUNCTION_NAME,
        help=f"Lambda function name (default: {DEFAULT_FUNCTION_NAME})",
    )
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help=f"AWS region (default: {DEFAULT_REGION})",
    )

    # ── Display ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--details",
        action="store_true",
        help="Show per-prediction metadata (candles, features, inference time)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock results — no AWS call made (useful for testing)",
    )

    args = parser.parse_args()

    # Validate limit
    if args.limit < 50:
        parser.error("--limit must be >= 50 (Lambda validator requires at least 50 candles)")

    try:
        # ── Step 1: Resolve symbols ───────────────────────────────────────────
        data_loader = BinanceDataLoader()

        if args.symbol:
            symbols = [args.symbol.upper()]
        elif args.all_symbols:
            symbols = data_loader.get_usdt_symbols()
        else:
            symbols = data_loader.get_usdt_symbols(limit=args.symbols)

        if not symbols:
            logger.error("No symbols resolved. Exiting.")
            sys.exit(1)

        preview = ", ".join(symbols[:10]) + ("..." if len(symbols) > 10 else "")
        logger.info(f"Processing {len(symbols)} symbol(s): {preview}")

        # ── Step 2: Fetch OHLCV and build request items ───────────────────────
        logger.info(f"Fetching OHLCV ({args.timeframe}, {args.limit} candles)...")
        request_items: list[dict] = []

        for symbol in symbols:
            ohlcv = data_loader.get_ohlcv(symbol, args.timeframe, limit=args.limit)
            if ohlcv is None:
                logger.warning(f"Skipping {symbol}: no data available")
                continue

            item: dict[str, Any] = {
                "symbol": symbol,
                "timeframe": args.timeframe,
                "model_version": args.model_version,
                "data": ohlcv,
            }

            # Auto-construct S3 key if not explicitly provided
            s3_key = args.model_s3_key or f"{symbol}_{args.timeframe}_{args.model_version}.json"
            item["model_s3_key"] = s3_key

            request_items.append(item)
            time.sleep(0.05)  # gentle rate limiting

        if not request_items:
            logger.error("No OHLCV data retrieved for any symbol. Exiting.")
            sys.exit(1)

        logger.info(f"Built {len(request_items)} request item(s)")

        # ── Step 3: Invoke Lambda (or mock) ───────────────────────────────────
        t0 = time.time()

        if args.mock:
            logger.info("Using mock results (--mock flag set, no AWS call)")
            response = mock_invoke(request_items)
        else:
            client = XGBoostLambdaClient(
                function_name=args.function_name,
                region=args.region,
            )
            response = client.predict(request_items)

        duration = time.time() - t0
        logger.info(f"Total round-trip: {duration:.2f}s")

        # ── Step 4: Display results ───────────────────────────────────────────
        predictions = response.get("predictions", [])
        timing = response.get("timing", {})

        display_results(predictions, timing, show_details=args.details)

        logger.info("Demo completed successfully!")

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

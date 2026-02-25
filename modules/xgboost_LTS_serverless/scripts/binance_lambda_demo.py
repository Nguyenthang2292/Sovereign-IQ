#!/usr/bin/env python3
"""
Binance Lambda Demo Script for XGBoost Serverless

Full pipeline:
  1. Fetch USDT symbols from Binance
  2. Fetch OHLCV data for each symbol × timeframe
  3. [AUTO-TRAIN] If a model is missing on S3 → invoke xgboost-trainer Lambda
     (synchronous, up to 900s per model) — runs in parallel per symbol
  4. Invoke xgboost-serverless-predict Lambda with batched OHLCV payloads
  5. Display prediction results

Usage:
    # Full pipeline (train missing models then predict):
    python binance_lambda_demo.py --symbols 10

    # Skip training step (fail fast if model missing):
    python binance_lambda_demo.py --symbols 5 --skip-train

    # Only train models, don't predict:
    python binance_lambda_demo.py --symbols 5 --train-only

    # Mock mode (no AWS required):
    python binance_lambda_demo.py --symbols 5 --mock

    # Full pipeline with custom timeframes:
    python binance_lambda_demo.py --timeframes 15m 1h --symbols 5
"""

import argparse
import concurrent.futures
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import boto3
import botocore.exceptions
from dotenv import load_dotenv

# Load .env from project root (4 levels up from this script)
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# Add project root to path to import common modules
sys.path.insert(0, str(_PROJECT_ROOT))

from modules.common.core.data_fetcher import DataFetcher, SymbolFetchError
from modules.common.core.exchange_manager import ExchangeManager

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
DEFAULT_PREDICT_FUNCTION = "xgboost-serverless-predict"
DEFAULT_TRAINER_FUNCTION = "xgboost-trainer"
DEFAULT_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "ap-southeast-1"))
DEFAULT_S3_BUCKET = os.environ.get("S3_BUCKET_NAME", os.environ.get("MODEL_BUCKET", "xgboost-models-store"))
DEFAULT_MODEL_VERSION = "v1"


# ─── AWS Credentials helper ───────────────────────────────────────────────────
def _make_boto3_client(service: str, region: str) -> Any:
    """Create a boto3 client using credentials from .env."""
    aws_key = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY")

    if not aws_key or not aws_secret:
        raise RuntimeError(
            "AWS credentials not found. "
            "Ensure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set in .env"
        )

    return boto3.client(
        service,
        region_name=region,
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
    )


# ─── Normalisation (matches Lambda trainer logic) ─────────────────────────────
def _normalize(symbol: str) -> str:
    """Normalize symbol to filename format, e.g. 'BTC/USDT' → 'BTCUSDT'."""
    return "".join(ch for ch in symbol.upper() if ch.isalnum())


def _model_s3_key(symbol: str, timeframe: str, version: str = DEFAULT_MODEL_VERSION) -> str:
    return f"{_normalize(symbol)}_{timeframe}_{version}.json"


# ─── S3 Model checker ─────────────────────────────────────────────────────────
class S3ModelChecker:
    """Check which (symbol, timeframe) models exist on S3."""

    def __init__(self, bucket: str, region: str):
        self.bucket = bucket
        self._s3 = _make_boto3_client("s3", region)

    def exists(self, symbol: str, timeframe: str, version: str = DEFAULT_MODEL_VERSION) -> bool:
        key = _model_s3_key(symbol, timeframe, version)
        try:
            self._s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return False
            raise

    def missing_models(
        self,
        symbol_tf_pairs: List[Tuple[str, str]],
        version: str = DEFAULT_MODEL_VERSION,
    ) -> List[Tuple[str, str]]:
        """Return list of (symbol, timeframe) pairs whose models are absent from S3."""
        missing = []
        for symbol, tf in symbol_tf_pairs:
            if not self.exists(symbol, tf, version):
                missing.append((symbol, tf))
        return missing


# ─── XGBoost Trainer Client ───────────────────────────────────────────────────
class XGBoostTrainerClient:
    """
    Invokes the xgboost-trainer Lambda synchronously (RequestResponse).

    Each invocation trains a model for ONE (symbol, timeframe) pair and uploads
    the result directly to S3. Timeout on Lambda side is 900 s.
    """

    def __init__(
        self,
        function_name: str = DEFAULT_TRAINER_FUNCTION,
        region: str = DEFAULT_REGION,
        s3_bucket: str = DEFAULT_S3_BUCKET,
    ):
        self.function_name = function_name
        self.region = region
        self.s3_bucket = s3_bucket
        self._lambda = _make_boto3_client("lambda", region)

        logger.info(
            f"XGBoostTrainerClient: function='{function_name}', "
            f"region={region}, bucket={s3_bucket}"
        )

    def train_one(
        self,
        symbol: str,
        timeframe: str,
        version: str = DEFAULT_MODEL_VERSION,
        fetch_limit: int = 1000,
    ) -> Dict[str, Any]:
        """
        Invoke trainer Lambda for a single (symbol, timeframe).

        Returns the Lambda response dict on success, raises on error.
        """
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "model_version": version,
            "s3_bucket": self.s3_bucket,
            "fetch_limit": fetch_limit,
        }

        logger.info(f"[trainer] Invoking for {symbol} {timeframe} ...")

        response = self._lambda.invoke(
            FunctionName=self.function_name,
            InvocationType="RequestResponse",  # synchronous — wait for completion
            Payload=json.dumps(payload).encode("utf-8"),
        )

        status = response["StatusCode"]
        raw_body = response["Payload"].read().decode("utf-8")

        if status != 200:
            raise RuntimeError(
                f"[trainer] HTTP {status} for {symbol} {timeframe}: {raw_body}"
            )

        if response.get("FunctionError"):
            body = json.loads(raw_body) if raw_body and raw_body != "null" else {}
            raise RuntimeError(
                f"[trainer] Lambda error [{response['FunctionError']}] "
                f"for {symbol} {timeframe}: {body.get('errorMessage', raw_body)}"
            )

        result = json.loads(raw_body)
        elapsed = result.get("elapsed_s", "?")
        s3_key = result.get("s3_key", "?")
        size = result.get("size_bytes", 0)
        logger.info(
            f"[trainer] ✓ {symbol} {timeframe} → s3://{self.s3_bucket}/{s3_key} "
            f"({size:,} bytes, {elapsed}s)"
        )
        return result

    def train_batch(
        self,
        symbol_tf_pairs: List[Tuple[str, str]],
        version: str = DEFAULT_MODEL_VERSION,
        fetch_limit: int = 1000,
        max_workers: int = 4,
    ) -> Dict[Tuple[str, str], Any]:
        """
        Train multiple (symbol, timeframe) pairs in parallel using a thread pool.

        Returns dict mapping (symbol, tf) → result or Exception.
        """
        if not symbol_tf_pairs:
            return {}

        logger.info(
            f"[trainer] Starting parallel training for {len(symbol_tf_pairs)} models "
            f"(max_workers={max_workers}) ..."
        )

        results: Dict[Tuple[str, str], Any] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pair = {
                executor.submit(self.train_one, sym, tf, version, fetch_limit): (sym, tf)
                for sym, tf in symbol_tf_pairs
            }

            for future in concurrent.futures.as_completed(future_to_pair):
                pair = future_to_pair[future]
                try:
                    results[pair] = future.result()
                except Exception as exc:
                    logger.error(f"[trainer] ✗ {pair[0]} {pair[1]} — {exc}")
                    results[pair] = exc

        success = sum(1 for v in results.values() if not isinstance(v, Exception))
        failed = len(results) - success
        logger.info(f"[trainer] Batch complete: {success} succeeded, {failed} failed")
        return results


# ─── XGBoost Predict Client ───────────────────────────────────────────────────
class XGBoostLambdaClient:
    """Invokes the xgboost-serverless-predict Lambda via boto3."""

    def __init__(
        self,
        function_name: str = DEFAULT_PREDICT_FUNCTION,
        region: str = DEFAULT_REGION,
    ):
        self.function_name = function_name
        self.region = region
        self._lambda = _make_boto3_client("lambda", region)
        logger.info(f"XGBoostLambdaClient: function='{function_name}', region={region}")

    def invoke(self, requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Invoke predict Lambda with a batch of OHLCV prediction requests.

        Returns the XGBoostResponse dict.
        """
        payload = {
            "version": "1.0",
            "mode": "batch",
            "requests": requests,
            "options": {"return_features": False},
        }

        logger.info(f"Invoking predict Lambda with {len(requests)} requests ...")

        response = self._lambda.invoke(
            FunctionName=self.function_name,
            InvocationType="RequestResponse",
            Payload=json.dumps(payload).encode("utf-8"),
        )

        status = response["StatusCode"]
        raw_body = response["Payload"].read().decode("utf-8")
        logger.debug(f"Lambda HTTP status: {status}")

        if status != 200:
            raise RuntimeError(f"Lambda invocation failed (HTTP {status}): {raw_body}")

        if response.get("FunctionError"):
            body = json.loads(raw_body) if raw_body and raw_body != "null" else {}
            raise RuntimeError(
                f"Lambda function error [{response['FunctionError']}]: "
                f"{body.get('errorMessage', raw_body)}"
            )

        return json.loads(raw_body)


# ─── Binance Data Loader ──────────────────────────────────────────────────────
class BinanceDataLoader:
    """Loads Binance market data using the DataFetcher module."""

    def __init__(self):
        self.exchange_manager = ExchangeManager()
        self.data_fetcher = DataFetcher(self.exchange_manager)

    def get_usdt_symbols(self, limit: Optional[int] = None) -> List[str]:
        """Get all USDT trading pairs from Binance."""
        try:
            logger.info("Fetching USDT symbols from Binance ...")
            symbols = self.data_fetcher.get_spot_symbols(
                exchange_name="binance", quote_currency="USDT"
            )
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
        self,
        symbols: List[str],
        timeframes: List[str],
        limit: int = 100,
    ) -> Dict[str, Dict[str, Dict[str, List]]]:
        """Fetch OHLCV data for symbols across multiple timeframes."""
        results = {}
        for symbol in symbols:
            logger.info(f"Fetching data for {symbol} ...")
            symbol_data = {}
            for tf in timeframes:
                try:
                    df = self.data_fetcher.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=tf,
                        limit=limit,
                        check_freshness=False,
                    )
                    if df is not None and not df.empty:
                        symbol_data[tf] = {
                            "timestamp": [int(ts.timestamp()) for ts in df.index],
                            "open": df["open"].tolist(),
                            "high": df["high"].tolist(),
                            "low": df["low"].tolist(),
                            "close": df["close"].tolist(),
                            "volume": df["volume"].tolist(),
                        }
                        logger.debug(f"  {symbol} {tf}: {len(df)} candles")
                    else:
                        logger.warning(f"  No data for {symbol} {tf}")
                except Exception as e:
                    logger.warning(f"  Failed to fetch {symbol} {tf}: {e}")
            if symbol_data:
                results[symbol] = symbol_data
            time.sleep(0.05)
        return results


# ─── Mock predict ─────────────────────────────────────────────────────────────
def mock_invoke(requests: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a mock predict result (no AWS required)."""
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
        total = sum(probs)
        probs = [p / total for p in probs]
        predictions.append(
            {
                "symbol": req["symbol"],
                "timeframe": req.get("timeframe", "15m"),
                "prediction": {
                    "label": label,
                    "probabilities": probs,
                    "confidence": max(probs),
                },
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


# ─── Display ──────────────────────────────────────────────────────────────────
def display_training_summary(train_results: Dict[Tuple[str, str], Any]) -> None:
    """Print a summary table of trainer results."""
    if not train_results:
        return

    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"{'Symbol':<18} {'TF':<6} {'Status':<10} {'S3 Key':<35} {'Time (s)'}")
    print("-" * 80)

    for (sym, tf), result in sorted(train_results.items()):
        if isinstance(result, Exception):
            print(f"{sym:<18} {tf:<6} \033[91mFAILED\033[0m    {'—':<35} —")
        else:
            s3_key = result.get("s3_key", "?")
            elapsed = result.get("elapsed_s", "?")
            print(f"{sym:<18} {tf:<6} \033[92mOK\033[0m        {s3_key:<35} {elapsed}")

    print("=" * 80)


def display_results(response: Dict[str, Any]) -> None:
    """Display prediction results in a formatted table."""
    predictions = response.get("predictions", [])
    if not predictions:
        print("No prediction results to display.")
        return

    print("\n" + "=" * 100)
    print("XGBOOST SIGNAL RESULTS")
    print("=" * 100)

    sorted_results = sorted(
        predictions,
        key=lambda x: x.get("prediction", {}).get("confidence", 0),
        reverse=True,
    )

    print(
        f"{'#':<4} {'Symbol':<18} {'TF':<6} {'Signal':<10} "
        f"{'Confidence':<12} {'[Down, Neutral, Up]'}"
    )
    print("-" * 100)

    for idx, result in enumerate(sorted_results, 1):
        # Use _display_symbol (original e.g. "0G/USDT") when present, fall back to symbol
        symbol = result.get("_display_symbol") or result.get("symbol", "?")
        tf = result.get("timeframe", "?")
        pred = result.get("prediction", {})
        signal = pred.get("label", "NEUTRAL")
        confidence = pred.get("confidence", 0.0)
        probs = pred.get("probabilities", [0.0, 0.0, 0.0])

        if signal == "UP":
            signal_disp = f"\033[92m{signal}\033[0m"
        elif signal == "DOWN":
            signal_disp = f"\033[91m{signal}\033[0m"
        else:
            signal_disp = f"\033[93m{signal}\033[0m"

        probs_fmt = f"[{probs[0]:.2f}, {probs[1]:.2f}, {probs[2]:.2f}]"
        print(f"{idx:<4} {symbol:<18} {tf:<6} {signal_disp:<18} {confidence:<12.2%} {probs_fmt}")

    print("=" * 100)

    up = sum(1 for r in predictions if r.get("prediction", {}).get("label") == "UP")
    down = sum(1 for r in predictions if r.get("prediction", {}).get("label") == "DOWN")
    neutral = len(predictions) - up - down
    total = len(predictions)

    print("\nSummary:")
    print(f"  Total Signals: {total}")
    print(f"  UP:      {up:>4} ({up / total * 100:.1f}%)")
    print(f"  DOWN:    {down:>4} ({down / total * 100:.1f}%)")
    print(f"  NEUTRAL: {neutral:>4} ({neutral / total * 100:.1f}%)")

    if "timing" in response:
        t = response["timing"]
        print("\nServer Execution Timing:")
        print(f"  Total Lambda Time: {t.get('total_ms', 0)} ms")
        print(f"  Model Load Time:   {t.get('model_load_ms', 0)} ms")
        print(f"  Feature Calc Time: {t.get('feature_calc_ms', 0)} ms")
        print(f"  Inference Time:    {t.get('inference_ms', 0)} ms")
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Binance Lambda Demo — full pipeline: fetch → auto-train → predict.\n"
            "Missing models are automatically trained via xgboost-trainer Lambda."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Symbol / data args
    parser.add_argument("--symbols", type=int, default=10, help="Number of symbols to process (default: 10)")
    parser.add_argument("--all-symbols", action="store_true", help="Process ALL USDT symbols")
    parser.add_argument("--timeframes", nargs="+", default=["15m", "1h"], help="Timeframes (default: 15m 1h)")
    parser.add_argument("--limit", type=int, default=100, help="Candles per timeframe (min 50, default: 100)")

    # Pipeline control
    parser.add_argument("--mock", action="store_true", help="Use mock results (no AWS needed)")
    parser.add_argument("--skip-train", action="store_true", help="Skip training step — fail if model missing")
    parser.add_argument("--train-only", action="store_true", help="Only train models, skip predict step")

    # Trainer args
    parser.add_argument("--train-fetch-limit", type=int, default=1000, help="Candles to fetch for training (default: 1000)")
    parser.add_argument("--train-workers", type=int, default=4, help="Parallel trainer threads (default: 4)")
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION, help=f"Model version (default: {DEFAULT_MODEL_VERSION})")
    parser.add_argument("--s3-bucket", default=DEFAULT_S3_BUCKET, help=f"S3 bucket for models (default: {DEFAULT_S3_BUCKET})")
    parser.add_argument("--trainer-function", default=DEFAULT_TRAINER_FUNCTION, help=f"Trainer Lambda name (default: {DEFAULT_TRAINER_FUNCTION})")

    # Predict args
    parser.add_argument("--function-name", default=DEFAULT_PREDICT_FUNCTION, help=f"Predict Lambda name (default: {DEFAULT_PREDICT_FUNCTION})")
    parser.add_argument("--region", default=DEFAULT_REGION, help=f"AWS region (default: {DEFAULT_REGION}, from .env)")

    args = parser.parse_args()

    if args.limit < 50:
        logger.error("XGBoost requires at least 50 candles.")
        sys.exit(1)

    try:
        # ── Step 1: Fetch Binance symbols ──────────────────────────────────────
        data_loader = BinanceDataLoader()
        sym_limit = None if args.all_symbols else args.symbols
        symbols = data_loader.get_usdt_symbols(limit=sym_limit)
        if not symbols:
            logger.error("No symbols found")
            sys.exit(1)

        preview = ", ".join(symbols[:10]) + ("..." if len(symbols) > 10 else "")
        logger.info(f"Processing {len(symbols)} symbols: {preview}")

        # ── Step 2: Fetch OHLCV data ───────────────────────────────────────────
        logger.info(f"Fetching OHLCV data for timeframes: {args.timeframes}")
        ohlcv_data = data_loader.get_ohlcv_data(symbols, args.timeframes, limit=args.limit)
        if not ohlcv_data:
            logger.error("No OHLCV data could be fetched")
            sys.exit(1)

        # Build flat list of all (symbol, timeframe) pairs that have data
        available_pairs: List[Tuple[str, str]] = [
            (sym, tf)
            for sym, tf_data in ohlcv_data.items()
            for tf in tf_data
        ]

        # Pre-build S3 key map so predict payload can supply model_s3_key
        # Rust handler (handler.rs) only downloads from S3 when model_s3_key is provided.
        def _s3_key_for(symbol: str, tf: str) -> str:
            return _model_s3_key(symbol, tf, args.model_version)

        # ── Step 3: Auto-train missing models ──────────────────────────────────
        train_results: Dict[Tuple[str, str], Any] = {}

        if not args.mock and not args.skip_train:
            logger.info(f"[trainer] Checking S3 for existing models (bucket: {args.s3_bucket}) ...")
            checker = S3ModelChecker(bucket=args.s3_bucket, region=args.region)
            missing = checker.missing_models(available_pairs, version=args.model_version)

            if missing:
                logger.info(
                    f"[trainer] {len(missing)} model(s) missing on S3 — "
                    f"training now (this may take several minutes per model) ..."
                )
                for sym, tf in missing:
                    logger.info(f"  → needs training: {sym} {tf}")

                trainer = XGBoostTrainerClient(
                    function_name=args.trainer_function,
                    region=args.region,
                    s3_bucket=args.s3_bucket,
                )
                train_results = trainer.train_batch(
                    missing,
                    version=args.model_version,
                    fetch_limit=args.train_fetch_limit,
                    max_workers=args.train_workers,
                )
                display_training_summary(train_results)

                # Check if any training failed — remove those pairs from predict
                failed_pairs: Set[Tuple[str, str]] = {
                    p for p, v in train_results.items() if isinstance(v, Exception)
                }
                if failed_pairs:
                    logger.warning(
                        f"[trainer] {len(failed_pairs)} model(s) failed to train — "
                        f"those pairs will be skipped in predict step."
                    )
                    available_pairs = [p for p in available_pairs if p not in failed_pairs]
            else:
                logger.info("[trainer] All models already exist on S3 — skipping training.")

        elif args.mock:
            logger.info("[trainer] Mock mode — skipping training check.")
        else:
            logger.info("[trainer] --skip-train set — training check skipped.")

        # ── Stop here if --train-only ──────────────────────────────────────────
        if args.train_only:
            logger.info("--train-only mode: skipping predict step.")
            return

        # ── Step 4: Build predict request payloads ─────────────────────────────
        requests = []
        for symbol, tf in available_pairs:
            if symbol in ohlcv_data and tf in ohlcv_data[symbol]:
                # Rust handler (handler.rs:226) uses item.symbol as the /tmp cache filename.
                # Symbols with "/" (e.g. "0G/USDT") produce invalid paths like "/tmp/0G/USDT_...".
                # Pass the normalized form (alphanumeric only) so the cache path is safe.
                norm_sym = _normalize(symbol)
                requests.append(
                    {
                        "symbol": norm_sym,           # e.g. "0GUSDT"  (safe for /tmp)
                        "_display_symbol": symbol,    # original, for local display only
                        "timeframe": tf,
                        "model_version": args.model_version,
                        "timestamp": int(time.time() * 1000),
                        "data": ohlcv_data[symbol][tf],
                        # Rust handler only downloads from S3 when this key is provided
                        "model_s3_key": _s3_key_for(symbol, tf),
                    }
                )

        if not requests:
            logger.error("No prediction requests to send (all models may have failed training).")
            sys.exit(1)

        logger.info(f"Prepared {len(requests)} prediction requests across {len(ohlcv_data)} symbols")

        # ── Step 5: Invoke predict Lambda in batches ───────────────────────────
        BATCH_SIZE = 50
        all_responses: Dict[str, Any] = {
            "predictions": [],
            "success": True,
            "timing": {"total_ms": 0, "model_load_ms": 0, "feature_calc_ms": 0, "inference_ms": 0},
        }

        predict_client: Optional[XGBoostLambdaClient] = None
        if not args.mock:
            predict_client = XGBoostLambdaClient(
                function_name=args.function_name,
                region=args.region,
            )

        t0 = time.time()
        num_batches = (len(requests) - 1) // BATCH_SIZE + 1

        for i in range(0, len(requests), BATCH_SIZE):
            batch = requests[i : i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            logger.info(f"Predict batch {batch_num}/{num_batches} ({len(batch)} items) ...")

            if args.mock:
                response = mock_invoke(batch)
            else:
                response = predict_client.invoke(batch)  # type: ignore[union-attr]

            if response.get("success"):
                all_responses["predictions"].extend(response.get("predictions", []))
                if "timing" in response:
                    for k in all_responses["timing"]:
                        all_responses["timing"][k] += response["timing"].get(k, 0)
            else:
                logger.error(f"Predict batch {batch_num} failed: {response}")

        duration = time.time() - t0

        # ── Step 6: Display results ────────────────────────────────────────────
        display_results(all_responses)

        print("\nOverall Client Performance:")
        print(f"  Total Time:      {duration:.2f}s")
        print(f"  Requests:        {len(requests)}")
        if duration > 0:
            print(f"  Throughput:      {len(requests) / duration:.2f} inferences/s")

        logger.info("Pipeline completed successfully!")

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

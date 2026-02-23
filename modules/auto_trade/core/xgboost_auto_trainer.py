"""Auto-trainer for XGBoost Serverless models.

Triggered automatically by XGBoostServerlessFilter when Lambda returns
"Failed to download model from S3" — meaning no model exists for that symbol.

Flow:
  1. Fetch OHLCV history for the symbol
  2. Compute indicators (IndicatorEngine, add_advanced_features)
  3. Apply directional labels (apply_directional_labels)
  4. Train XGBoost model via train_and_predict()
  5. Save model as XGBoost native JSON (not joblib)
  6. Upload to S3 bucket at the key that Lambda expects
  7. Return the local path so caller can optionally load it

S3 key format:  {SYMBOL_NORMALIZED}_{TIMEFRAME}_{VERSION}.json
  Examples:  BTCUSDT_15m_v1.json  ETHUSDT_15m_v1.json

The trainer runs in a background thread so it never blocks the pipeline.
A per-symbol lock prevents duplicate concurrent training runs.
Results are cached in-memory:
  - "pending"  → training in progress
  - "ready"    → model uploaded to S3; Lambda can serve it on next invocation
  - "failed"   → training failed; won't retry until TTL expires
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from modules.common.ui.logging import log_error, log_info

if TYPE_CHECKING:
    from modules.common.core.data_fetcher import DataFetcher


# ── Constants ─────────────────────────────────────────────────────────────────

# Minimum candles required to train a meaningful model
_MIN_TRAIN_CANDLES = 300

# How many candles to fetch for training (more = better model, slower fetch)
_FETCH_LIMIT = 1_000

# How long (seconds) a "failed" status is cached before a retry is allowed
_FAILURE_TTL = 600  # 10 minutes

# Temp directory for saving models before upload
_TMP_DIR = Path("/tmp")

# Lambda trainer config
_TRAINER_FUNCTION_NAME = os.environ.get("XGBOOST_TRAINER_FUNCTION_NAME", "xgboost-trainer")
_TRAINER_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")


# ── State ─────────────────────────────────────────────────────────────────────

# status: Dict[cache_key → {"status": str, "ts": float, "path": str|None}]
_STATUS: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()
_SYMBOL_LOCKS: Dict[str, threading.Lock] = {}

# S3 check cache: Dict[cache_key → timestamp_checked]
_S3_CHECK_CACHE: Dict[str, float] = {}
_S3_CHECK_TTL = 300  # 5 minutes


def _symbol_lock(cache_key: str) -> threading.Lock:
    with _LOCK:
        if cache_key not in _SYMBOL_LOCKS:
            _SYMBOL_LOCKS[cache_key] = threading.Lock()
        return _SYMBOL_LOCKS[cache_key]


def _set_status(cache_key: str, status: str, path: Optional[str] = None) -> None:
    with _LOCK:
        _STATUS[cache_key] = {"status": status, "ts": time.monotonic(), "path": path}


def get_training_status(cache_key: str) -> Optional[str]:
    """Return 'pending' | 'ready' | 'failed' | None (unknown)."""
    with _LOCK:
        entry = _STATUS.get(cache_key)
    if entry is None:
        return None
    if entry["status"] == "failed" and time.monotonic() - entry["ts"] > _FAILURE_TTL:
        return None  # TTL expired → allow retry
    return entry["status"]


# ── Normalizer (mirrors common.domain.symbols.normalize_symbol_key) ──────────


def _normalize(symbol: str) -> str:
    """Strip separators and uppercase: 'BTC/USDT' → 'BTCUSDT'."""
    return "".join(ch for ch in symbol.upper() if ch.isalnum())


# ── S3 Check Helper ───────────────────────────────────────────────────────────


def _model_exists_in_s3(symbol: str, timeframe: str, version: str, bucket: str) -> bool:
    """Check if model exists in S3 (caches 'not found' result for 5 minutes)."""
    cache_key = f"{_normalize(symbol)}_{timeframe}_{version}"
    
    with _LOCK:
        last_checked = _S3_CHECK_CACHE.get(cache_key, 0)
        if time.monotonic() - last_checked < _S3_CHECK_TTL:
            return False

    try:
        import boto3
        import botocore.exceptions
        s3 = boto3.client("s3", region_name=_TRAINER_REGION)
        s3_key = f"{_normalize(symbol)}_{timeframe}_{version}.json"
        
        s3.head_object(Bucket=bucket, Key=s3_key)
        return True
    except botocore.exceptions.ClientError as exc:
        if exc.response['Error']['Code'] == '404':
            with _LOCK:
                _S3_CHECK_CACHE[cache_key] = time.monotonic()
            return False
        log_error("XGBoostAutoTrainer: [%s] S3 head_object error: %s", symbol, exc)
        return False
    except Exception as exc:
        log_error("XGBoostAutoTrainer: [%s] S3 head_object unexpected error: %s", symbol, exc)
        return False


# ── Core training function ────────────────────────────────────────────────────


def _train_and_upload(
    symbol: str,
    timeframe: str,
    model_version: str,
    s3_bucket: str,
    data_fetcher: "DataFetcher",
    cache_key: str,
) -> None:
    """Train model for *symbol* and upload to S3. Runs in a daemon thread."""
    log_info("XGBoostAutoTrainer: [%s] starting training (timeframe=%s, version=%s)", symbol, timeframe, model_version)
    t0 = time.perf_counter()

    try:
        # ── 1. Import heavy deps lazily ───────────────────────────────────────
        from modules.common.core.indicator_engine import (
            IndicatorConfig,
            IndicatorEngine,
            IndicatorProfile,
        )
        from modules.xgboost_LTS.core.labeling import apply_directional_labels
        from modules.xgboost_LTS.core.model import train_and_predict
        from modules.xgboost_LTS.utils.features import add_advanced_features

        # ── 2. Fetch OHLCV ────────────────────────────────────────────────────
        df = data_fetcher.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            limit=_FETCH_LIMIT,
            check_freshness=False,
        )
        if df is None or df.empty:
            raise ValueError(f"fetch_ohlcv returned empty data for {symbol}")

        if len(df) < _MIN_TRAIN_CANDLES:
            raise ValueError(f"Insufficient candles for {symbol}: {len(df)} < {_MIN_TRAIN_CANDLES}")

        log_info("XGBoostAutoTrainer: [%s] fetched %s candles", symbol, len(df))

        # ── 3. Indicators ─────────────────────────────────────────────────────
        engine = IndicatorEngine(IndicatorConfig.for_profile(IndicatorProfile.XGBOOST))
        result = engine.compute_features(df)
        df = result[0] if isinstance(result, tuple) else result
        if df is None or df.empty:
            raise ValueError(f"Indicator computation returned empty DataFrame for {symbol}")

        df = add_advanced_features(df)
        if df is None or df.empty:
            raise ValueError(f"Advanced feature computation failed for {symbol}")

        # ── 4. Labels ─────────────────────────────────────────────────────────
        df = apply_directional_labels(df, use_cache=False)
        df = df.dropna(subset=["Target"])
        if df.empty:
            raise ValueError(f"No labeled rows after dropna for {symbol}")

        # ── 5. Train ──────────────────────────────────────────────────────────
        model = train_and_predict(df, use_cache=False)
        log_info("XGBoostAutoTrainer: [%s] training done in %.1fs", symbol, time.perf_counter() - t0)

        # ── 6. Save as XGBoost native JSON ────────────────────────────────────
        normalized = _normalize(symbol)
        model_filename = f"{normalized}_{timeframe}_{model_version}.json"
        _TMP_DIR.mkdir(parents=True, exist_ok=True)
        local_path = _TMP_DIR / model_filename

        # Prefer saving via booster directly to avoid sklearn-compat metadata issues
        # in some xgboost/sklearn version combinations (e.g. `_estimator_type` errors).
        saved = False

        get_booster = getattr(model, "get_booster", None)
        if callable(get_booster):
            try:
                booster = get_booster()
                if booster is not None and hasattr(booster, "save_model"):
                    booster.save_model(str(local_path))
                    saved = True
            except Exception:
                saved = False

        if not saved:
            # Fallback 1: sklearn-wrapper save_model
            model_save = getattr(model, "save_model", None)
            if callable(model_save):
                try:
                    model_save(str(local_path))
                    saved = True
                except Exception:
                    saved = False

        if not saved:
            # Fallback 2: raw booster attributes
            booster = getattr(model, "_Booster", None) or getattr(model, "booster_", None)
            if booster is not None and hasattr(booster, "save_model"):
                booster.save_model(str(local_path))
                saved = True

        if not saved:
            raise RuntimeError("Cannot save model: no known save interface found")

        log_info("XGBoostAutoTrainer: [%s] saved model to %s", symbol, local_path)

        # ── 7. Upload to S3 ───────────────────────────────────────────────────
        import boto3

        s3 = boto3.client("s3")
        s3_key = model_filename  # bare key — matches Lambda model_s3_key lookup
        s3.upload_file(
            str(local_path),
            s3_bucket,
            s3_key,
            ExtraArgs={
                "ContentType": "application/json",
                "Metadata": {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "version": model_version,
                    "trained_at": str(int(time.time())),
                },
            },
        )
        elapsed = time.perf_counter() - t0
        log_info(
            "XGBoostAutoTrainer: [%s] uploaded s3://%s/%s in %.1fs total",
            symbol,
            s3_bucket,
            s3_key,
            elapsed,
        )
        _set_status(cache_key, "ready", path=str(local_path))

    except Exception as exc:
        log_error("XGBoostAutoTrainer: [%s] training failed: %s", symbol, exc, exc_info=True)
        _set_status(cache_key, "failed")


def _invoke_lambda_trainer(
    symbol: str,
    timeframe: str,
    model_version: str,
    s3_bucket: str,
    data_fetcher: "DataFetcher",
    cache_key: str,
) -> None:
    """Invoke Lambda trainer asynchronously; fallback to local training on failure."""
    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "model_version": model_version,
        "s3_bucket": s3_bucket,
        "fetch_limit": _FETCH_LIMIT,
    }

    try:
        import boto3

        lambda_client = boto3.client("lambda", region_name=_TRAINER_REGION)
        lambda_client.invoke(
            FunctionName=_TRAINER_FUNCTION_NAME,
            InvocationType="Event",
            Payload=json.dumps(payload).encode("utf-8"),
        )
        log_info(
            "XGBoostAutoTrainer: [%s] invoked Lambda trainer '%s' (region=%s)",
            symbol,
            _TRAINER_FUNCTION_NAME,
            _TRAINER_REGION,
        )
    except Exception as exc:
        log_error(
            "XGBoostAutoTrainer: [%s] Lambda trainer unavailable, fallback to local training: %s",
            symbol,
            exc,
            exc_info=True,
        )
        _train_and_upload(symbol, timeframe, model_version, s3_bucket, data_fetcher, cache_key)


# ── Public API ────────────────────────────────────────────────────────────────


def request_training(
    symbol: str,
    timeframe: str,
    model_version: str,
    s3_bucket: str,
    data_fetcher: "DataFetcher",
) -> str:
    """Trigger background training for *symbol* if not already in progress.

    Returns the current status:
        "pending"  → training just started (or already running)
        "ready"    → a previous run already uploaded the model
        "failed"   → previous run failed; retry allowed after TTL
    """
    cache_key = f"{_normalize(symbol)}_{timeframe}_{model_version}"
    current = get_training_status(cache_key)

    if current == "ready":
        log_info("XGBoostAutoTrainer: [%s] model already uploaded, skipping training", symbol)
        return "ready"

    if current == "pending":
        log_info("XGBoostAutoTrainer: [%s] training already in progress", symbol)
        return "pending"

    # Check S3 before starting new training
    if _model_exists_in_s3(symbol, timeframe, model_version, s3_bucket):
        log_info("XGBoostAutoTrainer: [%s] model found in S3, skipping training", symbol)
        _set_status(cache_key, "ready")
        return "ready"

    # "failed" (expired) or None → start new training
    sym_lock = _symbol_lock(cache_key)
    if not sym_lock.acquire(blocking=False):
        # Another thread just grabbed the lock → already starting
        return "pending"

    try:
        # Double-check after acquiring lock
        current = get_training_status(cache_key)
        if current in ("pending", "ready"):
            return current

        _set_status(cache_key, "pending")
        thread = threading.Thread(
            target=_invoke_lambda_trainer,
            args=(symbol, timeframe, model_version, s3_bucket, data_fetcher, cache_key),
            daemon=True,
            name=f"xgb-train-{cache_key}",
        )
        thread.start()
        log_info(
            "XGBoostAutoTrainer: [%s] background training thread started (key=%s)",
            symbol,
            cache_key,
        )
        return "pending"
    finally:
        sym_lock.release()

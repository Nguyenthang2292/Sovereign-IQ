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

from modules.common.ui.logging import log_error, log_info, log_warn

if TYPE_CHECKING:
    from modules.common.core.data_fetcher import DataFetcher


# ── Constants ─────────────────────────────────────────────────────────────────

# Minimum candles required to train a meaningful model
_MIN_TRAIN_CANDLES = 300

# How many candles to fetch for training (more = better model, slower fetch)
_FETCH_LIMIT = 1_000

# How long (seconds) a "failed" status is cached before a retry is allowed
_FAILURE_TTL = 600  # 10 minutes

# How long (seconds) a "skipped" status (class imbalance) is cached before a retry
# Set to 1 hour — give the market time to generate more diverse price movements
_SKIP_TTL = 3600  # 1 hour

# How long (seconds) an "infra_error" status (AWS IAM/S3 permission) is cached
# No point retrying frequently — fix requires IAM policy change on AWS side
_INFRA_ERROR_TTL = 1800  # 30 minutes

# Model age TTL: retrain if model in S3 is older than this threshold.
# Configurable via XGBOOST_MODEL_AGE_DAYS env var (default: 7 days).
# Set to 0 to disable periodic retraining.
_MODEL_AGE_DAYS: float = float(os.environ.get("XGBOOST_MODEL_AGE_DAYS", "7"))
_MODEL_AGE_TTL: float = _MODEL_AGE_DAYS * 86_400  # convert to seconds

# Temp directory for saving models before upload
_TMP_DIR = Path("/tmp")

# Lambda trainer config
_TRAINER_FUNCTION_NAME = os.environ.get("XGBOOST_TRAINER_FUNCTION_NAME", "xgboost-trainer")
# Prefer AWS_REGION (set in .env); fall back to AWS_DEFAULT_REGION then literal fallback
_TRAINER_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "ap-southeast-1"))


def _boto3_lambda(region: str = _TRAINER_REGION):
    """Create a boto3 Lambda client with explicit env credentials."""
    import boto3
    return boto3.client(
        "lambda",
        region_name=region,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )


def _boto3_s3(region: str = _TRAINER_REGION):
    """Create a boto3 S3 client with explicit env credentials."""
    import boto3
    return boto3.client(
        "s3",
        region_name=region,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )


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
    """Return 'pending' | 'ready' | 'failed' | 'skipped' | 'infra_error' | None (unknown)."""
    with _LOCK:
        entry = _STATUS.get(cache_key)
    if entry is None:
        return None
    elapsed = time.monotonic() - entry["ts"]
    if entry["status"] == "failed" and elapsed > _FAILURE_TTL:
        return None  # TTL expired → allow retry
    if entry["status"] == "skipped" and elapsed > _SKIP_TTL:
        return None  # TTL expired → retry (market may have more class diversity now)
    if entry["status"] == "infra_error" and elapsed > _INFRA_ERROR_TTL:
        return None  # TTL expired → retry in case IAM was fixed
    return entry["status"]


# ── Normalizer (mirrors common.domain.symbols.normalize_symbol_key) ──────────


def _normalize(symbol: str) -> str:
    """Strip separators and uppercase: 'BTC/USDT' → 'BTCUSDT'."""
    return "".join(ch for ch in symbol.upper() if ch.isalnum())


# ── S3 Check Helper ───────────────────────────────────────────────────────────


def _model_exists_in_s3(symbol: str, timeframe: str, version: str, bucket: str) -> bool:
    """Check if a fresh model exists in S3.

    Returns False when:
      - the model key is absent (404)
      - the model is older than _MODEL_AGE_TTL (triggers background retrain)
    Caches negative results for _S3_CHECK_TTL seconds.
    """
    cache_key = f"{_normalize(symbol)}_{timeframe}_{version}"

    with _LOCK:
        last_checked = _S3_CHECK_CACHE.get(cache_key, 0)
        if time.monotonic() - last_checked < _S3_CHECK_TTL:
            return False

    try:
        import botocore.exceptions  # type: ignore[import-untyped]

        s3 = _boto3_s3()
        s3_key = f"{_normalize(symbol)}_{timeframe}_{version}.json"
        head = s3.head_object(Bucket=bucket, Key=s3_key)

        # ── Model Age Check ───────────────────────────────────────────────────
        if _MODEL_AGE_TTL > 0:
            trained_at_str = head.get("Metadata", {}).get("trained_at", "")
            if trained_at_str:
                try:
                    trained_at_ts = float(trained_at_str)
                    age_seconds = time.time() - trained_at_ts
                    if age_seconds > _MODEL_AGE_TTL:
                        age_days = age_seconds / 86_400
                        log_warn(
                            "XGBoostAutoTrainer: [%s] model is %.1f days old (limit=%.1f days) — "
                            "will schedule background retrain",
                            symbol, age_days, _MODEL_AGE_DAYS,
                        )
                        # Clear in-memory "ready" status so request_training() re-evaluates
                        with _LOCK:
                            _STATUS.pop(cache_key, None)
                            _S3_CHECK_CACHE.pop(cache_key, None)
                        return False  # triggers retrain path in request_training()
                except (ValueError, TypeError):
                    pass  # malformed metadata → treat as fresh

        return True

    except botocore.exceptions.ClientError as exc:
        if exc.response["Error"]["Code"] == "404":
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
        from modules.xgboost_LTS.core.model import ClassDiversityError, train_and_predict
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
        try:
            model = train_and_predict(df, use_cache=False)
        except ClassDiversityError as cde:
            log_warn(
                "XGBoostAutoTrainer: [%s] skipping training — insufficient class diversity: %s",
                symbol,
                cde,
            )
            _set_status(cache_key, "skipped")
            return
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
                    booster.save_model(str(local_path))  # type: ignore[attr-defined]
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
                booster.save_model(str(local_path))  # type: ignore[attr-defined]
                saved = True

        if not saved:
            raise RuntimeError("Cannot save model: no known save interface found")

        log_info("XGBoostAutoTrainer: [%s] saved model to %s", symbol, local_path)

        # ── 7. Upload to S3 ───────────────────────────────────────────────────
        s3 = _boto3_s3()
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
        lambda_client = _boto3_lambda()
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

    if current == "skipped":
        log_info(
            "XGBoostAutoTrainer: [%s] previously skipped due to class imbalance, will retry after TTL expires",
            symbol,
        )
        return "skipped"

    if current == "infra_error":
        log_warn(
            "XGBoostAutoTrainer: [%s] previously failed due to AWS infrastructure error (IAM/S3). "
            "Fix the IAM policy for the Lambda role and retry will happen automatically after TTL.",
            symbol,
        )
        return "infra_error"

    # Check S3 before starting new training.
    # If model is absent OR stale (older than _MODEL_AGE_TTL), _model_exists_in_s3 returns False.
    if _model_exists_in_s3(symbol, timeframe, model_version, s3_bucket):
        log_info("XGBoostAutoTrainer: [%s] model found in S3 (fresh), skipping training", symbol)
        _set_status(cache_key, "ready")
        return "ready"

    # Determine log message: new model vs stale refresh
    stale_msg = f" (model older than {_MODEL_AGE_DAYS:.0f} days — refreshing)" if _MODEL_AGE_TTL > 0 else ""
    log_info("XGBoostAutoTrainer: [%s] scheduling background training%s", symbol, stale_msg)
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


def train_model_sync(
    symbol: str,
    timeframe: str,
    model_version: str,
    s3_bucket: str,
    data_fetcher: "DataFetcher",
    wait_timeout_seconds: int = 120,
) -> str:
    """Train model synchronously and return final status.

    Strict mode contract:
      1) Trigger trainer Lambda with RequestResponse (or local fallback)
      2) Wait until model is visible in S3
      3) Return "ready" only when S3 key is confirmed

    Returns:
      "ready" | "failed"
    """
    cache_key = f"{_normalize(symbol)}_{timeframe}_{model_version}"

    if _model_exists_in_s3(symbol, timeframe, model_version, s3_bucket):
        _set_status(cache_key, "ready")
        return "ready"

    sym_lock = _symbol_lock(cache_key)
    with sym_lock:
        if _model_exists_in_s3(symbol, timeframe, model_version, s3_bucket):
            _set_status(cache_key, "ready")
            return "ready"

        _set_status(cache_key, "pending")
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "model_version": model_version,
            "s3_bucket": s3_bucket,
            "fetch_limit": _FETCH_LIMIT,
        }

        try:
            lambda_client = _boto3_lambda()
            response = lambda_client.invoke(
                FunctionName=_TRAINER_FUNCTION_NAME,
                InvocationType="RequestResponse",
                Payload=json.dumps(payload).encode("utf-8"),
            )

            if response.get("FunctionError"):
                body = response.get("Payload").read().decode("utf-8") if response.get("Payload") else ""
                _body_lower = body.lower()
                # Detect class imbalance errors from Lambda — skip instead of fallback local training
                if (
                    "missing classes" in _body_lower
                    or "class diversity" in _body_lower
                    or "biased predictions" in _body_lower
                ):
                    log_warn(
                        "XGBoostAutoTrainer: [%s] Lambda reports insufficient class diversity — skipping training. Details: %s",
                        symbol,
                        body,
                    )
                    _set_status(cache_key, "skipped")
                    return "failed"
                # Detect AWS infrastructure errors (IAM/S3) — no point doing local fallback
                if (
                    "accessdenied" in _body_lower
                    or "putobject" in _body_lower
                    or "getobject" in _body_lower
                    or ("s3" in _body_lower and "not authorized" in _body_lower)
                ):
                    log_error(
                        "XGBoostAutoTrainer: [%s] Lambda IAM/S3 permission error — skipping local fallback. "
                        "Action required: add s3:PutObject to Lambda role policy for bucket 'xgboost-models-store'. "
                        "Details: %s",
                        symbol,
                        body,
                    )
                    _set_status(cache_key, "infra_error")
                    return "failed"
                raise RuntimeError(f"Trainer Lambda function error: {body}")

            status_code = int(response.get("StatusCode", 0))
            if status_code != 200:
                body = response.get("Payload").read().decode("utf-8") if response.get("Payload") else ""
                raise RuntimeError(f"Trainer Lambda failed with status {status_code}: {body}")

            log_info(
                "XGBoostAutoTrainer: [%s] synchronous trainer invoke completed (function=%s)",
                symbol,
                _TRAINER_FUNCTION_NAME,
            )
        except Exception as exc:
            exc_msg = str(exc).lower()
            # Detect class imbalance before trying expensive local fallback
            if "missing classes" in exc_msg or "class diversity" in exc_msg or "biased predictions" in exc_msg:
                log_warn(
                    "XGBoostAutoTrainer: [%s] skipping local fallback — class imbalance detected: %s",
                    symbol,
                    exc,
                )
                _set_status(cache_key, "skipped")
                return "failed"
            # Detect AWS infrastructure errors — local training won't help (S3 upload will also fail)
            if (
                "accessdenied" in exc_msg
                or ("putobject" in exc_msg)
                or ("s3" in exc_msg and "not authorized" in exc_msg)
            ):
                log_error(
                    "XGBoostAutoTrainer: [%s] AWS IAM/S3 permission error — skipping local fallback. "
                    "Fix: add s3:PutObject to Lambda role 'xgboost-trainer' for bucket 'xgboost-models-store'. "
                    "Error: %s",
                    symbol,
                    exc,
                )
                _set_status(cache_key, "infra_error")
                return "failed"
            log_error(
                "XGBoostAutoTrainer: [%s] sync Lambda trainer failed, fallback local training: %s",
                symbol,
                exc,
                exc_info=True,
            )
            try:
                _train_and_upload(symbol, timeframe, model_version, s3_bucket, data_fetcher, cache_key)
            except Exception:
                _set_status(cache_key, "failed")
                return "failed"

        deadline = time.monotonic() + max(wait_timeout_seconds, 1)
        while time.monotonic() < deadline:
            if _model_exists_in_s3(symbol, timeframe, model_version, s3_bucket):
                _set_status(cache_key, "ready")
                return "ready"
            time.sleep(2)

        log_error(
            "XGBoostAutoTrainer: [%s] timeout waiting model on S3 after sync training",
            symbol,
        )
        _set_status(cache_key, "failed")
        return "failed"

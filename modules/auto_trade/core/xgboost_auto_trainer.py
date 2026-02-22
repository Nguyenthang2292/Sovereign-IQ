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


# ── State ─────────────────────────────────────────────────────────────────────

# status: Dict[cache_key → {"status": str, "ts": float, "path": str|None}]
_STATUS: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()
_SYMBOL_LOCKS: Dict[str, threading.Lock] = {}


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

        # model is an XGBClassifier (sklearn wrapper) — use save_model for native format
        if hasattr(model, "save_model"):
            model.save_model(str(local_path))
        elif hasattr(model, "get_booster"):
            model.get_booster().save_model(str(local_path))
        else:
            # Fallback: save via the underlying booster attribute
            booster = getattr(model, "_Booster", None) or getattr(model, "booster_", None)
            if booster is not None:
                booster.save_model(str(local_path))
            else:
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
        log_error("XGBoostAutoTrainer: [%s] training failed: %s", symbol, exc)
        _set_status(cache_key, "failed")


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
            target=_train_and_upload,
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

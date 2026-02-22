"""XGBoost Serverless filter adapter for Auto Trade pipeline.

Delegates XGBoost inference to the AWS Lambda backend instead of running
a local model.  Implements the same ``filter_signals()`` interface as
``XGBoostFilter`` and ``XGBoostPerSymbolFilter`` so it is a transparent
drop-in replacement in the ``SignalPipeline``.

Architecture
------------
The XGBoost Lambda is **synchronous** — it returns predictions directly
in the invoke response (unlike ATC Serverless which uses SQS).

Fallback Strategy
-----------------
If the Lambda client fails to initialise (no AWS credentials, wrong region,
network issue) OR a runtime invoke call fails, the filter **automatically
falls back to the local pre-trained XGBoostFilter** (from ``xgboost_LTS``).

  Lambda OK         → predictions from AWS Lambda
  Lambda init fail  → local XGBoostFilter (pre-trained .json/.joblib model)
  Lambda call fail  → local XGBoostFilter
  No local model    → pass signals through unchanged (pipeline not blocked)

Flow:
  1. Receive ATC signals (LONG/SHORT)
  2. Fetch OHLCV for each symbol
  3. Build batch request and invoke Lambda
  4. Keep signals where Lambda direction confirms ATC direction (UP→LONG, DOWN→SHORT)
  5. Attach xgboost_conf, xgboost_dir, xgboost_backend to signal details
"""

from __future__ import annotations

import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, cast

import pandas as pd

from modules.auto_trade.core.atc_scanner import SignalResult
from modules.auto_trade.core.xgboost_auto_trainer import get_training_status, request_training
from modules.common.core.data_fetcher import DataFetcher
from modules.common.domain.symbols import normalize_symbol_key
from modules.common.ui.logging import (
    log_debug,
    log_error,
    log_info,
    log_warn,
)

# ── Default constants ────────────────────────────────────────────────────────

DEFAULT_FUNCTION_NAME = "xgboost-serverless-predict"
DEFAULT_REGION = "us-east-1"
DEFAULT_MODEL_VERSION = "v1"
DEFAULT_TIMEFRAME = "15m"
DEFAULT_CANDLE_LIMIT = 200
DEFAULT_MIN_CONFIDENCE = 0.55
DEFAULT_MIN_CANDLES = 50


# ── Mock helper (for unit tests / dry-run only) ──────────────────────────────


def _mock_predict(requests: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a realistic mock response without hitting AWS.

    Used only in mock_mode=True (unit tests / dry-run).
    NOT used as a fallback in production — local XGBoostFilter is.
    """
    predictions = []
    for req in requests:
        probs = [random.random() for _ in range(3)]
        total = sum(probs)
        probs = [p / total for p in probs]
        idx = probs.index(max(probs))
        label = ["DOWN", "NEUTRAL", "UP"][idx]
        predictions.append(
            {
                "symbol": req.get("symbol", ""),
                "timeframe": req.get("timeframe", DEFAULT_TIMEFRAME),
                "prediction": {
                    "label": label,
                    "probabilities": probs,
                    "confidence": max(probs),
                },
                "metadata": {
                    "candles_processed": len(req.get("data", {}).get("close", [])),
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


# ── Lambda client ────────────────────────────────────────────────────────────


class XGBoostLambdaClient:
    """Thin wrapper around boto3 Lambda invoke for XGBoost predictions."""

    def __init__(
        self,
        function_name: str = DEFAULT_FUNCTION_NAME,
        region: str = DEFAULT_REGION,
    ) -> None:
        import boto3

        self.function_name = function_name
        self.region = region
        self._lambda = boto3.client("lambda", region_name=region)
        log_info(f"XGBoostLambdaClient: function='{function_name}', region={region}")

    def predict(self, requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Invoke Lambda with a batch of prediction requests."""
        payload = {
            "version": "1.0",
            "mode": "batch" if len(requests) > 1 else "single",
            "requests": requests,
            "options": {"return_features": False},
        }

        log_info(f"XGBoostLambdaClient: invoking '{self.function_name}' with {len(requests)} request(s)...")

        try:
            response = self._lambda.invoke(
                FunctionName=self.function_name,
                InvocationType="RequestResponse",
                Payload=json.dumps(payload).encode("utf-8"),
            )
        except Exception as exc:
            raise RuntimeError(f"Lambda invoke failed: {exc}") from exc

        status = response["StatusCode"]
        raw_body = response["Payload"].read().decode("utf-8")

        if response.get("FunctionError"):
            detail: dict = {}
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
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Failed to parse Lambda response JSON: {exc}\nRaw: {raw_body}") from exc

        if not body.get("success"):
            raise RuntimeError(f"Lambda returned unsuccessful response: {body}")

        return body


# ── Local model discovery ─────────────────────────────────────────────────────


def _find_local_model() -> Optional[str]:
    """Search standard locations for a pre-trained XGBoost model file.

    Returns the path of the most recently modified model found, or None.
    """
    search_roots = [
        Path("models"),
        Path("modules/auto_trade/models"),
        Path("modules/xgboost_LTS/models"),
    ]
    for root in search_roots:
        for pattern in ("*.json", "*.ubj", "*.joblib"):
            if root.exists():
                matches = list(root.glob(pattern))
                if matches:
                    return str(max(matches, key=lambda p: p.stat().st_mtime))
    return None


# ── Serverless filter ─────────────────────────────────────────────────────────


class XGBoostServerlessFilter:
    """Filters signals using a remote XGBoost Lambda instead of a local model.

    Drop-in replacement for ``XGBoostFilter`` / ``XGBoostPerSymbolFilter``
    in the ``SignalPipeline``.

    **Automatic fallback:** when Lambda is unavailable (init or runtime),
    automatically delegates to the local ``XGBoostFilter`` (pre-trained model).
    If no local model exists, signals pass through unchanged.

    Config keys (all optional):
        xgboost_serverless_function_name    Lambda function name
        xgboost_serverless_region           AWS region
        xgboost_serverless_model_version    Model version tag for S3
        xgboost_serverless_timeframe        OHLCV timeframe to send
        xgboost_serverless_candle_limit     Number of candles to fetch
        xgboost_serverless_min_confidence   Minimum prediction confidence
        xgboost_serverless_min_candles      Minimum candles required
        xgboost_serverless_mock_mode        True → use random mock (tests only)
        xgboost_serverless_local_model_path Override path for local fallback model
        xgboost_serverless_s3_bucket        S3 bucket holding models (default: xgboost-models-store)
        xgboost_serverless_auto_train       True → auto-train missing models (default: True)
    """

    def __init__(
        self,
        data_fetcher: DataFetcher,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.data_fetcher = data_fetcher
        self.config = config or {}

        self.function_name: str = str(self.config.get("xgboost_serverless_function_name", DEFAULT_FUNCTION_NAME))
        self.region: str = str(self.config.get("xgboost_serverless_region", DEFAULT_REGION))
        self.model_version: str = str(self.config.get("xgboost_serverless_model_version", DEFAULT_MODEL_VERSION))
        self.timeframe: str = str(self.config.get("xgboost_serverless_timeframe", DEFAULT_TIMEFRAME))
        self.candle_limit: int = int(self.config.get("xgboost_serverless_candle_limit", DEFAULT_CANDLE_LIMIT))
        self.min_confidence: float = float(self.config.get("xgboost_serverless_min_confidence", DEFAULT_MIN_CONFIDENCE))
        self.min_candles: int = int(self.config.get("xgboost_serverless_min_candles", DEFAULT_MIN_CANDLES))
        self.s3_bucket: str = str(self.config.get("xgboost_serverless_s3_bucket", "xgboost-models-store"))
        self.auto_train: bool = bool(self.config.get("xgboost_serverless_auto_train", True))

        # Test-only mock mode — bypasses both Lambda AND local fallback
        self.mock_mode: bool = bool(self.config.get("xgboost_serverless_mock_mode", False))

        # ── Lambda client ────────────────────────────────────────────────────
        self._lambda_client: Optional[XGBoostLambdaClient] = None
        self._lambda_available: bool = False

        if self.mock_mode:
            log_info("XGBoostServerlessFilter: MOCK mode (unit test / dry-run)")
        else:
            try:
                self._lambda_client = XGBoostLambdaClient(
                    function_name=self.function_name,
                    region=self.region,
                )
                self._lambda_available = True
            except Exception as exc:
                log_error(f"XGBoostServerlessFilter: Lambda client init failed: {exc}")
                log_warn("XGBoostServerlessFilter: Lambda unavailable — will fall back to local XGBoostFilter")

        # ── Local fallback filter ────────────────────────────────────────────
        # Always built in non-mock mode so it is ready if Lambda fails later
        self._local_fallback: Optional[Any] = None
        if not self.mock_mode:
            self._local_fallback = self._build_local_fallback()

        mode_label = "MOCK" if self.mock_mode else ("LIVE" if self._lambda_available else "LOCAL_FALLBACK")
        log_info(
            f"XGBoostServerlessFilter ready ({mode_label}, "
            f"function={self.function_name}, region={self.region}, "
            f"model={self.model_version}, timeframe={self.timeframe}, "
            f"min_conf={self.min_confidence:.2f})"
        )

    # ── Local fallback builder ────────────────────────────────────────────────

    def _build_local_fallback(self) -> Optional[Any]:
        """Attempt to build a local XGBoostFilter as a fallback.

        Returns the filter instance, or None if no model is available.
        """
        try:
            from config import XGBOOST_FILTER_DEFAULTS
            from modules.auto_trade.core.xgboost_filter import XGBoostFilter, XGBoostFilterConfig

            # Honour explicit override first, then auto-discover
            model_path_str: str = str(self.config.get("xgboost_serverless_local_model_path", ""))
            model_path: Optional[str] = model_path_str if model_path_str and Path(model_path_str).exists() else None

            if not model_path:
                model_path = _find_local_model()

            if not model_path:
                log_warn("XGBoostServerlessFilter: no local model found — fallback will pass signals through unchanged")
                return None

            fallback_cfg = dict(XGBOOST_FILTER_DEFAULTS)
            fallback_cfg["require_model"] = False  # non-fatal if load fails
            fallback_cfg["prediction_timeframe"] = self.timeframe
            fallback_cfg["min_confidence"] = self.min_confidence

            flt = XGBoostFilter(
                data_fetcher=self.data_fetcher,
                model_path=model_path,
                config=cast(XGBoostFilterConfig, fallback_cfg),
            )
            log_info(f"XGBoostServerlessFilter: local fallback ready (model={Path(model_path).name})")
            return flt
        except Exception as exc:
            log_error(f"XGBoostServerlessFilter: could not build local fallback: {exc}")
            return None

    # ── Public API (XGBoostFilterLike protocol) ───────────────────────────────

    def filter_signals(self, signals: List[SignalResult]) -> List[SignalResult]:
        """Filter signals using XGBoost Lambda predictions.

        When Lambda fails because a model doesn't exist in S3, automatically
        triggers background training (via xgboost_auto_trainer) for those symbols.
        Signals whose model is still training are passed through unchanged.
        On the next pipeline cycle the model will be available in S3.

        Args:
            signals: List of SignalResult from ATC Scanner.

        Returns:
            Filtered list of SignalResult with xgboost metadata attached.
        """
        if not signals:
            return []

        # ── Mock mode (unit tests / dry-run) ─────────────────────────────────
        if self.mock_mode:
            log_info(f"XGBoostServerlessFilter [MOCK]: filtering {len(signals)} signal(s)")
            req_items = self._build_requests(signals)[0]  # items only
            return self._parse_predictions(
                req_items, self._build_requests(signals)[1], signals, _mock_predict(req_items)
            )

        # ── Lambda unavailable at init → go to local fallback immediately ─────
        if not self._lambda_available:
            return self._use_local_fallback(signals, "Lambda not available at init")

        # ── Fetch OHLCV + build request payload ───────────────────────────────
        request_items, signal_map = self._build_requests(signals)

        if not request_items:
            log_warn("XGBoostServerlessFilter: no valid OHLCV data, falling back to local")
            return self._use_local_fallback(signals, "no OHLCV data")

        # ── Invoke Lambda ─────────────────────────────────────────────────────
        try:
            t0 = time.perf_counter()
            assert self._lambda_client is not None
            response = self._lambda_client.predict(request_items)
            log_info(f"XGBoostServerlessFilter: Lambda returned in {time.perf_counter() - t0:.2f}s")
            return self._parse_predictions(request_items, signal_map, signals, response)

        except Exception as exc:
            exc_str = str(exc)
            # ── Detect "no model in S3" error and auto-train ──────────────────
            if self.auto_train and "Failed to download model from S3" in exc_str:
                return self._handle_missing_models(signals, request_items, signal_map, exc_str)
            # ── Other Lambda errors → local fallback ──────────────────────────
            log_error(f"XGBoostServerlessFilter: Lambda call failed: {exc}")
            return self._use_local_fallback(signals, exc_str)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _use_local_fallback(self, signals: List[SignalResult], reason: str = "") -> List[SignalResult]:
        """Route signals to the local XGBoostFilter fallback.

        If no local fallback is configured, passes all signals through so
        the pipeline is never hard-blocked.
        """
        if self._local_fallback is not None:
            log_warn(f"XGBoostServerlessFilter: switching to LOCAL fallback ({reason or 'Lambda unavailable'})")
            return self._local_fallback.filter_signals(signals)

        # No local model at all → pass-through
        log_warn(
            f"XGBoostServerlessFilter: no local fallback available "
            f"({reason}), passing {len(signals)} signal(s) through unchanged"
        )
        return signals

    def _handle_missing_models(
        self,
        signals: List[SignalResult],
        request_items: List[Dict[str, Any]],
        signal_map: Dict[str, SignalResult],
        exc_str: str,
    ) -> List[SignalResult]:
        """Handle Lambda failure due to missing S3 model(s).

        Strategy per symbol:
          - If model is already training → pass signal through unchanged (non-blocking)
          - If model just became ready  → retry Lambda for that symbol alone
          - Otherwise                  → start background training, pass through for now
        """
        log_warn(
            "XGBoostServerlessFilter: Lambda S3 model missing — checking %s symbol(s) for auto-training",
            len(request_items),
        )

        ready_items: List[Dict[str, Any]] = []
        passthrough_signals: List[SignalResult] = []

        for item in request_items:
            sym_normalized = item["symbol"]
            # Reverse-lookup original SignalResult
            signal = signal_map.get(sym_normalized)
            if signal is None:
                continue

            cache_key = f"{sym_normalized}_{self.timeframe}_{self.model_version}"
            status = get_training_status(cache_key)

            if status == "ready":
                # Model just finished training on a previous cycle → retry Lambda
                log_info(
                    "XGBoostServerlessFilter: [%s] model now ready in S3, will retry Lambda",
                    signal.symbol,
                )
                ready_items.append(item)
            elif status == "pending":
                log_info(
                    "XGBoostServerlessFilter: [%s] training in progress — passing signal through",
                    signal.symbol,
                )
                passthrough_signals.append(signal)
            else:
                # Not started or failed+TTL expired → kick off training
                status = request_training(
                    symbol=signal.symbol,
                    timeframe=self.timeframe,
                    model_version=self.model_version,
                    s3_bucket=self.s3_bucket,
                    data_fetcher=self.data_fetcher,
                )
                log_warn(
                    "XGBoostServerlessFilter: [%s] auto-training triggered (status=%s) — "
                    "passing signal through this cycle",
                    signal.symbol,
                    status,
                )
                passthrough_signals.append(signal)

        # ── Retry Lambda for symbols whose models are now in S3 ───────────────
        confirmed: List[SignalResult] = []
        if ready_items:
            try:
                assert self._lambda_client is not None
                retry_response = self._lambda_client.predict(ready_items)
                ready_signal_map = {item["symbol"]: signal_map[item["symbol"]] for item in ready_items}
                confirmed = self._parse_predictions(
                    ready_items, ready_signal_map, list(ready_signal_map.values()), retry_response
                )
                log_info(
                    "XGBoostServerlessFilter: Lambda retry confirmed %s/%s signals",
                    len(confirmed),
                    len(ready_items),
                )
            except Exception as retry_exc:
                log_warn(
                    "XGBoostServerlessFilter: Lambda retry also failed (%s) — passing ready signals through",
                    retry_exc,
                )
                passthrough_signals.extend(signal_map[i["symbol"]] for i in ready_items if i["symbol"] in signal_map)

        result = confirmed + passthrough_signals
        log_info(
            "XGBoostServerlessFilter: _handle_missing_models → %s confirmed + %s pass-through = %s total",
            len(confirmed),
            len(passthrough_signals),
            len(result),
        )
        return result

    def _build_requests(self, signals: List[SignalResult]) -> tuple:
        """Fetch OHLCV and build Lambda request payloads.

        Returns:
            (request_items: List[dict], signal_map: Dict[normalized_sym → SignalResult])
        """
        request_items: List[Dict[str, Any]] = []
        signal_map: Dict[str, SignalResult] = {}

        for signal in signals:
            sym = signal.symbol
            ohlcv = self._fetch_ohlcv(sym)
            if ohlcv is None:
                log_warn(f"XGBoostServerlessFilter: skipping {sym} (no OHLCV data)")
                continue

            normalized = normalize_symbol_key(sym)
            s3_key = f"{normalized}_{self.timeframe}_{self.model_version}.json"
            request_items.append(
                {
                    "symbol": normalized,
                    "timeframe": self.timeframe,
                    "model_version": self.model_version,
                    "model_s3_key": s3_key,
                    "data": ohlcv,
                }
            )
            signal_map[normalized] = signal

        return request_items, signal_map

    def _parse_predictions(
        self,
        request_items: List[Dict[str, Any]],
        signal_map: Dict[str, SignalResult],
        original_signals: List[SignalResult],
        response: Dict[str, Any],
    ) -> List[SignalResult]:
        """Parse Lambda/mock response and apply direction + confidence filtering."""
        pred_map: Dict[str, Dict[str, Any]] = {str(p.get("symbol", "")): p for p in response.get("predictions", [])}

        filtered: List[SignalResult] = []
        for item in request_items:
            norm = item["symbol"]
            signal = signal_map.get(norm)
            if signal is None:
                continue

            pred = pred_map.get(norm)
            if pred is None:
                log_debug(f"XGBoostServerlessFilter: no prediction for {norm}, skipping")
                continue

            prediction = pred.get("prediction", {})
            label = str(prediction.get("label", "NEUTRAL")).upper()
            confidence = float(prediction.get("confidence", 0.0))

            xgboost_dir = "LONG" if label == "UP" else ("SHORT" if label == "DOWN" else "NEUTRAL")

            if confidence < self.min_confidence:
                log_debug(
                    f"XGBoostServerlessFilter: {signal.symbol} "
                    f"conf {confidence:.2f} < {self.min_confidence:.2f}, skipping"
                )
                continue

            if xgboost_dir != signal.signal_type:
                log_debug(
                    f"XGBoostServerlessFilter: {signal.symbol} direction mismatch "
                    f"(ATC={signal.signal_type}, Lambda={xgboost_dir}), skipping"
                )
                continue

            new_details = {
                **signal.details,
                "xgboost_conf": round(confidence, 4),
                "xgboost_dir": xgboost_dir,
                "xgboost_backend": "serverless",
                "xgboost_label": label,
            }
            probs = prediction.get("probabilities", [])
            if probs:
                new_details["xgboost_probabilities"] = [round(p, 4) for p in probs]

            filtered.append(signal._replace(details=new_details))

        log_info(f"XGBoostServerlessFilter: {len(filtered)}/{len(original_signals)} signals confirmed by Lambda")
        return filtered

    # ── OHLCV fetching ────────────────────────────────────────────────────────

    def _fetch_ohlcv(self, symbol: str) -> Optional[Dict[str, list]]:
        """Fetch OHLCV data for a symbol and return as a dict for Lambda."""
        try:
            df = self.data_fetcher.fetch_ohlcv(
                symbol=symbol,
                timeframe=self.timeframe,
                limit=self.candle_limit,
                check_freshness=False,
            )
        except Exception as exc:
            log_warn(f"XGBoostServerlessFilter: fetch error {symbol} {self.timeframe}: {exc}")
            return None

        if df is None or df.empty:
            return None

        if len(df) < self.min_candles:
            log_debug(f"XGBoostServerlessFilter: {symbol} only {len(df)} candles (need {self.min_candles})")
            return None

        timestamp_values = self._extract_timestamps(df)
        open_values = self._extract_numeric(df, ("open", "Open"))
        high_values = self._extract_numeric(df, ("high", "High"))
        low_values = self._extract_numeric(df, ("low", "Low"))
        close_values = self._extract_numeric(df, ("close", "Close"))
        volume_values = self._extract_numeric(df, ("volume", "Volume"))

        if not all((timestamp_values, open_values, high_values, low_values, close_values, volume_values)):
            return None

        min_len = min(
            len(timestamp_values),
            len(open_values),
            len(high_values),
            len(low_values),
            len(close_values),
            len(volume_values),
        )
        if min_len < self.min_candles:
            return None

        return {
            "timestamp": [int(v) for v in timestamp_values[-min_len:]],
            "open": [float(v) for v in open_values[-min_len:]],
            "high": [float(v) for v in high_values[-min_len:]],
            "low": [float(v) for v in low_values[-min_len:]],
            "close": [float(v) for v in close_values[-min_len:]],
            "volume": [float(v) for v in volume_values[-min_len:]],
        }

    def _extract_numeric(self, df: pd.DataFrame, candidates: Sequence[str]) -> List[float]:
        for name in candidates:
            if name in df.columns:
                return pd.to_numeric(df[name], errors="coerce").dropna().astype(float).tolist()
        return []

    def _extract_timestamps(self, df: pd.DataFrame) -> List[int]:
        for col in ("timestamp", "time", "datetime", "date", "Date"):
            if col in df.columns:
                return self._to_unix_ms(df[col].tolist())
        if isinstance(df.index, pd.DatetimeIndex):
            return [int(ts.timestamp() * 1000) for ts in df.index.to_pydatetime()]
        return []

    def _to_unix_ms(self, values: Sequence[Any]) -> List[int]:
        result: List[int] = []
        for v in values:
            ts = self._normalize_timestamp(v)
            if ts is not None:
                result.append(ts)
        return result

    def _normalize_timestamp(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return int(value.timestamp() * 1000)
        if hasattr(value, "to_pydatetime"):
            try:
                return int(value.to_pydatetime().timestamp() * 1000)
            except Exception:
                pass
        try:
            numeric = float(value)
            return int(numeric) if numeric > 1e12 else int(numeric * 1000)
        except (TypeError, ValueError):
            pass
        try:
            parsed = pd.to_datetime(value, utc=True)
            if pd.isna(parsed):
                return None
            return int(parsed.timestamp() * 1000)
        except Exception:
            return None

    # ── Backward-compat helpers ───────────────────────────────────────────────

    @staticmethod
    def _passthrough_all(signals: List[SignalResult]) -> List[SignalResult]:
        """Kept for backward compatibility with tests that patch this.

        In production, prefer _use_local_fallback().
        """
        log_warn(f"XGBoostServerlessFilter: passing through {len(signals)} signals (no fallback)")
        return list(signals)

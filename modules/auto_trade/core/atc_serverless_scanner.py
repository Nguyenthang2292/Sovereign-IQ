"""ATC Serverless scanner adapter for Auto Trade pipeline."""

from __future__ import annotations

from modules.common.ui.logging import log_info, log_error, log_warn, log_debug, log_success, log_system
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import pandas as pd

from modules.adaptive_trend_LTS_serverless import DEFAULT_ATC_CONFIG, ATCLambdaClient
from modules.auto_trade.core.atc_scanner import SignalResult
from modules.common.core.data_fetcher import DataFetcher
from modules.common.domain.symbol_codec import SymbolCodec

_SYMBOL_CODEC = SymbolCodec()


class ATCServerlessScanner:
    """Scanner adapter that delegates ATC computation to AWS Lambda serverless backend."""

    def __init__(self, data_fetcher: DataFetcher, config: Optional[Dict[str, Any]] = None) -> None:
        self.data_fetcher = data_fetcher
        self.config = config or {}
        self.timeframes: List[str] = cast(List[str], self.config.get("timeframes", ["15m", "1h", "4h"]))
        self.threshold: float = float(self.config.get("threshold", 0.6))
        self.ohlcv_limit: int = int(self.config.get("serverless_ohlcv_limit", 220))
        self.min_candles_per_tf: int = int(self.config.get("serverless_min_candles_per_tf", 50))

        self.lambda_client = ATCLambdaClient(
            function_name=str(self.config.get("serverless_function_name", "atc-serverless")),
            sqs_queue_name=str(self.config.get("serverless_sqs_queue", "atc-results")),
            region=str(self.config.get("serverless_region", "us-east-1")),
            sqs_poll_timeout=int(self.config.get("serverless_sqs_poll_timeout", 60)),
            sqs_poll_interval=float(self.config.get("serverless_sqs_poll_interval", 2.0)),
            mock_mode=bool(self.config.get("serverless_mock_mode", False)),
        )

    def scan_symbols(self, symbols: List[str]) -> List[SignalResult]:
        if not symbols:
            return []

        symbols_data: List[Dict[str, Any]] = []
        symbol_map: Dict[str, str] = {}

        for symbol in symbols:
            payload = self._build_symbol_payload(symbol)
            if payload is None:
                continue
            normalized = payload["symbol"]
            symbol_map[normalized] = symbol
            symbols_data.append(payload)

        if not symbols_data:
            log_warn("ATCServerlessScanner: no valid symbols payload, skipping serverless scan")
            return []

        lambda_config = self._build_lambda_config()

        try:
            response = self.lambda_client.invoke(symbols_data=symbols_data, config=lambda_config)
        except Exception as exc:
            log_error("ATCServerlessScanner: Lambda invoke failed: %s", exc)
            return []

        raw_results = cast(List[Dict[str, Any]], response.get("results", []))
        raw_errors = cast(List[Dict[str, Any]], response.get("errors", []))

        if raw_errors:
            log_warn("ATCServerlessScanner: %s symbol errors from serverless backend", len(raw_errors))

        signals: List[SignalResult] = []
        for item in raw_results:
            signal_type = str(item.get("signal_type", "NEUTRAL")).upper()
            if signal_type not in {"LONG", "SHORT"}:
                continue

            normalized_symbol = str(item.get("symbol", ""))
            symbol = symbol_map.get(normalized_symbol, normalized_symbol)

            details = cast(Dict[str, Any], item.get("details", {}))
            strengths = cast(Dict[str, float], item.get("strengths", {}))
            score = float(item.get("score", 0.0))

            if abs(score) < self.threshold:
                continue

            signals.append(
                SignalResult(
                    symbol=symbol,
                    score=round(score, 4),
                    signal_type=signal_type,
                    details=details,
                    strengths=strengths,
                )
            )

        log_info(
            "ATCServerlessScanner: %s/%s symbols produced actionable signals",
            len(signals),
            len(symbols_data),
        )
        return signals

    def _build_lambda_config(self) -> Dict[str, Any]:
        merged = DEFAULT_ATC_CONFIG.copy()
        for key in (
            "weights",
            "threshold",
            "min_signal",
            "use_signal_strength",
            "lambda_param",
            "decay",
            "cutout",
            "equity_floor",
            "robustness",
            "ma_configs",
        ):
            if key in self.config:
                merged[key] = self.config[key]

        if "weights" not in self.config:
            equal_weight = 1.0 / max(len(self.timeframes), 1)
            merged["weights"] = {tf: equal_weight for tf in self.timeframes}

        return merged

    def _build_symbol_payload(self, symbol: str) -> Optional[Dict[str, Any]]:
        normalized_symbol = _SYMBOL_CODEC.to_db(symbol)
        timeframe_payload: Dict[str, Dict[str, List[float]]] = {}

        for timeframe in self.timeframes:
            ohlcv = self._fetch_timeframe_ohlcv(symbol, timeframe)
            if ohlcv is None:
                return None
            timeframe_payload[timeframe] = ohlcv

        return {"symbol": normalized_symbol, "timeframes": timeframe_payload}

    def _fetch_timeframe_ohlcv(self, symbol: str, timeframe: str) -> Optional[Dict[str, List[float]]]:
        try:
            df = self.data_fetcher.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=self.ohlcv_limit,
                check_freshness=False,
            )
        except Exception as exc:
            log_warn("ATCServerlessScanner: fetch error %s %s: %s", symbol, timeframe, exc)
            return None

        if df is None or df.empty:
            return None

        timestamp_values = self._extract_timestamps(df)
        open_values = self._extract_numeric_series(df, ("open", "Open"))
        high_values = self._extract_numeric_series(df, ("high", "High"))
        low_values = self._extract_numeric_series(df, ("low", "Low"))
        close_values = self._extract_numeric_series(df, ("close", "Close"))
        volume_values = self._extract_numeric_series(df, ("volume", "Volume"))

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
        if min_len < self.min_candles_per_tf:
            return None

        return {
            "timestamp": [int(v) for v in timestamp_values[-min_len:]],
            "open": [float(v) for v in open_values[-min_len:]],
            "high": [float(v) for v in high_values[-min_len:]],
            "low": [float(v) for v in low_values[-min_len:]],
            "close": [float(v) for v in close_values[-min_len:]],
            "volume": [float(v) for v in volume_values[-min_len:]],
        }

    def _extract_numeric_series(self, df: pd.DataFrame, candidates: Sequence[str]) -> List[float]:
        for name in candidates:
            if name in df.columns:
                return pd.to_numeric(df[name], errors="coerce").dropna().astype(float).tolist()
        return []

    def _extract_timestamps(self, df: pd.DataFrame) -> List[int]:
        for column in ("timestamp", "time", "datetime", "date", "Date"):
            if column in df.columns:
                series = df[column]
                return self._to_unix_seconds(series.tolist())

        if isinstance(df.index, pd.DatetimeIndex):
            return [int(ts.timestamp()) for ts in df.index.to_pydatetime()]

        return []

    def _to_unix_seconds(self, values: Sequence[Any]) -> List[int]:
        unix_values: List[int] = []
        for value in values:
            ts = self._normalize_timestamp(value)
            if ts is not None:
                unix_values.append(ts)
        return unix_values

    def _normalize_timestamp(self, value: Any) -> Optional[int]:
        if value is None:
            return None

        if isinstance(value, datetime):
            return int(value.timestamp())

        if hasattr(value, "to_pydatetime"):
            try:
                dt = value.to_pydatetime()
                return int(dt.timestamp())
            except Exception:
                pass

        try:
            numeric = float(value)
            if numeric > 1e12:
                return int(numeric / 1000)
            return int(numeric)
        except (TypeError, ValueError):
            pass

        try:
            parsed = pd.to_datetime(value, utc=True)
            if pd.isna(parsed):
                return None
            return int(parsed.timestamp())
        except Exception:
            return None

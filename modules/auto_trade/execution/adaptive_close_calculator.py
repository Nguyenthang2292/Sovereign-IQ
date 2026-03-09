"""
auto_trade/execution/adaptive_close_calculator.py
==================================================
Adaptive Close Time Calculator.

Táº¡i thá»i Ä‘iá»ƒm má»Ÿ order, gá»i RegimeDurationAnalyzer Ä‘á»ƒ tÃ­nh
adaptive_duration_hours cho symbol Ä‘Ã³, rá»“i set lÃªn order record.

TÃ­ch há»£p vÃ o flow má»Ÿ order hiá»‡n cÃ³.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Literal, Optional

from modules.common.ui.logging import log_error, log_info, log_warn

# Import RegimeLambdaClient for optional Lambda offloading
try:
    from modules.detect_regime_change.regime_lambda_client import (
        RegimeLambdaClient,
        RegimeDurationResult as LambdaRegimeResult,
    )
except Exception:
    RegimeLambdaClient = None
    LambdaRegimeResult = None

# Keep a module-level symbol so tests can patch
# `modules.auto_trade.execution.adaptive_close_calculator.RegimeDurationAnalyzer`.
try:
    from modules.detect_regime_change.regime_duration_analyzer import RegimeDurationAnalyzer
except Exception:
    RegimeDurationAnalyzer = None


# â”€â”€â”€ Configuration defaults â”€â”€â”€
DEFAULT_MIN_DURATION_HOURS = 1.0
DEFAULT_MAX_DURATION_HOURS = 12.0
DEFAULT_FALLBACK_DURATION_HOURS = 4.0
DEFAULT_LOOKBACK_DAYS = 60


@dataclass
class AdaptiveCloseResult:
    """
    Result of adaptive close deadline calculation with metadata for DB persistence.

    Fields:
        deadline_utc: Calculated deadline in UTC (or None if adaptive disabled)
        source: Source of the deadline calculation
        duration_hours: Final duration used (after clamping)
        pelt_hours: PELT average duration from analysis (None if not available)
        hmm_hours: HMM next state duration from analysis (None if not available)
    """
    deadline_utc: Optional[datetime]
    source: Literal["adaptive", "static", "adaptive_fallback"]
    duration_hours: float
    pelt_hours: Optional[float]
    hmm_hours: Optional[float]


class AdaptiveCloseCalculator:
    """
    Calculator for adaptive close deadlines based on regime change analysis.

    Usage:
        calculator = AdaptiveCloseCalculator(settings_manager)
        deadline_utc = calculator.compute_adaptive_deadline(
            symbol="BTC/USDT",
            opened_at=datetime.now(timezone.utc),
        )

        if deadline_utc:
            order_updates["auto_close_deadline_utc"] = deadline_utc.isoformat()
    """

    def __init__(self, settings_manager):
        self.settings_manager = settings_manager

    def _get_config(self) -> Dict[str, Any]:
        """Get adaptive close configuration from settings."""
        cfg = self.settings_manager.get("auto_close", {}) or {}
        adaptive = cfg.get("adaptive", {}) or {}
        return {
            "enabled": bool(adaptive.get("enabled", False)),
            "min_duration_hours": float(adaptive.get("min_duration_hours", DEFAULT_MIN_DURATION_HOURS)),
            "max_duration_hours": float(adaptive.get("max_duration_hours", DEFAULT_MAX_DURATION_HOURS)),
            "fallback_duration_hours": float(cfg.get("max_duration_hours", DEFAULT_FALLBACK_DURATION_HOURS)),
            "lookback_days": int(adaptive.get("lookback_days", DEFAULT_LOOKBACK_DAYS)),
            "timeframe": str(adaptive.get("timeframe", "") or cfg.get("timeframe", "15m")),
            # Lambda offloading config (Feature B)
            "use_lambda": bool(adaptive.get("use_lambda", False)),
            "lambda_endpoint": str(adaptive.get("lambda_endpoint", "")),
            "lambda_timeout_seconds": float(adaptive.get("lambda_timeout_seconds", 3.0)),
        }

    def compute_adaptive_deadline(
        self,
        symbol: str,
        opened_at: datetime,
        ohlcv_df=None,
    ) -> Optional[datetime]:
        """
        Compute adaptive close deadline for an order.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            opened_at: Order open timestamp (UTC)
            ohlcv_df: Optional pre-fetched OHLCV DataFrame.
                       If None, will be fetched internally.

        Returns:
            datetime: Adaptive close deadline in UTC, or None if disabled/failed.
        """
        cfg = self._get_config()

        if not cfg["enabled"]:
            return None

        try:
            # === 1. Fetch data if not provided ===
            if ohlcv_df is None:
                ohlcv_df = self._fetch_ohlcv(
                    symbol=symbol,
                    timeframe=cfg["timeframe"],
                    lookback_days=cfg["lookback_days"],
                )

            if ohlcv_df is None or len(ohlcv_df) < 100:
                log_warn(
                    f"Adaptive close: insufficient data for {symbol} "
                    f"({len(ohlcv_df) if ohlcv_df is not None else 0} candles), "
                    f"falling back to static {cfg['fallback_duration_hours']}h"
                )
                return opened_at + timedelta(hours=cfg["fallback_duration_hours"])

            # === 2. Run regime analysis ===
            if RegimeDurationAnalyzer is None:
                raise ImportError(
                    "RegimeDurationAnalyzer is unavailable "
                    "(modules.detect_regime_change.regime_duration_analyzer import failed)"
                )

            analyzer = RegimeDurationAnalyzer(
                lookback_days=cfg["lookback_days"],
            )
            analysis = analyzer.analyze(
                df=ohlcv_df,
                symbol=symbol,
                timeframe=cfg["timeframe"],
            )

            # === 4. Extract and clamp ===
            if analysis.is_valid and analysis.recommended_duration_hours is not None:
                raw_hours = analysis.recommended_duration_hours
                clamped_hours = max(
                    cfg["min_duration_hours"],
                    min(cfg["max_duration_hours"], raw_hours),
                )

                log_info(
                    f"Adaptive close [{symbol}]: "
                    f"raw={raw_hours:.2f}h â†’ clamped={clamped_hours:.2f}h "
                    f"(min={cfg['min_duration_hours']}h, max={cfg['max_duration_hours']}h)"
                )

                return opened_at + timedelta(hours=clamped_hours)

            # === 4. Fallback ===
            log_warn(
                f"Adaptive close [{symbol}]: analysis invalid "
                f"(error={analysis.error}), falling back to {cfg['fallback_duration_hours']}h"
            )
            return opened_at + timedelta(hours=cfg["fallback_duration_hours"])

        except Exception as e:
            log_error(f"Adaptive close calculation failed for {symbol}: {e}")
            return opened_at + timedelta(hours=cfg["fallback_duration_hours"])

    def _fetch_ohlcv(self, symbol: str, timeframe: str, lookback_days: int):
        """
        Fetch historical OHLCV data for regime analysis.

        Sá»­ dá»¥ng data fetcher Ä‘Ã£ cÃ³ trong project.
        """
        try:
            import ccxt
            import pandas as pd

            exchange = ccxt.binance({"enableRateLimit": True})
            since_ms = int(
                (datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp() * 1000
            )

            ohlcv = exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=since_ms, limit=1000
            )

            if not ohlcv:
                return None

            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            return df

        except Exception as e:
            log_error(f"Failed to fetch OHLCV for {symbol}: {e}")
            return None

    def compute_adaptive_deadline_with_meta(
        self,
        symbol: str,
        opened_at: datetime,
        ohlcv_df=None,
    ) -> AdaptiveCloseResult:
        """
        Compute adaptive close deadline with full metadata for DB persistence.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            opened_at: Order open timestamp (UTC)
            ohlcv_df: Optional pre-fetched OHLCV DataFrame.
                       If None, will be fetched internally.

        Returns:
            AdaptiveCloseResult with deadline and metadata.
        """
        cfg = self._get_config()

        # If adaptive is disabled, return static result
        if not cfg["enabled"]:
            static_deadline = opened_at + timedelta(hours=cfg['fallback_duration_hours'])
            return AdaptiveCloseResult(
                deadline_utc=static_deadline,
                source="static",
                duration_hours=cfg['fallback_duration_hours'],
                pelt_hours=None,
                hmm_hours=None,
            )

        try:
            # === 1. Fetch data if not provided ===
            if ohlcv_df is None:
                ohlcv_df = self._fetch_ohlcv(
                    symbol=symbol,
                    timeframe=cfg["timeframe"],
                    lookback_days=cfg["lookback_days"],
                )

            if ohlcv_df is None or len(ohlcv_df) < 100:
                log_warn(
                    f"Adaptive close: insufficient data for {symbol} "
                    f"({len(ohlcv_df) if ohlcv_df is not None else 0} candles), "
                    f"falling back to static {cfg['fallback_duration_hours']}h"
                )
                fallback_deadline = opened_at + timedelta(hours=cfg['fallback_duration_hours'])
                return AdaptiveCloseResult(
                    deadline_utc=fallback_deadline,
                    source="adaptive_fallback",
                    duration_hours=cfg['fallback_duration_hours'],
                    pelt_hours=None,
                    hmm_hours=None,
                )

            # === 2. Try Lambda first if enabled ===
            analysis = None
            if cfg["use_lambda"] and cfg["lambda_endpoint"] and RegimeLambdaClient is not None:
                client = RegimeLambdaClient(
                    endpoint=cfg["lambda_endpoint"],
                    timeout_seconds=cfg["lambda_timeout_seconds"],
                )
                lambda_result = client.invoke(ohlcv_df, symbol, cfg)

                if (
                    lambda_result is not None
                    and lambda_result.is_valid
                    and lambda_result.recommended_duration_hours is not None
                ):
                    # Use Lambda result
                    analysis = lambda_result
                    log_info(f"Adaptive close [{symbol}]: Using Lambda result")
                elif lambda_result is not None:
                    log_warn(
                        f"Adaptive close [{symbol}]: Lambda returned invalid result "
                        f"(error={lambda_result.error}), falling back to local analyzer"
                    )

            # === 3. Fallback to local analyzer if Lambda failed or not enabled ===
            if analysis is None:
                if RegimeDurationAnalyzer is None:
                    raise ImportError(
                        "RegimeDurationAnalyzer is unavailable "
                        "(modules.detect_regime_change.regime_duration_analyzer import failed)"
                    )

                analyzer = RegimeDurationAnalyzer(
                    lookback_days=cfg["lookback_days"],
                )
                analysis = analyzer.analyze(
                    df=ohlcv_df,
                    symbol=symbol,
                    timeframe=cfg["timeframe"],
                )
                log_info(f"Adaptive close [{symbol}]: Using local analyzer result")

            # === 4. Extract and clamp ===
            if analysis.is_valid and analysis.recommended_duration_hours is not None:
                raw_hours = analysis.recommended_duration_hours
                clamped_hours = max(
                    cfg["min_duration_hours"],
                    min(cfg["max_duration_hours"], raw_hours),
                )

                log_info(
                    f"Adaptive close [{symbol}]: "
                    f"raw={raw_hours:.2f}h → clamped={clamped_hours:.2f}h "
                    f"(min={cfg['min_duration_hours']}h, max={cfg['max_duration_hours']}h)"
                )

                deadline = opened_at + timedelta(hours=clamped_hours)
                return AdaptiveCloseResult(
                    deadline_utc=deadline,
                    source="adaptive",
                    duration_hours=clamped_hours,
                    pelt_hours=analysis.pelt_avg_duration_hours,
                    hmm_hours=analysis.hmm_next_state_duration_hours,
                )

            # === 5. Fallback when analysis invalid ===
            log_warn(
                f"Adaptive close [{symbol}]: analysis invalid "
                f"(error={analysis.error}), falling back to {cfg['fallback_duration_hours']}h"
            )
            fallback_deadline = opened_at + timedelta(hours=cfg['fallback_duration_hours'])
            return AdaptiveCloseResult(
                deadline_utc=fallback_deadline,
                source="adaptive_fallback",
                duration_hours=cfg['fallback_duration_hours'],
                pelt_hours=analysis.pelt_avg_duration_hours,
                hmm_hours=analysis.hmm_next_state_duration_hours,
            )

        except Exception as e:
            log_error(f"Adaptive close calculation failed for {symbol}: {e}")
            fallback_deadline = opened_at + timedelta(hours=cfg['fallback_duration_hours'])
            return AdaptiveCloseResult(
                deadline_utc=fallback_deadline,
                source="adaptive_fallback",
                duration_hours=cfg['fallback_duration_hours'],
                pelt_hours=None,
                hmm_hours=None,
            )

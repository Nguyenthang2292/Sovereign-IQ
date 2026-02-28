"""
Auto Close Timer Logic
======================

Core helper logic for auto-close timer:
- deadline parsing and calculation
- timeout/daily trigger checks
- close execution via quasi-market TP update
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

from modules.auto_trade.execution.binance_client import BinanceClient
from modules.common.ui.logging import log_error, log_info, log_warn


@dataclass
class AutoCloseDecision:
    should_close: bool
    reason: str
    deadline_utc: Optional[datetime]
    trigger_label: str


@dataclass
class AutoCloseExecutionResult:
    success: bool
    message: str
    target_tp: Optional[float]
    trigger_time_utc: datetime


def parse_utc_datetime(value: Any) -> Optional[datetime]:
    """Parse datetime-like value into timezone-aware UTC datetime."""
    if value is None:
        return None

    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError:
            return None
    else:
        return None

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_utc_iso(dt: datetime) -> str:
    """Format UTC datetime as ISO string with Z suffix."""
    utc_dt = dt.astimezone(timezone.utc)
    return utc_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def symbol_for_ccxt(symbol: str) -> str:
    """Convert DB symbol (e.g. BTCUSDT) to CCXT format (BTC/USDT)."""
    s = (symbol or "").strip()
    if "/" in s:
        return s
    if s.endswith("USDT"):
        return s[:-4] + "/USDT"
    return s + "/USDT" if s else s


def get_order_id(order: Dict[str, Any]) -> str:
    """Get best-effort order id from common fields."""
    for key in ("order_id", "id", "pk"):
        value = order.get(key)
        if value:
            return str(value)
    return ""


def get_opened_at(order: Dict[str, Any]) -> Optional[datetime]:
    """Get order open timestamp from common fields."""
    for key in ("opened_at", "created_at", "open_time", "entry_time"):
        dt = parse_utc_datetime(order.get(key))
        if dt is not None:
            return dt
    return None


def parse_daily_close_time(value: str) -> Optional[Tuple[int, int]]:
    """Parse daily close time HH:MM (UTC)."""
    try:
        parts = (value or "").strip().split(":")
        if len(parts) != 2:
            return None
        hour = int(parts[0])
        minute = int(parts[1])
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return None
        return hour, minute
    except (TypeError, ValueError):
        return None


def compute_deadline_utc(
    order: Dict[str, Any],
    max_duration_enabled: bool,
    max_duration_hours: float,
) -> Optional[datetime]:
    """Compute timeout deadline for an order."""
    override = parse_utc_datetime(order.get("auto_close_deadline_utc"))
    if override is not None:
        return override

    if not max_duration_enabled or max_duration_hours <= 0:
        return None

    opened_at = get_opened_at(order)
    if opened_at is None:
        return None

    return opened_at + timedelta(hours=max_duration_hours)


def evaluate_order_for_auto_close(
    order: Dict[str, Any],
    now_utc: datetime,
    auto_close_cfg: Dict[str, Any],
) -> AutoCloseDecision:
    """Evaluate if an open order should be auto-closed now."""
    if bool(order.get("auto_close_triggered", False)):
        return AutoCloseDecision(False, "already_triggered", None, "")

    grace_period_minutes = int(auto_close_cfg.get("grace_period_minutes", 5) or 5)
    opened_at = get_opened_at(order)
    if opened_at is not None and grace_period_minutes > 0:
        if now_utc < opened_at + timedelta(minutes=grace_period_minutes):
            return AutoCloseDecision(False, "in_grace_period", None, "")

    max_duration_enabled = bool(auto_close_cfg.get("max_duration_enabled", True))
    max_duration_hours = float(auto_close_cfg.get("max_duration_hours", 4.0) or 4.0)

    timeout_deadline = compute_deadline_utc(
        order=order,
        max_duration_enabled=max_duration_enabled,
        max_duration_hours=max_duration_hours,
    )

    if timeout_deadline is not None and now_utc >= timeout_deadline:
        return AutoCloseDecision(True, "max_duration", timeout_deadline, "timer")

    daily_close_enabled = bool(auto_close_cfg.get("daily_close_enabled", False))
    if daily_close_enabled:
        parsed_time = parse_daily_close_time(str(auto_close_cfg.get("daily_close_time", "22:00") or "22:00"))
        if parsed_time is not None:
            hour, minute = parsed_time
            allowed_days = str(auto_close_cfg.get("daily_close_days", "1234567") or "1234567")
            weekday = now_utc.isoweekday()
            if str(weekday) in allowed_days:
                today_cutoff = now_utc.replace(hour=hour, minute=minute, second=0, microsecond=0)
                last_daily_date = str(order.get("auto_close_last_daily_date", "") or "")
                today_str = now_utc.date().isoformat()

                if now_utc >= today_cutoff and last_daily_date != today_str:
                    return AutoCloseDecision(True, "daily_close", today_cutoff, "daily")

    return AutoCloseDecision(False, "no_trigger", timeout_deadline, "")


def _get_mark_price(binance_client: BinanceClient, symbol: str) -> Optional[float]:
    try:
        ticker = binance_client.fetch_ticker(symbol_for_ccxt(symbol))
        if not ticker:
            return None
        info = ticker.get("info") or {}
        mark = info.get("markPrice") if isinstance(info, dict) else None
        if mark is not None:
            return float(mark)
        if ticker.get("last") is not None:
            return float(ticker["last"])
        return None
    except Exception as exc:
        log_error(f"[AutoClose] Could not fetch mark price for {symbol}: {exc}")
        return None


def _calc_quasi_market_tp(mark_price: float, side: str, offset_pct: float) -> float:
    if str(side).upper() == "SHORT":
        return mark_price * (1.0 + offset_pct / 100.0)
    return mark_price * (1.0 - offset_pct / 100.0)


def execute_auto_close(
    *,
    order: Dict[str, Any],
    reason: str,
    binance_client: Optional[BinanceClient],
    tp_offset_pct: float,
) -> AutoCloseExecutionResult:
    """Execute auto-close by updating TP to a near-market trigger price."""
    now_utc = datetime.now(timezone.utc)
    symbol = str(order.get("symbol", "") or "")
    side = str(order.get("side", "LONG") or "LONG").upper()

    if not symbol:
        return AutoCloseExecutionResult(False, "Missing symbol", None, now_utc)

    if tp_offset_pct <= 0:
        tp_offset_pct = 0.05

    if binance_client is None:
        log_warn(f"[AutoClose] No Binance client for {symbol}; marking as dry-run success")
        return AutoCloseExecutionResult(True, "Dry run (no client)", None, now_utc)

    mark_price = _get_mark_price(binance_client, symbol)
    if mark_price is None or mark_price <= 0:
        return AutoCloseExecutionResult(False, f"Could not fetch mark price for {symbol}", None, now_utc)

    target_tp = _calc_quasi_market_tp(mark_price, side, tp_offset_pct)

    try:
        ccxt_symbol = symbol_for_ccxt(symbol)
        result = binance_client.modify_take_profit(
            symbol=ccxt_symbol,
            position_id=None,
            take_profit_price=target_tp,
        )
        success = result is not None and (
            bool(result.get("success")) or bool(result.get("id")) or bool(result.get("dry_run"))
        )
        if not success:
            err = str((result or {}).get("error", "Unknown error"))
            return AutoCloseExecutionResult(False, f"Failed to place TP close order: {err}", target_tp, now_utc)

        log_info(
            f"[AutoClose] Triggered {symbol} {side} | reason={reason} | mark={mark_price:.6f} | target_tp={target_tp:.6f}"
        )
        return AutoCloseExecutionResult(True, "Auto-close trigger sent", target_tp, now_utc)
    except Exception as exc:
        log_error(f"[AutoClose] Execute failed for {symbol}: {exc}")
        return AutoCloseExecutionResult(False, f"Execute failed: {exc}", target_tp, now_utc)

from datetime import datetime, timedelta, timezone
from typing import Any, cast

import pytest

from modules.auto_trade.execution.auto_close_timer import (
    AutoCloseExecutionResult,
    _calc_quasi_market_tp,
    compute_deadline_utc,
    evaluate_order_for_auto_close,
    execute_auto_close,
    parse_utc_datetime,
)
from modules.auto_trade.execution.auto_close_timer_job import AutoCloseTimerJob


class _SettingsStub:
    def __init__(self, payload):
        self.payload = payload

    def get(self, key, default=None):
        if key == "auto_close":
            return self.payload
        return default


class _OrderRepoStub:
    def __init__(self):
        self.calls = []

    def update(self, order_id, updates):
        self.calls.append((order_id, updates))
        return True


class _RepoContextStub:
    def __init__(self):
        self.orders = _OrderRepoStub()


class _BinanceClientStub:
    def __init__(self, mark_price=100.0, success=True):
        self.mark_price = mark_price
        self.success = success
        self.calls = []

    def fetch_ticker(self, symbol):
        return {"info": {"markPrice": str(self.mark_price)}}

    def modify_take_profit(self, symbol, position_id, take_profit_price):
        self.calls.append({"symbol": symbol, "take_profit_price": take_profit_price})
        if self.success:
            return {"success": True, "id": "test_id"}
        return {"success": False, "error": "Mocked failure"}


def _base_cfg():
    return {
        "enabled": True,
        "max_duration_enabled": True,
        "max_duration_hours": 4.0,
        "daily_close_enabled": True,
        "daily_close_time": "22:00",
        "daily_close_days": "1234567",
        "grace_period_minutes": 5,
        "tp_offset_pct": 0.05,
    }


def test_parse_utc_datetime_with_z_suffix():
    dt = parse_utc_datetime("2026-02-28T22:00:00Z")
    assert dt is not None
    assert dt.tzinfo is not None
    assert dt.isoformat().endswith("+00:00")


def test_compute_deadline_utc_with_override():
    now = datetime(2026, 2, 28, 12, 0, tzinfo=timezone.utc)
    override_time = now + timedelta(hours=1)
    order = {
        "order_id": "o-override",
        "opened_at": (now - timedelta(hours=2)).isoformat(),
        "auto_close_deadline_utc": override_time.isoformat(),
    }

    # Override should take precedence over max_duration
    deadline = compute_deadline_utc(order, max_duration_enabled=True, max_duration_hours=4.0)
    assert deadline == override_time


def test_trigger_max_duration_timeout():
    cfg = _base_cfg()
    now = datetime(2026, 2, 28, 12, 0, tzinfo=timezone.utc)
    order = {
        "order_id": "o-1",
        "symbol": "BTCUSDT",
        "created_at": (now - timedelta(hours=5)).isoformat(),
    }

    decision = evaluate_order_for_auto_close(order=order, now_utc=now, auto_close_cfg=cfg)
    assert decision.should_close is True
    assert decision.reason == "max_duration"
    assert decision.trigger_label == "timer"


def test_trigger_daily_close_when_past_cutoff():
    cfg = _base_cfg()
    cfg["max_duration_enabled"] = False

    now = datetime(2026, 2, 28, 22, 1, tzinfo=timezone.utc)
    order = {
        "order_id": "o-2",
        "symbol": "ETHUSDT",
        "created_at": (now - timedelta(hours=1)).isoformat(),
    }

    decision = evaluate_order_for_auto_close(order=order, now_utc=now, auto_close_cfg=cfg)
    assert decision.should_close is True
    assert decision.reason == "daily_close"
    assert decision.trigger_label == "daily"


def test_skip_daily_close_if_already_triggered_today():
    cfg = _base_cfg()
    cfg["max_duration_enabled"] = False

    now = datetime(2026, 2, 28, 22, 1, tzinfo=timezone.utc)
    order = {
        "order_id": "o-daily-triggered",
        "symbol": "ETHUSDT",
        "created_at": (now - timedelta(hours=1)).isoformat(),
        "auto_close_last_daily_date": "2026-02-28",  # Already triggered today
    }

    decision = evaluate_order_for_auto_close(order=order, now_utc=now, auto_close_cfg=cfg)
    assert decision.should_close is False


def test_skip_when_already_triggered_idempotent():
    cfg = _base_cfg()
    now = datetime(2026, 2, 28, 12, 0, tzinfo=timezone.utc)
    order = {
        "order_id": "o-3",
        "symbol": "SOLUSDT",
        "created_at": (now - timedelta(hours=8)).isoformat(),
        "auto_close_triggered": True,
    }

    decision = evaluate_order_for_auto_close(order=order, now_utc=now, auto_close_cfg=cfg)
    assert decision.should_close is False
    assert decision.reason == "already_triggered"


def test_skip_when_in_grace_period():
    cfg = _base_cfg()
    now = datetime(2026, 2, 28, 12, 0, tzinfo=timezone.utc)
    order = {
        "order_id": "o-4",
        "symbol": "ADAUSDT",
        "created_at": (now - timedelta(minutes=2)).isoformat(),
    }

    decision = evaluate_order_for_auto_close(order=order, now_utc=now, auto_close_cfg=cfg)
    assert decision.should_close is False
    assert decision.reason == "in_grace_period"


def test_calc_quasi_market_tp():
    # Long: TP slightly below mark price
    tp_long = _calc_quasi_market_tp(mark_price=100.0, side="LONG", offset_pct=0.05)
    assert tp_long == 99.95

    # Short: TP slightly above mark price
    tp_short = _calc_quasi_market_tp(mark_price=100.0, side="SHORT", offset_pct=0.05)
    assert tp_short == 100.05


def test_execute_auto_close_success():
    client = _BinanceClientStub(mark_price=50000.0, success=True)
    order = {"symbol": "BTCUSDT", "side": "LONG"}

    result = execute_auto_close(order=order, reason="max_duration", binance_client=client, tp_offset_pct=0.05)

    assert result.success is True
    assert result.target_tp == 49975.0  # 50000 * (1 - 0.0005)
    assert len(client.calls) == 1
    assert client.calls[0]["symbol"] == "BTC/USDT"
    assert client.calls[0]["take_profit_price"] == 49975.0


def test_execute_auto_close_failure():
    client = _BinanceClientStub(mark_price=50000.0, success=False)
    order = {"symbol": "BTCUSDT", "side": "LONG"}

    result = execute_auto_close(order=order, reason="max_duration", binance_client=client, tp_offset_pct=0.05)

    assert result.success is False
    assert "Failed to place TP close order" in result.message


def test_job_updates_db_on_trigger(monkeypatch):
    cfg = _base_cfg()
    settings = _SettingsStub(cfg)
    repo = _RepoContextStub()

    open_order = {
        "order_id": "o-5",
        "symbol": "BTCUSDT",
        "side": "LONG",
        "created_at": (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat(),
    }

    monkeypatch.setattr(
        "modules.auto_trade.execution.auto_close_timer_job.get_open_positions",
        lambda: [open_order],
    )

    class _ExecutionResult:
        def __init__(self):
            self.success = True
            self.message = "ok"
            self.target_tp = 100.0
            self.trigger_time_utc = datetime.now(timezone.utc)

    monkeypatch.setattr(
        "modules.auto_trade.execution.auto_close_timer_job.execute_auto_close",
        lambda **kwargs: _ExecutionResult(),
    )

    job = AutoCloseTimerJob(settings_manager=settings, repo_context=cast(Any, repo), binance_client=None)
    result = job.run()

    assert result["orders_checked"] == 1
    assert result["orders_triggered"] == 1
    assert len(repo.orders.calls) == 1
    updated = repo.orders.calls[0][1]
    assert updated.get("auto_close_triggered") is True
    assert updated.get("auto_close_reason") == "max_duration"
    assert updated.get("auto_close_target_tp") == 100.0

from __future__ import annotations

import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.auto_trade.core import xgboost_auto_trainer as trainer


class _ImmediateThread:
    def __init__(self, target=None, args=(), daemon=None, name=None):
        self._target = target
        self._args = args
        self.daemon = daemon
        self.name = name
        self.started = False

    def start(self):
        self.started = True
        if self._target is not None:
            self._target(*self._args)


@pytest.fixture(autouse=True)
def _reset_trainer_state(monkeypatch):
    trainer._STATUS.clear()
    trainer._SYMBOL_LOCKS.clear()
    trainer._S3_CHECK_CACHE.clear()
    monkeypatch.setattr(trainer, "_TRAINER_FUNCTION_NAME", "xgboost-trainer")
    monkeypatch.setattr(trainer, "_TRAINER_REGION", "us-east-1")
    monkeypatch.setattr(trainer, "_model_exists_in_s3", lambda *a, **kw: False)


def test_request_training_lambda_success_returns_pending_and_finishes_fast(monkeypatch):
    invoke_mock = MagicMock()
    lambda_client = SimpleNamespace(invoke=invoke_mock)
    boto3_stub = SimpleNamespace(client=lambda *_args, **_kwargs: lambda_client)
    monkeypatch.setitem(sys.modules, "boto3", boto3_stub)
    monkeypatch.setattr(trainer.threading, "Thread", _ImmediateThread)

    data_fetcher = MagicMock()

    t0 = time.perf_counter()
    status = trainer.request_training(
        symbol="BTC/USDT",
        timeframe="15m",
        model_version="v1",
        s3_bucket="xgboost-models-store",
        data_fetcher=data_fetcher,
    )
    elapsed = time.perf_counter() - t0

    assert status == "pending"
    assert elapsed < 2.0
    assert invoke_mock.call_count == 1


def test_request_training_lambda_failure_falls_back_to_local(monkeypatch):
    def _raise_invoke(*_args, **_kwargs):
        raise RuntimeError("lambda unavailable")

    lambda_client = SimpleNamespace(invoke=_raise_invoke)
    boto3_stub = SimpleNamespace(client=lambda *_args, **_kwargs: lambda_client)
    monkeypatch.setitem(sys.modules, "boto3", boto3_stub)
    monkeypatch.setattr(trainer.threading, "Thread", _ImmediateThread)

    fallback_mock = MagicMock()
    monkeypatch.setattr(trainer, "_train_and_upload", fallback_mock)

    data_fetcher = MagicMock()
    status = trainer.request_training(
        symbol="ETH/USDT",
        timeframe="15m",
        model_version="v1",
        s3_bucket="xgboost-models-store",
        data_fetcher=data_fetcher,
    )

    assert status == "pending"
    fallback_mock.assert_called_once_with(
        "ETH/USDT",
        "15m",
        "v1",
        "xgboost-models-store",
        data_fetcher,
        "ETHUSDT_15m_v1",
    )


def test_get_training_status_returns_expected_state_and_ttl_expiry():
    cache_key = "BTCUSDT_15m_v1"

    trainer._set_status(cache_key, "pending")
    assert trainer.get_training_status(cache_key) == "pending"

    trainer._set_status(cache_key, "ready", path="/tmp/BTCUSDT_15m_v1.json")
    assert trainer.get_training_status(cache_key) == "ready"

    trainer._set_status(cache_key, "failed")
    assert trainer.get_training_status(cache_key) == "failed"

    trainer._STATUS[cache_key]["ts"] = time.monotonic() - trainer._FAILURE_TTL - 1
    assert trainer.get_training_status(cache_key) is None

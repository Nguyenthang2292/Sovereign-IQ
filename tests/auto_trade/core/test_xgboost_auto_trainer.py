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


# ── New tests: class imbalance / skipped status ───────────────────────────────


def test_get_training_status_skipped_ttl():
    """'skipped' status is cached until _SKIP_TTL expires, then allows retry."""
    cache_key = "IOSTUSDT_15m_v1"

    trainer._set_status(cache_key, "skipped")
    assert trainer.get_training_status(cache_key) == "skipped"

    # Simulate TTL expired
    trainer._STATUS[cache_key]["ts"] = time.monotonic() - trainer._SKIP_TTL - 1
    assert trainer.get_training_status(cache_key) is None  # allow retry


def test_request_training_skipped_returns_skipped_without_new_thread(monkeypatch):
    """request_training must NOT spawn a new thread when status is 'skipped' and TTL is still active."""
    cache_key = "IOSTUSDT_15m_v1"
    trainer._set_status(cache_key, "skipped")

    thread_cls = MagicMock()
    monkeypatch.setattr(trainer.threading, "Thread", thread_cls)

    status = trainer.request_training(
        symbol="IOST/USDT",
        timeframe="15m",
        model_version="v1",
        s3_bucket="xgboost-models-store",
        data_fetcher=MagicMock(),
    )

    assert status == "skipped"
    thread_cls.assert_not_called()


def test_train_model_sync_skips_on_lambda_missing_classes_error(monkeypatch):
    """train_model_sync must set status 'skipped' and NOT fallback to local training
    when Lambda FunctionError body contains 'missing classes'."""
    import json
    import sys

    cache_key = "IOSTUSDT_15m_v1"
    trainer._STATUS.clear()

    # Patch _model_exists_in_s3 so we go straight to Lambda invoke
    monkeypatch.setattr(trainer, "_model_exists_in_s3", lambda *a, **kw: False)

    # Build a Lambda response that signals class imbalance
    missing_class_body = json.dumps({
        "errorMessage": (
            "Training set has 2 class(es) ['DOWN', 'NEUTRAL'], "
            "but model expects 3 classes. Missing: ['UP']. "
            "Training with missing classes produces biased predictions."
        )
    }).encode()
    fake_payload = MagicMock()
    fake_payload.read.return_value = missing_class_body

    lambda_client = MagicMock()
    lambda_client.invoke.return_value = {
        "StatusCode": 200,
        "FunctionError": "Unhandled",
        "Payload": fake_payload,
    }

    boto3_stub = MagicMock()
    boto3_stub.client.return_value = lambda_client
    monkeypatch.setitem(sys.modules, "boto3", boto3_stub)

    fallback_mock = MagicMock()
    monkeypatch.setattr(trainer, "_train_and_upload", fallback_mock)

    result = trainer.train_model_sync(
        symbol="IOST/USDT",
        timeframe="15m",
        model_version="v1",
        s3_bucket="xgboost-models-store",
        data_fetcher=MagicMock(),
        wait_timeout_seconds=1,
    )

    # Caller gets "failed" but root cause is class imbalance
    assert result == "failed"
    # Status must be "skipped" (not "failed") so TTL is _SKIP_TTL, not _FAILURE_TTL
    assert trainer.get_training_status(cache_key) == "skipped"
    # MUST NOT fall back to expensive local training
    fallback_mock.assert_not_called()

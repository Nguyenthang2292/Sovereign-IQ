from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from modules.adaptive_trend_LTS_serverless.lambda_client import ATCLambdaClient, DEFAULT_ATC_CONFIG


class _FakePayload:
    def __init__(self, text: str):
        self._text = text

    def read(self) -> bytes:
        return self._text.encode("utf-8")


class _FakeLambdaClient:
    def __init__(self, response: dict[str, Any] | None = None, exc: Exception | None = None):
        self._response = response or {"StatusCode": 200, "Payload": _FakePayload("null")}
        self._exc = exc

    def invoke(self, **kwargs: Any) -> dict[str, Any]:
        if self._exc is not None:
            raise self._exc
        return self._response


class _FakeSQSClient:
    def __init__(self, batches: list[list[dict[str, Any]]] | None = None):
        self._batches = batches or []
        self.deleted: list[str] = []
        self.visibility_reset: list[str] = []

    def get_queue_url(self, **kwargs: Any) -> dict[str, Any]:
        return {"QueueUrl": "https://sqs.test.local/queue"}

    def receive_message(self, **kwargs: Any) -> dict[str, Any]:
        if self._batches:
            return {"Messages": self._batches.pop(0)}
        return {"Messages": []}

    def delete_message(self, **kwargs: Any) -> dict[str, Any]:
        self.deleted.append(kwargs["ReceiptHandle"])
        return {}

    def change_message_visibility(self, **kwargs: Any) -> dict[str, Any]:
        self.visibility_reset.append(kwargs["ReceiptHandle"])
        return {}


@pytest.fixture
def sample_symbols() -> list[dict[str, Any]]:
    return [
        {
            "symbol": "BTCUSDT",
            "timeframes": {
                "1h": {
                    "timestamp": [1, 2, 3],
                    "open": [100.0, 101.0, 102.0],
                    "high": [101.0, 102.0, 103.0],
                    "low": [99.0, 100.0, 101.0],
                    "close": [100.5, 101.5, 102.5],
                    "volume": [10.0, 11.0, 12.0],
                }
            },
        }
    ]


def test_mock_mode_invoke_returns_valid_result(sample_symbols: list[dict[str, Any]]) -> None:
    client = ATCLambdaClient(mock_mode=True)

    result = client.invoke(sample_symbols)

    assert result["batch_id"].startswith("mock-")
    assert result["success_count"] == len(sample_symbols)
    assert result["error_count"] == 0
    assert len(result["results"]) == len(sample_symbols)
    assert result["results"][0]["symbol"] == "BTCUSDT"


def test_mock_mode_batch_invoke_returns_all_symbols(sample_symbols: list[dict[str, Any]]) -> None:
    symbols = sample_symbols + [deepcopy(sample_symbols[0])]
    symbols[1]["symbol"] = "ETHUSDT"

    client = ATCLambdaClient(mock_mode=True)
    result = client.invoke_batch(symbols)

    returned_symbols = {entry["symbol"] for entry in result["results"]}
    assert returned_symbols == {"BTCUSDT", "ETHUSDT"}
    assert result["success_count"] == 2


def test_poll_sqs_timeout_returns_partial_results() -> None:
    client = ATCLambdaClient(mock_mode=True, sqs_poll_timeout=1, sqs_poll_interval=0)
    client._queue_url = "https://sqs.test.local/queue"
    client._sqs = _FakeSQSClient(
        batches=[
            [
                {
                    "Body": '{"batch_id":"req-123","results":[{"symbol":"BTCUSDT"}]}',
                    "ReceiptHandle": "rh-1",
                }
            ]
        ]
    )

    result = client._poll_sqs_for_batch("req-123")

    assert result["batch_id"] == "req-123"
    assert result["success_count"] >= 1
    assert any(item["symbol"] == "BTCUSDT" for item in result["results"])


def test_invoke_lambda_error_returns_error_dict(sample_symbols: list[dict[str, Any]]) -> None:
    client = ATCLambdaClient(mock_mode=False)
    client._lambda = _FakeLambdaClient(exc=RuntimeError("invoke failed"))

    result = client.invoke(sample_symbols)

    assert result["error_count"] == len(sample_symbols)
    assert result["success_count"] == 0
    assert result["errors"]
    assert "invoke failed" in result["errors"][0]["error"]


def test_poll_sqs_skips_malformed_messages_gracefully() -> None:
    client = ATCLambdaClient(mock_mode=True, sqs_poll_timeout=2, sqs_poll_interval=0)
    client._queue_url = "https://sqs.test.local/queue"
    client._sqs = _FakeSQSClient(
        batches=[
            [{"Body": "not-json", "ReceiptHandle": "bad-1"}],
            [
                {
                    "Body": '{"batch_id":"req-ok","results":[{"symbol":"BTCUSDT"}],"success_count":1,"error_count":0}',
                    "ReceiptHandle": "ok-1",
                }
            ],
        ]
    )

    result = client._poll_sqs_for_batch("req-ok")

    assert result["success_count"] >= 1
    assert result["results"][0]["symbol"] == "BTCUSDT"


def test_default_config_deepcopy_on_invoke(sample_symbols: list[dict[str, Any]]) -> None:
    client = ATCLambdaClient(mock_mode=True)
    original_length = DEFAULT_ATC_CONFIG["ma_configs"][0]["length"]

    def _mutating_mock(symbols_data: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
        config["ma_configs"][0]["length"] = 999
        return {
            "batch_id": "mock-test",
            "results": [{"symbol": symbols_data[0]["symbol"]}],
            "errors": [],
            "success_count": 1,
            "error_count": 0,
        }

    client._mock_invoke = _mutating_mock  # type: ignore[method-assign]

    _ = client.invoke(sample_symbols)

    assert DEFAULT_ATC_CONFIG["ma_configs"][0]["length"] == original_length

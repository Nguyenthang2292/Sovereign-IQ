import io
import json
import unittest

from modules.adaptive_trend_LTS_serverless.lambda_client import ATCLambdaClient
from modules.adaptive_trend_LTS_serverless.lambda_client import botocore_exceptions


class _FakePayloadStream:
    def __init__(self, body: str) -> None:
        self._stream = io.BytesIO(body.encode("utf-8"))

    def read(self) -> bytes:
        return self._stream.read()


class _FakeLambdaClient:
    def __init__(self, response: dict[str, object]) -> None:
        self._response = response
        self.last_invoke_kwargs: dict[str, object] | None = None

    def invoke(self, **kwargs: object) -> dict[str, object]:
        self.last_invoke_kwargs = kwargs
        return self._response


class _FailingLambdaClient:
    def invoke(self, **kwargs: object) -> dict[str, object]:
        _ = kwargs
        raise RuntimeError("simulated invoke failure")


class _ClientErrorLambdaClient:
    def invoke(self, **kwargs: object) -> dict[str, object]:
        _ = kwargs
        if botocore_exceptions is None:
            raise RuntimeError("botocore not available")
        raise botocore_exceptions.ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "Denied"}},
            "Invoke",
        )


class ATCLambdaClientDirectResponseTests(unittest.TestCase):
    def test_invoke_reads_scan_result_directly_from_lambda_payload(self) -> None:
        lambda_response = {
            "StatusCode": 200,
            "Payload": _FakePayloadStream(
                json.dumps(
                    {
                        "batch_id": "req-direct01",
                        "results": [
                            {
                                "symbol": "BTCUSDT",
                                "score": 0.87,
                                "signal_type": "LONG",
                                "details": {"1h": "LONG"},
                                "strengths": {"1h": 0.87},
                            }
                        ],
                        "errors": [],
                        "success_count": 1,
                        "error_count": 0,
                    }
                )
            ),
        }

        client = ATCLambdaClient(mock_mode=True)
        fake_lambda = _FakeLambdaClient(lambda_response)
        client.mock_mode = False
        client._lambda = fake_lambda

        result = client.invoke(
            symbols_data=[
                {
                    "symbol": "BTCUSDT",
                    "timeframes": {"1h": {"close": [1, 2, 3]}},
                }
            ]
        )

        self.assertEqual(result["batch_id"], "req-direct01")
        self.assertEqual(result["success_count"], 1)
        self.assertEqual(result["error_count"], 0)
        self.assertEqual(result["results"][0]["symbol"], "BTCUSDT")
        self.assertEqual(fake_lambda.last_invoke_kwargs["InvocationType"], "RequestResponse")

    def test_invoke_raises_when_transport_call_fails(self) -> None:
        client = ATCLambdaClient(mock_mode=True)
        client.mock_mode = False
        client._lambda = _FailingLambdaClient()

        symbols_data = [
            {"symbol": "BTCUSDT", "timeframes": {"1h": {"close": [1, 2, 3]}}},
            {"symbol": "ETHUSDT", "timeframes": {"1h": {"close": [4, 5, 6]}}},
        ]
        with self.assertRaises(RuntimeError):
            client.invoke(symbols_data=symbols_data)

    def test_invoke_reraises_client_error_as_infra_failure(self) -> None:
        if botocore_exceptions is None:
            self.skipTest("botocore not available")

        client = ATCLambdaClient(mock_mode=True)
        client.mock_mode = False
        client._lambda = _ClientErrorLambdaClient()

        with self.assertRaises(botocore_exceptions.ClientError):
            client.invoke(
                symbols_data=[{"symbol": "BTCUSDT", "timeframes": {"1h": {"close": [1, 2, 3]}}}]
            )

    def test_invoke_raises_when_payload_shape_is_invalid(self) -> None:
        lambda_response = {
            "StatusCode": 200,
            "Payload": _FakePayloadStream(json.dumps({"batch_id": "bad-payload"})),
        }

        client = ATCLambdaClient(mock_mode=True)
        fake_lambda = _FakeLambdaClient(lambda_response)
        client.mock_mode = False
        client._lambda = fake_lambda

        with self.assertRaises(RuntimeError):
            client.invoke(
                symbols_data=[{"symbol": "BTCUSDT", "timeframes": {"1h": {"close": [1, 2, 3]}}}]
            )


if __name__ == "__main__":
    unittest.main()

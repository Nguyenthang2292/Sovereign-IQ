"""
ATC Lambda Client

Main client for invoking the ATC Serverless Lambda function.
Architecture:
  client --[RequestResponse invoke]--> Lambda (atc-serverless) --> direct ScanResult payload

This client:
  1. Invokes Lambda via boto3 (IAM-signed, bypassing Function URL auth/SCP restrictions)
  2. Parses the returned ScanResult payload directly
"""

import copy
import importlib.util
import json
import logging
import uuid
import warnings
from typing import Any, Optional, Protocol, cast

# Boto3 is required for Lambda access
try:
    import boto3  # type: ignore[import-untyped]
except ImportError:
    boto3 = None

# botocore exceptions are used to distinguish infra misconfiguration vs symbol-level failures.
try:
    from botocore import exceptions as botocore_exceptions  # type: ignore[import-untyped]
except ImportError:
    botocore_exceptions = None

# AWS Requests Auth is optional (only for direct HTTP w/ IAM auth)
HAS_AWS_AUTH = importlib.util.find_spec("aws_requests_auth") is not None

logger = logging.getLogger(__name__)

# Constants
DEFAULT_FUNCTION_NAME = "atc-serverless"
# Deprecated in v0.2.0: retained only for backward-compatible __init__ signatures.
DEFAULT_SQS_QUEUE_NAME = "atc-results"
DEFAULT_REGION = "us-east-1"
# Deprecated in v0.2.0: retained only for backward-compatible __init__ signatures.
DEFAULT_SQS_POLL_TIMEOUT = 60
# Deprecated in v0.2.0: retained only for backward-compatible __init__ signatures.
DEFAULT_SQS_POLL_INTERVAL = 2.0

# Default ATC configuration
DEFAULT_ATC_CONFIG = {
    "weights": {"1h": 0.6, "4h": 0.4},
    "threshold": 0.3,
    "min_signal": 0.0,
    "use_signal_strength": True,
    "lambda_param": 0.02,
    "decay": 0.03,
    "cutout": 0,
    "equity_floor": 0.25,
    "robustness": "Medium",
    "ma_configs": [
        {"ma_type": "EMA", "length": 12, "weight": 1.0},
        {"ma_type": "HMA", "length": 12, "weight": 1.0},
        {"ma_type": "WMA", "length": 12, "weight": 1.0},
        {"ma_type": "DEMA", "length": 12, "weight": 1.0},
        {"ma_type": "LSMA", "length": 12, "weight": 1.0},
        {"ma_type": "KAMA", "length": 12, "weight": 1.0},
    ],
}


class _LambdaClientProtocol(Protocol):
    def invoke(self, **kwargs: Any) -> dict[str, Any]:
        ...


class ATCLambdaClient:
    """Invokes the ATC Serverless Lambda and parses the direct response payload.

    The Lambda function returns a serialized ScanResult in the synchronous
    RequestResponse payload, so the client does not coordinate through a shared
    SQS results queue anymore.
    """

    def __init__(
        self,
        function_name: str = DEFAULT_FUNCTION_NAME,
        sqs_queue_name: str = DEFAULT_SQS_QUEUE_NAME,
        region: str = DEFAULT_REGION,
        sqs_poll_timeout: int = DEFAULT_SQS_POLL_TIMEOUT,
        sqs_poll_interval: float = DEFAULT_SQS_POLL_INTERVAL,
        mock_mode: bool = False,
    ):
        """Initialize the client.

        Args:
            function_name: Name or ARN of the Lambda function
            sqs_queue_name: Deprecated. Retained for backwards-compatible init signatures.
            region: AWS region
            sqs_poll_timeout: Deprecated. Retained for backwards-compatible init signatures.
            sqs_poll_interval: Deprecated. Retained for backwards-compatible init signatures.
            mock_mode: If True, returns mock data without calling AWS
        """
        if sqs_queue_name != DEFAULT_SQS_QUEUE_NAME:
            warnings.warn(
                "sqs_queue_name is deprecated since v0.2.0 and has no effect. "
                "The client now uses synchronous RequestResponse invocation.",
                DeprecationWarning,
                stacklevel=2,
            )
        if sqs_poll_timeout != DEFAULT_SQS_POLL_TIMEOUT:
            warnings.warn(
                "sqs_poll_timeout is deprecated since v0.2.0 and has no effect. "
                "The client now uses synchronous RequestResponse invocation.",
                DeprecationWarning,
                stacklevel=2,
            )
        if sqs_poll_interval != DEFAULT_SQS_POLL_INTERVAL:
            warnings.warn(
                "sqs_poll_interval is deprecated since v0.2.0 and has no effect. "
                "The client now uses synchronous RequestResponse invocation.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.function_name = function_name
        self.sqs_queue_name = sqs_queue_name
        self.region = region
        self.sqs_poll_timeout = sqs_poll_timeout
        self.sqs_poll_interval = sqs_poll_interval
        self.mock_mode = mock_mode

        self._lambda: Optional[_LambdaClientProtocol] = None

        if not self.mock_mode:
            if boto3 is None:
                raise ImportError("boto3 is required for ATCLambdaClient (run: pip install boto3)")
            self._lambda = cast(_LambdaClientProtocol, boto3.client("lambda", region_name=region))

        logger.debug(
            f"ATCLambdaClient initialized: function='{function_name}', "
            f"region={region}, mock={mock_mode}"
        )

    def _require_lambda_client(self) -> _LambdaClientProtocol:
        if self._lambda is None:
            raise RuntimeError("Lambda client is not initialized")
        return self._lambda

    def _parse_lambda_payload(self, raw_body: str) -> dict[str, Any]:
        if not raw_body or raw_body == "null":
            raise RuntimeError("Lambda returned an empty response payload")

        body: Any = json.loads(raw_body)

        if isinstance(body, dict) and "body" in body and isinstance(body["body"], str):
            body = json.loads(body["body"])

        if not isinstance(body, dict):
            raise RuntimeError(f"Unexpected Lambda payload type: {type(body).__name__}")

        required_fields = {"batch_id", "results", "errors", "success_count", "error_count"}
        missing_fields = required_fields.difference(body.keys())
        if missing_fields:
            raise RuntimeError(
                "Lambda payload is missing required fields: "
                + ", ".join(sorted(missing_fields))
            )

        return cast(dict[str, Any], body)

    def invoke(
        self,
        symbols_data: list[dict[str, Any]],
        config: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Invoke Lambda and parse the direct ScanResult response.

        Args:
            symbols_data: List of {symbol, timeframes} dicts.
                          Each timeframe contains OHLVC lists.
            config: ATC config (uses DEFAULT_ATC_CONFIG if not provided)

        Returns:
            ScanResult dict: {batch_id, results, errors, success_count, error_count}
        """
        if config is None:
            config = copy.deepcopy(DEFAULT_ATC_CONFIG)

        if self.mock_mode:
            return self._mock_invoke(symbols_data, config)

        lambda_client = self._require_lambda_client()

        batch_id = f"req-{uuid.uuid4().hex[:8]}"
        payload = {
            "batch_id": batch_id,
            "symbols": symbols_data,
            "config": config,
        }

        logger.info(f"Invoking Lambda batch '{batch_id}' with {len(symbols_data)} symbols...")

        infra_exception_types: tuple[type[BaseException], ...] = ()
        if botocore_exceptions is not None:
            infra_exception_types = (
                botocore_exceptions.ClientError,
                botocore_exceptions.NoCredentialsError,
                botocore_exceptions.PartialCredentialsError,
                botocore_exceptions.EndpointConnectionError,
                botocore_exceptions.NoRegionError,
            )

        try:
            response = lambda_client.invoke(
                FunctionName=self.function_name,
                InvocationType="RequestResponse",
                Payload=json.dumps(payload).encode("utf-8"),
            )
        except Exception as e:
            if infra_exception_types and isinstance(e, infra_exception_types):
                logger.error(f"Lambda invocation failed due to AWS client misconfiguration: {e}")
            else:
                logger.error(f"Lambda invocation transport failed: {e}")
            raise

        try:
            status = response["StatusCode"]
            payload_stream = response["Payload"]
        except KeyError as e:
            raise RuntimeError(f"Lambda response missing required field: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Invalid Lambda response envelope: {e}") from e

        raw_body = payload_stream.read().decode("utf-8")

        if status != 200:
            raise RuntimeError(f"Lambda invocation failed (HTTP {status}): {raw_body}")

        if response.get("FunctionError"):
            body = json.loads(raw_body) if raw_body and raw_body != "null" else {}
            err_msg = body.get("errorMessage", raw_body)
            raise RuntimeError(f"Lambda function error [{response['FunctionError']}]: {err_msg}")

        result = self._parse_lambda_payload(raw_body)
        logger.info(
            "Lambda returned direct result payload "
            f"(request_batch_id={batch_id}, result_batch_id={result.get('batch_id')})"
        )
        return result

    def invoke_batch(
        self,
        symbols_data: list[dict[str, Any]],
        config: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Convenience alias for invoking a multi-symbol batch."""
        return self.invoke(symbols_data=symbols_data, config=config)

    def _mock_invoke(self, symbols_data: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
        """Return deterministic mock data when mock_mode is enabled."""
        _ = config
        batch_id = f"mock-{uuid.uuid4().hex[:8]}"
        results = [
            {
                "symbol": symbol_data.get("symbol", "UNKNOWN"),
                "score": 0.0,
                "signal_type": "NEUTRAL",
                "details": {tf: "MOCK" for tf in symbol_data.get("timeframes", {}).keys()},
                "strengths": {tf: 0.0 for tf in symbol_data.get("timeframes", {}).keys()},
            }
            for symbol_data in symbols_data
        ]
        return {
            "batch_id": batch_id,
            "results": results,
            "errors": [],
            "success_count": len(results),
            "error_count": 0,
        }

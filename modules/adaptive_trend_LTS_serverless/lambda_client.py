"""
ATC Lambda Client

Main client for invoking the ATC Serverless Lambda function.
Architecture:
  Lambda (atc-serverless) --[process]--> SQS queue (atc-results) --> this client polls SQS

This client:
  1. Invokes Lambda via boto3 (IAM-signed, bypassing Function URL auth/SCP restrictions)
  2. Polls SQS for the results
"""

import copy
import importlib.util
import json
import logging
import time
import uuid
from typing import Any, Optional, Protocol, cast

# Boto3 is required for Lambda/SQS access
try:
    import boto3  # type: ignore[import-untyped]
except ImportError:
    boto3 = None

# AWS Requests Auth is optional (only for direct HTTP w/ IAM auth)
HAS_AWS_AUTH = importlib.util.find_spec("aws_requests_auth") is not None

logger = logging.getLogger(__name__)

# Constants
DEFAULT_FUNCTION_NAME = "atc-serverless"
DEFAULT_SQS_QUEUE_NAME = "atc-results"
DEFAULT_REGION = "us-east-1"
DEFAULT_SQS_POLL_TIMEOUT = 60
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


class _SQSClientProtocol(Protocol):
    def get_queue_url(self, **kwargs: Any) -> dict[str, Any]:
        ...

    def receive_message(self, **kwargs: Any) -> dict[str, Any]:
        ...

    def delete_message(self, **kwargs: Any) -> dict[str, Any]:
        ...

    def change_message_visibility(self, **kwargs: Any) -> dict[str, Any]:
        ...


class ATCLambdaClient:
    """Invokes the ATC Serverless Lambda and polls SQS for results.

    The Lambda function does NOT return JSON results directly. It processes the batch
    and sends results to an SQS queue. This client handles the async pattern:
      invoke -> wait -> poll SQS -> return result
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
            sqs_queue_name: Name of the SQS queue for results
            region: AWS region
            sqs_poll_timeout: Seconds to wait for SQS result
            sqs_poll_interval: Seconds between SQS polls
            mock_mode: If True, returns mock data without calling AWS
        """
        self.function_name = function_name
        self.sqs_queue_name = sqs_queue_name
        self.region = region
        self.sqs_poll_timeout = sqs_poll_timeout
        self.sqs_poll_interval = sqs_poll_interval
        self.mock_mode = mock_mode

        self._queue_url: Optional[str] = None
        self._lambda: Optional[_LambdaClientProtocol] = None
        self._sqs: Optional[_SQSClientProtocol] = None

        if not self.mock_mode:
            if boto3 is None:
                raise ImportError("boto3 is required for ATCLambdaClient (run: pip install boto3)")
            self._lambda = cast(_LambdaClientProtocol, boto3.client("lambda", region_name=region))
            self._sqs = cast(_SQSClientProtocol, boto3.client("sqs", region_name=region))

        logger.debug(
            f"ATCLambdaClient initialized: function='{function_name}', "
            f"sqs='{sqs_queue_name}', region={region}, mock={mock_mode}"
        )

    def _require_lambda_client(self) -> _LambdaClientProtocol:
        if self._lambda is None:
            raise RuntimeError("Lambda client is not initialized")
        return self._lambda

    def _require_sqs_client(self) -> _SQSClientProtocol:
        if self._sqs is None:
            raise RuntimeError("SQS client is not initialized")
        return self._sqs

    def _get_queue_url(self) -> str:
        sqs_client = self._require_sqs_client()

        if not self._queue_url:
            response = sqs_client.get_queue_url(QueueName=self.sqs_queue_name)
            queue_url = response.get("QueueUrl")
            if not isinstance(queue_url, str) or not queue_url:
                raise RuntimeError(f"Unable to resolve queue URL for '{self.sqs_queue_name}'")
            self._queue_url = queue_url

        if self._queue_url is None:
            raise RuntimeError("Queue URL is not initialized")

        return self._queue_url

    def invoke(
        self,
        symbols_data: list[dict[str, Any]],
        config: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Invoke Lambda and poll SQS for the result.

        Args:
            symbols_data: List of {symbol, timeframes} dicts.
                          Each timeframe contains OHLVC lists.
            config: ATC config (uses DEFAULT_ATC_CONFIG if not provided)

        Returns:
            ScanResult dict: {batch_id, results, errors, success_count, error_count}
        """
        if config is None:
            config = copy.deepcopy(DEFAULT_ATC_CONFIG)

        # Handle mock mode
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

        try:
            # Step 1: Invoke Lambda (fire-and-forget, result goes to SQS)
            response = lambda_client.invoke(
                FunctionName=self.function_name,
                InvocationType="RequestResponse",  # Wait for Lambda to ACK receipt
                Payload=json.dumps(payload).encode("utf-8"),
            )

            status = response["StatusCode"]
            raw_body = response["Payload"].read().decode("utf-8")

            if status != 200:
                raise RuntimeError(f"Lambda invocation failed (HTTP {status}): {raw_body}")

            # Check for Lambda runtime errors (synchronous failure)
            if response.get("FunctionError"):
                body = json.loads(raw_body) if raw_body and raw_body != "null" else {}
                err_msg = body.get("errorMessage", raw_body)
                raise RuntimeError(f"Lambda function error [{response['FunctionError']}]: {err_msg}")

            logger.info(f"Lambda invoked OK (batch_id={batch_id}). Polling SQS for results...")

            # Step 2: Poll SQS for the result message matching our batch_id
            return self._poll_sqs_for_batch(batch_id)

        except Exception as e:
            logger.error(f"Lambda invocation failed: {e}")
            return {
                "batch_id": batch_id,
                "results": [],
                "errors": [{"symbol": "_batch", "error": str(e)}],
                "success_count": 0,
                "error_count": len(symbols_data),
            }

    def invoke_batch(
        self,
        symbols_data: list[dict[str, Any]],
        config: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Convenience alias for invoking a multi-symbol batch."""
        return self.invoke(symbols_data=symbols_data, config=config)

    def _poll_sqs_for_batch(self, batch_id: str) -> dict[str, Any]:
        """Poll SQS until we find the message for our batch_id."""
        sqs_client = self._require_sqs_client()

        queue_url = self._get_queue_url()
        deadline = time.time() + self.sqs_poll_timeout
        attempts = 0
        partial: dict[str, Any] = {
            "batch_id": batch_id,
            "results": [],
            "errors": [],
            "success_count": 0,
            "error_count": 0,
        }

        while time.time() < deadline:
            attempts += 1
            messages = sqs_client.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=10,
                WaitTimeSeconds=5,  # long-polling: up to 5s
                VisibilityTimeout=30,  # Hide message from others while processing
            ).get("Messages", [])

            for msg in messages:
                try:
                    body = json.loads(msg["Body"])
                    if body.get("batch_id") == batch_id:
                        # Found our result — delete from queue
                        sqs_client.delete_message(
                            QueueUrl=queue_url,
                            ReceiptHandle=msg["ReceiptHandle"],
                        )
                        # Merge partial data in case producer emits multiple messages.
                        partial["results"].extend(body.get("results", []))
                        partial["errors"].extend(body.get("errors", []))
                        partial["success_count"] = partial.get("success_count", 0) + int(
                            body.get("success_count", len(body.get("results", [])))
                        )
                        partial["error_count"] = partial.get("error_count", 0) + int(
                            body.get("error_count", len(body.get("errors", [])))
                        )

                        # If this looks like a complete ScanResult payload, return immediately.
                        if "success_count" in body and "error_count" in body:
                            logger.info(f"Got result from SQS (batch_id={batch_id}) after {attempts} attempt(s)")
                            return partial
                        logger.debug(
                            f"Received partial SQS payload for batch_id={batch_id}; continuing to poll for completion"
                        )
                    else:
                        # Not our message — put it back immediately
                        sqs_client.change_message_visibility(
                            QueueUrl=queue_url,
                            ReceiptHandle=msg["ReceiptHandle"],
                            VisibilityTimeout=0,
                        )
                except json.JSONDecodeError:
                    logger.debug("Skipping malformed SQS message")
                    continue

            remaining = int(deadline - time.time())
            if remaining > 0:
                logger.debug(f"Waiting for SQS result... ({remaining}s remaining)")
                time.sleep(self.sqs_poll_interval)

        if partial.get("results") or partial.get("errors"):
            logger.warning(
                f"SQS polling timeout for batch_id='{batch_id}' after {self.sqs_poll_timeout}s; "
                "returning partial results"
            )
            return partial

        raise TimeoutError(
            f"No SQS result received for batch_id='{batch_id}' "
            f"within {self.sqs_poll_timeout}s. "
            "Check CloudWatch logs for Lambda errors."
        )

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

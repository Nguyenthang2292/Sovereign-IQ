#!/usr/bin/env python3
"""
Example Python client for ATC Serverless.

This script demonstrates how to use the ATC Serverless module with Python,
including both synchronous and asynchronous invocation patterns.
"""

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List

import boto3

# Add the parent directory to Python path to import atc_serverless
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ATCServerlessClient:
    def __init__(
        self,
        lambda_function_name: str = "atc-serverless",
        sqs_queue_name: str = "atc-results",
        region: str = "us-east-1",
    ):
        """
        Initialize the ATC Serverless client.

        Args:
            lambda_function_name: Name of the Lambda function
            sqs_queue_name: Name of the SQS queue for results
            region: AWS region
        """
        self.lambda_client = boto3.client("lambda", region_name=region)
        self.sqs_client = boto3.client("sqs", region_name=region)
        self.lambda_name = lambda_function_name
        self.sqs_name = sqs_queue_name
        self.sqs_url = self._get_sqs_url()

        # Set up logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

    def _get_sqs_url(self) -> str:
        """Get the SQS queue URL."""
        try:
            response = self.sqs_client.get_queue_url(QueueName=self.sqs_name)
            return response["QueueUrl"]
        except Exception as e:
            self.logger.error(f"Failed to get SQS queue URL: {e}")
            raise

    def _wait_for_results(self, batch_id: str, timeout: int = 30, poll_interval: float = 1.0) -> List[Dict[str, Any]]:
        """
        Wait for results from the SQS queue.

        Args:
            batch_id: The batch ID to wait for
            timeout: Maximum wait time in seconds
            poll_interval: Time between polls in seconds

        Returns:
            List of results for the batch
        """
        start_time = time.time()
        results = []

        self.logger.info(f"Waiting for results for batch {batch_id} (timeout: {timeout}s)")

        while time.time() - start_time < timeout:
            try:
                response = self.sqs_client.receive_message(
                    QueueUrl=self.sqs_url, MaxNumberOfMessages=10, WaitTimeSeconds=0, AttributeNames=["All"]
                )

                if "Messages" in response:
                    for message in response["Messages"]:
                        body = json.loads(message["Body"])
                        if body.get("batch_id") == batch_id:
                            results.append(body)
                            # Delete the message after processing
                            self.sqs_client.delete_message(
                                QueueUrl=self.sqs_url, ReceiptHandle=message["ReceiptHandle"]
                            )
                            self.logger.debug(f"Processed result for symbol {body.get('symbol')}")

                if results:
                    self.logger.info(f"Received {len(results)} results for batch {batch_id}")
                    return results

                time.sleep(poll_interval)

            except Exception as e:
                self.logger.error(f"Error waiting for results: {e}")
                time.sleep(poll_interval)

        self.logger.warning(f"Timeout waiting for batch {batch_id} results")
        raise TimeoutError(f"Timeout waiting for batch {batch_id} results after {timeout}s")

    def invoke_lambda_sync(self, symbols: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke the Lambda function synchronously (for small batches).

        Args:
            symbols: List of symbol data
            config: ATC configuration

        Returns:
            Dictionary containing results and errors
        """
        payload = {"batch_id": f"batch-{int(time.time())}", "symbols": symbols, "config": config}

        self.logger.info(f"Invoking Lambda synchronously with {len(symbols)} symbols")

        try:
            response = self.lambda_client.invoke(
                FunctionName=self.lambda_name, InvocationType="RequestResponse", Payload=json.dumps(payload)
            )

            if response["StatusCode"] != 200:
                error_msg = f"Lambda invocation failed with status {response['StatusCode']}"
                self.logger.error(error_msg)
                raise Exception(error_msg)

            result = json.loads(response["Payload"].read().decode("utf-8"))
            self.logger.info(f"Synchronous invocation completed: {len(result.get('results', []))} results")
            return result

        except Exception as e:
            self.logger.error(f"Lambda invocation failed: {e}")
            raise

    def invoke_lambda_async(self, symbols: List[Dict[str, Any]], config: Dict[str, Any]) -> str:
        """
        Invoke the Lambda function asynchronously (recommended for production).

        Args:
            symbols: List of symbol data
            config: ATC configuration

        Returns:
            Batch ID to track results
        """
        payload = {"batch_id": f"batch-{int(time.time())}", "symbols": symbols, "config": config}

        self.logger.info(f"Invoking Lambda asynchronously with {len(symbols)} symbols")

        try:
            response = self.lambda_client.invoke(
                FunctionName=self.lambda_name, InvocationType="Event", Payload=json.dumps(payload)
            )

            if response["StatusCode"] != 202:
                error_msg = f"Lambda async invocation failed with status {response['StatusCode']}"
                self.logger.error(error_msg)
                raise Exception(error_msg)

            batch_id = payload["batch_id"]
            self.logger.info(f"Async invocation started, batch ID: {batch_id}")
            return batch_id

        except Exception as e:
            self.logger.error(f"Lambda async invocation failed: {e}")
            raise

    def process_batch_sync(self, symbols: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a batch of symbols synchronously.

        Args:
            symbols: List of symbol data
            config: ATC configuration

        Returns:
            Dictionary containing results and errors
        """
        return self.invoke_lambda_sync(symbols, config)

    def process_batch_async(
        self, symbols: List[Dict[str, Any]], config: Dict[str, Any], timeout: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of symbols asynchronously and wait for results.

        Args:
            symbols: List of symbol data
            config: ATC configuration
            timeout: Maximum wait time for results

        Returns:
            List of results for the batch
        """
        batch_id = self.invoke_lambda_async(symbols, config)
        return self._wait_for_results(batch_id, timeout=timeout)


def create_sample_symbols(num_symbols: int = 1) -> List[Dict[str, Any]]:
    """Create sample symbol data for testing.

    Generates 50 bars of OHLCV data per timeframe to meet MIN_DATA_LENGTH requirements.
    """
    symbols = []

    # Generate 50 bars of sample data
    num_bars_1h = 50
    num_bars_4h = 50

    for i in range(num_symbols):
        # Generate 1h timeframe data with 50 bars
        base_price = 42000.0 + i * 100
        timestamps_1h = [1704067200 + j * 3600 for j in range(num_bars_1h)]
        closes_1h = [base_price + j * 0.5 + (j % 5) * 10 for j in range(num_bars_1h)]
        opens_1h = [closes_1h[j] - 50.0 for j in range(num_bars_1h)]
        highs_1h = [closes_1h[j] + 100.0 for j in range(num_bars_1h)]
        lows_1h = [closes_1h[j] - 100.0 for j in range(num_bars_1h)]
        volumes_1h = [100.0 + j * 10 for j in range(num_bars_1h)]

        # Generate 4h timeframe data with 50 bars
        timestamps_4h = [1704067200 + j * 14400 for j in range(num_bars_4h)]
        closes_4h = [base_price + j * 2.0 + (j % 3) * 20 for j in range(num_bars_4h)]
        opens_4h = [closes_4h[j] - 100.0 for j in range(num_bars_4h)]
        highs_4h = [closes_4h[j] + 200.0 for j in range(num_bars_4h)]
        lows_4h = [closes_4h[j] - 200.0 for j in range(num_bars_4h)]
        volumes_4h = [400.0 + j * 50 for j in range(num_bars_4h)]

        symbol_data = {
            "symbol": f"BTCUSDT_{i}",
            "timeframes": {
                "1h": {
                    "timestamp": timestamps_1h,
                    "open": opens_1h,
                    "high": highs_1h,
                    "low": lows_1h,
                    "close": closes_1h,
                    "volume": volumes_1h,
                },
                "4h": {
                    "timestamp": timestamps_4h,
                    "open": opens_4h,
                    "high": highs_4h,
                    "low": lows_4h,
                    "close": closes_4h,
                    "volume": volumes_4h,
                },
            },
        }
        symbols.append(symbol_data)

    return symbols


def create_atc_config() -> Dict[str, Any]:
    """Create sample ATC configuration."""
    return {
        "weights": {"1h": 0.6, "4h": 0.4},
        "threshold": 0.3,
        "min_signal": 0.0,
        "use_signal_strength": True,
        "lambda_param": 0.02,
        "decay": 0.03,
        "cutout": 0,
        "equity_floor": 0.25,
        "ma_configs": [{"ma_type": "EMA", "length": 12, "weight": 1.0}],
    }


def main():
    """Main function to demonstrate the Python client."""
    print("ATC Serverless Python Client Example")
    print("=" * 40)

    try:
        # Initialize client
        client = ATCServerlessClient()

        # Create sample data
        symbols = create_sample_symbols(num_symbols=3)
        config = create_atc_config()

        print(f"Created {len(symbols)} sample symbols")
        print(f"Configuration: {config}")
        print()

        # Process batch synchronously (for demonstration)
        print("Processing batch synchronously...")
        start_time = time.time()
        result = client.process_batch_sync(symbols, config)
        sync_time = time.time() - start_time

        print(f"Synchronous processing completed in {sync_time:.2f}s")
        print(f"Results: {len(result.get('results', []))} successful, {len(result.get('errors', []))} errors")
        print()

        # Process batch asynchronously (recommended for production)
        print("Processing batch asynchronously...")
        start_time = time.time()
        results = client.process_batch_async(symbols, config, timeout=10)
        async_time = time.time() - start_time

        print(f"Asynchronous processing completed in {async_time:.2f}s")
        print(f"Received {len(results)} results:")

        for result in results:
            symbol = result.get("symbol", "unknown")
            signal_type = result.get("signal_type", "unknown")
            score = result.get("score", 0.0)
            print(f"  {symbol}: {signal_type} (score: {score:.4f})")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

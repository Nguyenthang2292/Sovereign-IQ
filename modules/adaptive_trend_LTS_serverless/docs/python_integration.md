# Python Integration Guide

This guide covers how to integrate the ATC Serverless Rust module with Python applications, including deployment patterns, API usage, and best practices.

## Overview

The ATC Serverless module is designed to be integrated with Python applications through various deployment patterns:

1. **API Gateway → Lambda → SQS**: Python client → AWS API Gateway → Rust Lambda → SQS Results
2. **Direct Lambda Invocation**: Python client → Rust Lambda (async)
3. **Event-Driven**: Python application → SQS → Rust Lambda
4. **Batch Processing**: Python batch processor → Rust Lambda

## Deployment Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Python Client │────▶│ AWS API Gateway  │───▶│   Rust Lambda   │────▶│   SQS Queue   │
│   (Your App)    │     │   (REST API)     │     │ (ATC Serverless)│      │   (Results)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Python Client Example

Here's a complete Python client example that demonstrates how to use the ATC Serverless module:

```python
import boto3
import json
import time
from typing import List, Dict, Any, Optional
import requests
import logging

class ATCServerlessClient:
    def __init__(self, 
                 lambda_function_name: str = "atc-serverless",
                 sqs_queue_name: str = "atc-results",
                 region: str = "us-east-1"):
        """
        Initialize the ATC Serverless client.
        
        Args:
            lambda_function_name: Name of the Lambda function
            sqs_queue_name: Name of the SQS queue for results
            region: AWS region
        """
        self.lambda_client = boto3.client('lambda', region_name=region)
        self.sqs_client = boto3.client('sqs', region_name=region)
        self.lambda_name = lambda_function_name
        self.sqs_name = sqs_queue_name
        self.sqs_url = self._get_sqs_url()
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _get_sqs_url(self) -> str:
        """Get the SQS queue URL."""
        response = self.sqs_client.get_queue_url(QueueName=self.sqs_name)
        return response['QueueUrl']

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
        
        while time.time() - start_time < timeout:
            try:
                response = self.sqs_client.receive_message(
                    QueueUrl=self.sqs_url,
                    MaxNumberOfMessages=10,
                    WaitTimeSeconds=0,
                    AttributeNames=['All']
                )
                
                if 'Messages' in response:
                    for message in response['Messages']:
                        body = json.loads(message['Body'])
                        if body.get('batch_id') == batch_id:
                            results.append(body)
                            # Delete the message after processing
                            self.sqs_client.delete_message(
                                QueueUrl=self.sqs_url,
                                ReceiptHandle=message['ReceiptHandle']
                            )
                
                if results:
                    return results
                
                time.sleep(poll_interval)
                
            except Exception as e:
                self.logger.error(f"Error waiting for results: {e}")
                time.sleep(poll_interval)
        
        raise TimeoutError(f"Timeout waiting for batch {batch_id} results")

    def invoke_lambda_sync(self, symbols: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke the Lambda function synchronously (for small batches).
        
        Args:
            symbols: List of symbol data
            config: ATC configuration
            
        Returns:
            Dictionary containing results and errors
        """
        payload = {
            "batch_id": f"batch-{int(time.time())}",
            "symbols": symbols,
            "config": config
        }
        
        try:
            response = self.lambda_client.invoke(
                FunctionName=self.lambda_name,
                InvocationType='RequestResponse',
                Payload=json.dumps(payload)
            )
            
            if response['StatusCode'] != 200:
                raise Exception(f"Lambda invocation failed with status {response['StatusCode']}")
                
            result = json.loads(response['Payload'].read().decode('utf-8'))
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
        payload = {
            "batch_id": f"batch-{int(time.time())}",
            "symbols": symbols,
            "config": config
        }
        
        try:
            response = self.lambda_client.invoke(
                FunctionName=self.lambda_name,
                InvocationType='Event',
                Payload=json.dumps(payload)
            )
            
            if response['StatusCode'] != 202:
                raise Exception(f"Lambda async invocation failed with status {response['StatusCode']}")
                
            # Extract batch ID from the payload
            batch_id = payload['batch_id']
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

    def process_batch_async(self, symbols: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a batch of symbols asynchronously and wait for results.
        
        Args:
            symbols: List of symbol data
            config: ATC configuration
            
        Returns:
            List of results for the batch
        """
        batch_id = self.invoke_lambda_async(symbols, config)
        return self._wait_for_results(batch_id)

# Example usage
if __name__ == "__main__":
    # Initialize client
    client = ATCServerlessClient()
    
    # Sample symbol data (BTC/USDT 1h and 4h timeframes)
    symbols = [
        {
            "symbol": "BTCUSDT",
            "timeframes": {
                "1h": {
                    "timestamp": [1704067200, 1704070800, 1704074400],
                    "open": [42000.0, 42100.0, 42200.0],
                    "high": [42200.0, 42300.0, 42400.0],
                    "low": [41900.0, 42000.0, 42100.0],
                    "close": [42100.0, 42200.0, 42300.0],
                    "volume": [100.0, 150.0, 200.0]
                },
                "4h": {
                    "timestamp": [1704067200, 1704070800],
                    "open": [41900.0, 42100.0],
                    "high": [42200.0, 42300.0],
                    "low": [41800.0, 42000.0],
                    "close": [42000.0, 42200.0],
                    "volume": [400.0, 450.0]
                }
            }
        }
    ]
    
    # ATC configuration
    config = {
        "weights": {"1h": 0.6, "4h": 0.4},
        "threshold": 0.3,
        "min_signal": 0.0,
        "use_signal_strength": True,
        "lambda_param": 0.02,
        "decay": 0.03,
        "cutout": 0,
        "equity_floor": 0.25,
        "ma_configs": [
            {"ma_type": "EMA", "length": 12, "weight": 1.0}
        ]
    }
    
    # Process batch synchronously (for testing)
    print("Processing batch synchronously...")
    result = client.process_batch_sync(symbols, config)
    print(f"Results: {len(result.get('results', []))} successful, {len(result.get('errors', []))} errors")
    
    # Process batch asynchronously (recommended for production)
    print("\nProcessing batch asynchronously...")
    results = client.process_batch_async(symbols, config)
    print(f"Async results received: {len(results)}")
```

## Error Handling Patterns

### Retry Logic

```python
import backoff

class ATCServerlessClient:
    # ... (previous code) ...
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def invoke_lambda_with_retry(self, symbols, config, async_mode=True):
        if async_mode:
            return self.invoke_lambda_async(symbols, config)
        else:
            return self.invoke_lambda_sync(symbols, config)
```

### Timeout Handling

```python
def process_batch_with_timeout(self, symbols, config, timeout=30):
    try:
        return self.process_batch_async(symbols, config, timeout=timeout)
    except TimeoutError:
        self.logger.error("Processing timed out")
        # Implement fallback logic or alerting
        raise
```

## Performance Considerations

### Batch Size Optimization

- **Small batches (10-50 symbols)**: Use synchronous invocation for low latency
- **Medium batches (50-500 symbols)**: Use asynchronous invocation with appropriate timeout
- **Large batches (500+ symbols)**: Consider splitting into multiple Lambda invocations

### Memory Management

Monitor Lambda memory usage and adjust batch sizes accordingly:

```python
def estimate_memory_usage(symbols):
    """Estimate memory usage based on number of symbols and timeframes"""
    total_bars = 0
    for symbol in symbols:
        for tf_data in symbol['timeframes'].values():
            total_bars += len(tf_data['close'])
    # Rough estimate: ~55KB per symbol
    return len(symbols) * 55  # KB
```

## Configuration Mapping

### Python ATC vs Rust ATC Configuration

| Python Parameter | Rust Parameter | Description |
|-----------------|----------------|-------------|
| `timeframe_weights` | `weights` | Timeframe weighting for aggregation |
| `signal_threshold` | `threshold` | Signal classification threshold |
| `min_signal_strength` | `min_signal` | Minimum signal strength to consider |
| `use_signal_strength` | `use_signal_strength` | Enable signal strength weighting |
| `lambda_param` | `lambda_param` | Lambda parameter for equity calculation |
| `decay_factor` | `decay` | Decay factor for equity weighting |
| `initial_bars_to_skip` | `cutout` | Number of initial bars to cut out |
| `equity_floor` | `equity_floor` | Minimum equity value for stability |
| `ma_configurations` | `ma_configs` | List of MA type configurations |

## Migration Guide

For detailed migration instructions from Python ATC to Rust ATC, see the [Migration Guide](MIGRATION_GUIDE.md).

## Best Practices

1. **Use Async Invocation**: Always prefer asynchronous invocation for production workloads
2. **Implement Retry Logic**: Add exponential backoff for Lambda invocations
3. **Monitor Results**: Use SQS to track processing status and results
4. **Batch Optimization**: Tune batch sizes based on Lambda memory and performance requirements
5. **Error Handling**: Implement comprehensive error handling and fallback mechanisms
6. **Logging**: Enable detailed logging for debugging and monitoring

## Next Steps

1. [Migration Guide](MIGRATION_GUIDE.md) - Detailed migration instructions
2. [Python Client Example](examples/python_client.py) - Complete working example
3. [API Reference](../src/lib.rs) - Full Rust API documentation
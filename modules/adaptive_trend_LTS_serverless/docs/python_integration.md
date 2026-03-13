# Python Integration Guide

This guide documents Python integration for `adaptive_trend_LTS_serverless`
using the **current invoke contract**.

## Runtime Contract

```
Python -> ATCLambdaClient -> boto3 Lambda Invoke (RequestResponse) -> ScanResult
```

The response payload already contains final results:

- `batch_id`
- `results`
- `errors`
- `success_count`
- `error_count`
- per-result optional fields:
  - `average_signal_raw` (raw causal score contract)
  - `average_signal_exec` (optional execution-view field, often omitted for snapshot API)

No SQS polling is required.

## Recommended Client

Use:

- `modules/adaptive_trend_LTS_serverless/lambda_client.py`
- `ATCLambdaClient.invoke(...)`

## Minimal Example

```python
from modules.adaptive_trend_LTS_serverless.lambda_client import (
    ATCLambdaClient,
    DEFAULT_ATC_CONFIG,
)

symbols_data = [
    {
        "symbol": "BTCUSDT",
        "timeframes": {
            "1h": {
                "timestamp": [1704067200, 1704070800, 1704074400],
                "open": [42000.0, 42100.0, 42200.0],
                "high": [42200.0, 42300.0, 42400.0],
                "low": [41900.0, 42000.0, 42100.0],
                "close": [42100.0, 42200.0, 42300.0],
                "volume": [100.0, 120.0, 130.0],
            },
            "4h": {
                "timestamp": [1704067200, 1704081600, 1704096000],
                "open": [41900.0, 42100.0, 42200.0],
                "high": [42200.0, 42400.0, 42500.0],
                "low": [41800.0, 42000.0, 42100.0],
                "close": [42000.0, 42200.0, 42300.0],
                "volume": [300.0, 320.0, 340.0],
            },
        },
    }
]

client = ATCLambdaClient(function_name="atc-serverless", region="us-east-1")
result = client.invoke(
    symbols_data=symbols_data,
    config=DEFAULT_ATC_CONFIG,
    apply_strategy_shift=False,  # adapter hint; core stays raw
)

print(result["success_count"], result["error_count"])
for row in result["results"]:
    print(row["symbol"], row["signal_type"], row["score"])
```

## Error Handling Behavior

`ATCLambdaClient` distinguishes:

- infrastructure misconfiguration (for example missing credentials) -> re-raises
- invocation/runtime issues -> returns error payload in `errors`

Recommended caller pattern:

```python
try:
    out = client.invoke(symbols_data, config)
except Exception as exc:
    # infra/config failure
    raise

if out["error_count"] > 0:
    # symbol-level or batch-level processing errors
    ...
```

## Configuration Notes

- `threshold` controls long/short decision boundary.
- `weights` must match provided timeframes.
- `ma_configs` controls MA types, lengths, and static weights.
- `equity_floor` sets minimum Layer 2 equity contribution.

## Deprecated Constructor Arguments

`ATCLambdaClient` still accepts legacy SQS-related init arguments for backward
signature compatibility:

- `sqs_queue_name`
- `sqs_poll_timeout`
- `sqs_poll_interval`

They are ignored by the current direct-response architecture.

## Validation and Data Contract

Input must satisfy:

- non-empty symbol list
- OHLCV arrays with equal lengths
- configured timeframes present in each symbol payload
- finite numeric values

Invalid payloads are rejected by server-side validation.

## Execution Shift Migration (2026-03-14)

- Core Rust signal computation is **raw-only**.
- `apply_strategy_shift` is an adapter-level request hint and does not mutate core score logic.
- For snapshot responses, `average_signal_raw` mirrors `score`.
- `average_signal_exec` may be omitted when historical bars are not returned.
- Legacy consumers that treated `score`/`Average_Signal` as already shifted must migrate
  execution logic to explicit adapter-side shift handling.

## Related Files

- `modules/adaptive_trend_LTS_serverless/lambda_client.py`
- `modules/adaptive_trend_LTS_serverless/scripts/binance_lambda_demo.py`
- `modules/adaptive_trend_LTS_serverless/docs/aws/aws_setup_deployment_guide.md`


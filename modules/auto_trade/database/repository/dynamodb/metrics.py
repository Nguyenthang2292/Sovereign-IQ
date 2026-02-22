"""
DynamoDB Metrics and Structured Logging
======================================

Emits CloudWatch EMF-compatible metrics and structured logs for DynamoDB operations.

Created: 2026-02-20
"""

import json
from modules.common.ui.logging import log_info, log_error, log_warn, log_debug, log_success, log_system
import time
from typing import Any, Dict, Optional


NAMESPACE = "AutoTrade/DynamoDB"

_OPERATION_TO_METRIC = {
    "GetItem": "GetItemLatency",
    "Query": "QueryLatency",
    "Scan": "QueryLatency",
    "PutItem": "WriteLatency",
    "UpdateItem": "WriteLatency",
    "DeleteItem": "WriteLatency",
    "TransactWriteItems": "WriteLatency",
}


def _sanitize_log_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _emit_emf(metric_name: str, value: float, operation: str, table: str) -> None:
    payload = {
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [
                {
                    "Namespace": NAMESPACE,
                    "Dimensions": [["Operation", "Table"]],
                    "Metrics": [{"Name": metric_name, "Unit": "Milliseconds"}],
                }
            ],
        },
        "Operation": operation,
        "Table": table,
        metric_name: float(value),
    }
    log_info(json.dumps(payload, default=str))


def _emit_error_count(operation: str, table: str) -> None:
    payload = {
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [
                {
                    "Namespace": NAMESPACE,
                    "Dimensions": [["Operation", "Table"]],
                    "Metrics": [{"Name": "ErrorCount", "Unit": "Count"}],
                }
            ],
        },
        "Operation": operation,
        "Table": table,
        "ErrorCount": 1.0,
    }
    log_info(json.dumps(payload, default=str))


def log_dynamodb_success(
    operation: str,
    table: str,
    latency_ms: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    metric_name = _OPERATION_TO_METRIC.get(operation)
    if metric_name:
        _emit_emf(metric_name, latency_ms, operation, table)

    log_payload: Dict[str, Any] = {
        "event": f"dynamodb.{operation.lower()}",
        "table": table,
        "latency_ms": round(latency_ms, 2),
    }

    if metadata:
        log_payload.update({key: _sanitize_log_value(value) for key, value in metadata.items()})

    log_info(json.dumps(log_payload, default=str))


def log_dynamodb_error(
    operation: str,
    table: str,
    latency_ms: float,
    error: Exception,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    _emit_error_count(operation, table)

    log_payload: Dict[str, Any] = {
        "event": f"dynamodb.{operation.lower()}.error",
        "table": table,
        "latency_ms": round(latency_ms, 2),
        "error_type": type(error).__name__,
        "error_message": str(error),
    }

    if metadata:
        log_payload.update({key: _sanitize_log_value(value) for key, value in metadata.items()})

    log_error(json.dumps(log_payload, default=str))

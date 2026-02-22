"""
DynamoDB Client Initialization
==============================

Singleton client and table resources for DynamoDB.
Safe for use in AWS Lambda.

Created: 2026-02-20
"""

import os
import time
from functools import lru_cache
from typing import Any, Callable, Dict

import boto3

from .metrics import log_dynamodb_error, log_dynamodb_success

# Configuration
TABLE_NAME = os.getenv("DYNAMODB_TABLE_NAME", "AutoTrade")
AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ap-southeast-1"
ENDPOINT_URL = os.getenv("DYNAMODB_ENDPOINT_URL")


class InstrumentedDynamoTable:
    """Table proxy that logs structured events and emits EMF metrics."""

    def __init__(self, table: Any):
        self._table = table

    @property
    def name(self) -> str:
        return self._table.name

    @property
    def meta(self) -> Any:
        return self._table.meta

    def _call(self, operation: str, method: Callable[..., Any], **kwargs: Any) -> Any:
        start = time.perf_counter()
        metadata: Dict[str, Any] = {
            "pk": kwargs.get("Key", {}).get("pk") if isinstance(kwargs.get("Key"), dict) else None,
            "index": kwargs.get("IndexName"),
        }

        try:
            result = method(**kwargs)
            latency_ms = (time.perf_counter() - start) * 1000
            metadata["hit"] = bool(result.get("Item")) if isinstance(result, dict) and operation == "GetItem" else None
            log_dynamodb_success(operation=operation, table=self.name, latency_ms=latency_ms, metadata=metadata)
            return result
        except Exception as error:
            latency_ms = (time.perf_counter() - start) * 1000
            log_dynamodb_error(
                operation=operation,
                table=self.name,
                latency_ms=latency_ms,
                error=error,
                metadata=metadata,
            )
            raise

    def get_item(self, **kwargs: Any) -> Any:
        return self._call("GetItem", self._table.get_item, **kwargs)

    def query(self, **kwargs: Any) -> Any:
        return self._call("Query", self._table.query, **kwargs)

    def scan(self, **kwargs: Any) -> Any:
        return self._call("Scan", self._table.scan, **kwargs)

    def put_item(self, **kwargs: Any) -> Any:
        return self._call("PutItem", self._table.put_item, **kwargs)

    def update_item(self, **kwargs: Any) -> Any:
        return self._call("UpdateItem", self._table.update_item, **kwargs)

    def delete_item(self, **kwargs: Any) -> Any:
        return self._call("DeleteItem", self._table.delete_item, **kwargs)

    def batch_writer(self, **kwargs: Any) -> Any:
        return self._table.batch_writer(**kwargs)


class InstrumentedDynamoClient:
    """Client proxy that logs structured events and emits EMF metrics."""

    def __init__(self, client: Any):
        self._client = client

    def transact_write_items(self, **kwargs: Any) -> Any:
        start = time.perf_counter()
        item_count = len(kwargs.get("TransactItems", []))
        metadata = {"items": item_count}

        try:
            result = self._client.transact_write_items(**kwargs)
            latency_ms = (time.perf_counter() - start) * 1000
            log_dynamodb_success(
                operation="TransactWriteItems",
                table=TABLE_NAME,
                latency_ms=latency_ms,
                metadata=metadata,
            )
            return result
        except Exception as error:
            latency_ms = (time.perf_counter() - start) * 1000
            log_dynamodb_error(
                operation="TransactWriteItems",
                table=TABLE_NAME,
                latency_ms=latency_ms,
                error=error,
                metadata=metadata,
            )
            raise

    def __getattr__(self, item: str) -> Any:
        return getattr(self._client, item)


@lru_cache(maxsize=1)
def get_dynamodb_resource():
    """
    Get a cached boto3 DynamoDB resource.
    Safe for reuse across Lambda invocations.
    """
    kwargs = {"region_name": AWS_REGION}
    if ENDPOINT_URL:
        kwargs["endpoint_url"] = ENDPOINT_URL

    return boto3.resource("dynamodb", **kwargs)


@lru_cache(maxsize=1)
def get_dynamodb_client():
    """
    Get a cached boto3 DynamoDB client.
    Safe for reuse across Lambda invocations.
    """
    kwargs = {"region_name": AWS_REGION}
    if ENDPOINT_URL:
        kwargs["endpoint_url"] = ENDPOINT_URL

    client = boto3.client("dynamodb", **kwargs)
    return InstrumentedDynamoClient(client)


@lru_cache(maxsize=1)
def get_dynamodb_table():
    """
    Get a cached Table resource for AutoTrade.
    Safe for reuse across Lambda invocations.
    """
    dynamodb = get_dynamodb_resource()
    table = dynamodb.Table(TABLE_NAME)
    return InstrumentedDynamoTable(table)


def reset_connections():
    """
    Clear the cached connections. Useful for testing.
    """
    get_dynamodb_resource.cache_clear()
    get_dynamodb_client.cache_clear()
    get_dynamodb_table.cache_clear()

"""
DynamoDB Type Serializer
========================

Handles conversion between Python types (float, datetime, None)
and DynamoDB supported types (Decimal, ISO strings, missing keys).

Created: 2026-02-20
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict


def _normalize_for_dynamo(value: Any) -> Any:
    """Recursively normalize Python values for DynamoDB compatibility."""
    if isinstance(value, dict):
        return {key: _normalize_for_dynamo(val) for key, val in value.items() if val is not None}

    if isinstance(value, list):
        return [_normalize_for_dynamo(item) for item in value]

    if isinstance(value, float):
        return float_to_decimal(value)

    if isinstance(value, datetime):
        return datetime_to_iso(value)

    return value


def clean_none_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove all None values from a dictionary, as DynamoDB doesn't accept None.
    Also handles nested dictionaries.
    """
    cleaned = {}
    for k, v in data.items():
        if v is None:
            continue
        elif isinstance(v, dict):
            cleaned[k] = clean_none_values(v)
        else:
            cleaned[k] = v
    return cleaned


def float_to_decimal(value: Any) -> Any:
    """Convert float to Decimal for DynamoDB storage."""
    if isinstance(value, float):
        # Convert through string to avoid precision issues
        # e.g. float 1.1 -> Decimal('1.1') instead of Decimal('1.100000000000000088...')
        return Decimal(str(value))
    return value


def decimal_to_float(value: Any) -> Any:
    """Convert Decimal back to float for application use."""
    if isinstance(value, Decimal):
        return float(value)
    return value


def datetime_to_iso(value: Any) -> Any:
    """Convert datetime objects to ISO-8601 strings."""
    if isinstance(value, datetime):
        # Ensure timezone info is present, default to UTC
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return value


def to_dynamo_item(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare a dictionary for DynamoDB storage:
    1. Remove None values
    2. Convert floats to Decimals
    3. Convert datetimes to ISO strings
    """
    item = clean_none_values(data)
    return _normalize_for_dynamo(item)


def from_dynamo_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a DynamoDB item back to application format:
    1. Convert Decimals to floats
    """
    if not item:
        return {}

    result = {}
    for key, value in item.items():
        if isinstance(value, Decimal):
            result[key] = decimal_to_float(value)
        else:
            result[key] = value

    return result

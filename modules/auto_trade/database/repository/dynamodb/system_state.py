"""
DynamoDB System State Repository
================================

Phase 2: DynamoDB implementation of SystemStateRepository.

Created: 2026-02-20
"""

from datetime import datetime, timezone
from typing import Any, Optional

from ..base import SystemStateRepository
from .client import get_dynamodb_table
from .keys import DynamoKeys
from .serializer import from_dynamo_item, to_dynamo_item


class DynamoDBSystemStateRepository(SystemStateRepository):
    """DynamoDB implementation of SystemStateRepository."""

    def __init__(self):
        self._table = get_dynamodb_table()

    def get_system_state(self, key: str, default: Optional[Any] = None) -> Optional[Any]:
        # Parse category if it's dot notation, e.g., "trading.enabled"
        parts = key.split(".")
        category = parts[0] if len(parts) > 1 else "global"

        response = self._table.get_item(Key={"pk": DynamoKeys.state_pk(category), "sk": DynamoKeys.state_sk(key)})
        item = response.get("Item")

        # If not found under 'global' and key has no dot, do a targeted scan
        # (handles cases where state was written with explicit category like category='system')
        if not item and len(parts) == 1:
            from boto3.dynamodb.conditions import Attr

            scan_response = self._table.scan(
                FilterExpression=Attr("entity_type").eq("STATE") & Attr("key").eq(key),
            )
            items = scan_response.get("Items", [])
            # If multiple items (different categories), pick most recently updated
            if items:
                items.sort(key=lambda x: x.get("updated_at", x.get("gsi2sk", "")), reverse=True)
            item = items[0] if items else None

        if not item:
            return default

        item = from_dynamo_item(item)

        # Apply type casting based on value_type
        value = item.get("value")
        value_type = item.get("value_type", "string")

        if value is None:
            return default

        if value_type == "integer":
            return int(value)
        elif value_type == "float":
            return float(value)
        elif value_type == "boolean":
            # DynamoDB handles booleans natively, but handle strings just in case
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes")
            return bool(value)
        elif value_type == "json":
            # Assuming the serializer handles nested dicts
            return value
        else:
            return str(value)

    def set_system_state(
        self,
        key: str,
        value: Any,
        value_type: str = "string",
        description: Optional[str] = None,
        category: Optional[str] = None,
    ) -> bool:
        # Determine category
        if not category:
            parts = key.split(".")
            category = parts[0] if len(parts) > 1 else "global"

        now = datetime.now(timezone.utc).isoformat()

        # Build the item
        entity = {
            "pk": DynamoKeys.state_pk(category),
            "sk": DynamoKeys.state_sk(key),
            "entity_type": "STATE",
            "key": key,
            "value": value,
            "value_type": value_type,
            "category": category,
            "updated_at": now,
            # GSI index keys so STATE items can be queried via GSI2
            "gsi2pk": "STATE",
            "gsi2sk": now,
        }

        if description:
            entity["description"] = description

        item = to_dynamo_item(entity)

        try:
            # We use put_item which does an upsert natively
            self._table.put_item(Item=item)
            return True
        except Exception:
            return False

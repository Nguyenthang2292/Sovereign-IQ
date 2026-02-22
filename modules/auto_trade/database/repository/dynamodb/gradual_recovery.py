"""
DynamoDB Gradual Recovery Repository
====================================

Phase 2: DynamoDB implementation of GradualRecoveryRepository.

Created: 2026-02-20
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from boto3.dynamodb.conditions import Attr, Key

from ..base import GradualRecoveryRepository
from .client import get_dynamodb_table
from .keys import DynamoKeys
from .serializer import from_dynamo_item, to_dynamo_item


class DynamoDBGradualRecoveryRepository(GradualRecoveryRepository):
    """DynamoDB implementation of GradualRecoveryRepository."""

    def __init__(self):
        self._table = get_dynamodb_table()

    def create_gradual_recovery(self, data: Dict[str, Any]) -> Dict[str, Any]:
        recovery_id = data.get("recovery_id")
        initial_loss = data.get("initial_loss", 0.0)
        config = data.get("config", {})
        symbol = data.get("symbol")

        created_at = datetime.now(timezone.utc).isoformat()
        symbol_val = symbol or "GLOBAL"

        entity = {
            "pk": DynamoKeys.recovery_pk(recovery_id),
            "sk": "METADATA",
            "entity_type": "RECOVERY",
            "recovery_id": recovery_id,
            "symbol": symbol_val,
            "initial_loss": initial_loss,
            "config": config,
            "remaining_loss": initial_loss,
            "total_profit_accumulated": 0.0,
            "recovery_percentage": 0.0,
            "trades_count": 0,
            "win_streak": 0,
            "status": "ACTIVE",
            "created_at": created_at,
            "updated_at": created_at,
            # GSI-1: By symbol
            "gsi1pk": symbol_val,
            "gsi1sk": DynamoKeys.gsi1_sk_recovery("ACTIVE", created_at),
            # GSI-2: Global timeline
            "gsi2pk": "RECOVERY",
            "gsi2sk": created_at,
        }

        item = to_dynamo_item(entity)

        self._table.put_item(Item=item, ConditionExpression=Attr("pk").not_exists())

        return from_dynamo_item(item)

    def get_active_gradual_recovery(self, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        symbol_val = symbol or "GLOBAL"

        response = self._table.query(
            IndexName="GSI1",
            KeyConditionExpression=Key("gsi1pk").eq(symbol_val) & Key("gsi1sk").begins_with("RECOVERY#ACTIVE#"),
            Limit=1,
        )

        items = response.get("Items", [])
        if not items:
            return None

        return from_dynamo_item(items[0])

    def update_gradual_recovery(self, recovery_id: str, updates: Dict[str, Any]) -> bool:
        now = datetime.now(timezone.utc).isoformat()

        response = self._table.get_item(Key={"pk": DynamoKeys.recovery_pk(recovery_id), "sk": "METADATA"})
        existing = response.get("Item")
        if not existing:
            return False

        status = updates.get("status", existing.get("status", "ACTIVE"))
        created_at = existing.get("created_at")

        update_expr_parts = ["SET updated_at = :now", "gsi1sk = :gsi1sk"]
        expr_names = {}
        expr_values = {":now": now, ":gsi1sk": DynamoKeys.gsi1_sk_recovery(status, created_at)}

        mapped_updates = to_dynamo_item(updates)

        for k, v in mapped_updates.items():
            if k == "status":
                if status == "COMPLETE":
                    update_expr_parts.append("completed_at = :now")
                elif status in ("FAILED", "CANCELLED"):
                    update_expr_parts.append("failed_at = :now")

            attr_name = f"#{k}"
            attr_val = f":{k}"

            update_expr_parts.append(f"{attr_name} = {attr_val}")
            expr_names[attr_name] = k
            expr_values[attr_val] = v

        update_expr = "SET " + ", ".join(update_expr_parts).replace("SET ", "")

        try:
            self._table.update_item(
                Key={"pk": DynamoKeys.recovery_pk(recovery_id), "sk": "METADATA"},
                UpdateExpression=update_expr,
                ExpressionAttributeNames=expr_names,
                ExpressionAttributeValues=expr_values,
                ConditionExpression=Attr("pk").exists(),
            )
            return True
        except Exception:
            return False

    def cancel_gradual_recovery(self, recovery_id: str) -> bool:
        return self.update_gradual_recovery(recovery_id, {"status": "CANCELLED"})

    def get_gradual_recovery_by_id(self, recovery_id: str) -> Optional[Dict[str, Any]]:
        response = self._table.get_item(Key={"pk": DynamoKeys.recovery_pk(recovery_id), "sk": "METADATA"})

        item = response.get("Item")
        if not item:
            return None

        return from_dynamo_item(item)

    def get_all_gradual_recoveries(
        self, status: Optional[str] = None, limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        target_count = limit + offset
        items = []

        kwargs = {
            "IndexName": "GSI2",
            "KeyConditionExpression": Key("gsi2pk").eq("RECOVERY"),
            "ScanIndexForward": False,
        }

        if status:
            kwargs["FilterExpression"] = Attr("status").eq(status)

        response = self._table.query(**kwargs)
        items.extend(response.get("Items", []))

        while "LastEvaluatedKey" in response and len(items) < target_count:
            kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]
            response = self._table.query(**kwargs)
            items.extend(response.get("Items", []))

        sliced_items = items[offset : offset + limit]
        return [from_dynamo_item(item) for item in sliced_items]

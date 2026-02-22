"""
DynamoDB Martingale Repository
==============================

Phase 2: DynamoDB implementation of MartingaleRepository.

Created: 2026-02-20
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from boto3.dynamodb.conditions import Attr, Key  # type: ignore[import-untyped]

from ..base import MartingaleRepository
from .client import get_dynamodb_table
from .keys import DynamoKeys
from .serializer import from_dynamo_item, to_dynamo_item


class DynamoDBMartingaleRepository(MartingaleRepository):
    """DynamoDB implementation of MartingaleRepository."""

    def __init__(self):
        self._table = get_dynamodb_table()

    def find_or_create_martingale_chain(
        self, symbol: str, initial_order_id: str, loss: float, chain_id: Optional[str] = None
    ) -> Dict[str, Any]:
        # 1. Check for active chain
        active = self.get_martingale_state(symbol)
        if active:
            return active

        # 2. Create new chain
        import uuid

        actual_chain_id = chain_id or str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()

        entity = {
            "pk": DynamoKeys.chain_pk(actual_chain_id),
            "sk": "METADATA",
            "entity_type": "CHAIN",
            "chain_id": actual_chain_id,
            "symbol": symbol,
            "original_loss": loss,
            "initial_order_id": initial_order_id,
            "current_step": 0,
            "latest_order_id": initial_order_id,
            "total_loss": loss,
            "max_step_reached": 0,
            "recovered": False,
            "recovery_pnl": 0.0,
            "status": "ACTIVE",
            "created_at": created_at,
            "updated_at": created_at,
            # GSI-1: By symbol
            "gsi1pk": symbol,
            "gsi1sk": DynamoKeys.gsi1_sk_chain("ACTIVE", created_at),
            # GSI-2: Global timeline
            "gsi2pk": "CHAIN",
            "gsi2sk": created_at,
        }

        item = to_dynamo_item(entity)

        try:
            self._table.put_item(Item=item, ConditionExpression=Attr("pk").not_exists())
            return from_dynamo_item(item)
        except Exception as e:
            if "ConditionalCheckFailedException" in str(e):
                # Race condition: someone else created it
                response = self._table.get_item(Key={"pk": DynamoKeys.chain_pk(actual_chain_id), "sk": "METADATA"})
                return from_dynamo_item(response.get("Item", {}))
            raise

    def get_martingale_state(self, symbol: str) -> Optional[Dict[str, Any]]:
        response = self._table.query(
            IndexName="GSI1",
            KeyConditionExpression=Key("gsi1pk").eq(symbol) & Key("gsi1sk").begins_with("CHAIN#ACTIVE#"),
            Limit=1,
        )

        items = response.get("Items", [])
        if not items:
            return None

        return from_dynamo_item(items[0])

    def update_martingale_chain(self, chain_id: str, updates: Dict[str, Any]) -> bool:
        # Build update expression
        now = datetime.now(timezone.utc).isoformat()

        # We need the existing item to update GSIs if status changes
        response = self._table.get_item(Key={"pk": DynamoKeys.chain_pk(chain_id), "sk": "METADATA"})
        existing = response.get("Item")
        if not existing:
            return False

        status = updates.get("status", existing.get("status", "ACTIVE"))
        created_at = existing.get("created_at")

        update_expr_parts = ["SET updated_at = :now", "gsi1sk = :gsi1sk"]
        expr_names = {}
        expr_values = {":now": now, ":gsi1sk": DynamoKeys.gsi1_sk_chain(status, created_at)}

        mapped_updates = to_dynamo_item(updates)

        for k, v in mapped_updates.items():
            attr_name = f"#{k}"
            attr_val = f":{k}"

            update_expr_parts.append(f"{attr_name} = {attr_val}")
            expr_names[attr_name] = k
            expr_values[attr_val] = v

        # Max step reached logic
        if "current_step" in updates:
            current = updates["current_step"]
            max_step = from_dynamo_item(existing).get("max_step_reached", 0)
            if current > max_step:
                update_expr_parts.append("#max_step = :max_step")
                expr_names["#max_step"] = "max_step_reached"
                expr_values[":max_step"] = current

        update_expr = ", ".join(update_expr_parts).replace("SET updated_at", "SET updated_at")
        # Fix the SET string construction
        update_expr = "SET " + ", ".join(update_expr_parts).replace("SET ", "")

        try:
            self._table.update_item(
                Key={"pk": DynamoKeys.chain_pk(chain_id), "sk": "METADATA"},
                UpdateExpression=update_expr,
                ExpressionAttributeNames=expr_names,
                ExpressionAttributeValues=expr_values,
                ConditionExpression=Attr("pk").exists(),
            )
            return True
        except Exception:
            return False

    def get_active_martingale_chains(self) -> List[Dict[str, Any]]:
        # This requires scanning GSI-2 since we only want ACTIVE ones
        # A sparse GSI would be better in a real high-scale system
        response = self._table.query(
            IndexName="GSI2",
            KeyConditionExpression=Key("gsi2pk").eq("CHAIN"),
            FilterExpression=Attr("status").eq("ACTIVE"),
        )

        items = response.get("Items", [])
        return [from_dynamo_item(item) for item in items]

    def get_martingale_chains_cursor(self, last_id: Optional[int] = None, limit: int = 50) -> List[Dict[str, Any]]:
        # Simplified simulation of pagination
        response = self._table.query(
            IndexName="GSI2", KeyConditionExpression=Key("gsi2pk").eq("CHAIN"), ScanIndexForward=False, Limit=limit
        )

        items = response.get("Items", [])
        return [from_dynamo_item(item) for item in items]

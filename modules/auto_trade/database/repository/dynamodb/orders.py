"""
DynamoDB Order Repository
=========================

Phase 2: DynamoDB implementation of OrderRepository.

Created: 2026-02-20
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from boto3.dynamodb.conditions import Attr, Key

from ..base import OrderRepository
from .client import get_dynamodb_table
from .keys import DynamoKeys
from .serializer import from_dynamo_item, to_dynamo_item


class DynamoDBOrderRepository(OrderRepository):
    """DynamoDB implementation of OrderRepository."""

    def __init__(self):
        self._table = get_dynamodb_table()

    def create_order(self, data: Dict[str, Any]) -> Dict[str, Any]:
        order_id = data.get("order_id")
        if not order_id:
            raise ValueError("order_id is required")

        symbol = data.get("symbol")
        if not symbol:
            raise ValueError("symbol is required")

        status = data.get("status", "PENDING")
        created_at = data.get("created_at")
        if not created_at:
            created_at = datetime.now(timezone.utc)
            data["created_at"] = created_at

        created_iso = created_at.isoformat() if isinstance(created_at, datetime) else str(created_at)
        order_source = data.get("order_source", "PROGRAMMATIC")

        # Base entity properties
        entity = {
            "pk": DynamoKeys.order_pk(order_id),
            "sk": "METADATA",
            "entity_type": "ORDER",
            "order_source": order_source,  # Store the order_source
            # GSI-1: Symbol queries
            "gsi1pk": symbol,
            "gsi1sk": DynamoKeys.gsi1_sk_order(status, created_iso),
            # GSI-2: Global timeline
            "gsi2pk": "ORDER",
            "gsi2sk": created_iso,
            # GSI-3: Programmatic Open Orders
            "gsi3pk": DynamoKeys.gsi3_pk(order_source, status),
            "gsi3sk": created_iso,
        }

        # Merge with user data
        entity.update(data)
        item = to_dynamo_item(entity)

        self._table.put_item(Item=item, ConditionExpression=Attr("pk").not_exists())

        return from_dynamo_item(item)

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        if symbol:
            # GSI-1: by symbol + status=OPEN
            response = self._table.query(
                IndexName="GSI1",
                KeyConditionExpression=Key("gsi1pk").eq(symbol) & Key("gsi1sk").begins_with("ORDER#OPEN#"),
            )
        else:
            # GSI-3: all PROGRAMMATIC#OPEN
            response = self._table.query(IndexName="GSI3", KeyConditionExpression=Key("gsi3pk").eq("PROGRAMMATIC#OPEN"))

        items = response.get("Items", [])
        return [from_dynamo_item(item) for item in items]

    def get_order_by_id(self, order_id: str, verify_programmatic: bool = True) -> Optional[Dict[str, Any]]:
        response = self._table.get_item(Key={"pk": DynamoKeys.order_pk(order_id), "sk": "METADATA"})

        item = response.get("Item")
        if not item:
            return None

        if verify_programmatic and item.get("order_source") != "PROGRAMMATIC":
            return None

        return from_dynamo_item(item)

    def get_order_by_client_id(self, client_order_id: str) -> Optional[Dict[str, Any]]:
        # Using GSI-2 with a FilterExpression since client_order_id is not indexed
        # For high scale, this should get its own GSI or reverse mapping item
        response = self._table.query(
            IndexName="GSI2",
            KeyConditionExpression=Key("gsi2pk").eq("ORDER"),
            FilterExpression=Attr("client_order_id").eq(client_order_id),
            Limit=1,
        )

        items = response.get("Items", [])
        if not items:
            return None

        return from_dynamo_item(items[0])

    def update_order_status(
        self, order_id: str, status: str, pnl: Optional[float] = None, verify_programmatic: bool = True
    ) -> bool:
        # First get the order to know created_at, symbol, order_source for GSI updates
        order = self.get_order_by_id(order_id, verify_programmatic)
        if not order:
            return False

        created_at = order.get("created_at")
        symbol = order.get("symbol")
        order_source = order.get("order_source", "PROGRAMMATIC")

        if isinstance(created_at, datetime):
            created_iso = created_at.isoformat()
        else:
            created_iso = str(created_at)

        update_expr = "SET #status = :status, gsi1sk = :gsi1sk, gsi3pk = :gsi3pk"
        expr_names = {"#status": "status"}
        expr_values = {
            ":status": status,
            ":gsi1sk": DynamoKeys.gsi1_sk_order(status, created_iso),
            ":gsi3pk": DynamoKeys.gsi3_pk(order_source, status),
        }

        if pnl is not None:
            update_expr += ", pnl = :pnl"
            expr_values[":pnl"] = to_dynamo_item({"pnl": pnl})["pnl"]

        try:
            self._table.update_item(
                Key={"pk": DynamoKeys.order_pk(order_id), "sk": "METADATA"},
                UpdateExpression=update_expr,
                ExpressionAttributeNames=expr_names,
                ExpressionAttributeValues=expr_values,
                ConditionExpression=Attr("pk").exists(),
            )
            return True
        except Exception as e:
            if "ConditionalCheckFailedException" in str(e):
                return False
            raise

    def update_order_status_by_client_id(
        self, client_order_id: str, status: str, closed_at: Optional[datetime] = None, pnl: Optional[float] = None
    ) -> bool:
        order = self.get_order_by_client_id(client_order_id)
        if not order:
            return False

        order_id = order["order_id"]

        # Custom logic similar to update_order_status
        created_at = order.get("created_at")
        symbol = order.get("symbol")
        order_source = order.get("order_source", "PROGRAMMATIC")
        created_iso = created_at.isoformat() if isinstance(created_at, datetime) else str(created_at)

        update_expr = "SET #status = :status, gsi1sk = :gsi1sk, gsi3pk = :gsi3pk"
        expr_names = {"#status": "status"}
        expr_values = {
            ":status": status,
            ":gsi1sk": DynamoKeys.gsi1_sk_order(status, created_iso),
            ":gsi3pk": DynamoKeys.gsi3_pk(order_source, status),
        }

        if pnl is not None:
            update_expr += ", pnl = :pnl"
            expr_values[":pnl"] = to_dynamo_item({"pnl": pnl})["pnl"]

        if closed_at is not None:
            update_expr += ", closed_at = :closed_at"
            expr_values[":closed_at"] = closed_at.isoformat() if isinstance(closed_at, datetime) else str(closed_at)

        try:
            self._table.update_item(
                Key={"pk": DynamoKeys.order_pk(order_id), "sk": "METADATA"},
                UpdateExpression=update_expr,
                ExpressionAttributeNames=expr_names,
                ExpressionAttributeValues=expr_values,
                ConditionExpression=Attr("pk").exists(),
            )
            return True
        except Exception:
            return False

    def update(
        self,
        order_id: str,
        updates: Dict[str, Any],
        verify_programmatic: bool = True,
    ) -> bool:
        """Generic update for order fields."""
        if not updates:
            return True

        if verify_programmatic:
            order = self.get_order_by_id(order_id, verify_programmatic=True)
            if not order:
                return False

        update_expr = "SET "
        expr_values = {}
        expr_names = {}
        first = True

        # Fields that can be updated directly
        allowed_fields = {
            "stop_loss",
            "trailing_step_index",
            "take_profit",
            "entry_price",
            "amount",
            "side",
            "status",
            "pnl",
            "closed_at",
            "be_moved",
            "be_moved_at",
            "original_stop_loss",
            "notes",
            "auto_close_deadline_utc",
            "auto_close_triggered",
            "auto_close_reason",
            "auto_close_triggered_at",
            "auto_close_target_tp",
            "auto_close_last_daily_date",
        }

        for i, (key, value) in enumerate(updates.items()):
            if key not in allowed_fields:
                continue
            if not first:
                update_expr += ", "
            first = False

            # Handle attribute name conflicts
            attr_name = f"#attr{i}"
            expr_names[attr_name] = key

            # Convert value to DynamoDB format
            if value is not None:
                expr_values[f":val{i}"] = to_dynamo_item({key: value})[key]
            else:
                expr_values[f":val{i}"] = None

            if value is None:
                update_expr += f"{attr_name} = :val{i}"
            else:
                update_expr += f"{attr_name} = :val{i}"

        if not expr_values:
            return True

        try:
            self._table.update_item(
                Key={"pk": DynamoKeys.order_pk(order_id), "sk": "METADATA"},
                UpdateExpression=update_expr,
                ExpressionAttributeNames=expr_names,
                ExpressionAttributeValues=expr_values,
                ConditionExpression=Attr("pk").exists(),
            )
            return True
        except Exception:
            return False

    def mark_be_moved(
        self,
        order_id: str,
        new_stop_loss: Optional[float] = None,
        new_take_profit: Optional[float] = None,
        verify_programmatic: bool = True,
    ) -> bool:
        # Get item to verify programmatic if needed
        if verify_programmatic:
            order = self.get_order_by_id(order_id, verify_programmatic=True)
            if not order:
                return False

        update_expr = "SET be_moved = :true"
        expr_values = {":true": True, ":false": False}

        if new_stop_loss is not None:
            update_expr += ", stop_loss = :sl"
            expr_values[":sl"] = to_dynamo_item({"sl": new_stop_loss})["sl"]

        if new_take_profit is not None:
            update_expr += ", take_profit = :tp"
            expr_values[":tp"] = to_dynamo_item({"tp": new_take_profit})["tp"]

        try:
            self._table.update_item(
                Key={"pk": DynamoKeys.order_pk(order_id), "sk": "METADATA"},
                UpdateExpression=update_expr,
                ExpressionAttributeValues=expr_values,
                # Condition: must exist and be_moved must be missing or False
                ConditionExpression=Attr("pk").exists() & (Attr("be_moved").not_exists() | Attr("be_moved").eq(False)),
            )
            return True
        except Exception as e:
            if "ConditionalCheckFailedException" in str(e):
                return False
            raise

    def get_all_programmatic_orders(
        self, status: Optional[str] = None, symbol: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        # In DynamoDB, offset is inefficient. The API supports offset, but we'll fetch limit + offset
        # and slice it to simulate SQL behavior.
        # For a true cursor API, use get_orders_cursor

        target_count = limit + offset
        items = []

        if status:
            gsi3pk = f"PROGRAMMATIC#{status}"
            kwargs = {
                "IndexName": "GSI3",
                "KeyConditionExpression": Key("gsi3pk").eq(gsi3pk),
                "ScanIndexForward": False,
            }
        else:
            kwargs = {
                "IndexName": "GSI2",
                "KeyConditionExpression": Key("gsi2pk").eq("ORDER"),
                "FilterExpression": Attr("order_source").eq("PROGRAMMATIC"),
                "ScanIndexForward": False,
            }

        if symbol:
            if "FilterExpression" in kwargs:
                kwargs["FilterExpression"] = kwargs["FilterExpression"] & Attr("symbol").eq(symbol)
            else:
                kwargs["FilterExpression"] = Attr("symbol").eq(symbol)

        response = self._table.query(**kwargs)
        items.extend(response.get("Items", []))

        while "LastEvaluatedKey" in response and len(items) < target_count:
            kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]
            response = self._table.query(**kwargs)
            items.extend(response.get("Items", []))

        # Return the slice based on limit and offset
        sliced_items = items[offset : offset + limit]
        return [from_dynamo_item(item) for item in sliced_items]

    def get_orders_cursor(
        self,
        last_id: Optional[int] = None,  # Not used directly in DynamoDB, we'd prefer a dict token
        limit: int = 50,
        status: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        # Simplified for now. A true cursor requires the LastEvaluatedKey dict
        # We simulate using get_all_programmatic_orders
        return self.get_all_programmatic_orders(status=status, symbol=symbol, limit=limit)

    def get_last_closed_order(self, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if symbol:
            # Use GSI-1 for symbol
            response = self._table.query(
                IndexName="GSI1",
                KeyConditionExpression=Key("gsi1pk").eq(symbol) & Key("gsi1sk").begins_with("ORDER#CLOSED#"),
                ScanIndexForward=False,
                Limit=1,
            )
        else:
            # Use GSI-3 for programmatic closed
            response = self._table.query(
                IndexName="GSI3",
                KeyConditionExpression=Key("gsi3pk").eq("PROGRAMMATIC#CLOSED"),
                ScanIndexForward=False,
                Limit=1,
            )

        items = response.get("Items", [])
        if not items:
            return None

        return from_dynamo_item(items[0])

    def get_orders_by_symbol(
        self, symbol: str, status: Optional[str] = None, limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        target_count = limit + offset
        items = []

        if status:
            gsi1sk_prefix = f"ORDER#{status}#"
        else:
            gsi1sk_prefix = "ORDER#"

        kwargs = {
            "IndexName": "GSI1",
            "KeyConditionExpression": Key("gsi1pk").eq(symbol) & Key("gsi1sk").begins_with(gsi1sk_prefix),
            "ScanIndexForward": False,
        }

        response = self._table.query(**kwargs)
        items.extend(response.get("Items", []))

        while "LastEvaluatedKey" in response and len(items) < target_count:
            kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]
            response = self._table.query(**kwargs)
            items.extend(response.get("Items", []))

        sliced_items = items[offset : offset + limit]
        return [from_dynamo_item(item) for item in sliced_items]

    def is_programmatic_order(self, order_id: str) -> bool:
        order = self.get_order_by_id(order_id, verify_programmatic=False)
        if not order:
            return False
        return order.get("order_source") == "PROGRAMMATIC"

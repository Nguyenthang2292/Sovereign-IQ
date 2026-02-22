"""
DynamoDB Signal Repository
==========================

Phase 2: DynamoDB implementation of SignalRepository.

Created: 2026-02-20
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from boto3.dynamodb.conditions import Attr, Key  # type: ignore[import-untyped]

from ..base import SignalRepository
from .client import get_dynamodb_table
from .keys import DynamoKeys
from .serializer import from_dynamo_item, to_dynamo_item


class DynamoDBSignalRepository(SignalRepository):
    """DynamoDB implementation of SignalRepository."""

    def __init__(self):
        self._table = get_dynamodb_table()

    def save_signal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        correlation_id = data.get("correlation_id")
        if not correlation_id:
            raise ValueError("correlation_id is required")

        symbol = data.get("symbol")
        if not symbol:
            raise ValueError("symbol is required")

        status = "PENDING"
        created_at = data.get("created_at")
        if not created_at:
            created_at = datetime.now(timezone.utc)
            data["created_at"] = created_at

        created_iso = created_at.isoformat() if isinstance(created_at, datetime) else str(created_at)

        entity = {
            "pk": DynamoKeys.signal_pk(correlation_id),
            "sk": "METADATA",
            "entity_type": "SIGNAL",
            # GSI-1: Symbol queries
            "gsi1pk": symbol,
            "gsi1sk": DynamoKeys.gsi1_sk_signal(status, created_iso),
            # GSI-2: Global timeline
            "gsi2pk": "SIGNAL",
            "gsi2sk": created_iso,
        }

        entity.update(data)
        item = to_dynamo_item(entity)

        self._table.put_item(Item=item)
        return from_dynamo_item(item)

    def get_recent_signals(
        self, limit: int = 50, symbol: Optional[str] = None, executed_only: bool = False, offset: int = 0
    ) -> List[Dict[str, Any]]:
        target_count = limit + offset
        items = []

        kwargs: Dict[str, Any] = {"ScanIndexForward": False}

        if symbol:
            kwargs["IndexName"] = "GSI1"
            kwargs["KeyConditionExpression"] = Key("gsi1pk").eq(symbol) & Key("gsi1sk").begins_with("SIGNAL#")
        else:
            kwargs["IndexName"] = "GSI2"
            kwargs["KeyConditionExpression"] = Key("gsi2pk").eq("SIGNAL")

        if executed_only:
            kwargs["FilterExpression"] = Attr("executed").eq(True)

        response = self._table.query(**kwargs)
        items.extend(response.get("Items", []))

        while "LastEvaluatedKey" in response and len(items) < target_count:
            kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]
            response = self._table.query(**kwargs)
            items.extend(response.get("Items", []))

        sliced_items = items[offset : offset + limit]
        return [from_dynamo_item(item) for item in sliced_items]

    def mark_signal_executed(self, correlation_id: str, order_id: str) -> bool:
        # Get signal to update GSI
        response = self._table.get_item(Key={"pk": DynamoKeys.signal_pk(correlation_id), "sk": "METADATA"})
        item = response.get("Item")
        if not item:
            return False

        created_at = item.get("created_at")
        symbol = item.get("symbol")

        now = datetime.now(timezone.utc).isoformat()

        update_expr = "SET executed = :true, execution_order_id = :order_id, executed_at = :now, gsi1sk = :gsi1sk"
        expr_values = {
            ":true": True,
            ":order_id": order_id,
            ":now": now,
            ":gsi1sk": DynamoKeys.gsi1_sk_signal("EXECUTED", created_at),
        }

        try:
            self._table.update_item(
                Key={"pk": DynamoKeys.signal_pk(correlation_id), "sk": "METADATA"},
                UpdateExpression=update_expr,
                ExpressionAttributeValues=expr_values,
                ConditionExpression=Attr("pk").exists(),
            )
            return True
        except Exception:
            return False

    def update_signal_outcome(
        self,
        correlation_id: str,
        outcome: str,
        outcome_pnl: Optional[float] = None,
        outcome_duration_minutes: Optional[int] = None,
    ) -> bool:
        now = datetime.now(timezone.utc).isoformat()

        update_expr = "SET outcome = :outcome, outcome_pnl = :pnl, outcome_at = :now"
        converted_pnl = to_dynamo_item({"pnl": outcome_pnl}).get("pnl", 0.0)
        expr_values = {
            ":outcome": outcome,
            ":pnl": converted_pnl,
            ":now": now,
        }

        if outcome_duration_minutes is not None:
            update_expr += ", outcome_duration_minutes = :dur"
            expr_values[":dur"] = outcome_duration_minutes

        try:
            self._table.update_item(
                Key={"pk": DynamoKeys.signal_pk(correlation_id), "sk": "METADATA"},
                UpdateExpression=update_expr,
                ExpressionAttributeValues=expr_values,
                ConditionExpression=Attr("pk").exists(),
            )
            return True
        except Exception:
            return False

    def get_signal_performance_stats(self, symbol: Optional[str] = None, days: int = 30) -> Dict[str, Any]:
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        kwargs: Dict[str, Any] = {"ScanIndexForward": False}

        if symbol:
            kwargs["IndexName"] = "GSI1"
            kwargs["KeyConditionExpression"] = Key("gsi1pk").eq(symbol) & Key("gsi1sk").between(f"SIGNAL#", f"SIGNAL#Z")
            # DynamoDB doesn't allow multiple conditions on SK with begin_with AND >
            # We filter by date locally or use FilterExpression
            kwargs["FilterExpression"] = Attr("created_at").gte(cutoff_date)
        else:
            kwargs["IndexName"] = "GSI2"
            kwargs["KeyConditionExpression"] = Key("gsi2pk").eq("SIGNAL") & Key("gsi2sk").gte(cutoff_date)

        items = []
        response = self._table.query(**kwargs)
        items.extend(response.get("Items", []))

        while "LastEvaluatedKey" in response:
            kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]
            response = self._table.query(**kwargs)
            items.extend(response.get("Items", []))

        # Aggregate in Python
        total = len(items)
        wins = 0
        losses = 0
        breakevens = 0
        total_pnl = 0.0

        for item in items:
            py_item = from_dynamo_item(item)
            outcome = py_item.get("outcome")

            if outcome == "WIN":
                wins += 1
            elif outcome == "LOSS":
                losses += 1
            elif outcome == "BREAKEVEN":
                breakevens += 1

            pnl = py_item.get("outcome_pnl")
            if pnl is not None:
                total_pnl += float(pnl)

        win_rate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0.0
        avg_pnl = total_pnl / total if total > 0 else 0.0

        return {
            "total_signals": total,
            "wins": wins,
            "losses": losses,
            "breakevens": breakevens,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
        }

    def get_signals_cursor(
        self,
        last_id: Optional[int] = None,
        limit: int = 50,
        symbol: Optional[str] = None,
        executed: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        return self.get_recent_signals(limit=limit, symbol=symbol, executed_only=executed or False)

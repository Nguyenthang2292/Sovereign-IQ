"""
DynamoDB Audit Log Repository
=============================

Phase 2: DynamoDB implementation of AuditLogRepository.
Uses DynamoDB TTL for automatic log expiration.

Created: 2026-02-20
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from boto3.dynamodb.conditions import Attr, Key

from ..base import AuditLogRepository
from .client import get_dynamodb_table
from .keys import DynamoKeys
from .serializer import from_dynamo_item, to_dynamo_item


class DynamoDBAuditLogRepository(AuditLogRepository):
    """DynamoDB implementation of AuditLogRepository."""

    def __init__(self):
        self._table = get_dynamodb_table()

    def create_audit_log(self, data: Dict[str, Any]) -> Dict[str, Any]:
        correlation_id = data.get("correlation_id", str(uuid.uuid4()))
        event_type = data.get("event_type", "UNKNOWN")
        event_category = data.get("event_category", "SYSTEM")
        severity = data.get("severity", "INFO")
        event_summary = data.get("event_summary", "")

        timestamp = datetime.now(timezone.utc)
        created_iso = timestamp.isoformat()

        # TTL for 90 days (AutoTrade default)
        # Stored as epoch seconds for DynamoDB TTL feature
        expire_at = int((timestamp + timedelta(days=90)).timestamp())

        entity = {
            "pk": DynamoKeys.audit_pk(correlation_id),
            "sk": DynamoKeys.audit_sk(created_iso),
            "entity_type": "AUDIT",
            "timestamp": created_iso,
            "event_type": event_type,
            "event_category": event_category,
            "severity": severity,
            "event_summary": event_summary,
            "details": data,
            # GSI-2: Global timeline
            "gsi2pk": "AUDIT",
            "gsi2sk": created_iso,
            # TTL Field
            "expire_at": expire_at,
        }

        item = to_dynamo_item(entity)
        self._table.put_item(Item=item)

        return from_dynamo_item(item)

    def get_recent_audit_logs(
        self, limit: int = 100, severity: Optional[str] = None, event_type: Optional[str] = None, offset: int = 0
    ) -> List[Dict[str, Any]]:
        target_count = limit + offset
        items = []

        kwargs = {"IndexName": "GSI2", "KeyConditionExpression": Key("gsi2pk").eq("AUDIT"), "ScanIndexForward": False}

        filters = []
        if severity:
            filters.append(Attr("severity").eq(severity))
        if event_type:
            filters.append(Attr("event_type").eq(event_type))

        if filters:
            if len(filters) == 1:
                kwargs["FilterExpression"] = filters[0]
            else:
                combined = filters[0]
                for f in filters[1:]:
                    combined = combined & f
                kwargs["FilterExpression"] = combined

        response = self._table.query(**kwargs)
        items.extend(response.get("Items", []))

        while "LastEvaluatedKey" in response and len(items) < target_count:
            kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]
            response = self._table.query(**kwargs)
            items.extend(response.get("Items", []))

        sliced_items = items[offset : offset + limit]
        return [from_dynamo_item(item) for item in sliced_items]

    def get_audit_log_cursor(
        self,
        last_id: Optional[int] = None,
        limit: int = 50,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return self.get_recent_audit_logs(limit=limit, severity=severity, event_type=event_type)

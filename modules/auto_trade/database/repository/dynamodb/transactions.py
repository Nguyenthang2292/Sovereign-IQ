"""
DynamoDB Transactions
=====================

Provides atomic multi-item operations for DynamoDB using TransactWriteItems.

Created: 2026-02-20
"""

from datetime import datetime, timezone
from typing import Any, Dict

from .client import TABLE_NAME, get_dynamodb_client
from .exceptions import DynamoDBTransactionFailed
from .keys import DynamoKeys
from .serializer import to_dynamo_item


def transact_create_order_with_signal(
    order_data: Dict[str, Any], signal_correlation_id: str, signal_created_at: str
) -> bool:
    """
    Atomic operation:
    1. Create the new order
    2. Update the signal as executed
    """
    client = get_dynamodb_client()

    # 1. Prepare Order Put
    order_id = order_data["order_id"]
    symbol = order_data["symbol"]
    status = order_data.get("status", "PENDING")
    order_source = order_data.get("order_source", "PROGRAMMATIC")

    # Convert created_at to ISO string if it's a datetime object
    created_at_raw = order_data["created_at"]
    if isinstance(created_at_raw, datetime):
        created_iso = created_at_raw.isoformat()
    else:
        created_iso = str(created_at_raw)

    # Create the full order entity (similar to what DynamoDBOrderRepository does)
    order_entity = {
        "pk": DynamoKeys.order_pk(order_id),
        "sk": "METADATA",
        "entity_type": "ORDER",
        "order_source": order_source,
        "created_at": created_at_raw,  # Keep original for storage
        "gsi1pk": symbol,
        "gsi1sk": DynamoKeys.gsi1_sk_order(status, created_iso),
        "gsi2pk": "ORDER",
        "gsi2sk": created_iso,
        "gsi3pk": DynamoKeys.gsi3_pk(order_source, status),
        "gsi3sk": created_iso,
    }
    order_entity.update(order_data)

    # Convert item to DynamoDB format using boto3 TypeSerializer style
    # We use a custom util because boto3 expects Typed dicts for transact API
    import boto3.dynamodb.types as ddb_types

    serializer = ddb_types.TypeSerializer()

    formatted_order_data = to_dynamo_item(order_entity)
    ddb_order_item = {k: serializer.serialize(v) for k, v in formatted_order_data.items()}

    # 2. Prepare Signal Update
    now = datetime.now(timezone.utc).isoformat()

    try:
        client.transact_write_items(
            TransactItems=[
                {
                    "Put": {
                        "TableName": TABLE_NAME,
                        "Item": ddb_order_item,
                        "ConditionExpression": "attribute_not_exists(pk)",
                    }
                },
                {
                    "Update": {
                        "TableName": TABLE_NAME,
                        "Key": {"pk": {"S": DynamoKeys.signal_pk(signal_correlation_id)}, "sk": {"S": "METADATA"}},
                        "UpdateExpression": "SET executed = :true, execution_order_id = :oid, executed_at = :now, gsi1sk = :gsi1sk",
                        "ExpressionAttributeValues": {
                            ":true": {"BOOL": True},
                            ":oid": {"S": order_id},
                            ":now": {"S": now},
                            ":gsi1sk": {"S": DynamoKeys.gsi1_sk_signal("EXECUTED", signal_created_at)},
                        },
                        "ConditionExpression": "attribute_exists(pk)",
                    }
                },
            ]
        )
        return True
    except Exception as e:
        raise DynamoDBTransactionFailed(f"Transaction failed: {str(e)}")

"""
DynamoDB Repository Implementations
===================================

Phase 2: DynamoDB Single-Table Design repositories.

Created: 2026-02-20
"""

from .audit_log import DynamoDBAuditLogRepository
from .client import get_dynamodb_client, get_dynamodb_table, reset_connections
from .exceptions import (
    DynamoDBConditionalCheckFailed,
    DynamoDBError,
    DynamoDBItemNotFound,
    DynamoDBTransactionFailed,
)
from .gradual_recovery import DynamoDBGradualRecoveryRepository
from .keys import DynamoKeys
from .martingale import DynamoDBMartingaleRepository
from .metrics import log_dynamodb_error, log_dynamodb_success
from .orders import DynamoDBOrderRepository
from .serializer import from_dynamo_item, to_dynamo_item
from .signals import DynamoDBSignalRepository
from .system_state import DynamoDBSystemStateRepository
from .transactions import transact_create_order_with_signal

__all__ = [
    "get_dynamodb_table",
    "get_dynamodb_client",
    "reset_connections",
    "DynamoKeys",
    "to_dynamo_item",
    "from_dynamo_item",
    "DynamoDBError",
    "DynamoDBItemNotFound",
    "DynamoDBConditionalCheckFailed",
    "DynamoDBTransactionFailed",
    "DynamoDBOrderRepository",
    "DynamoDBSignalRepository",
    "DynamoDBMartingaleRepository",
    "DynamoDBGradualRecoveryRepository",
    "DynamoDBSystemStateRepository",
    "DynamoDBAuditLogRepository",
    "transact_create_order_with_signal",
    "log_dynamodb_success",
    "log_dynamodb_error",
]

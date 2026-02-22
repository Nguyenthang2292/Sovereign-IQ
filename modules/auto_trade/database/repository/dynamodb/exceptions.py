"""
DynamoDB Exceptions
===================

Custom exceptions for DynamoDB operations.

Created: 2026-02-20
"""


class DynamoDBError(Exception):
    """Base exception for DynamoDB errors."""

    pass


class DynamoDBItemNotFound(DynamoDBError):
    """Raised when an expected item is not found."""

    pass


class DynamoDBConditionalCheckFailed(DynamoDBError):
    """Raised when a condition expression fails (e.g. item already exists, wrong status)."""

    pass


class DynamoDBTransactionFailed(DynamoDBError):
    """Raised when a transact_write_items or transact_get_items operation fails."""

    pass

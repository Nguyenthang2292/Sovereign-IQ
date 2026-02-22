"""
DynamoDB Key Builder
====================

Centralizes all Partition Key (PK) and Sort Key (SK) patterns for the single-table design.

Created: 2026-02-20
"""


class DynamoKeys:
    """Builds DynamoDB keys ensuring consistent patterns."""

    # -------------------------------------------------------------------------
    # Core Entity PKs
    # -------------------------------------------------------------------------

    @staticmethod
    def order_pk(order_id: str) -> str:
        return f"ORDER#{order_id}"

    @staticmethod
    def signal_pk(correlation_id: str) -> str:
        return f"SIGNAL#{correlation_id}"

    @staticmethod
    def chain_pk(chain_id: str) -> str:
        return f"CHAIN#{chain_id}"

    @staticmethod
    def recovery_pk(recovery_id: str) -> str:
        return f"RECOVERY#{recovery_id}"

    @staticmethod
    def state_pk(category: str) -> str:
        return f"STATE#{category}"

    @staticmethod
    def audit_pk(correlation_id: str) -> str:
        return f"AUDIT#{correlation_id}"

    # -------------------------------------------------------------------------
    # Core Entity SKs
    # -------------------------------------------------------------------------

    @staticmethod
    def state_sk(key: str) -> str:
        return f"KEY#{key}"

    @staticmethod
    def audit_sk(timestamp: str) -> str:
        # ISO-8601 string
        return f"TS#{timestamp}"

    # -------------------------------------------------------------------------
    # GSI-1: Symbol-based Queries
    # Pattern: gsi1pk = {symbol}, gsi1sk = {entity_type}#{status}#{created_at}
    # -------------------------------------------------------------------------

    @staticmethod
    def gsi1_sk_order(status: str, created_at: str) -> str:
        return f"ORDER#{status}#{created_at}"

    @staticmethod
    def gsi1_sk_signal(status: str, created_at: str) -> str:
        return f"SIGNAL#{status}#{created_at}"

    @staticmethod
    def gsi1_sk_chain(status: str, created_at: str) -> str:
        return f"CHAIN#{status}#{created_at}"

    @staticmethod
    def gsi1_sk_recovery(status: str, created_at: str) -> str:
        return f"RECOVERY#{status}#{created_at}"

    # -------------------------------------------------------------------------
    # GSI-3: Programmatic Open Orders
    # Pattern: gsi3pk = {order_source}#{status}, gsi3sk = {created_at}
    # -------------------------------------------------------------------------

    @staticmethod
    def gsi3_pk(order_source: str, status: str) -> str:
        return f"{order_source}#{status}"

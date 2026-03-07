"""Database Service Layer.

Extracts database operations from UI components to provide a clean service layer.
Supports DynamoDB (RepositoryContext) backend.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import boto3

from modules.auto_trade.database.repository.context import RepositoryContext
from modules.auto_trade.gui.config.database_panel_config import DatabasePanelConfig
from modules.common.ui.logging import log_error, log_info

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_ctx() -> RepositoryContext:
    """Return a RepositoryContext for the active backend (DynamoDB)."""
    return RepositoryContext.from_env()


# ---------------------------------------------------------------------------
# DatabaseService
# ---------------------------------------------------------------------------


class DatabaseService:
    """Service for database operations using DynamoDB."""

    @staticmethod
    def get_stats() -> Dict[str, Any]:
        """Get database statistics."""
        try:
            ctx = _get_ctx()
            orders = ctx.orders.get_all_programmatic_orders(limit=99999)
            open_positions = ctx.orders.get_open_positions()
            signals = ctx.signals.get_recent_signals(limit=99999)
            chains = ctx.martingale.get_active_martingale_chains()
            logs = ctx.audit_log.get_recent_audit_logs(limit=99999)

            return {
                "total_orders": len(orders),
                "open_positions": len(open_positions),
                "total_signals": len(signals),
                "active_chains": len(chains),
                "audit_logs": len(logs),
            }
        except Exception as e:
            log_error(f"Failed to get stats: {e}")
            return {}

    @staticmethod
    def get_last_backup_time() -> Optional[str]:
        """Get last backup timestamp."""
        return "DynamoDB PITR"

    @staticmethod
    def create_backup() -> Optional[str]:
        """Create database backup. For DynamoDB, directs to AWS PITR."""
        log_info("DynamoDB backup: use AWS Console → DynamoDB → AutoTrade → Backups (PITR enabled)")
        return "DynamoDB: PITR enabled — use AWS Console for on-demand backups"

    @staticmethod
    def run_migrations() -> Tuple[bool, str]:
        """Run database migrations. Not applicable for DynamoDB."""
        return (True, "DynamoDB uses schema-less design — no SQL migrations needed")

    @staticmethod
    def cleanup_old_records(days_to_keep: Optional[int] = None) -> Tuple[bool, str]:
        """Cleanup old records."""
        return (
            True,
            "DynamoDB: TTL (expire_at) auto-expires records — no manual cleanup needed",
        )

    @staticmethod
    def check_integrity() -> Tuple[bool, str]:
        """Check database integrity for DynamoDB."""
        try:
            region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ap-southeast-1"
            table_name = os.getenv("DYNAMODB_TABLE_NAME", "AutoTrade")
            client = boto3.client("dynamodb", region_name=region)
            resp = client.describe_table(TableName=table_name)
            status = resp["Table"]["TableStatus"]
            is_ok = status == "ACTIVE"
            return (is_ok, f"DynamoDB table '{table_name}' ({region}): {status}")
        except Exception as e:
            return (False, str(e))


# ---------------------------------------------------------------------------
# DataViewerService
# ---------------------------------------------------------------------------


class DataViewerService:
    """Service for data viewer operations."""

    @staticmethod
    def get_table_count(table_name: str) -> int:
        """Get total count for a table."""
        try:
            ctx = _get_ctx()
            if table_name == "Orders":
                return len(ctx.orders.get_all_programmatic_orders(limit=99999))
            elif table_name == "Signals":
                return len(ctx.signals.get_recent_signals(limit=99999))
            elif table_name == "Martingale Chains":
                return len(ctx.martingale.get_active_martingale_chains())
            elif table_name == "Audit Log":
                return len(ctx.audit_log.get_recent_audit_logs(limit=99999))
            return 0
        except Exception as e:
            log_error(f"Failed to get count for {table_name}: {e}")
            return 0

    @staticmethod
    def get_table_data(
        table_name: str,
        limit: Optional[int] = None,
        last_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get paginated data from a table. Returns list of dicts."""
        if limit is None:
            limit = DatabasePanelConfig.DEFAULT_PAGE_SIZE

        try:
            ctx = _get_ctx()
            if table_name == "Orders":
                return ctx.orders.get_all_programmatic_orders(limit=limit)
            elif table_name == "Signals":
                return ctx.signals.get_recent_signals(limit=limit)
            elif table_name == "Martingale Chains":
                return ctx.martingale.get_active_martingale_chains()
            elif table_name == "Audit Log":
                return ctx.audit_log.get_recent_audit_logs(limit=limit)
            return []
        except Exception as e:
            log_error(f"Failed to get data for {table_name}: {e}")
            return []

    @staticmethod
    def get_audit_logs(limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit logs. Returns list of dicts."""
        try:
            ctx = _get_ctx()
            return ctx.audit_log.get_recent_audit_logs(limit=limit)
        except Exception as e:
            log_error(f"Failed to get audit logs: {e}")
            return []

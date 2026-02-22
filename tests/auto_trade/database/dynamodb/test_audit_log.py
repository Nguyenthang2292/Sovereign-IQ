"""
Tests for DynamoDB Audit Log Repository.

Created: 2026-02-20
"""

from modules.auto_trade.database.repository.dynamodb.audit_log import DynamoDBAuditLogRepository


class TestDynamoDBAuditLogRepository:
    def test_create_sets_ttl(self, setup_dynamodb_table):
        repo = DynamoDBAuditLogRepository()

        entry = repo.create_audit_log(
            {
                "correlation_id": "corr_1",
                "event_type": "ORDER_CREATED",
                "event_category": "TRADING",
                "severity": "INFO",
                "event_summary": "Order created",
            }
        )

        assert entry["pk"] == "AUDIT#corr_1"
        assert "expire_at" in entry
        assert isinstance(entry["expire_at"], (int, float))

    def test_get_recent_with_severity_filter(self, setup_dynamodb_table):
        repo = DynamoDBAuditLogRepository()

        repo.create_audit_log(
            {
                "correlation_id": "corr_info",
                "event_type": "SIGNAL",
                "event_category": "TRADING",
                "severity": "INFO",
                "event_summary": "Informational",
            }
        )
        repo.create_audit_log(
            {
                "correlation_id": "corr_error",
                "event_type": "SIGNAL",
                "event_category": "TRADING",
                "severity": "ERROR",
                "event_summary": "Error happened",
            }
        )

        only_error = repo.get_recent_audit_logs(limit=10, severity="ERROR")
        assert len(only_error) == 1
        assert only_error[0]["severity"] == "ERROR"

"""
SQLite Audit Log Repository
===========================

SQLite implementation of AuditLogRepository interface.
Wraps existing query functions from queries/audit_logs.py.

Created: 2026-02-20
"""

from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from ..base import AuditLogRepository
from ...queries import audit_logs as audit_logs_queries


class SQLiteAuditLogRepository(AuditLogRepository):
    """SQLite implementation wrapping existing audit log query functions."""

    def __init__(self, session: Session):
        self._session = session

    def create_audit_log(self, data: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
        payload = dict(data or {})
        payload.update(kwargs)

        event_type = payload.pop("event_type")
        event_category = payload.pop("event_category")
        severity = payload.pop("severity")
        event_summary = payload.pop("event_summary")

        log = audit_logs_queries.create_audit_log(
            self._session,
            event_type=event_type,
            event_category=event_category,
            severity=severity,
            event_summary=event_summary,
            **payload,
        )
        return log.to_dict()

    def get_recent_audit_logs(
        self, limit: int = 100, severity: Optional[str] = None, event_type: Optional[str] = None, offset: int = 0
    ) -> List[Dict[str, Any]]:
        logs = audit_logs_queries.get_recent_audit_logs(
            self._session, limit=limit, severity=severity, event_type=event_type, offset=offset
        )
        return [log.to_dict() for log in logs]

    def get_audit_log_cursor(
        self,
        last_id: Optional[int] = None,
        limit: int = 50,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return audit_logs_queries.get_audit_log_cursor(
            self._session, last_id=last_id, limit=limit, event_type=event_type, severity=severity
        )

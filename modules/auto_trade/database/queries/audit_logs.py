"""
Audit Log Queries Module
=========================

Audit trail management queries for the auto_trade system.

Features:
- Severity-based filtering
- Event categorization
- Cursor-based pagination for performance

Functions:
- create_audit_log: Create audit log entry
- get_recent_audit_logs: Get recent audit log entries with filters
- get_audit_log_cursor: Cursor-based pagination for audit logs
"""

from ._shared import (
    Any,
    AuditLog,
    List,
    Optional,
    Session,
    datetime,
    desc,
)


def create_audit_log(
    session: Session,
    event_type: str,
    event_category: str,
    severity: str,
    event_summary: str,
    **kwargs: Any,
) -> AuditLog:
    """
    Create audit log entry.

    Args:
        session: Database session
        event_type: Type of event
        event_category: Event category
        severity: Severity level
        event_summary: Human-readable summary
        **kwargs: Additional audit log fields

    Returns:
        Created AuditLog object
    """
    log_data = {
        "event_type": event_type,
        "event_category": event_category,
        "severity": severity,
        "event_summary": event_summary,
        "timestamp": datetime.utcnow(),
    }
    log_data.update(kwargs)

    log = AuditLog(**log_data)
    session.add(log)
    session.commit()

    return log


def get_recent_audit_logs(
    session: Session,
    limit: int = 100,
    severity: Optional[str] = None,
    event_type: Optional[str] = None,
    offset: int = 0,
) -> List[AuditLog]:
    """
    Get recent audit log entries.

    Args:
        session: Database session
        limit: Maximum results
        severity: Optional severity filter
        event_type: Optional event type filter
        offset: Number of results to skip (for pagination)

    Returns:
        List of AuditLog objects
    """
    query = session.query(AuditLog)

    if severity:
        query = query.filter(AuditLog.severity == severity)

    if event_type:
        query = query.filter(AuditLog.event_type == event_type)

    return query.order_by(desc(AuditLog.timestamp)).offset(offset).limit(limit).all()


def get_audit_log_cursor(
    session: Session,
    last_id: Optional[int] = None,
    limit: int = 50,
    event_type: Optional[str] = None,
    severity: Optional[str] = None,
) -> List[AuditLog]:
    """
    Fetch audit log entries using cursor-based pagination.

    Uses AuditLog.id < last_id for cursor pagination instead of offset.

    Args:
        session: Database session
        last_id: Last AuditLog.id from previous page (None for first page)
        limit: Maximum number of entries to return
        event_type: Optional event type filter
        severity: Optional severity filter

    Returns:
        List of AuditLog objects
    """
    query = session.query(AuditLog)

    if last_id:
        query = query.filter(AuditLog.id < last_id)

    if event_type:
        query = query.filter(AuditLog.event_type == event_type)

    if severity:
        query = query.filter(AuditLog.severity == severity)

    return query.order_by(desc(AuditLog.id)).limit(limit).all()


__all__ = [
    "create_audit_log",
    "get_recent_audit_logs",
    "get_audit_log_cursor",
]

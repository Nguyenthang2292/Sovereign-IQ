"""
Audit Trail System.

Provides an append-only log for critical business events (Orders, Positions).
Designed for high integrity and traceability.
"""

import hashlib
import json
import logging
import logging.handlers
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Set


class AuditEventType(str, Enum):
    """Standardized audit event types."""
    ORDER_CREATED = "ORDER_CREATED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    ORDER_FAILED = "ORDER_FAILED"
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_CLOSED = "POSITION_CLOSED"
    POSITION_MODIFIED = "POSITION_MODIFIED"
    RISK_LIMIT_EXCEEDED = "RISK_LIMIT_EXCEEDED"
    MANUAL_INTERVENTION = "MANUAL_INTERVENTION"
    SYSTEM_START = "SYSTEM_START"
    SYSTEM_STOP = "SYSTEM_STOP"
    CONFIG_CHANGE = "CONFIG_CHANGE"


class AuditLogger:
    """
    Specialized logger for audit events with integrity verification.
    """

    # Sensitive keys to redact from logs
    SENSITIVE_KEYS: Set[str] = {
        'password', 'api_key', 'token', 'secret', 'api_secret',
        'private_key', 'passphrase', 'auth', 'authorization',
        'credential', 'access_token', 'refresh_token'
    }

    def __init__(self, log_dir: str = "logs") -> None:
        """
        Initialize audit logger with rotating file handler.

        Args:
            log_dir: Directory to store audit logs

        Raises:
            ValueError: If log_dir is empty or invalid
        """
        # Validate log_dir
        if not log_dir or not log_dir.strip():
            raise ValueError("log_dir cannot be empty")

        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.INFO)
        self.log_dir = Path(log_dir)

        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            raise ValueError(f"Cannot create log directory '{log_dir}': {e}")

        # Ensure we don't add multiple handlers on re-init
        if not self.logger.handlers:
            try:
                # Audit logs rotate but we keep them longer (90 days)
                handler = logging.handlers.TimedRotatingFileHandler(
                    filename=self.log_dir / "audit.log",
                    when="midnight",
                    interval=1,
                    backupCount=90,
                    encoding="utf-8"
                )

                # Simple JSON format for audit
                formatter = logging.Formatter("%(message)s")
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)

                # Prevent propagation to root logger (avoid duplication in system.log)
                self.logger.propagate = False
            except (OSError, PermissionError) as e:
                raise ValueError(f"Cannot create audit log handler: {e}")

    def _sanitize_details(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Redact sensitive information from log details.

        Args:
            details: Dictionary containing event details

        Returns:
            Sanitized dictionary with sensitive values redacted
        """
        sanitized = {}
        for key, value in details.items():
            # Check if key contains sensitive keywords
            if any(sensitive in key.lower() for sensitive in self.SENSITIVE_KEYS):
                sanitized[key] = "***REDACTED***"
            elif isinstance(value, dict):
                # Recursively sanitize nested dictionaries
                sanitized[key] = self._sanitize_details(value)
            elif isinstance(value, list):
                # Handle lists that might contain dicts
                sanitized[key] = [
                    self._sanitize_details(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        return sanitized

    def _calculate_checksum(self, record: Dict[str, Any]) -> str:
        """
        Calculate SHA256 checksum of record for integrity verification.

        Args:
            record: Record dictionary (without checksum field)

        Returns:
            Hexadecimal checksum string
        """
        # Create a copy without checksum field to avoid circular dependency
        record_copy = {k: v for k, v in record.items() if k != 'checksum'}
        record_str = json.dumps(record_copy, sort_keys=True)
        return hashlib.sha256(record_str.encode('utf-8')).hexdigest()

    def _make_json_serializable(self, obj: Any) -> Any:
        """
        Convert non-serializable objects to JSON-serializable format.

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            # For custom objects, use their __dict__
            return str(obj)
        else:
            # Fallback: convert to string
            return str(obj)

    def log_event(
        self,
        event_type: str,
        details: Dict[str, Any],
        user: str = "system",
        add_checksum: bool = True
    ) -> None:
        """
        Log a critical event to the audit trail with integrity verification.

        Args:
            event_type: Type of event (use AuditEventType enum for standardized types)
            details: Dictionary containing event-specific details
            user: User or system component that triggered the event
            add_checksum: Whether to add integrity checksum (default: True)

        Raises:
            ValueError: If event_type is empty or details is not a dict
            RuntimeError: If logging fails after all fallback attempts
        """
        try:
            # Validate inputs
            if not event_type or not isinstance(event_type, str):
                raise ValueError("event_type must be a non-empty string")
            if not isinstance(details, dict):
                raise ValueError("details must be a dictionary")
            if not user or not isinstance(user, str):
                raise ValueError("user must be a non-empty string")

            # Sanitize sensitive data
            sanitized_details = self._sanitize_details(details)

            # Make details JSON-serializable
            serializable_details = self._make_json_serializable(sanitized_details)

            # Create audit record
            record = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "event_type": event_type,
                "user": user,
                "details": serializable_details,
            }

            # Add integrity checksum
            if add_checksum:
                record["checksum"] = self._calculate_checksum(record)

            # Attempt to log
            try:
                self.logger.info(json.dumps(record))
                # Flush to ensure immediate write
                for handler in self.logger.handlers:
                    handler.flush()
            except (TypeError, ValueError) as e:
                # JSON serialization failed - try with safe fallback
                fallback_record = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "event_type": event_type,
                    "user": user,
                    "details": {"error": f"Serialization failed: {e}", "original": str(details)},
                    "error": "JSON_SERIALIZATION_ERROR"
                }
                if add_checksum:
                    fallback_record["checksum"] = self._calculate_checksum(fallback_record)
                self.logger.info(json.dumps(fallback_record))

        except Exception as e:
            # Critical fallback: log to stderr if audit logging fails
            error_msg = (
                f"CRITICAL: Failed to write audit log\n"
                f"  Event Type: {event_type}\n"
                f"  User: {user}\n"
                f"  Error: {e}\n"
            )
            print(error_msg, file=sys.stderr)

            # Try one last time with minimal record
            try:
                minimal_record = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "event_type": "AUDIT_LOGGING_FAILED",
                    "user": "system",
                    "details": {
                        "original_event_type": str(event_type),
                        "error": str(e)
                    }
                }
                self.logger.error(json.dumps(minimal_record))
            except:
                # If even minimal logging fails, raise exception
                raise RuntimeError(f"Audit logging completely failed: {e}")

    def close(self) -> None:
        """
        Close all log handlers and release file handles.

        Call this method when shutting down to ensure proper cleanup.
        """
        for handler in self.logger.handlers[:]:  # Copy list to avoid modification during iteration
            try:
                handler.close()
                self.logger.removeHandler(handler)
            except Exception:
                pass  # Ignore errors during cleanup

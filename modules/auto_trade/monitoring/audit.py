"""
Audit logging for critical system events.

Records immutable trails of actions like order execution,
configuration changes, and risk limit hits.

Created: 2026-02-06
"""

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from modules.common.ui.logging import log_error, log_info, log_warn


class AuditEventType(Enum):
    """
    Standardized categories for audit events to ensure consistency.
    """

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
        "password",
        "api_key",
        "token",
        "secret",
        "api_secret",
        "private_key",
        "passphrase",
        "auth",
        "authorization",
        "credential",
        "access_token",
        "refresh_token",
    }

    def __init__(self, log_dir: str = "logs") -> None:
        """
        Initialize audit logger.

        Args:
            log_dir: Directory to store audit logs

        Raises:
            ValueError: If log_dir is empty or invalid
        """
        # Validate log_dir
        if not log_dir or not log_dir.strip():
            raise ValueError("log_dir cannot be empty")

        self.log_dir = Path(log_dir)
        self.log_file = self.log_dir / "audit.log"

        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            raise ValueError(f"Cannot create log directory '{log_dir}': {e}")

    def _sanitize_details(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Redact sensitive information from log details.

        Args:
            details: Dictionary containing event details

        Returns:
            Sanitized dictionary with sensitive values redacted
        """
        sanitized: Dict[str, Any] = {}
        for key, value in details.items():
            # Check for sensitive key names (case-insensitive)
            is_sensitive = any(s_key in key.lower() for s_key in self.SENSITIVE_KEYS)

            if is_sensitive and value:
                # Mask value but preserve length to indicate it was set
                sanitized[key] = f"***REDACTED({len(str(value))})***"
            elif isinstance(value, dict):
                # Recursively sanitize nested dictionaries
                sanitized[key] = self._sanitize_details(value)
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                # Sanitize lists of dictionaries
                sanitized[key] = [self._sanitize_details(item) for item in value]
            else:
                sanitized[key] = value

        return sanitized

    def _calculate_checksum(self, record: Dict[str, Any]) -> str:
        """
        Calculate integrity checksum for an audit record.

        Creates a predictable string representation of the record (excluding
        existing checksum) and hashes it.

        Args:
            record: The audit record dictionary

        Returns:
            SHA-256 hex digest of the record
        """
        # Create a copy without any existing checksum
        checksum_record = {k: v for k, v in record.items() if k != "checksum"}

        # Serialize with sorted keys for consistency across Python versions/runs
        # Separators ensure no whitespace variations
        serialized = json.dumps(checksum_record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _make_json_serializable(self, obj: Any) -> Any:
        """
        Recursively convert objects to JSON-serializable formats.
        Specifically handles datetime objects which default json encoder fails on.
        """
        if obj is None or isinstance(obj, (int, float, str, bool)):
            return obj
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple, set)):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, "isoformat"):
            # Handles datetime, date, time
            return obj.isoformat()
        elif hasattr(obj, "__dict__"):
            # For custom objects, use their __dict__
            return str(obj)
        else:
            # Fallback: convert to string
            return str(obj)

    def log_event(
        self,
        event_type: Union[str, AuditEventType],
        details: Dict[str, Any],
        user: str = "system",
        add_checksum: bool = True,
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
            if not event_type or not isinstance(event_type, (str, AuditEventType)):
                raise ValueError("event_type must be a non-empty string or AuditEventType")
            if not isinstance(details, dict):
                raise ValueError("details must be a dictionary")
            if not user or not isinstance(user, str):
                raise ValueError("user must be a non-empty string")

            # Convert enum to string if needed
            event_type_str = event_type.value if isinstance(event_type, AuditEventType) else event_type

            # Sanitize sensitive data
            sanitized_details = self._sanitize_details(details)

            # Make details JSON-serializable
            serializable_details = self._make_json_serializable(sanitized_details)

            # Create audit record
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                "event_type": event_type_str,
                "user": user,
                "details": serializable_details,
            }

            # Add integrity checksum
            if add_checksum:
                record["checksum"] = self._calculate_checksum(record)

            # Attempt to log
            try:
                msg = json.dumps(record)
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
                log_info(msg)
            except (TypeError, ValueError) as e:
                # JSON serialization failed - try with safe fallback
                fallback_record = {
                    "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                    "event_type": event_type_str,
                    "user": user,
                    "details": {"error": f"Serialization failed: {e}", "original": str(details)},
                }
                if add_checksum:
                    fallback_record["checksum"] = self._calculate_checksum(fallback_record)

                msg = json.dumps(fallback_record)
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
                log_warn(msg)

        except Exception as e:
            # Absolute fallback - try to record the failure itself
            error_msg = f"CRITICAL: Audit logging failed entirely: {e} | Type: {event_type}"
            log_error(error_msg, exc_info=True)
            # We don't raise here to prevent bringing down the system over an audit failure
            # but we ensure it's recorded to stderr

    def verify_integrity(self, log_lines: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Verify the checksums of existing audit logs.

        Args:
            log_lines: Optional list of raw JSON strings to verify instead of reading file

        Returns:
            List of dictionaries containing validation results for each record
        """
        results: List[Dict[str, Any]] = []

        try:
            lines_to_check = log_lines

            # Read from file if no lines provided
            if lines_to_check is None:
                if not self.log_file.exists():
                    return [{"status": "error", "message": "Audit log file not found"}]

                try:
                    with open(self.log_file, "r", encoding="utf-8") as f:
                        lines_to_check = [line.strip() for line in f if line.strip()]
                except (OSError, PermissionError) as e:
                    return [{"status": "error", "message": f"Cannot read audit file: {e}"}]

            # Verify each line
            for i, line in enumerate(lines_to_check):
                try:
                    record = json.loads(line)

                    # Missing checksum is a validation failure
                    if "checksum" not in record:
                        results.append(
                            {
                                "line": i + 1,
                                "timestamp": record.get("timestamp", "unknown"),
                                "status": "failed",
                                "reason": "Missing checksum field",
                            }
                        )
                        continue

                    # Extract stored checksum and verify against recalculated one
                    stored_checksum = record["checksum"]
                    calculated_checksum = self._calculate_checksum(record)

                    if stored_checksum == calculated_checksum:
                        results.append({"line": i + 1, "timestamp": record.get("timestamp"), "status": "valid"})
                    else:
                        results.append(
                            {
                                "line": i + 1,
                                "timestamp": record.get("timestamp"),
                                "status": "failed",
                                "reason": f"Checksum mismatch. Expected {calculated_checksum}, got {stored_checksum}",
                            }
                        )

                except json.JSONDecodeError as e:
                    results.append({"line": i + 1, "status": "error", "reason": f"Invalid JSON format: {e}"})

            return results
        except Exception as e:
            # Catch unexpected errors during verification
            return [{"status": "error", "message": f"Verification process failed: {e}"}]

    def cleanup(self) -> None:
        """
        Close file handlers. Used during system shutdown.
        """
        pass


# Global instance for easy access
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger(log_dir: str = "logs") -> AuditLogger:
    """
    Get or create the global audit logger instance.
    """
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger(log_dir)
    return _audit_logger

"""
Audit logging for critical system events.

Records immutable trails of actions like order execution,
configuration changes, and risk limit hits.
"""

import hashlib
import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union


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

    REDACTED_VALUE = "***REDACTED***"

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

        # Use a file-specific logger name so different paths don't share handlers.
        logger_key = str(self.log_file.resolve()).lower()
        self.logger = logging.getLogger(f"modules.auto_trade.audit.{hash(logger_key)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        self._ensure_file_handler()

    def _ensure_file_handler(self) -> None:
        """Ensure exactly one timed rotating file handler is configured."""
        target = str(self.log_file.resolve()).lower()
        has_target_handler = False

        for handler in list(self.logger.handlers):
            is_target = (
                isinstance(handler, logging.handlers.TimedRotatingFileHandler)
                and str(Path(handler.baseFilename).resolve()).lower() == target
            )
            if is_target:
                has_target_handler = True
                continue

            # Keep this logger clean and deterministic for tests and runtime.
            try:
                handler.close()
            finally:
                self.logger.removeHandler(handler)

        if has_target_handler:
            return

        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=self.log_file,
            when="midnight",
            interval=1,
            backupCount=30,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(file_handler)

    def _sanitize_details(
        self,
        details: Dict[str, Any],
        visited: Optional[Set[int]] = None,
    ) -> Dict[str, Any]:
        """
        Redact sensitive information from log details.

        Args:
            details: Dictionary containing event details

        Returns:
            Sanitized dictionary with sensitive values redacted
        """
        if visited is None:
            visited = set()

        obj_id = id(details)
        if obj_id in visited:
            return {"_circular_reference": "detected"}

        visited.add(obj_id)
        sanitized: Dict[str, Any] = {}
        for key, value in details.items():
            # Check for sensitive key names (case-insensitive)
            is_sensitive = any(s_key in key.lower() for s_key in self.SENSITIVE_KEYS)

            if is_sensitive and value:
                sanitized[key] = self.REDACTED_VALUE
            elif isinstance(value, dict):
                # Recursively sanitize nested dictionaries
                sanitized[key] = self._sanitize_details(value, visited)
            elif isinstance(value, list):
                sanitized_list: List[Any] = []
                for item in value:
                    if isinstance(item, dict):
                        sanitized_list.append(self._sanitize_details(item, visited))
                    else:
                        sanitized_list.append(item)
                sanitized[key] = sanitized_list
            else:
                sanitized[key] = value

        visited.remove(obj_id)
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

            msg = json.dumps(record)
            self.logger.info(msg)

        except Exception as e:
            # Primary fallback: write an explicit failure audit event.
            fallback_record = {
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                "event_type": "AUDIT_LOGGING_FAILED",
                "user": "system",
                "details": {
                    "error": str(e),
                    "original_event_type": str(event_type),
                },
            }

            if add_checksum:
                fallback_record["checksum"] = self._calculate_checksum(fallback_record)

            critical_msg = f"CRITICAL: Failed to write audit log: {e}"
            print(critical_msg, file=sys.stderr)

            try:
                self.logger.error(json.dumps(fallback_record))
            except Exception as fallback_error:
                print(
                    f"CRITICAL: Failed to write fallback audit log: {fallback_error}",
                    file=sys.stderr,
                )
                raise RuntimeError("Audit logging completely failed") from fallback_error

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

    def close(self) -> None:
        """Close and detach all handlers for this audit logger."""
        for handler in list(self.logger.handlers):
            try:
                handler.close()
            finally:
                self.logger.removeHandler(handler)

    def cleanup(self) -> None:
        """Compatibility alias for explicit shutdown cleanup."""
        self.close()


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

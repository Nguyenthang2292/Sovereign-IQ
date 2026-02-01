"""
Audit Trail System.

Provides an append-only log for critical business events (Orders, Positions).
Designed for high integrity and traceability.
"""

import json
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


class AuditLogger:
    """
    Specialized logger for audit events.
    """

    def __init__(self, log_dir: str = "logs"):
        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.INFO)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Ensure we don't add multiple handlers on re-init
        if not self.logger.handlers:
            # Audit logs rotate but we keep them longer (90 days)
            handler = logging.handlers.TimedRotatingFileHandler(
                filename=self.log_dir / "audit.log", when="midnight", interval=1, backupCount=90, encoding="utf-8"
            )

            # Simple JSON format for audit
            formatter = logging.Formatter("%(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

            # Prevent propagation to root logger (avoid duplication in system.log)
            self.logger.propagate = False

    def log_event(self, event_type: str, details: Dict[str, Any], user: str = "system") -> None:
        """
        Log a critical event to the audit trail.
        """
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": event_type,
            "user": user,
            "details": details,
        }
        self.logger.info(json.dumps(record))

"""
Shared types and imports for query modules.
"""
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, cast

from sqlalchemy import case, desc, func
from sqlalchemy.orm import Session
from sqlalchemy.types import DateTime, Integer

# Re-export models for convenience
from ..models import (
    AuditLog,
    GradualRecovery,
    MartingaleChain,
    Order,
    Signal,
    SystemState,
)

# Constants
DEFAULT_ORDER_SOURCE = "PROGRAMMATIC"
DEFAULT_EXECUTION_MODE = "AUTO"

__all__ = [
    # Types
    "Any",
    "Dict",
    "List",
    "Optional",
    "cast",
    "datetime",
    "timedelta",
    # SQLAlchemy
    "desc",
    "case",
    "func",
    "Session",
    "DateTime",
    "Integer",
    # Models
    "AuditLog",
    "GradualRecovery",
    "MartingaleChain",
    "Order",
    "Signal",
    "SystemState",
    # Constants
    "DEFAULT_ORDER_SOURCE",
    "DEFAULT_EXECUTION_MODE",
]

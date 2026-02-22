"""
SQLite System State Repository
==============================

SQLite implementation of SystemStateRepository interface.
Wraps existing query functions from queries/system_state.py.

Created: 2026-02-20
"""

from typing import Any, Optional

from sqlalchemy.orm import Session

from ..base import SystemStateRepository
from ...queries import system_state as system_state_queries


class SQLiteSystemStateRepository(SystemStateRepository):
    """SQLite implementation wrapping existing system state query functions."""

    def __init__(self, session: Session):
        self._session = session

    def get_system_state(self, key: str, default: Optional[Any] = None) -> Optional[Any]:
        value = system_state_queries.get_system_state(self._session, key)
        return value if value is not None else default

    def set_system_state(
        self,
        key: str,
        value: Any,
        value_type: str = "string",
        description: Optional[str] = None,
        category: Optional[str] = None,
    ) -> bool:
        return system_state_queries.set_system_state(
            self._session, key, value, value_type=value_type, description=description, category=category
        )

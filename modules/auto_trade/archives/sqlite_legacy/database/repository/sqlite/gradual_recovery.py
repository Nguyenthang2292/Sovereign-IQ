"""
SQLite Gradual Recovery Repository
==================================

SQLite implementation of GradualRecoveryRepository interface.
Wraps existing query functions from queries/gradual_recovery.py.

Created: 2026-02-20
"""

from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from ...queries import gradual_recovery as gradual_recovery_queries
from ..base import GradualRecoveryRepository


class SQLiteGradualRecoveryRepository(GradualRecoveryRepository):
    """SQLite implementation wrapping existing gradual recovery query functions."""

    def __init__(self, session: Session):
        self._session = session

    def create_gradual_recovery(self, data: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
        payload = dict(data or {})
        payload.update(kwargs)

        recovery_id = payload["recovery_id"]
        initial_loss = payload["initial_loss"]
        config = payload["config"]
        symbol = payload.get("symbol")

        recovery = gradual_recovery_queries.create_gradual_recovery(
            self._session, recovery_id, initial_loss, config, symbol=symbol
        )
        return recovery.to_dict()

    def get_active_gradual_recovery(self, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        recovery = gradual_recovery_queries.get_active_gradual_recovery(self._session, symbol=symbol)
        return recovery.to_dict() if recovery else None

    def update_gradual_recovery(self, recovery_id: str, updates: Dict[str, Any]) -> bool:
        return gradual_recovery_queries.update_gradual_recovery(
            self._session,
            recovery_id,
            remaining_loss=updates.get("remaining_loss"),
            total_profit_accumulated=updates.get("total_profit_accumulated"),
            recovery_percentage=updates.get("recovery_percentage"),
            trades_count=updates.get("trades_count"),
            win_streak=updates.get("win_streak"),
            estimated_trades_remaining=updates.get("estimated_trades_remaining"),
            status=updates.get("status"),
        )

    def cancel_gradual_recovery(self, recovery_id: str) -> bool:
        return gradual_recovery_queries.cancel_gradual_recovery(self._session, recovery_id)

    def get_gradual_recovery_by_id(self, recovery_id: str) -> Optional[Dict[str, Any]]:
        recovery = gradual_recovery_queries.get_gradual_recovery_by_id(self._session, recovery_id)
        return recovery.to_dict() if recovery else None

    def get_all_gradual_recoveries(
        self, status: Optional[str] = None, limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        recoveries = gradual_recovery_queries.get_all_gradual_recoveries(
            self._session, status=status, limit=limit, offset=offset
        )
        return [recovery.to_dict() for recovery in recoveries]

"""
SQLite Signal Repository
========================

SQLite implementation of SignalRepository interface.
Wraps existing query functions from queries/signals.py.

Created: 2026-02-20
"""

from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from ..base import SignalRepository
from ...queries import signals as signal_queries


class SQLiteSignalRepository(SignalRepository):
    """SQLite implementation wrapping existing signal query functions."""

    def __init__(self, session: Session):
        self._session = session

    def save_signal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        signal = signal_queries.save_signal(self._session, **data)
        return signal.to_dict()

    def get_recent_signals(
        self, limit: int = 50, symbol: Optional[str] = None, executed_only: bool = False, offset: int = 0
    ) -> List[Dict[str, Any]]:
        signals = signal_queries.get_recent_signals(
            self._session, limit=limit, symbol=symbol, executed_only=executed_only, offset=offset
        )
        return [signal.to_dict() for signal in signals]

    def mark_signal_executed(self, correlation_id: str, order_id: str) -> bool:
        return signal_queries.mark_signal_executed(self._session, correlation_id, order_id)

    def update_signal_outcome(
        self,
        correlation_id: str,
        outcome: str,
        outcome_pnl: Optional[float] = None,
        outcome_duration_minutes: Optional[int] = None,
    ) -> bool:
        return signal_queries.update_signal_outcome(
            self._session,
            correlation_id,
            outcome,
            outcome_pnl if outcome_pnl is not None else 0.0,
            outcome_duration_minutes=outcome_duration_minutes,
        )

    def get_signal_performance_stats(self, symbol: Optional[str] = None, days: int = 30) -> Dict[str, Any]:
        return signal_queries.get_signal_performance_stats(self._session, symbol=symbol, days=days)

    def get_signals_cursor(
        self,
        last_id: Optional[int] = None,
        limit: int = 50,
        symbol: Optional[str] = None,
        executed: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        return signal_queries.get_signals_cursor(
            self._session, last_id=last_id, limit=limit, symbol=symbol, executed=executed
        )

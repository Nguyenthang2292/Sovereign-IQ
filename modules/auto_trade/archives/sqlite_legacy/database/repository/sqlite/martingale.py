"""
SQLite Martingale Repository
============================

SQLite implementation of MartingaleRepository interface.
Wraps existing query functions from queries/martingale.py.

Created: 2026-02-20
"""

import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from ..base import MartingaleRepository
from ...queries import martingale as martingale_queries


class SQLiteMartingaleRepository(MartingaleRepository):
    """SQLite implementation wrapping existing martingale query functions."""

    def __init__(self, session: Session):
        self._session = session

    def find_or_create_martingale_chain(
        self,
        symbol: str,
        initial_order_id: str,
        loss: Optional[float] = None,
        chain_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        resolved_loss = loss if loss is not None else kwargs.get("original_loss")
        if resolved_loss is None:
            raise ValueError("loss (or original_loss) is required")

        resolved_chain_id = chain_id or f"chain_{symbol}_{uuid.uuid4().hex[:8]}"
        chain = martingale_queries.find_or_create_martingale_chain(
            self._session, resolved_chain_id, symbol, float(resolved_loss), initial_order_id
        )
        return chain.to_dict()

    def get_martingale_state(self, symbol: str) -> Optional[Dict[str, Any]]:
        chain = martingale_queries.get_martingale_state(self._session, symbol)
        return chain.to_dict() if chain else None

    def update_martingale_chain(self, chain_id: str, updates: Dict[str, Any]) -> bool:
        current_step = updates.get("current_step")
        latest_order_id = updates.get("latest_order_id")
        total_loss = updates.get("total_loss")
        recovered = updates.get("recovered", False)
        recovery_pnl = updates.get("recovery_pnl", 0.0)

        if current_step is None or latest_order_id is None or total_loss is None:
            return False

        return martingale_queries.update_martingale_chain(
            self._session,
            chain_id,
            current_step=current_step,
            latest_order_id=latest_order_id,
            total_loss=total_loss,
            recovered=recovered,
            recovery_pnl=recovery_pnl,
        )

    def get_active_martingale_chains(self) -> List[Dict[str, Any]]:
        chains = martingale_queries.get_active_martingale_chains(self._session)
        return [chain.to_dict() for chain in chains]

    def get_martingale_chains_cursor(self, last_id: Optional[int] = None, limit: int = 50) -> List[Dict[str, Any]]:
        return martingale_queries.get_martingale_chains_cursor(self._session, last_id=last_id, limit=limit)

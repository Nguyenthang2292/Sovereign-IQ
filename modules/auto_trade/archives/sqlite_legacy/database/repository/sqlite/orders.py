"""
SQLite Order Repository
=======================

SQLite implementation of OrderRepository interface.
Wraps existing query functions from queries/orders.py.

Created: 2026-02-20
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from ..base import OrderRepository
from ...queries import orders as order_queries


class SQLiteOrderRepository(OrderRepository):
    """SQLite implementation wrapping existing order query functions."""

    def __init__(self, session: Session):
        self._session = session

    def create_order(self, data: Dict[str, Any]) -> Dict[str, Any]:
        order = order_queries.create_order(self._session, data)
        return order.to_dict()

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        orders = order_queries.get_open_positions(self._session, symbol=symbol)
        return [order.to_dict() for order in orders]

    def get_order_by_id(self, order_id: str, verify_programmatic: bool = True) -> Optional[Dict[str, Any]]:
        order = order_queries.get_order_by_id(self._session, order_id, verify_programmatic=verify_programmatic)
        return order.to_dict() if order else None

    def get_order_by_client_id(self, client_order_id: str) -> Optional[Dict[str, Any]]:
        order = order_queries.get_order_by_client_id(self._session, client_order_id)
        return order.to_dict() if order else None

    def update_order_status(
        self, order_id: str, status: str, pnl: Optional[float] = None, verify_programmatic: bool = True
    ) -> bool:
        return order_queries.update_order_status(
            self._session, order_id, status, pnl=pnl, verify_programmatic=verify_programmatic
        )

    def update_order_status_by_client_id(
        self, client_order_id: str, status: str, closed_at: Optional[datetime] = None, pnl: Optional[float] = None
    ) -> bool:
        return order_queries.update_order_status_by_client_id(
            self._session, client_order_id, status, closed_at=closed_at, pnl=pnl
        )

    def mark_be_moved(
        self,
        order_id: str,
        new_stop_loss: Optional[float] = None,
        new_take_profit: Optional[float] = None,
        verify_programmatic: bool = True,
    ) -> bool:
        return order_queries.mark_be_moved(
            self._session,
            order_id,
            new_stop_loss=new_stop_loss,
            new_take_profit=new_take_profit,
            verify_programmatic=verify_programmatic,
        )

    def get_all_programmatic_orders(
        self, status: Optional[str] = None, symbol: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        orders = order_queries.get_all_programmatic_orders(
            self._session, status=status, symbol=symbol, limit=limit, offset=offset
        )
        return [order.to_dict() for order in orders]

    def get_orders_cursor(
        self, last_id: Optional[int] = None, limit: int = 50, status: Optional[str] = None, symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        return order_queries.get_orders_cursor(
            self._session, last_id=last_id, limit=limit, status=status, symbol=symbol
        )

    def get_last_closed_order(self, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        order = order_queries.get_last_closed_order(self._session, symbol=symbol)
        return order.to_dict() if order else None

    def get_orders_by_symbol(
        self, symbol: str, status: Optional[str] = None, limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        orders = order_queries.get_orders_by_symbol(
            self._session, symbol=symbol, status=status, limit=limit, offset=offset
        )
        return [order.to_dict() for order in orders]

    def is_programmatic_order(self, order_id: str) -> bool:
        return order_queries.is_programmatic_order(self._session, order_id)

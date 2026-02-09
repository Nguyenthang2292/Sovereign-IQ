"""
Order Queries Module
====================

All order-related database queries for the auto_trade system.

CRITICAL: All order queries ONLY return PROGRAMMATIC orders by default.
Manual trades from Binance are excluded from queries.

Functions:
- get_open_positions: Get all open positions
- get_last_closed_order: Get most recent closed order
- get_all_programmatic_orders: Fetch all programmatic orders with pagination
- get_orders_cursor: Cursor-based pagination for orders (more performant)
- is_programmatic_order: Check if order is programmatic
- get_order_by_id: Get order by ID
- get_order_by_client_id: Get order by client_order_id
- update_order_status_by_client_id: Update order status by client_order_id
- update_order_status: Update order status and optionally P&L
- mark_be_moved: Mark that break-even has been triggered
- create_order: Create a new programmatic order
- get_orders_by_symbol: Get orders for a specific symbol
"""

from ._shared import (
    Any,
    DateTime,
    Dict,
    List,
    Optional,
    Order,
    Session,
    cast,
    datetime,
    desc,
    timezone,
)


def get_open_positions(
    session: Session, symbol: Optional[str] = None, order_source: str = "PROGRAMMATIC"
) -> List[Order]:
    """
    Get all open positions.

    **Only returns PROGRAMMATIC orders by default.**

    Args:
        session: Database session
        symbol: Optional symbol filter
        order_source: Order source filter (default: PROGRAMMATIC only)

    Returns:
        List of open Order objects
    """
    query = session.query(Order).filter(Order.order_source == order_source, Order.status == "OPEN")

    if symbol:
        query = query.filter(Order.symbol == symbol)

    return query.order_by(desc(Order.created_at)).all()


def get_last_closed_order(
    session: Session, symbol: Optional[str] = None, order_source: str = "PROGRAMMATIC"
) -> Optional[Order]:
    """
    Get the most recent closed order.

    **Only queries PROGRAMMATIC orders by default.**

    Args:
        session: Database session
        symbol: Optional symbol filter
        order_source: Order source filter (default: PROGRAMMATIC only)

    Returns:
        Most recent closed Order or None
    """
    query = session.query(Order).filter(Order.order_source == order_source, Order.status == "CLOSED")

    if symbol:
        query = query.filter(Order.symbol == symbol)

    return query.order_by(desc(Order.closed_at)).first()


def get_all_programmatic_orders(
    session: Session, status: Optional[str] = None, symbol: Optional[str] = None, limit: int = 100, offset: int = 0
) -> List[Order]:
    """
    Fetch all programmatic orders (auto_trade system orders only).

    Args:
        session: Database session
        status: Optional status filter ('OPEN', 'CLOSED', etc.)
        symbol: Optional symbol filter
        limit: Maximum number of orders to return
        offset: Number of orders to skip (for pagination)

    Returns:
        List of programmatic Order objects
    """
    query = session.query(Order).filter(Order.order_source == "PROGRAMMATIC")

    if status:
        query = query.filter(Order.status == status)

    if symbol:
        query = query.filter(Order.symbol == symbol)
    if limit <= 0:
        return []

    if offset < 0:
        offset = 0

    if offset > 0:
        total_count = query.count()
        if offset >= total_count:
            return []

    return query.order_by(desc(Order.created_at)).offset(offset).limit(limit).all()


def get_orders_cursor(
    session: Session,
    last_id: Optional[int] = None,
    limit: int = 50,
    order_source: str = "PROGRAMMATIC",
    status: Optional[str] = None,
    symbol: Optional[str] = None,
) -> List[Order]:
    """
    Fetch orders using cursor-based pagination (more performant for large datasets).

    Uses Order.id < last_id for cursor pagination instead of offset.

    Args:
        session: Database session
        last_id: Last Order.id from previous page (None for first page)
        limit: Maximum number of orders to return
        order_source: Order source filter (default: PROGRAMMATIC)
        status: Optional status filter
        symbol: Optional symbol filter

    Returns:
        List of Order objects
    """
    query = session.query(Order).filter(Order.order_source == order_source)

    if last_id:
        query = query.filter(Order.id < last_id)

    if status:
        query = query.filter(Order.status == status)

    if symbol:
        query = query.filter(Order.symbol == symbol)

    return query.order_by(desc(Order.id)).limit(limit).all()


def is_programmatic_order(session: Session, order_id: str) -> bool:
    """
    Check if an order was created by the auto_trade system.

    Args:
        session: Database session
        order_id: Order ID to check

    Returns:
        True if order is programmatic, False otherwise
    """
    order = session.query(Order).filter(Order.order_id == order_id).first()
    if not order:
        return False
    return cast(bool, order.order_source == "PROGRAMMATIC")


def get_order_by_id(session: Session, order_id: str, verify_programmatic: bool = True) -> Optional[Order]:
    """
    Get order by ID.

    Args:
        session: Database session
        order_id: Order ID
        verify_programmatic: If True, only return if order is programmatic

    Returns:
        Order object or None
    """
    query = session.query(Order).filter(Order.order_id == order_id)

    if verify_programmatic:
        query = query.filter(Order.order_source == "PROGRAMMATIC")

    return query.first()


def get_order_by_client_id(session: Session, client_order_id: str) -> Optional[Order]:
    """
    Get order by client_order_id (typically starts with 'AT_' for auto_trade).

    Args:
        session: Database session
        client_order_id: Client order ID (e.g., AT_1706947200_BTCUSDT_abc123)

    Returns:
        Order object or None
    """
    return session.query(Order).filter(Order.client_order_id == client_order_id).first()


def update_order_status_by_client_id(
    session: Session,
    client_order_id: str,
    status: str,
    closed_at: Optional[datetime] = None,
    pnl: Optional[float] = None,
) -> bool:
    """
    Update order status by client_order_id.

    **Only updates if row exists and status == 'OPEN'**.
    Sets closed_at when provided.

    Args:
        session: Database session
        client_order_id: Client order ID to update
        status: New status (e.g., 'CLOSED', 'CANCELLED', 'FAILED')
        closed_at: Optional close timestamp
        pnl: Optional P&L value

    Returns:
        True if updated, False otherwise
    """
    order = session.query(Order).filter(Order.client_order_id == client_order_id, Order.status == "OPEN").first()

    if order:
        setattr(order, "status", status)
        if pnl is not None:
            setattr(order, "pnl", pnl)
            ep = cast(Optional[float], getattr(order, "entry_price", None))
            am = cast(Optional[float], getattr(order, "amount", None))
            base = (ep or 0.0) * (am or 0.0)
            setattr(order, "pnl_percentage", (pnl / base) * 100 if base else 0.0)
        if closed_at is not None:
            setattr(order, "closed_at", closed_at)
        session.commit()
        return True

    return False


def update_order_status(
    session: Session, order_id: str, status: str, pnl: Optional[float] = None, verify_programmatic: bool = True
) -> bool:
    """
    Update order status and optionally P&L.

    **Verifies order is PROGRAMMATIC by default.**

    Args:
        session: Database session
        order_id: Order ID to update
        status: New status
        pnl: Optional P&L value
        verify_programmatic: If True, only update programmatic orders

    Returns:
        True if updated, False otherwise
    """
    query = session.query(Order).filter(Order.order_id == order_id)

    if verify_programmatic:
        query = query.filter(Order.order_source == "PROGRAMMATIC")

    order = query.first()

    if order:
        setattr(order, "status", status)
        if pnl is not None:
            setattr(order, "pnl", pnl)
            ep = cast(Optional[float], getattr(order, "entry_price", None))
            am = cast(Optional[float], getattr(order, "amount", None))
            base = (ep or 0.0) * (am or 0.0)
            setattr(order, "pnl_percentage", (pnl / base) * 100 if base else 0.0)

        if status == "CLOSED":
            setattr(order, "closed_at", datetime.now(timezone.utc))

        session.commit()
        return True

    return False


def mark_be_moved(
    session: Session,
    order_id: str,
    new_stop_loss: Optional[float] = None,
    new_take_profit: Optional[float] = None,
    verify_programmatic: bool = True,
) -> bool:
    """
    Mark that break-even has been triggered for an order.

    **Verifies order is PROGRAMMATIC by default.**
    Updates stop_loss and/or take_profit, sets be_moved flag.

    Args:
        session: Database session
        order_id: Order ID
        new_stop_loss: New stop loss price (optional)
        new_take_profit: New take profit price (optional)
        verify_programmatic: If True, only update programmatic orders

    Returns:
        True if updated, False otherwise
    """
    query = session.query(Order).filter(Order.order_id == order_id)

    if verify_programmatic:
        query = query.filter(Order.order_source == "PROGRAMMATIC")

    order = query.first()

    if order and not getattr(order, "be_moved", False):
        if new_stop_loss is not None:
            setattr(order, "original_stop_loss", getattr(order, "stop_loss", None))
            setattr(order, "stop_loss", cast(float, new_stop_loss))
        if new_take_profit is not None:
            setattr(order, "take_profit", cast(float, new_take_profit))
        setattr(order, "be_moved", True)
        setattr(order, "be_moved_at", cast(DateTime, cast(datetime, datetime.now(timezone.utc))))
        session.commit()
        return True

    return False


def create_order(session: Session, order_data: Dict[str, Any]) -> Order:
    """
    Create a new programmatic order in the database.

    Args:
        session: Database session
        order_data: Dictionary with order details

    Returns:
        Created Order object

    Raises:
        ValueError: If required fields are missing or invalid
    """
    # Validate required fields
    required_fields = ["order_id", "symbol", "side", "entry_price", "amount"]
    missing_fields = [field for field in required_fields if field not in order_data]

    if missing_fields:
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

    # Validate side
    if order_data.get("side") not in ("LONG", "SHORT"):
        raise ValueError(f"Invalid side: {order_data.get('side')}. Must be 'LONG' or 'SHORT'")

    # Validate numeric fields
    entry_price = order_data.get("entry_price")
    if not isinstance(entry_price, (int, float)) or entry_price <= 0:
        raise ValueError(f"Invalid entry_price: {entry_price}")

    amount = order_data.get("amount")
    if not isinstance(amount, (int, float)) or amount <= 0:
        raise ValueError(f"Invalid amount: {amount}")

    # Validate leverage if provided
    if "leverage" in order_data:
        leverage = order_data.get("leverage")
        if not isinstance(leverage, int) or leverage < 1 or leverage > 125:
            raise ValueError(f"Invalid leverage: {leverage}. Must be between 1 and 125")

    # Ensure order is marked as PROGRAMMATIC
    order_data.setdefault("order_source", "PROGRAMMATIC")
    order_data.setdefault("execution_mode", "AUTO")
    order_data.setdefault("created_at", datetime.now(timezone.utc))

    order = Order(**order_data)
    session.add(order)
    session.commit()
    session.refresh(order)

    return order


def get_orders_by_symbol(
    session: Session, symbol: str, status: Optional[str] = None, limit: int = 50, offset: int = 0
) -> List[Order]:
    """
    Get orders for a specific symbol (programmatic only).

    Args:
        session: Database session
        symbol: Trading symbol
        status: Optional status filter
        limit: Maximum results
        offset: Number of results to skip (for pagination)

    Returns:
        List of Order objects
    """
    query = session.query(Order).filter(Order.order_source == "PROGRAMMATIC", Order.symbol == symbol)

    if status:
        query = query.filter(Order.status == status)

    return query.order_by(desc(Order.created_at)).offset(offset).limit(limit).all()


__all__ = [
    "get_open_positions",
    "get_last_closed_order",
    "get_all_programmatic_orders",
    "get_orders_cursor",
    "is_programmatic_order",
    "get_order_by_id",
    "get_order_by_client_id",
    "update_order_status_by_client_id",
    "update_order_status",
    "mark_be_moved",
    "create_order",
    "get_orders_by_symbol",
]

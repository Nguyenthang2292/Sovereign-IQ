"""
Database Query Layer for Auto Trading System
==============================================

Provides high-level query functions with programmatic order filtering.

CRITICAL: All order queries ONLY return PROGRAMMATIC orders by default.
Manual trades from Binance are excluded from queries.

Created: 2026-02-03
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, cast

from sqlalchemy import desc
from sqlalchemy.orm import Session
from sqlalchemy.types import DateTime, Integer

from .models import AuditLog, GradualRecovery, MartingaleChain, Order, Signal, SystemState

# ============================================================================
# ORDER QUERIES - Only Programmatic Orders
# ============================================================================


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


def get_signals_cursor(
    session: Session,
    last_id: Optional[int] = None,
    limit: int = 50,
    symbol: Optional[str] = None,
    executed: Optional[bool] = None,
) -> List[Signal]:
    """
    Fetch signals using cursor-based pagination.

    Uses Signal.id < last_id for cursor pagination instead of offset.

    Args:
        session: Database session
        last_id: Last Signal.id from previous page (None for first page)
        limit: Maximum number of signals to return
        symbol: Optional symbol filter
        executed: Optional executed filter

    Returns:
        List of Signal objects
    """
    query = session.query(Signal)

    if last_id:
        query = query.filter(Signal.id < last_id)

    if symbol:
        query = query.filter(Signal.symbol == symbol)

    if executed is not None:
        query = query.filter(Signal.executed == executed)

    return query.order_by(desc(Signal.id)).limit(limit).all()


def get_audit_log_cursor(
    session: Session,
    last_id: Optional[int] = None,
    limit: int = 50,
    event_type: Optional[str] = None,
    severity: Optional[str] = None,
) -> List[AuditLog]:
    """
    Fetch audit log entries using cursor-based pagination.

    Uses AuditLog.id < last_id for cursor pagination instead of offset.

    Args:
        session: Database session
        last_id: Last AuditLog.id from previous page (None for first page)
        limit: Maximum number of entries to return
        event_type: Optional event type filter
        severity: Optional severity filter

    Returns:
        List of AuditLog objects
    """
    query = session.query(AuditLog)

    if last_id:
        query = query.filter(AuditLog.id < last_id)

    if event_type:
        query = query.filter(AuditLog.event_type == event_type)

    if severity:
        query = query.filter(AuditLog.severity == severity)

    return query.order_by(desc(AuditLog.id)).limit(limit).all()


def get_martingale_chains_cursor(
    session: Session,
    last_id: Optional[int] = None,
    limit: int = 50,
) -> List[MartingaleChain]:
    """
    Fetch Martingale chains using cursor-based pagination.

    Uses MartingaleChain.id < last_id for cursor pagination.

    Args:
        session: Database session
        last_id: Last MartingaleChain.id from previous page (None for first page)
        limit: Maximum number of chains to return

    Returns:
        List of MartingaleChain objects
    """
    query = session.query(MartingaleChain)
    if last_id:
        query = query.filter(MartingaleChain.id < last_id)
    return query.order_by(desc(MartingaleChain.id)).limit(limit).all()


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
            ep = getattr(order, "entry_price", None)
            am = getattr(order, "amount", None)
            base = (ep or 0) * (am or 0)
            setattr(order, "pnl_percentage", (pnl / base) * 100 if base else 0)
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
            ep = getattr(order, "entry_price", None)
            am = getattr(order, "amount", None)
            base = (ep or 0) * (am or 0)
            setattr(order, "pnl_percentage", (pnl / base) * 100 if base else 0)

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


# ============================================================================
# MARTINGALE QUERIES
# ============================================================================


def get_martingale_state(session: Session, symbol: str) -> Optional[MartingaleChain]:
    """
    Get active Martingale chain for a symbol.

    **Only tracks PROGRAMMATIC order chains.**

    Args:
        session: Database session
        symbol: Trading symbol

    Returns:
        Active MartingaleChain or None
    """
    return (
        session.query(MartingaleChain)
        .filter(MartingaleChain.symbol == symbol, MartingaleChain.status == "ACTIVE")
        .first()
    )


def find_or_create_martingale_chain(
    session: Session, chain_id: str, symbol: str, original_loss: float, initial_order_id: str
) -> MartingaleChain:
    """
    Find existing or create new Martingale chain.

    Args:
        session: Database session
        chain_id: Unique chain identifier
        symbol: Trading symbol
        original_loss: Initial loss amount
        initial_order_id: First order in chain

    Returns:
        MartingaleChain object
    """
    chain = session.query(MartingaleChain).filter(MartingaleChain.chain_id == chain_id).first()

    if not chain:
        chain = MartingaleChain(
            chain_id=chain_id,
            symbol=symbol,
            original_loss=original_loss,
            total_loss=original_loss,
            initial_order_id=initial_order_id,
            current_step=0,
            status="ACTIVE",
        )
        session.add(chain)
        session.commit()
        session.refresh(chain)

    return chain


def update_martingale_chain(
    session: Session,
    chain_id: str,
    current_step: int,
    latest_order_id: str,
    total_loss: float,
    recovered: bool = False,
    recovery_pnl: float = 0.0,
) -> bool:
    """
    Update Martingale chain progress.

    Args:
        session: Database session
        chain_id: Chain ID
        current_step: Current step number
        latest_order_id: Most recent order ID
        total_loss: Updated total loss
        recovered: Whether chain has recovered
        recovery_pnl: Recovery P&L if recovered

    Returns:
        True if updated, False otherwise
    """
    chain = session.query(MartingaleChain).filter(MartingaleChain.chain_id == chain_id).first()

    if chain:
        chain.current_step = current_step
        chain.latest_order_id = latest_order_id
        chain.total_loss = total_loss
        chain.max_step_reached = max(chain.max_step_reached, current_step)

        if recovered:
            chain.recovered = True
            chain.recovery_pnl = recovery_pnl
            chain.status = "RECOVERED"
            chain.recovered_at = datetime.now(timezone.utc)
            chain.recovery_order_id = latest_order_id

        session.commit()
        return True

    return False


def get_active_martingale_chains(session: Session) -> List[MartingaleChain]:
    """
    Get all active Martingale chains.

    Returns:
        List of active MartingaleChain objects
    """
    return (
        session.query(MartingaleChain)
        .filter(MartingaleChain.status == "ACTIVE")
        .order_by(desc(MartingaleChain.created_at))
        .all()
    )


# ============================================================================
# SIGNAL QUERIES
# ============================================================================


def save_signal(
    session: Session,
    correlation_id: str,
    symbol: str,
    signal_type: str,
    confidence: float,
    atc_score: Optional[float] = None,
    xgboost_score: Optional[float] = None,
    gemini_score: Optional[float] = None,
    **kwargs,
) -> Signal:
    """
    Save a new signal to the database.

    Args:
        session: Database session
        correlation_id: Unique signal ID
        symbol: Trading symbol
        signal_type: 'LONG', 'SHORT', or 'NEUTRAL'
        confidence: Signal confidence (0-1)
        atc_score: Optional ATC score
        xgboost_score: Optional XGBoost score
        gemini_score: Optional Gemini score
        **kwargs: Additional signal attributes

    Returns:
        Created Signal object
    """
    signal_data = {
        "correlation_id": correlation_id,
        "symbol": symbol,
        "signal_type": signal_type,
        "confidence": confidence,
        "atc_score": atc_score,
        "xgboost_score": xgboost_score,
        "gemini_score": gemini_score,
        "created_at": datetime.now(timezone.utc),
    }
    signal_data.update(kwargs)

    signal = Signal(**signal_data)
    session.add(signal)
    session.commit()
    session.refresh(signal)

    return signal


def mark_signal_executed(session: Session, correlation_id: str, order_id: str) -> bool:
    """
    Mark a signal as executed with order ID.

    Args:
        session: Database session
        correlation_id: Signal correlation ID
        order_id: Executed order ID

    Returns:
        True if updated, False otherwise
    """
    signal = session.query(Signal).filter(Signal.correlation_id == correlation_id).first()

    if signal:
        signal.executed = True
        signal.execution_order_id = order_id
        signal.executed_at = datetime.now(timezone.utc)
        session.commit()
        return True

    return False


def update_signal_outcome(
    session: Session,
    correlation_id: str,
    outcome: str,
    outcome_pnl: float,
    outcome_duration_minutes: Optional[int] = None,
) -> bool:
    """
    Update signal outcome after order closes.

    Args:
        session: Database session
        correlation_id: Signal correlation ID
        outcome: 'WIN', 'LOSS', 'BREAKEVEN'
        outcome_pnl: Final P&L
        outcome_duration_minutes: Time until outcome

    Returns:
        True if updated, False otherwise
    """
    signal = session.query(Signal).filter(Signal.correlation_id == correlation_id).first()

    if signal:
        signal.outcome = outcome
        signal.outcome_pnl = outcome_pnl
        signal.outcome_duration_minutes = outcome_duration_minutes
        signal.outcome_at = datetime.now(timezone.utc)
        session.commit()
        return True

    return False


def get_recent_signals(session: Session, limit: int = 50, executed_only: bool = False, offset: int = 0) -> List[Signal]:
    """
    Get recent signals.

    Args:
        session: Database session
        limit: Maximum results
        executed_only: If True, only return executed signals
        offset: Number of results to skip (for pagination)

    Returns:
        List of Signal objects
    """
    query = session.query(Signal)

    if executed_only:
        query = query.filter(Signal.executed.is_(True))

    return query.order_by(desc(Signal.created_at)).offset(offset).limit(limit).all()


def get_signal_performance_stats(session: Session, symbol: Optional[str] = None, days: int = 30) -> Dict[str, Any]:
    """
    Calculate signal performance statistics.

    Args:
        session: Database session
        symbol: Optional symbol filter
        days: Number of days to analyze

    Returns:
        Dictionary with performance metrics
    """
    start_date = datetime.now(timezone.utc) - timedelta(days=days)

    query = session.query(Signal).filter(
        Signal.executed.is_(True), Signal.outcome.isnot(None), Signal.created_at >= start_date
    )

    if symbol:
        query = query.filter(Signal.symbol == symbol)

    signals = query.all()

    if not signals:
        return {"total_signals": 0, "win_rate": 0.0, "avg_pnl": 0.0, "total_pnl": 0.0}

    wins = sum(1 for s in signals if s.outcome == "WIN")
    total = len(signals)
    total_pnl = sum(s.outcome_pnl or 0 for s in signals)

    return {
        "total_signals": total,
        "win_rate": (wins / total) * 100 if total > 0 else 0.0,
        "avg_pnl": total_pnl / total if total > 0 else 0.0,
        "total_pnl": total_pnl,
        "wins": wins,
        "losses": sum(1 for s in signals if s.outcome == "LOSS"),
        "breakevens": sum(1 for s in signals if s.outcome == "BREAKEVEN"),
    }


# ============================================================================
# SYSTEM STATE QUERIES
# ============================================================================


def get_system_state(session: Session, key: str) -> Optional[Any]:
    """
    Get system state value by key.

    Args:
        session: Database session
        key: State key

    Returns:
        State value with correct type or None
    """
    state = session.query(SystemState).filter(SystemState.key == key).first()
    return state.get_typed_value() if state else None


def set_system_state(
    session: Session,
    key: str,
    value: Any,
    value_type: str = "string",
    description: Optional[str] = None,
    category: Optional[str] = None,
) -> bool:
    """
    Set system state value.

    Args:
        session: Database session
        key: State key
        value: State value
        value_type: Type of value ('string', 'integer', 'float', 'boolean', 'json')
        description: Optional description
        category: Optional category

    Returns:
        True if updated/created, False otherwise
    """
    state = session.query(SystemState).filter(SystemState.key == key).first()

    # Convert value to string
    if value_type == "json":
        import json

        value_str = json.dumps(value)
    else:
        value_str = str(value)

    if state:
        state.value = value_str
        state.value_type = value_type
        if description:
            state.description = description
        if category:
            state.category = category
    else:
        state = SystemState(key=key, value=value_str, value_type=value_type, description=description, category=category)
        session.add(state)

    session.commit()
    return True


# ============================================================================
# AUDIT LOG QUERIES
# ============================================================================


def create_audit_log(
    session: Session, event_type: str, event_category: str, severity: str, event_summary: str, **kwargs
) -> AuditLog:
    """
    Create audit log entry.

    Args:
        session: Database session
        event_type: Type of event
        event_category: Event category
        severity: Severity level
        event_summary: Human-readable summary
        **kwargs: Additional audit log fields

    Returns:
        Created AuditLog object
    """
    log_data = {
        "event_type": event_type,
        "event_category": event_category,
        "severity": severity,
        "event_summary": event_summary,
        "timestamp": datetime.now(timezone.utc),
    }
    log_data.update(kwargs)

    log = AuditLog(**log_data)
    session.add(log)
    session.commit()

    return log


def get_recent_audit_logs(
    session: Session,
    limit: int = 100,
    severity: Optional[str] = None,
    event_type: Optional[str] = None,
    offset: int = 0,
) -> List[AuditLog]:
    """
    Get recent audit log entries.

    Args:
        session: Database session
        limit: Maximum results
        severity: Optional severity filter
        event_type: Optional event type filter
        offset: Number of results to skip (for pagination)

    Returns:
        List of AuditLog objects
    """
    query = session.query(AuditLog)

    if severity:
        query = query.filter(AuditLog.severity == severity)

    if event_type:
        query = query.filter(AuditLog.event_type == event_type)

    return query.order_by(desc(AuditLog.timestamp)).offset(offset).limit(limit).all()


# ============================================================================
# STATISTICS QUERIES
# ============================================================================


def get_daily_stats(session: Session, days: int = 30) -> List[Dict[str, Any]]:
    """
    Get daily trading statistics using database aggregation for performance.

    Args:
        session: Database session
        days: Number of days to retrieve

    Returns:
        List of daily stat dictionaries
    """
    start_date = datetime.now(timezone.utc) - timedelta(days=days)

    # Use SQLAlchemy aggregation instead of loading all orders into memory
    from sqlalchemy import case

    results = (
        session.query(
            func.date(Order.closed_at).label("date"),
            func.count(Order.id).label("total_trades"),
            func.sum(case((Order.pnl > 0, 1), else_=0)).label("winning_trades"),
            func.sum(case((Order.pnl < 0, 1), else_=0)).label("losing_trades"),
            func.sum(Order.pnl).label("total_pnl"),
            func.sum(Order.commission).label("total_fees"),
            func.max(Order.pnl).label("best_trade"),
            func.min(Order.pnl).label("worst_trade"),
        )
        .filter(Order.order_source == "PROGRAMMATIC", Order.status == "CLOSED", Order.closed_at >= start_date)
        .group_by(func.date(Order.closed_at))
        .order_by(func.date(Order.closed_at).desc())
        .all()
    )

    # Transform results into dictionaries
    stats_list = []
    for row in results:
        if row.date is None:
            continue

        stats = {
            "date": row.date.isoformat(),
            "total_trades": row.total_trades or 0,
            "winning_trades": row.winning_trades or 0,
            "losing_trades": row.losing_trades or 0,
            "total_pnl": float(row.total_pnl or 0),
            "total_fees": float(row.total_fees or 0),
            "best_trade": float(row.best_trade or 0),
            "worst_trade": float(row.worst_trade or 0),
        }

        # Calculate averages
        if stats["total_trades"] > 0:
            stats["avg_pnl"] = stats["total_pnl"] / stats["total_trades"]
            stats["win_rate"] = (stats["winning_trades"] / stats["total_trades"]) * 100
        else:
            stats["avg_pnl"] = 0.0
            stats["win_rate"] = 0.0

        stats_list.append(stats)

    return stats_list


def get_overall_stats(session: Session) -> Dict[str, Any]:
    """
    Get overall trading statistics (all-time) using single aggregation query.

    Args:
        session: Database session

    Returns:
        Dictionary with overall statistics
    """
    from sqlalchemy import case, func

    # Single aggregation query for all statistics
    result = (
        session.query(
            func.count(Order.id).label("total_trades"),
            func.sum(case((Order.pnl > 0, 1), else_=0)).label("winning_trades"),
            func.sum(case((Order.pnl < 0, 1), else_=0)).label("losing_trades"),
            func.sum(Order.pnl).label("total_pnl"),
            func.avg(Order.pnl).label("avg_pnl"),
            func.sum(Order.commission).label("total_fees"),
            func.max(Order.pnl).label("best_trade"),
            func.min(Order.pnl).label("worst_trade"),
        )
        .filter(Order.order_source == "PROGRAMMATIC", Order.status == "CLOSED")
        .first()
    )

    if not result or result.total_trades == 0:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "total_fees": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
        }

    total_trades = result.total_trades or 0
    winning_trades = result.winning_trades or 0
    losing_trades = result.losing_trades or 0
    total_pnl = result.total_pnl or 0.0
    avg_pnl = float(result.avg_pnl) if result.avg_pnl else 0.0
    total_fees = result.total_fees or 0.0
    best_trade = result.best_trade or 0.0
    worst_trade = result.worst_trade or 0.0

    return {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": (winning_trades / total_trades) * 100 if total_trades > 0 else 0.0,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "total_fees": total_fees,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
    }


# ============================================================================
# GRADUAL RECOVERY QUERIES
# ============================================================================


def get_active_gradual_recovery(session: Session, symbol: Optional[str] = None) -> Optional[GradualRecovery]:
    """
    Get active Gradual Recovery record.

    For GLOBAL recovery (symbol=None), returns the first active recovery.
    For per-symbol recovery, returns the active recovery for that symbol.

    Args:
        session: Database session
        symbol: Optional symbol filter (None for global recovery)

    Returns:
        Active GradualRecovery or None
    """
    query = session.query(GradualRecovery).filter(GradualRecovery.status == "ACTIVE")

    if symbol:
        query = query.filter(GradualRecovery.symbol == symbol)
    else:
        # For global recovery, use a special symbol marker
        query = query.filter(GradualRecovery.symbol == "GLOBAL")

    return query.order_by(desc(GradualRecovery.created_at)).first()


def create_gradual_recovery(
    session: Session,
    recovery_id: str,
    initial_loss: float,
    config: Dict[str, Any],
    symbol: Optional[str] = None,
) -> GradualRecovery:
    """
    Create a new Gradual Recovery record.

    Args:
        session: Database session
        recovery_id: Unique recovery identifier
        initial_loss: Initial loss amount to recover
        config: RecoveryConfig dictionary
        symbol: Symbol for per-symbol recovery (None for global)

    Returns:
        Created GradualRecovery object
    """
    recovery = GradualRecovery(
        recovery_id=recovery_id,
        symbol=symbol or "GLOBAL",
        status="ACTIVE",
        initial_loss=initial_loss,
        remaining_loss=initial_loss,
        total_profit_accumulated=0.0,
        recovery_percentage=0.0,
        trades_count=0,
        win_streak=0,
        estimated_trades_remaining=0,
    )
    recovery.set_config(config)

    session.add(recovery)
    session.commit()
    session.refresh(recovery)

    return recovery


def update_gradual_recovery(
    session: Session,
    recovery_id: str,
    remaining_loss: Optional[float] = None,
    total_profit_accumulated: Optional[float] = None,
    recovery_percentage: Optional[float] = None,
    trades_count: Optional[int] = None,
    win_streak: Optional[int] = None,
    estimated_trades_remaining: Optional[int] = None,
    status: Optional[str] = None,
) -> bool:
    """
    Update Gradual Recovery state fields.

    Args:
        session: Database session
        recovery_id: Recovery ID to update
        remaining_loss: Updated remaining loss
        total_profit_accumulated: Updated total profit
        recovery_percentage: Updated recovery percentage
        trades_count: Updated trade count
        win_streak: Updated win streak
        estimated_trades_remaining: Updated estimate
        status: Updated status

    Returns:
        True if updated, False otherwise
    """
    recovery = session.query(GradualRecovery).filter(GradualRecovery.recovery_id == recovery_id).first()

    if not recovery:
        return False

    if remaining_loss is not None:
        recovery.remaining_loss = remaining_loss
    if total_profit_accumulated is not None:
        recovery.total_profit_accumulated = total_profit_accumulated
    if recovery_percentage is not None:
        recovery.recovery_percentage = recovery_percentage
    if trades_count is not None:
        recovery.trades_count = trades_count
    if win_streak is not None:
        recovery.win_streak = win_streak
    if estimated_trades_remaining is not None:
        recovery.estimated_trades_remaining = estimated_trades_remaining
    if status is not None:
        recovery.status = status
        if status == "COMPLETE":
            recovery.completed_at = datetime.now(timezone.utc)
        elif status == "FAILED":
            recovery.failed_at = datetime.now(timezone.utc)

    session.commit()
    return True


def cancel_gradual_recovery(session: Session, recovery_id: str) -> bool:
    """
    Cancel a Gradual Recovery record.

    Args:
        session: Database session
        recovery_id: Recovery ID to cancel

    Returns:
        True if cancelled, False otherwise
    """
    recovery = session.query(GradualRecovery).filter(GradualRecovery.recovery_id == recovery_id).first()

    if not recovery:
        return False

    recovery.status = "CANCELLED"
    session.commit()
    return True


def get_gradual_recovery_by_id(session: Session, recovery_id: str) -> Optional[GradualRecovery]:
    """
    Get Gradual Recovery by ID.

    Args:
        session: Database session
        recovery_id: Recovery ID

    Returns:
        GradualRecovery object or None
    """
    return session.query(GradualRecovery).filter(GradualRecovery.recovery_id == recovery_id).first()


def get_all_gradual_recoveries(
    session: Session, status: Optional[str] = None, limit: int = 50, offset: int = 0
) -> List[GradualRecovery]:
    """
    Get all Gradual Recovery records.

    Args:
        session: Database session
        status: Optional status filter
        limit: Maximum results
        offset: Number to skip

    Returns:
        List of GradualRecovery objects
    """
    query = session.query(GradualRecovery)

    if status:
        query = query.filter(GradualRecovery.status == status)

    return query.order_by(desc(GradualRecovery.created_at)).offset(offset).limit(limit).all()

"""
Signal Queries Module
=====================

Signal lifecycle and performance tracking queries for the auto_trade system.

Features:
- Signal creation and execution tracking
- Correlation between signals and orders
- Performance analytics with win/loss tracking

Functions:
- save_signal: Save a new signal to database
- mark_signal_executed: Mark signal as executed with order ID
- update_signal_outcome: Update signal outcome after order closes
- get_recent_signals: Get recent signals with optional filters
- get_signal_performance_stats: Calculate performance statistics
- get_signals_cursor: Cursor-based pagination for signals
"""

from ._shared import (
    Any,
    Dict,
    List,
    Optional,
    Session,
    Signal,
    datetime,
    desc,
    timedelta,
)


def save_signal(
    session: Session,
    correlation_id: str,
    symbol: str,
    signal_type: str,
    confidence: float,
    atc_score: Optional[float] = None,
    xgboost_score: Optional[float] = None,
    gemini_score: Optional[float] = None,
    **kwargs: Any,
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
        "created_at": datetime.utcnow(),
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
        signal.executed_at = datetime.utcnow()
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
        signal.outcome_at = datetime.utcnow()
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
    start_date = datetime.utcnow() - timedelta(days=days)

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


__all__ = [
    "save_signal",
    "mark_signal_executed",
    "update_signal_outcome",
    "get_recent_signals",
    "get_signal_performance_stats",
    "get_signals_cursor",
]

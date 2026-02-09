"""
Gradual Recovery Queries Module
================================

Gradual recovery strategy management queries for the auto_trade system.

Features:
- Global vs. per-symbol recovery modes
- Progress tracking and estimation
- Status management (ACTIVE, COMPLETE, FAILED, CANCELLED)

Functions:
- get_active_gradual_recovery: Get active recovery record
- create_gradual_recovery: Create a new recovery record
- update_gradual_recovery: Update recovery state fields
- cancel_gradual_recovery: Cancel a recovery record
- get_gradual_recovery_by_id: Get recovery by ID
- get_all_gradual_recoveries: Get all recovery records with filters
"""

from ._shared import (
    Any,
    Dict,
    GradualRecovery,
    List,
    timezone,
    Optional,
    Session,
    datetime,
    desc,
)


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


__all__ = [
    "get_active_gradual_recovery",
    "create_gradual_recovery",
    "update_gradual_recovery",
    "cancel_gradual_recovery",
    "get_gradual_recovery_by_id",
    "get_all_gradual_recoveries",
]

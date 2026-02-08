"""
Statistics Queries Module
==========================

Performance analytics and reporting queries for the auto_trade system.

Features:
- Database-level aggregation for performance
- Comprehensive trading metrics (win rate, P&L, fees)
- Daily and overall statistics

Functions:
- get_daily_stats: Get daily trading statistics
- get_overall_stats: Get overall trading statistics (all-time)
"""

from ._shared import (
    Any,
    Dict,
    List,
    Order,
    Session,
    case,
    datetime,
    func,
    timedelta,
)


def get_daily_stats(session: Session, days: int = 30) -> List[Dict[str, Any]]:
    """
    Get daily trading statistics using database aggregation for performance.

    Args:
        session: Database session
        days: Number of days to retrieve

    Returns:
        List of daily stat dictionaries
    """
    start_date = datetime.utcnow() - timedelta(days=days)

    # Use SQLAlchemy aggregation instead of loading all orders into memory
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


__all__ = [
    "get_daily_stats",
    "get_overall_stats",
]

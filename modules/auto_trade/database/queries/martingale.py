"""
Martingale Chain Queries Module
================================

Martingale chain management queries for the auto_trade system.

CRITICAL: Only tracks PROGRAMMATIC order chains.
Manual trades are not included in Martingale chains.

Functions:
- get_martingale_state: Get active Martingale chain for a symbol
- find_or_create_martingale_chain: Find existing or create new chain
- update_martingale_chain: Update chain progress
- get_active_martingale_chains: Get all active chains
- get_martingale_chains_cursor: Cursor-based pagination for chains
"""

from ._shared import (
    List,
    MartingaleChain,
    Optional,
    Session,
    datetime,
    desc,
)


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
            chain.recovered_at = datetime.utcnow()
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


__all__ = [
    "get_martingale_state",
    "find_or_create_martingale_chain",
    "update_martingale_chain",
    "get_active_martingale_chains",
    "get_martingale_chains_cursor",
]

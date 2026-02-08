"""
System State Queries Module
============================

Key-value system state management queries for the auto_trade system.

Features:
- Type-aware value storage (string, integer, float, boolean, json)
- Automatic type conversion
- Category and description support

Functions:
- get_system_state: Get system state value by key
- set_system_state: Set system state value with type awareness
"""

from ._shared import (
    Any,
    Optional,
    Session,
    SystemState,
)


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
    value_str: str
    if value_type == "json":
        import json

        value_str = json.dumps(value)
    else:
        value_str = str(value)

    if state:
        setattr(state, "value", value_str)
        setattr(state, "value_type", value_type)
        if description:
            setattr(state, "description", description)
        if category:
            setattr(state, "category", category)
    else:
        state = SystemState(key=key, value=value_str, value_type=value_type, description=description, category=category)
        session.add(state)

    session.commit()
    return True


__all__ = [
    "get_system_state",
    "set_system_state",
]

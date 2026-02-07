"""MemOS API wrapper: optional dependency, add_message and search_memory."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_MEMOS_CLIENT: Any = None


def _get_memos_client(api_key: str | None) -> Any:
    """Import MemOSClient only when needed; return None if unavailable."""
    global _MEMOS_CLIENT
    if _MEMOS_CLIENT is not None:
        return _MEMOS_CLIENT
    api_key = api_key or __import__("os").environ.get("MEMOS_API_KEY")
    if not api_key:
        logger.debug("MEMOS_API_KEY not set; agent_memory store/recall will no-op")
        return None
    try:
        from memos.api.client import MemOSClient

        _MEMOS_CLIENT = MemOSClient(api_key=api_key)
        return _MEMOS_CLIENT
    except Exception as e:
        logger.warning("MemOS client unavailable: %s", e)
        return None


def get_client(api_key: str | None = None) -> Any:
    """Return MemOSClient instance or None if API key missing or package not installed."""
    return _get_memos_client(api_key)


def add_message(
    messages: list[dict[str, str]],
    user_id: str,
    conversation_id: str,
    *,
    api_key: str | None = None,
) -> dict | None:
    """Store messages in MemOS. Returns response dict or None on failure."""
    client = get_client(api_key)
    if not client:
        return None
    try:
        return client.add_message(
            messages=messages,
            user_id=user_id,
            conversation_id=conversation_id,
        )
    except Exception as e:
        logger.warning("MemOS add_message failed: %s", e)
        return None


def search_memory(
    query: str,
    user_id: str,
    conversation_id: str | None = None,
    *,
    api_key: str | None = None,
) -> list | dict | None:
    """Search MemOS for relevant memories. Returns result list/dict or None on failure."""
    client = get_client(api_key)
    if not client:
        return None
    try:
        method = getattr(client, "search_memory", None) or getattr(client, "search", None)
        if not method:
            logger.warning("MemOS client has no search_memory/search method")
            return None
        kwargs = {"query": query, "user_id": user_id}
        if conversation_id is not None:
            kwargs["conversation_id"] = conversation_id
        return method(**kwargs)
    except Exception as e:
        logger.warning("MemOS search failed: %s", e)
        return None

"""Agent memory module: MemOS-backed workflow and project-conventions context for Cursor, OpenCode, Claude Code, Antigravity."""

from modules.agent_memory.client import get_client, add_message, search_memory
from modules.agent_memory.config import get_user_id, get_conversation_id, get_context_path

__all__ = [
    "get_client",
    "add_message",
    "search_memory",
    "get_user_id",
    "get_conversation_id",
    "get_context_path",
]

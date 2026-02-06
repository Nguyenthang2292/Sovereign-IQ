"""Recall: search MemOS and write .cursor/agent_memory_context.md for agents."""

from __future__ import annotations

from pathlib import Path

from modules.agent_memory.client import search_memory
from modules.agent_memory.config import (
    get_context_path,
    get_conversation_id,
    get_user_id,
)


def _flatten_memories(result: list | dict | None) -> list[str]:
    """Turn MemOS search result into list of text strings."""
    if result is None:
        return []
    if isinstance(result, list):
        texts = []
        for item in result:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                texts.append(item.get("memory", item.get("content", str(item))))
            else:
                texts.append(str(item))
        return texts
    if isinstance(result, dict):
        # e.g. {"text_mem": [...]} or similar
        for key in ("text_mem", "memories", "results", "data"):
            if key in result and result[key]:
                return _flatten_memories(result[key])
        return [str(result)]
    return [str(result)]


def _section_block(title: str, lines: list[str], empty_msg: str = "(No memories found.)") -> str:
    """Build a Markdown section."""
    if not lines:
        return f"## {title}\n\n{empty_msg}\n\n"
    body = "\n".join(f"- {line}" for line in lines)
    return f"## {title}\n\n{body}\n\n"


def run_recall(
    *,
    user_id: str | None = None,
    conversation_id: str | None = None,
    context_path: Path | None = None,
    api_key: str | None = None,
) -> bool:
    """
    Search MemOS for conventions and recent workflow, write context file.
    Returns True if file was written (even if empty).
    """
    user_id = user_id or get_user_id()
    conversation_id = conversation_id or get_conversation_id()
    context_path = context_path or get_context_path()

    query_conventions = "project conventions structure coding style modules"
    query_workflow = "recent workflow tasks done commits"

    conventions = _flatten_memories(
        search_memory(query_conventions, user_id, conversation_id, api_key=api_key)
    )
    workflow = _flatten_memories(
        search_memory(query_workflow, user_id, conversation_id, api_key=api_key)
    )

    header = (
        "# Agent memory context (MemOS)\n\n"
        "Context below is for agents (Cursor, OpenCode, Claude Code, Antigravity). "
        "Use for project conventions and recent workflow.\n\n"
    )
    sections = [
        _section_block("Project conventions / Quy ước dự án", conventions),
        _section_block("Recent workflow / Workflow gần đây", workflow),
    ]
    content = header + "".join(sections)

    context_path.parent.mkdir(parents=True, exist_ok=True)
    context_path.write_text(content, encoding="utf-8")
    return True

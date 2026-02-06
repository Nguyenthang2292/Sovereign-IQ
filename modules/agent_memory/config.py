"""Configuration for agent_memory: user_id, conversation_id, context file path."""

import os
from datetime import date
from pathlib import Path


def _repo_root() -> Path:
    """Resolve repo root (directory containing .git or pyproject.toml)."""
    cur = Path(__file__).resolve().parent
    for _ in range(10):
        if (cur / ".git").exists() or (cur / "pyproject.toml").exists():
            return cur
        cur = cur.parent
    return Path.cwd()


def get_user_id() -> str:
    """MemOS user_id for this repo. Default: crypto_probability_repo."""
    return os.environ.get("MEMOS_USER_ID", "crypto_probability_repo")


def get_conversation_id() -> str:
    """MemOS conversation_id. Default: today's date (YYYY-MM-DD)."""
    return os.environ.get("MEMOS_CONVERSATION_ID", date.today().isoformat())


def get_context_path() -> Path:
    """Path to the context file written by recall (read by agents)."""
    rel = os.environ.get("AGENT_MEMORY_CONTEXT_PATH", ".cursor/agent_memory_context.md")
    if Path(rel).is_absolute():
        return Path(rel)
    return _repo_root() / rel

"""Store: add_message to MemOS (from summary or from commit)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from modules.agent_memory.client import add_message
from modules.agent_memory.config import get_conversation_id, get_user_id


def store_summary(
    summary: str,
    *,
    user_id: str | None = None,
    conversation_id: str | None = None,
    api_key: str | None = None,
) -> bool:
    """
    Store a single workflow summary as one user + one assistant message.
    Returns True if MemOS accepted the message.
    """
    user_id = user_id or get_user_id()
    conversation_id = conversation_id or get_conversation_id()
    messages = [
        {"role": "user", "content": f"Workflow summary: {summary}"},
        {"role": "assistant", "content": f"Recorded: {summary}"},
    ]
    return add_message(messages, user_id, conversation_id, api_key=api_key) is not None


def commit_summary(repo_root: Path | None = None) -> bool:
    """
    Store last git commit (message + short diff stat) as workflow.
    Call from post-commit hook. Returns True if stored successfully.
    """
    repo_root = repo_root or Path.cwd()
    try:
        msg = subprocess.run(
            ["git", "log", "-1", "--pretty=%s"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        commit_msg = (msg.stdout or "").strip() or "(no message)"
        diff = subprocess.run(
            ["git", "diff", "HEAD~1", "--stat"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        diff_stat = (diff.stdout or "").strip() or "(no diff)"
        summary = f"Commit: {commit_msg}\nDiff stat:\n{diff_stat}"
        return store_summary(summary)
    except Exception:
        return False

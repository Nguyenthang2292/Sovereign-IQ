"""Token limit helpers for interacting with Gemini."""

MAX_TOKENS_PER_REQUEST = 8192
PROMPT_TOKEN_WARNING_THRESHOLD = 0.8


def estimate_token_count(text: str) -> int:
    """Roughly estimate Gemini token usage by character length."""
    return len(text) // 4

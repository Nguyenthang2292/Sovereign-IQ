"""
Utility functions for masking sensitive information like API keys and secrets.
"""

def mask_api_key(key: str) -> str:
    """
    Masks an API key according to the following rules:
    - None or empty: Returns "—"
    - Length <= 8: Returns all asterisks
    - Length > 8: Returns first 4 + asterisks + last 4 characters
    """
    if not key:
        return "—"

    n: int = len(key)
    if n <= 8:
        return "*" * n

    return f"{key[:4]}{'*' * (n - 8)}{key[-4:]}"

def mask_secret(secret: str) -> str:
    """
    Masks a secret according to the following rules:
    - None or empty: Returns "—"
    - Length <= 8: Returns 8 asterisks
    - Length > 8: Returns first 4 + asterisks + last 4 characters (same as key)
    """
    if not secret:
        return "—"

    n: int = len(secret)
    if n <= 8:
        return "*" * 8

    return f"{secret[:4]}{'*' * (n - 8)}{secret[-4:]}"

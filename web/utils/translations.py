"""
Translation/i18n utilities for FastAPI backend.

Provides locale-aware message translation using JSON locale files.
"""

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

# Module-level logger
logger = logging.getLogger(__name__)

# Default locale
DEFAULT_LOCALE = "en"

# Supported locales
SUPPORTED_LOCALES = ["en", "vi"]


def _validate_and_normalize_locale(locale: str) -> str:
    """
    Validate and normalize locale string to prevent cache pollution and security issues.

    Args:
        locale: Locale code to validate

    Returns:
        Normalized, valid locale code

    Raises:
        ValueError: If locale is invalid (format violation or path traversal attempt)
    """
    if not locale or not isinstance(locale, str):
        raise ValueError(f"Invalid locale: must be a non-empty string, got {type(locale).__name__}")

    # Validate locale format to prevent path traversal attacks
    # Only allow letters, digits, hyphen, and underscore
    if not re.match(r"^[a-zA-Z0-9_-]+$", locale):
        raise ValueError(f"Invalid locale format: '{locale}' contains invalid characters")

    # Additional check: ensure Path(locale).name equals original (prevents hidden path traversal)
    if Path(locale).name != locale:
        raise ValueError(f"Invalid locale: '{locale}' appears to be a path traversal attempt")

    # Normalize: convert to lowercase and check if supported
    locale_lower = locale.lower()
    if locale_lower not in SUPPORTED_LOCALES:
        raise ValueError(f"Unsupported locale: '{locale}'. Supported locales: {SUPPORTED_LOCALES}")

    return locale_lower


@lru_cache(maxsize=10)
def _load_locale_messages(locale: str) -> Dict:
    """
    Load locale messages from JSON file.

    This function assumes the locale has already been validated and normalized.
    It should only be called with valid locale strings from SUPPORTED_LOCALES.

    Args:
        locale: Normalized, validated locale code (e.g., 'en', 'vi')

    Returns:
        Dict of translation messages

    Raises:
        FileNotFoundError: If locale file doesn't exist
    """
    base_dir = Path(__file__).parent
    locale_file = base_dir / "locales" / f"{locale}.json"

    if not locale_file.exists():
        # Fallback to English if locale file doesn't exist
        if locale != DEFAULT_LOCALE:
            return _load_locale_messages(DEFAULT_LOCALE)
        raise FileNotFoundError(f"Locale file not found: {locale_file}")

    with open(locale_file, "r", encoding="utf-8") as f:
        return json.load(f)


def get_locale_from_header(accept_language: Optional[str]) -> str:
    """
    Extract locale from Accept-Language header.

    Args:
        accept_language: Accept-Language header value (e.g., "vi,en;q=0.9")

    Returns:
        Locale code (default: "en")
    """
    if not accept_language:
        return DEFAULT_LOCALE

    # Parse Accept-Language header
    # Format: "vi,en;q=0.9" -> ["vi", "en"]
    languages = accept_language.split(",")
    for lang in languages:
        # Extract language code (e.g., "vi" from "vi;q=0.9")
        lang_code = lang.split(";")[0].strip().lower()
        # Check if it's a supported locale or starts with a supported locale
        for supported in SUPPORTED_LOCALES:
            if lang_code == supported or lang_code.startswith(f"{supported}-"):
                return supported

    return DEFAULT_LOCALE


def translate(key: str, locale: Optional[str] = None, **kwargs) -> str:
    """
    Translate a message key to the specified locale.

    Args:
        key: Translation key (e.g., "errors.logReadError")
        locale: Locale code (default: DEFAULT_LOCALE)
        **kwargs: Parameters to format into the message (e.g., error=str(e))

    Returns:
        Translated message string

    Examples:
        >>> translate("errors.logReadError", locale="vi", error="File not found")
        "Lỗi khi đọc logs: File not found"

        >>> translate("errors.logReadError", locale="en", error="File not found")
        "Error reading logs: File not found"
    """
    if locale is None:
        locale = DEFAULT_LOCALE

    # Validate and normalize locale before caching
    # Invalid locales will raise ValueError instead of polluting the cache
    try:
        locale = _validate_and_normalize_locale(locale)
    except ValueError:
        # Invalid locale provided, fallback to default
        logger.warning(f"Invalid locale provided, falling back to default: {locale}")
        locale = DEFAULT_LOCALE

    try:
        messages = _load_locale_messages(locale)

        # Navigate through nested keys (e.g., "errors.logReadError" -> messages["errors"]["logReadError"])
        keys = key.split(".")
        value = messages
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                # Key not found, try fallback to default locale
                if locale != DEFAULT_LOCALE:
                    logger.debug(
                        f"Translation key '{key}' not found in locale '{locale}', "
                        f"falling back to default locale '{DEFAULT_LOCALE}'"
                    )
                    return translate(key, locale=DEFAULT_LOCALE, **kwargs)
                # Even default locale doesn't have the key, return the key itself
                logger.debug(
                    f"Translation key '{key}' not found in default locale '{DEFAULT_LOCALE}', returning key as-is"
                )
                return key

        # Format the message with kwargs if provided
        if isinstance(value, str) and kwargs:
            try:
                return value.format(**kwargs)
            except (KeyError, ValueError):
                # If formatting fails, return the message as-is
                return value

        return str(value) if value is not None else key

    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        # Fallback: return the key itself if translation fails
        logger.warning(
            f"Translation error for key '{key}' in locale '{locale}': {type(e).__name__}: {e}. Returning key as-is"
        )
        return key


# Alias for convenience (similar to gettext convention)
_ = translate

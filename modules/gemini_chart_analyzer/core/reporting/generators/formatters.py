"""
HTML formatting utilities for report generation.

This module provides functions for formatting text, converting markdown-like
syntax to HTML, and handling signal colors.
"""

import re
from html import escape as html_escape


def format_text_to_html(text: str) -> str:
    """
    Convert analysis text (with simple markdown-like syntax) to HTML.

    Args:
        text: Plain text with markdown-like formatting

    Returns:
        HTML-formatted text
    """
    if not text:
        return ""

    # Escape HTML first
    html_text = html_escape(text)

    # Replace markdown-like syntax
    # Bold: **text**
    html_text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html_text)

    # Italic: *text*
    html_text = re.sub(r"\*(.*?)\*", r"<em>\1</em>", html_text)

    # Newlines to <p> tags
    paragraphs = html_text.split("\n\n")
    formatted_paragraphs = []
    for p in paragraphs:
        if p.strip():
            # Convert single newlines within paragraphs to <br>
            p_with_br = p.replace("\n", "<br>")
            formatted_paragraphs.append(f"<p>{p_with_br}</p>")

    return "\n".join(formatted_paragraphs)


def get_signal_color(signal: str) -> str:
    """
    Get color code for signal type.

    Args:
        signal: Signal type (LONG, SHORT, NONE)

    Returns:
        Hex color code
    """
    signal_upper = str(signal).upper()
    if signal_upper == "LONG":
        return "#48bb78"  # green
    elif signal_upper == "SHORT":
        return "#f56565"  # red
    else:
        return "#a0a0a0"  # gray


def escape_html(text: str) -> str:
    """
    Escape HTML special characters.

    Args:
        text: Text to escape

    Returns:
        HTML-escaped text
    """
    return html_escape(text)

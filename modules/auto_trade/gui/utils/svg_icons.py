"""SVG Icon helper for CustomTkinter GUI.

Converts SVG strings to ``CTkImage`` objects via ``cairosvg`` + ``Pillow``.
Results are **cached** by (svg_key, size, color) to avoid repeated re-renders.

Usage
-----
    from modules.auto_trade.gui.utils.svg_icons import get_icon, ICONS

    image = get_icon("save", size=(18, 18))
    ctk.CTkButton(..., image=image, compound="left")

Colour rewriting
----------------
The raw SVG strings use ``currentColor`` as the stroke/fill value.
``get_icon()`` replaces ``currentColor`` with the requested ``color`` argument
before rendering, so icons inherit whatever theme colour you pass.
"""

from __future__ import annotations

import io
from typing import Tuple

import customtkinter as ctk
from PIL import Image

# ---------------------------------------------------------------------------
# SVG source strings (Lucide icon set — MIT licence)
# Each string is the *inner* SVG markup; the wrapper <svg> tag is added by
# get_icon() so we can control width/height/viewBox freely.
# ---------------------------------------------------------------------------

_SVG_DEFS: dict[str, str] = {
    # floppy-disk / save
    "save": (
        '<path d="M15.2 3a2 2 0 0 1 1.4.6l3.8 3.8a2 2 0 0 1 .6 1.4V19a2 2 0 0 1-2 2H5'
        'a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z"/>'
        '<path d="M17 21v-7a1 1 0 0 0-1-1H8a1 1 0 0 0-1 1v7"/>'
        '<path d="M7 3v4a1 1 0 0 0 1 1h7"/>'
    ),
    # refresh / run migrations
    "refresh": (
        '<path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/>'
        '<path d="M21 3v5h-5"/>'
        '<path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/>'
        '<path d="M8 16H3v5"/>'
    ),
    # git-compare / reconcile
    "git_compare": (
        '<circle cx="18" cy="18" r="3"/>'
        '<circle cx="6" cy="6" r="3"/>'
        '<path d="M13 6h3a2 2 0 0 1 2 2v7"/>'
        '<path d="M11 18H8a2 2 0 0 1-2-2V9"/>'
        '<path d="m8 3-2 3 2 3"/>'
        '<path d="m16 21 2-3-2-3"/>'
    ),
    # trash-2 / remove orders
    "trash": (
        '<path d="M3 6h18"/>'
        '<path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/>'
        '<path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/>'
        '<line x1="10" x2="10" y1="11" y2="17"/>'
        '<line x1="14" x2="14" y1="11" y2="17"/>'
    ),
    # sparkles / cleanup
    "sparkles": (
        '<path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5'
        " 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0"
        " 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437"
        'l-1.582 6.135a.5.5 0 0 1-.963 0z"/>'
        '<path d="M20 3v4"/>'
        '<path d="M22 5h-4"/>'
        '<path d="M4 17v2"/>'
        '<path d="M5 18H3"/>'
    ),
    # file-up / export csv
    "file_up": (
        '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/>'
        '<path d="M14 2v4a2 2 0 0 0 2 2h4"/>'
        '<path d="M12 12v6"/>'
        '<path d="m15 15-3-3-3 3"/>'
    ),
    # scroll-text / audit log
    "scroll_text": (
        '<path d="M15 12h-5"/>'
        '<path d="M15 8h-5"/>'
        '<path d="M19 17V5a2 2 0 0 0-2-2H4"/>'
        '<path d="M8 21h12a2 2 0 0 0 2-2v-1a1 1 0 0 0-1-1H11a1 1 0 0 0-1 1v1a2 2 0 0 1'
        '-2 2h0a2 2 0 0 1-2-2v-6.5"/>'  # noqa: ISC001
    ),
    # shield-check / integrity
    "shield_check": (
        '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6'
        "a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5"
        ' 19 5a1 1 0 0 1 1 1z"/>'
        '<path d="m9 12 2 2 4-4"/>'
    ),
    "folder_open": (
        '<path d="m6 14 1.5-2.9A2 2 0 0 1 9.24 10H20a2 2 0 0 1 1.94 2.5l-1.54 6a2 2 0 0 1-1.95 1.5H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h3.9a2 2 0 0 1 1.69.9l.81 1.2a2 2 0 0 0 1.67.9H18a2 2 0 0 1 2 2v2"/>'
    ),
    "database": (
        '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/>'
    ),
    "bar_chart_2": (
        '<line x1="18" x2="18" y1="20" y2="10"/>'
        '<line x1="12" x2="12" y1="20" y2="4"/>'
        '<line x1="6" x2="6" y1="20" y2="14"/>'
    ),
    "play": ('<polygon points="6 3 20 12 6 21 6 3"/>'),
    "clipboard_list": (
        '<rect width="8" height="4" x="8" y="2" rx="1" ry="1"/>'
        '<path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/>'
        '<path d="M12 11h4"/><path d="M12 16h4"/><path d="M8 11h.01"/><path d="M8 16h.01"/>'
    ),
    "target": ('<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>'),
    "repeat": (
        '<path d="m17 2 4 4-4 4"/>'
        '<path d="M3 11v-1a4 4 0 0 1 4-4h14"/>'
        '<path d="m7 22-4-4 4-4"/>'
        '<path d="M21 13v1a4 4 0 0 1-4 4H3"/>'
    ),
    "link": (
        '<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/>'
        '<path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>'
    ),
    "chevron_left": ('<path d="m15 18-6-6 6-6"/>'),
    "chevron_right": ('<path d="m9 18 6-6-6-6"/>'),
    "calendar": (
        '<rect width="18" height="18" x="3" y="4" rx="2" ry="2"/>'
        '<line x1="16" x2="16" y1="2" y2="6"/>'
        '<line x1="8" x2="8" y1="2" y2="6"/>'
        '<line x1="3" x2="21" y1="10" y2="10"/>'
    ),
    "alert_triangle": (
        '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>'
        '<path d="M12 9v4"/>'
        '<path d="M12 17h.01"/>'
    ),
    "pencil": ('<path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/><path d="m15 5 4 4"/>'),
    "x": ('<path d="M18 6 6 18"/><path d="m6 6 12 12"/>'),
    "app_window": (
        '<rect x="2" y="4" width="20" height="16" rx="2" /><path d="M10 4v4" /><path d="M2 8h20" /><path d="M6 4v4" />'
    ),
    "file_text": (
        '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z" />'
        '<path d="M14 2v4a2 2 0 0 0 2 2h4" />'
        '<path d="M10 9H8" />'
        '<path d="M16 13H8" />'
        '<path d="M16 17H8" />'
    ),
    "rocket": (
        '<path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"/>'
        '<path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z"/>'
        '<path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5-4 5-4"/>'
        '<path d="M12 15v5s3.03-.55 4-2c1.08-1.62 4-5 4-5"/>'
    ),
    "zoom_in": (
        '<circle cx="11" cy="11" r="8"/>'
        '<line x1="21" x2="16.65" y1="21" y2="16.65"/>'
        '<line x1="11" x2="11" y1="8" y2="14"/>'
        '<line x1="8" x2="14" y1="11" y2="11"/>'
    ),
    "square": ('<rect width="18" height="18" x="3" y="3" rx="2" ry="2"/>'),
}

# Public alias — human-readable names used in actions_section
ICONS = _SVG_DEFS  # same dict, just re-exported

# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

# Cache: (icon_key, size_tuple, color_str) -> CTkImage
_cache: dict[tuple, ctk.CTkImage] = {}


def _build_svg(inner: str, size: Tuple[int, int], color: str) -> str:
    """Wrap inner SVG paths in a complete <svg> element."""
    w, h = size
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" '
        f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        f"{inner}</svg>"
    )


def _svg_to_pil(svg_str: str, size: Tuple[int, int]) -> Image.Image:
    """Render an SVG string to a PIL Image using cairosvg."""
    try:
        import cairosvg  # type: ignore

        png_bytes = cairosvg.svg2png(
            bytestring=svg_str.encode("utf-8"),
            output_width=size[0],
            output_height=size[1],
        )
        return Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    except ImportError as exc:
        raise ImportError("cairosvg is required for SVG icon rendering. Install it with: pip install cairosvg") from exc


def get_icon(
    key: str,
    size: Tuple[int, int] = (18, 18),
    light_color: str = "#1a1a2e",
    dark_color: str = "#e0e0e0",
) -> ctk.CTkImage | None:
    """Return a ``CTkImage`` for the named icon, with disk-level caching.

    Parameters
    ----------
    key:
        Icon identifier matching a key in :data:`ICONS` / ``_SVG_DEFS``.
    size:
        Pixel dimensions (width, height) for the rendered image.
    light_color:
        Stroke colour used when CustomTkinter is in *light* mode.
    dark_color:
        Stroke colour used when CustomTkinter is in *dark* mode.

    Returns
    -------
    ``ctk.CTkImage`` or ``None`` if the key is unknown or rendering fails.
    """
    cache_key = (key, size, light_color, dark_color)
    if cache_key in _cache:
        return _cache[cache_key]

    inner = _SVG_DEFS.get(key)
    if inner is None:
        return None

    try:
        light_img = _svg_to_pil(_build_svg(inner, size, light_color), size)
        dark_img = _svg_to_pil(_build_svg(inner, size, dark_color), size)
        icon = ctk.CTkImage(light_image=light_img, dark_image=dark_img, size=size)
        _cache[cache_key] = icon
        return icon
    except Exception as exc:  # noqa: BLE001
        # Graceful degradation — fall back to text-only button
        import logging

        logging.getLogger(__name__).warning("svg_icons: failed to render '%s': %s", key, exc)
        return None


def clear_cache() -> None:
    """Clear the in-memory icon cache (useful in tests)."""
    _cache.clear()

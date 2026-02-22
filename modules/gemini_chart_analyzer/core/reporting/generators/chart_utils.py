"""
Chart path utilities for report generation.

This module provides functions for handling chart paths, including
finding chart files, embedding images, and path sanitization.
"""

import base64
import os
from pathlib import Path
from typing import Dict, List, Optional

from modules.common.ui.logging import log_warn


def embed_chart_as_base64(chart_path: str) -> Optional[str]:
    """
    Embed chart image as base64 data URI.

    Args:
        chart_path: Path to the chart image file

    Returns:
        Base64 data URI string or None if embedding fails
    """
    try:
        if chart_path and os.path.exists(chart_path):
            with open(chart_path, "rb") as f:
                image_data = f.read()
                image_base64 = base64.b64encode(image_data).decode("utf-8")

                # Detect image format
                if chart_path.lower().endswith(".png"):
                    image_mime = "image/png"
                elif chart_path.lower().endswith((".jpg", ".jpeg")):
                    image_mime = "image/jpeg"
                else:
                    image_mime = "image/png"

                return f"data:{image_mime};base64,{image_base64}"
        else:
            log_warn(f"Chart path not found: {chart_path}")
            return None
    except Exception as e:
        log_warn(f"Cannot embed image: {e}, using placeholder")
        return None


def sanitize_chart_path(chart_path: str, output_dir: str) -> str:
    """
    Convert absolute chart path to a relative path for HTML use.

    Args:
        chart_path: Absolute path to chart file
        output_dir: Output directory for the HTML report

    Returns:
        Relative path suitable for HTML href/src attributes
    """
    try:
        rel_path = os.path.relpath(chart_path, output_dir)
        return rel_path.replace("\\", "/")
    except Exception as e:
        log_warn(f"[Chart Utils] Failed to get relative path for {chart_path}: {e}")
        return Path(chart_path).name


def find_chart_paths_for_timeframes(symbol: str, timeframes: List[str], charts_dir: str) -> Dict[str, str]:
    """
    Find chart image files for each timeframe of a symbol.

    Args:
        symbol: Trading symbol (e.g., "BTC/USDT")
        timeframes: List of timeframes to search for
        charts_dir: Directory containing chart images

    Returns:
        Dictionary mapping timeframe to chart path
    """
    results = {}
    if not os.path.exists(charts_dir):
        return results

    safe_symbol = symbol.replace("/", "_").replace(":", "_")
    all_files = os.listdir(charts_dir)

    for tf in timeframes:
        # Match pattern: {safe_symbol}_{tf}_{timestamp}.png
        matches = [f for f in all_files if f.startswith(f"{safe_symbol}_{tf}_") and f.endswith(".png")]
        if matches:
            # Get latest match (sorted by filename, which includes timestamp)
            latest = sorted(matches, reverse=True)[0]
            results[tf] = os.path.join(charts_dir, latest)

    return results


def sanitize_symbol_for_filename(symbol: str) -> str:
    """
    Sanitize symbol for use in filenames.

    Args:
        symbol: Trading symbol (e.g., "BTC/USDT")

    Returns:
        Filename-safe symbol string
    """
    return symbol.replace("/", "_").replace(":", "_")

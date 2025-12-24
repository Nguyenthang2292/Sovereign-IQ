"""
Chart path utilities for gemini chart analyzer.

Provides functions for determining chart directory paths.
"""

from pathlib import Path


def get_charts_dir() -> Path:
    """
    Get the charts directory path relative to module root.
    
    Creates the charts directory if it does not exist.
    
    Returns:
        Path object pointing to the charts directory
    """
    module_root = Path(__file__).resolve().parent.parent.parent
    charts_dir = module_root / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    return charts_dir


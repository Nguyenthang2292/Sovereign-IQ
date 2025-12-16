"""
Utility functions organized by domain.

This module provides backward compatibility by re-exporting all functions
from submodules, allowing existing imports like:
    from modules.common.utils import normalize_symbol, log_info, ...

to continue working.
"""

# System utilities
from .system import configure_windows_stdio

# Data utilities
from .data import dataframe_to_close_series, validate_ohlcv_input

# Domain utilities
from .domain import normalize_symbol, normalize_symbol_key, timeframe_to_minutes

# Component initialization
from .initialization import initialize_components

# Re-export UI utilities for backward compatibility
from ..ui.formatting import color_text, format_price, prompt_user_input, extract_dict_from_namespace
from ..ui.logging import (
    log_info,
    log_success,
    log_error,
    log_warn,
    log_debug,
    log_data,
    log_analysis,
    log_model,
    log_exchange,
    log_system,
    log_progress,
)

__all__ = [
    # System
    "configure_windows_stdio",
    # Data
    "dataframe_to_close_series",
    "validate_ohlcv_input",
    # Domain
    "normalize_symbol",
    "normalize_symbol_key",
    "timeframe_to_minutes",
    # UI/Formatting
    "color_text",
    "format_price",
    "prompt_user_input",
    "extract_dict_from_namespace",
    # UI/Logging
    "log_info",
    "log_success",
    "log_error",
    "log_warn",
    "log_debug",
    "log_data",
    "log_analysis",
    "log_model",
    "log_exchange",
    "log_system",
    "log_progress",
    # Initialization
    "initialize_components",
]

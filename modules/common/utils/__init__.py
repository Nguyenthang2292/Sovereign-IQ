"""
Utility functions organized by domain.

This module provides backward compatibility by re-exporting all functions
from submodules, allowing existing imports like:
    from modules.common.utils import normalize_symbol, log_info, ...

to continue working.
"""

# System utilities
from .system import (
    configure_windows_stdio,
    detect_gpu_availability,
    detect_pytorch_cuda_availability,
    detect_pytorch_gpu_availability,
    configure_gpu_memory,
    PyTorchGPUManager,
)

# Data utilities
from .data import dataframe_to_close_series, validate_ohlcv_input, fetch_ohlcv_data_dict

# Domain utilities
from .domain import normalize_symbol, normalize_symbol_key, normalize_timeframe, timeframe_to_minutes, days_to_candles

# Component initialization
from .initialization import initialize_components


_UNSET = object()


def safe_input(prompt: str, default: str | object = _UNSET) -> str:
    """
    Safely read input from stdin with Windows compatibility.
    
    Handles "I/O operation on closed file" errors on Windows by attempting
    to reopen stdin if it's closed. This should be used for all user input
    instead of calling input() directly.
    
    Args:
        prompt: Prompt message to display
        default: Default value to return if stdin is unavailable (optional)
    
    Returns:
        User input string, or default value if stdin unavailable or empty input
        
    Raises:
        OSError, IOError, EOFError, AttributeError, ValueError: 
            If stdin errors occur and no default value is provided
            
    Example:
        >>> name = safe_input("Enter your name: ", default="Anonymous")
        >>> confirm = safe_input("Continue? (y/n): ")
    """
    try:
        result = input(prompt)
        result = result.strip()
        
        if not result:
            if default is not _UNSET:
                return default
            return ""
        
        return result
    except (OSError, IOError, EOFError, AttributeError, ValueError):
        if default is not _UNSET:
            return default
        raise

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
    "detect_gpu_availability",
    "detect_pytorch_cuda_availability",
    "detect_pytorch_gpu_availability",
    "configure_gpu_memory",
    "PyTorchGPUManager",
    # Data
    "dataframe_to_close_series",
    "validate_ohlcv_input",
    "fetch_ohlcv_data_dict",
    # Domain
    "normalize_symbol",
    "normalize_symbol_key",
    "normalize_timeframe",
    "timeframe_to_minutes",
    "days_to_candles",
    # UI/Formatting
    "color_text",
    "format_price",
    "safe_input",
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

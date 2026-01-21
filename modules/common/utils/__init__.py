"""
Utility functions organized by domain.

This module provides backward compatibility by re-exporting all functions
from new organized packages, allowing existing imports like:
    from modules.common.utils import normalize_symbol, log_info, ...
to continue working.

NOTE: New code should import directly from organized packages:
    - from modules.common.domain import normalize_symbol
    - from modules.common.data import validate_ohlcv_input
    - from modules.common.io import cleanup_old_files
"""

from typing import Any, Iterable

# Re-export from domain package (trading domain utilities)
from modules.common.domain import (
    days_to_candles,
    normalize_symbol,
    normalize_symbol_key,
    normalize_timeframe,
    timeframe_to_minutes,
)

# Re-export from data package (DataFrame/Series utilities)
from modules.common.data import (
    dataframe_to_close_series,
    fetch_ohlcv_data_dict,
    validate_ohlcv_input,
)

# Re-export from io package (file operations)
from modules.common.io import cleanup_old_files

# System and hardware management (re-exported from system module)
from modules.common.system import (
    HardwareManager,
    HardwareResources,
    WorkloadConfig,
    PyTorchGPUManager,
    configure_gpu_memory,
    configure_windows_stdio,
    detect_gpu_availability,
    detect_pytorch_cuda_availability,
    detect_pytorch_gpu_availability,
    get_hardware_manager,
    reset_hardware_manager,
)


# Component initialization (re-exported from core, lazy import to avoid circular dependency)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Tuple
    from modules.common.core.data_fetcher import DataFetcher
    from modules.common.core.exchange_manager import ExchangeManager


def initialize_components() -> "Tuple[ExchangeManager, DataFetcher]":
    """
    Initialize ExchangeManager and DataFetcher components.
    
    This is a lazy import wrapper to avoid circular dependencies.
    For direct access, import from modules.common.core.initialization instead.
    
    Returns:
        Tuple containing (ExchangeManager, DataFetcher) instances
    """
    from modules.common.core.initialization import initialize_components as _initialize_components
    return _initialize_components()
# System imports removed - now re-exported from modules.common.system above
from .system_utils import (
    get_error_code,
    is_retryable_error,
    setup_windows_stdin,
)

_UNSET = object()


def safe_input(prompt: str, default: Any = _UNSET) -> str:
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
                return str(default)
            return ""

        return result
    except (OSError, IOError, EOFError, AttributeError, ValueError):
        if default is not _UNSET:
            return str(default)
        raise


# Re-export UI utilities for backward compatibility
from ..ui.formatting import color_text, extract_dict_from_namespace, format_price, prompt_user_input
from ..ui.logging import (
    log_analysis,
    log_data,
    log_debug,
    log_error,
    log_exchange,
    log_info,
    log_model,
    log_progress,
    log_success,
    log_system,
    log_warn,
)

__all__ = [
    # Hardware Management
    "HardwareManager",
    "HardwareResources",
    "WorkloadConfig",
    "get_hardware_manager",
    "reset_hardware_manager",
    # System
    "configure_windows_stdio",
    "detect_gpu_availability",
    "detect_pytorch_cuda_availability",
    "detect_pytorch_gpu_availability",
    "configure_gpu_memory",
    "PyTorchGPUManager",
    # System utilities
    "setup_windows_stdin",
    "get_error_code",
    "is_retryable_error",
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
    # File
    "cleanup_old_files",
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

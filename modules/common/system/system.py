"""
System utilities for platform-specific configuration.
"""

import io
import os
import sys
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple

# Import from managers and detection layers
from modules.common.system.detection import GPUDetector
from modules.common.system.managers.pytorch_gpu_manager import PyTorchGPUManager
from modules.common.ui.logging import log_info, log_warn


# ============================================================================
# XGBoost GPU Detection
# ============================================================================
def detect_gpu_availability(use_gpu: bool = True) -> bool:
    """
    Detect if GPU is available for XGBoost.

    Uses unified GPUDetector from detection layer.

    Args:
        use_gpu: Whether to check for GPU (if False, returns False immediately)

    Returns:
        True if GPU is available and use_gpu is True, False otherwise
    """
    if not use_gpu:
        return False
    return GPUDetector.detect_xgboost()


# ============================================================================
# PyTorch GPU Manager
# ============================================================================

# ============================================================================
# PyTorch GPU Convenience Functions
# ============================================================================


# Global singleton instance for convenience
# Note: This is kept for backward compatibility with tests
_pytorch_gpu_manager = PyTorchGPUManager()


def detect_pytorch_cuda_availability() -> Tuple[bool, Optional[str], int]:
    """
    Detect if CUDA is available for PyTorch, verify it's functional, and provide diagnostic info.

    This is a convenience wrapper that uses PyTorchGPUManager internally.
    For better performance and caching, use PyTorchGPUManager directly.

    Returns:
        Tuple (is_available, cuda_version, device_count):
            - is_available: True if CUDA is available and functional, otherwise False
            - cuda_version: CUDA version string or None
            - device_count: Number of CUDA devices (0 if not available)
    """
    return _pytorch_gpu_manager.detect_cuda_availability()


def detect_pytorch_gpu_availability() -> bool:
    """
    Detect if GPU is available for PyTorch.

    This is a convenience wrapper that detects PyTorch CUDA availability.
    Uses the global PyTorchGPUManager instance for caching.

    Returns:
        True if GPU is available and functional, False otherwise
    """
    return _pytorch_gpu_manager.is_available()


def configure_gpu_memory() -> bool:
    """
    Configure GPU memory settings for PyTorch.

    Sets up memory management options like memory fraction,
    empty cache, and other optimizations for better GPU utilization.

    Note: This function assumes GPU is available. Use detect_pytorch_gpu_availability()
    first to verify GPU availability before calling this function.

    Returns:
        True if GPU memory was configured successfully, False otherwise
    """
    return _pytorch_gpu_manager.configure_memory()


# ============================================================================
# PyTorch Environment Configuration
# ============================================================================


def _check_kmp_duplicate_lib_warning(env: dict) -> None:
    """
    Check if KMP_DUPLICATE_LIB_OK is enabled and log a warning with guidance.

    This function helps raise awareness about potential OpenMP library conflicts
    when the KMP_DUPLICATE_LIB_OK workaround is used.

    Args:
        env: Dictionary of environment variables being set
    """
    if env.get("KMP_DUPLICATE_LIB_OK") == "True":
        log_warn(
            "KMP_DUPLICATE_LIB_OK is enabled. This suppresses warnings about "
            "duplicate OpenMP libraries but may mask dependency conflicts."
        )
        log_warn(
            "Runtime monitoring is active. Watch for: "
            "1) Performance degradation (unusually slow operations), "
            "2) Unexpected crashes during multithreaded operations, "
            "3) Memory issues or resource leaks."
        )
        log_warn(
            "If issues occur, consider: "
            "1) Using conda environments with conda-forge packages, "
            "2) Ensuring compatible OpenMP versions across all packages, "
            "3) Managing library paths explicitly."
        )


def get_pytorch_env() -> dict:
    """
    Get PyTorch environment variables for the current mode.

    Returns production-safe environment by default. Merges debug environment
    variables only when DEBUG or DEV environment variable is set to 'true' or '1'.

    These environment variables should be set before importing PyTorch to ensure
    proper behavior. Use os.environ.update(get_pytorch_env()) before importing torch.

    Returns:
        dict: Environment variables dictionary to apply via os.environ.update()
    """
    from config.deep_learning import PYTORCH_DEBUG_ENV, PYTORCH_ENV

    env = PYTORCH_ENV.copy()

    # Check if debug mode is enabled via environment variable
    debug_flag = os.environ.get("DEBUG", "").lower() in ("true", "1", "yes")
    dev_flag = os.environ.get("DEV", "").lower() in ("true", "1", "yes")

    if debug_flag or dev_flag:
        env.update(PYTORCH_DEBUG_ENV)

    # Warn about KMP_DUPLICATE_LIB_OK in debug/dev mode to raise awareness
    if debug_flag or dev_flag:
        _check_kmp_duplicate_lib_warning(env)

    return env


# ============================================================================
# Runtime Monitoring for KMP_DUPLICATE_LIB_OK
# ============================================================================
# RuntimeMonitor moved to utils/monitoring.py
from modules.common.system.utils.monitoring import RuntimeMonitor, get_runtime_monitor


# RuntimeMonitor and get_runtime_monitor are imported from utils/monitoring.py above


# ============================================================================
# Windows Platform Configuration
# ============================================================================


def configure_windows_stdio() -> None:
    """
    Configure Windows stdio encoding for UTF-8 support.

    Only applies to interactive CLI runs, not during pytest.
    This function fixes encoding issues on Windows by reconfiguring
    stdout and stderr with UTF-8 encoding.

    Note:
        - Only runs on Windows (win32 platform)
        - Skips configuration during pytest runs
        - Uses reconfigure() if available for safety
    """
    if sys.platform != "win32":
        return
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return

    # Use reconfigure if available (Python 3.7+)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, io.UnsupportedOperation):
            pass
        return

    # Fallback for older versions or non-standard streams
    if not hasattr(sys.stdout, "buffer") or isinstance(sys.stdout, io.TextIOWrapper):
        return
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

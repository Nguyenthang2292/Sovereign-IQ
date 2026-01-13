"""
System utilities for platform-specific configuration.
"""

import io
import os
import sys
import time
import warnings
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple

from modules.common.ui.logging import log_info, log_system, log_warn


# ============================================================================
# XGBoost GPU Detection
# ============================================================================
def detect_gpu_availability(use_gpu: bool = True) -> bool:
    """
    Detect if GPU is available for XGBoost.

    Args:
        use_gpu: Whether to check for GPU (if False, returns False immediately)

    Returns:
        True if GPU is available and use_gpu is True, False otherwise
    """
    if not use_gpu:
        return False
    try:
        # import xgboost as xgb

        # Check if GPU is available by trying to get device info
        # This is a lightweight check that doesn't create a model
        try:
            # XGBoost 2.0+ has device parameter support
            # We'll try to detect GPU availability at runtime instead
            return True  # Will be checked more thoroughly at runtime
        except Exception:
            return False
    except ImportError:
        return False


# ============================================================================
# PyTorch GPU Manager
# ============================================================================


class PyTorchGPUManager:
    """
    Manager class for PyTorch GPU operations.

    This class provides a centralized way to manage GPU detection,
    configuration, and state for PyTorch operations. It caches torch module
    and GPU availability results to avoid repeated checks.

    Example:
        manager = PyTorchGPUManager()
        if manager.is_available():
            manager.configure_memory()
            device = manager.get_device()
    """

    def __init__(self):
        """Initialize the PyTorch GPU Manager."""
        self._torch = None
        self._gpu_available = None
        self._cuda_version = None
        self._device_count = 0
        self._device = None

    @staticmethod
    def _get_torch_module():
        """
        Helper method to import and return torch module.

        Returns:
            torch module if available, None otherwise
        """
        try:
            import torch

            return torch
        except ImportError:
            return None

    @staticmethod
    def _return_cuda_unavailable(reason: str) -> Tuple[bool, Optional[str], int]:
        """
        Helper method to return CUDA unavailable result with logging.

        Args:
            reason: Reason why CUDA is unavailable (for logging)

        Returns:
            Tuple (False, None, 0) indicating CUDA is not available
        """
        log_info(f"{reason} Using CPU mode.")
        return False, None, 0

    @property
    def torch(self):
        """Get or import torch module (cached)."""
        if self._torch is None:
            self._torch = self._get_torch_module()
        return self._torch

    def _detect_cuda_availability(self, torch_module) -> Tuple[bool, Optional[str], int]:
        """
        Detect if CUDA is available for PyTorch, verify it's functional, and provide diagnostic info.

        This is a private method that performs the actual CUDA detection logic.

        Args:
            torch_module: The imported torch module

        Returns:
            Tuple (is_available, cuda_version, device_count):
                - is_available: True if CUDA is available and functional, otherwise False
                - cuda_version: CUDA version string or None
                - device_count: Number of CUDA devices (0 if not available)
        """
        # Defensive: torch_module must have cuda property
        if not hasattr(torch_module, "cuda"):
            return self._return_cuda_unavailable("PyTorch does not have a 'cuda' attribute.")

        # Check availability flag
        if not callable(getattr(torch_module.cuda, "is_available", None)):
            return self._return_cuda_unavailable("torch.cuda.is_available not callable.")

        if not torch_module.cuda.is_available():
            return self._return_cuda_unavailable("CUDA not available,")

        # Try to actually allocate a tensor on the GPU
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                device_count = torch_module.cuda.device_count()
                if device_count == 0:
                    return self._return_cuda_unavailable("torch.cuda reports available but no GPU devices detected.")

                # Try all devices and track which ones are functional
                functional_devices = []
                for idx in range(device_count):
                    try:
                        torch_module.ones(1, device=f"cuda:{idx}")
                        functional_devices.append(idx)
                        log_system(f"CUDA device {idx} is functional")
                    except Exception as device_error:
                        log_warn(f"CUDA device {idx} not functional: {device_error}")
                        continue

                # If no devices were confirmed functional, CUDA is not usable
                if not functional_devices:
                    log_warn("No functional CUDA devices found. Using CPU mode.")
                    return False, None, 0

            # Get CUDA version
            cuda_version = None
            if hasattr(torch_module, "version") and hasattr(torch_module.version, "cuda"):
                cuda_version = torch_module.version.cuda
            elif hasattr(torch_module, "cuda") and hasattr(torch_module.cuda, "version"):
                # Fallback for rare cases
                cuda_version = getattr(torch_module.cuda, "version", None)
            if not cuda_version:
                cuda_version = "Unknown"

            functional_count = len(functional_devices)
            if functional_count > 0:
                log_system(
                    f"CUDA {cuda_version} available with {functional_count} functional device(s) "
                    f"out of {device_count} total (devices: {functional_devices})."
                )
            else:
                log_warn(
                    f"CUDA {cuda_version} reports {device_count} device(s) but none are functional. Using CPU mode."
                )
                return False, None, 0

            return True, cuda_version, device_count

        except Exception as cuda_error:
            log_warn(f"CUDA appears available but is not functional: {repr(cuda_error)}")
            log_warn("Falling back to CPU mode.")
            # Prevent future CUDA usage in this process
            # Do not monkey-patch torch.cuda.is_available, as this can cause
            # unintended side effects and break expectations for other code or libraries.
            # Instead, rely on returning False here and ensure that higher-level
            # application logic (such as the PyTorchGPUManager state) is used to
            # control CUDA usage in the rest of the application.
            return False, None, 0

    def detect_cuda_availability(self) -> Tuple[bool, Optional[str], int]:
        """
        Detect if CUDA is available for PyTorch and cache results.

        Returns:
            Tuple (is_available, cuda_version, device_count):
                - is_available: True if CUDA is available and functional
                - cuda_version: CUDA version string or None
                - device_count: Number of CUDA devices (0 if not available)
        """
        if self._gpu_available is not None:
            return self._gpu_available, self._cuda_version, self._device_count

        torch = self.torch
        if torch is None:
            self._gpu_available = False
            self._cuda_version = None
            self._device_count = 0
            return False, None, 0

        is_available, cuda_version, device_count = self._detect_cuda_availability(torch)
        self._gpu_available = is_available
        self._cuda_version = cuda_version
        self._device_count = device_count

        return is_available, cuda_version, device_count

    def is_available(self) -> bool:
        """
        Check if GPU is available for PyTorch (cached).

        Returns:
            True if GPU is available and functional, False otherwise
        """
        if self._gpu_available is None:
            self.detect_cuda_availability()
        return self._gpu_available or False

    def configure_memory(self) -> bool:
        """
        Configure GPU memory settings for PyTorch.

        Sets up memory management options like memory fraction,
        empty cache, and other optimizations for better GPU utilization.

        Returns:
            True if GPU memory was configured successfully, False otherwise
        """
        torch = self.torch
        if torch is None:
            log_info("PyTorch not installed, GPU memory configuration unavailable.")
            return False

        if not self.is_available():
            return False

        try:
            # Clear any existing cache
            torch.cuda.empty_cache()

            # Set memory fraction if needed (optional, can be configured)
            # torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory

            # Enable memory efficient attention if available (PyTorch 2.0+)
            backends_cuda = getattr(torch, "backends", None)
            if backends_cuda and hasattr(backends_cuda, "cuda"):
                cuda_backend = backends_cuda.cuda
                if hasattr(cuda_backend, "enable_flash_sdp"):
                    cuda_backend.enable_flash_sdp(True)
                if hasattr(cuda_backend, "enable_mem_efficient_sdp"):
                    cuda_backend.enable_mem_efficient_sdp(True)

            log_system("GPU memory configured successfully")
            return True
        except Exception as e:
            log_warn(f"Failed to configure GPU memory: {e}")
            return False

    def get_device(self, device_id: int = 0):
        """
        Get PyTorch device object for GPU or CPU.

        Args:
            device_id: GPU device ID (default: 0)

        Returns:
            torch.device object ('cuda:device_id' if available, 'cpu' otherwise)
        """
        if self.is_available() and self._device_count > 0:
            return self.torch.device(f"cuda:{device_id}")
        return self.torch.device("cpu") if self.torch else None

    def get_info(self) -> dict:
        """
        Get comprehensive GPU information.

        Returns:
            Dictionary with GPU information:
            - available: bool
            - cuda_version: str or None
            - device_count: int
            - device: torch.device or None
        """
        self.detect_cuda_availability()
        return {
            "available": self._gpu_available or False,
            "cuda_version": self._cuda_version,
            "device_count": self._device_count,
            "device": self.get_device(),
        }


# ============================================================================
# PyTorch GPU Convenience Functions
# ============================================================================


# Global singleton instance for convenience
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


class RuntimeMonitor:
    """
    Monitor runtime issues that may be related to KMP_DUPLICATE_LIB_OK.

    This class tracks performance, memory usage, and exceptions to help
    identify potential issues caused by OpenMP library conflicts.
    """

    def __init__(self):
        self.operation_times: Dict[str, list] = {}
        self.exception_count = 0
        self.memory_snapshots: list = []
        self.is_monitoring = os.environ.get("KMP_DUPLICATE_LIB_OK") == "True"

    def monitor_operation(self, operation_name: str):
        """
        Decorator to monitor operation execution time and exceptions.

        Args:
            operation_name: Name of the operation being monitored
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.is_monitoring:
                    return func(*args, **kwargs)

                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time

                    # Track execution time
                    if operation_name not in self.operation_times:
                        self.operation_times[operation_name] = []
                    self.operation_times[operation_name].append(execution_time)

                    # Warn if operation takes unusually long (> 10 seconds)
                    if execution_time > 10.0:
                        log_warn(
                            f"Slow operation detected: {operation_name} took {execution_time:.2f}s. "
                            "This may indicate performance issues related to OpenMP conflicts."
                        )

                    return result
                except Exception as e:
                    self.exception_count += 1
                    execution_time = time.time() - start_time

                    # Log exception details
                    log_warn(
                        f"Exception during {operation_name} (after {execution_time:.2f}s): {type(e).__name__}: {str(e)}"
                    )

                    # If multiple exceptions occur, suggest checking OpenMP conflicts
                    if self.exception_count >= 3:
                        log_warn(
                            f"Multiple exceptions detected ({self.exception_count}). "
                            "This may indicate OpenMP library conflicts. "
                            "Consider checking KMP_DUPLICATE_LIB_OK configuration."
                        )

                    raise

            return wrapper

        return decorator

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get summary of monitored performance metrics.

        Returns:
            Dictionary with performance statistics
        """
        summary = {
            "total_exceptions": self.exception_count,
            "monitored_operations": len(self.operation_times),
            "operation_stats": {},
        }

        for op_name, times in self.operation_times.items():
            if times:
                summary["operation_stats"][op_name] = {
                    "count": len(times),
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "total_time": sum(times),
                }

        return summary

    def log_summary(self):
        """Log a summary of monitored metrics."""
        if not self.is_monitoring:
            return

        summary = self.get_performance_summary()

        if summary["total_exceptions"] > 0:
            log_warn(f"Runtime monitoring detected {summary['total_exceptions']} exceptions. Review logs for details.")

        if summary["operation_stats"]:
            log_info("Runtime performance summary:")
            for op_name, stats in summary["operation_stats"].items():
                log_info(
                    f"  {op_name}: {stats['count']} operations, "
                    f"avg {stats['avg_time']:.2f}s, "
                    f"max {stats['max_time']:.2f}s"
                )


# Global runtime monitor instance
_runtime_monitor = RuntimeMonitor()


def get_runtime_monitor() -> RuntimeMonitor:
    """
    Get the global runtime monitor instance.

    Returns:
        RuntimeMonitor instance for tracking runtime issues
    """
    return _runtime_monitor


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

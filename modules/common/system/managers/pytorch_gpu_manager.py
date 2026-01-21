"""
PyTorch GPU Manager

Manager class for PyTorch GPU operations.

This module provides a centralized way to manage GPU detection,
configuration, and state for PyTorch operations. It caches torch module
and GPU availability results to avoid repeated checks.

Uses GPUDetector from detection layer for unified GPU detection.

Author: Crypto Probability Team
"""

import warnings
from typing import Optional, Tuple

from modules.common.system.detection import GPUDetector
from modules.common.ui.logging import log_info, log_system, log_warn


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

    def detect_cuda_availability(self) -> Tuple[bool, Optional[str], int]:
        """
        Detect if CUDA is available for PyTorch and cache results.

        Uses GPUDetector from detection layer for unified detection.

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

        # Direct call to GPUDetector (no wrapper indirection)
        is_available, cuda_version, device_count = GPUDetector._detect_pytorch_cuda(torch)
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

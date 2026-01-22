"""
Unified GPU detection for all backends.

Single source of truth for GPU detection across:
- PyTorch (CUDA)
- XGBoost (nvidia-smi)
- CuPy (CUDA)
- Numba (CUDA)
- OpenCL
"""

import subprocess
import warnings
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

from modules.common.ui.logging import log_debug, log_info, log_system, log_warn

# Type alias for GPU types
GpuType = Literal["cuda", "opencl", "pytorch", "xgboost"]

# Check library availability
try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    from numba import cuda

    NUMBA_CUDA_AVAILABLE = cuda.is_available()
except ImportError:
    NUMBA_CUDA_AVAILABLE = False

try:
    import pyopencl as cl

    PYOPENCL_AVAILABLE = True
except ImportError:
    PYOPENCL_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


@dataclass
class GPUInfo:
    """GPU information."""

    available: bool
    gpu_type: Optional[GpuType] = None
    gpu_count: int = 0
    gpu_memory_gb: Optional[float] = None
    pytorch_available: bool = False
    pytorch_cuda_version: Optional[str] = None
    pytorch_device_count: int = 0
    xgboost_available: bool = False


class GPUDetector:
    """
    Unified GPU detection across all backends.

    Single source of truth for GPU detection with priority order:
    1. PyTorch (most reliable)
    2. CuPy
    3. Numba
    4. OpenCL
    5. XGBoost (nvidia-smi)
    """

    @staticmethod
    def _detect_pytorch_cuda(torch_module) -> Tuple[bool, Optional[str], int]:
        """
        Detect PyTorch CUDA availability.

        Args:
            torch_module: The imported torch module

        Returns:
            Tuple (is_available, cuda_version, device_count)
        """
        # Defensive: torch_module must have cuda property
        if not hasattr(torch_module, "cuda"):
            return False, None, 0

        # Check availability flag
        if not callable(getattr(torch_module.cuda, "is_available", None)):
            return False, None, 0

        if not torch_module.cuda.is_available():
            return False, None, 0

        # Try to actually allocate a tensor on the GPU
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                device_count = torch_module.cuda.device_count()
                if device_count == 0:
                    return False, None, 0

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
            return False, None, 0

    @staticmethod
    def detect_pytorch() -> Tuple[bool, Optional[str], int]:
        """
        Detect PyTorch CUDA availability.

        Returns:
            Tuple (is_available, cuda_version, device_count)
        """
        if not TORCH_AVAILABLE:
            return False, None, 0

        try:
            return GPUDetector._detect_pytorch_cuda(torch)
        except ImportError:
            log_debug("PyTorch not installed")
            return False, None, 0
        except Exception as e:
            log_warn(f"PyTorch GPU detection error: {e}")
            return False, None, 0

    @staticmethod
    def detect_cupy() -> Tuple[bool, int, Optional[float]]:
        """
        Detect CuPy CUDA availability.

        Returns:
            Tuple (is_available, device_count, memory_gb)
        """
        if not CUPY_AVAILABLE:
            return False, 0, None

        try:
            cupy_device_count = cp.cuda.runtime.getDeviceCount()
            if cupy_device_count > 0:
                # Get memory of first GPU
                gpu_memory_gb = cp.cuda.Device(0).mem_info[1] / (1024**3)
                log_info(f"CuPy CUDA GPU detected: {cupy_device_count} device(s), {gpu_memory_gb:.2f} GB")
                return True, cupy_device_count, gpu_memory_gb
            return False, 0, None
        except ImportError:
            log_debug("CuPy not installed")
            return False, 0, None
        except Exception as e:
            log_warn(f"CuPy CUDA detection error: {e}")
            return False, 0, None

    @staticmethod
    def detect_numba() -> Tuple[bool, int]:
        """
        Detect Numba CUDA availability.

        Returns:
            Tuple (is_available, device_count)
        """
        if not NUMBA_CUDA_AVAILABLE:
            return False, 0

        try:
            numba_gpu_count = len(cuda.gpus)
            if numba_gpu_count > 0:
                log_info(f"CUDA GPU detected via Numba: {numba_gpu_count} device(s)")
                return True, numba_gpu_count
            return False, 0
        except ImportError:
            log_debug("Numba CUDA not available")
            return False, 0
        except Exception as e:
            log_warn(f"Numba CUDA detection error: {e}")
            return False, 0

    @staticmethod
    def detect_opencl() -> Tuple[bool, int]:
        """
        Detect OpenCL GPU availability.

        Returns:
            Tuple (is_available, device_count)
        """
        if not PYOPENCL_AVAILABLE:
            return False, 0

        try:
            platforms = cl.get_platforms()
            for platform in platforms:
                devices = platform.get_devices(device_type=cl.device_type.GPU)
                if devices:
                    gpu_count = len(devices)
                    log_info(f"OpenCL GPU detected: {gpu_count} device(s)")
                    return True, gpu_count
            return False, 0
        except ImportError:
            log_debug("PyOpenCL not installed")
            return False, 0
        except Exception as e:
            log_warn(f"OpenCL detection error: {e}")
            return False, 0

    @staticmethod
    def detect_xgboost() -> bool:
        """
        Detect XGBoost GPU availability via nvidia-smi.

        Returns:
            True if GPU is available for XGBoost, False otherwise
        """
        if not XGBOOST_AVAILABLE:
            return False

        try:
            result = subprocess.run(
                ["nvidia-smi", "--list-gpus"],
                capture_output=True,
                timeout=5,
                text=True,
            )
            if result.returncode == 0:
                log_info("XGBoost GPU available (nvidia-smi check passed)")
                log_debug(f"nvidia-smi output: {result.stdout.strip()}")
                return True
            return False
        except FileNotFoundError:
            log_debug("nvidia-smi not found in PATH")
            return False
        except subprocess.TimeoutExpired:
            log_debug("nvidia-smi command timed out")
            return False
        except Exception as e:
            log_warn(f"XGBoost GPU check error: {e}")
            return False

    @staticmethod
    def detect_all() -> GPUInfo:
        """
        Detect GPU across all backends with priority order.

        Priority:
        1. PyTorch (most reliable)
        2. CuPy
        3. Numba
        4. OpenCL
        5. XGBoost (nvidia-smi)

        Returns:
            GPUInfo with comprehensive GPU information
        """
        gpu_available = False
        gpu_type: Optional[GpuType] = None
        gpu_count = 0
        gpu_memory_gb: Optional[float] = None
        pytorch_available = False
        pytorch_cuda_version: Optional[str] = None
        pytorch_device_count = 0
        xgboost_available = False

        # Try PyTorch GPU detection first (most reliable)
        if TORCH_AVAILABLE:
            pytorch_available, pytorch_cuda_version, pytorch_device_count = GPUDetector.detect_pytorch()
            if pytorch_available:
                gpu_available = True
                gpu_type = "pytorch"
                gpu_count = pytorch_device_count
                # Try to get GPU memory from PyTorch
                try:
                    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                except Exception as e:
                    log_debug(f"Could not retrieve GPU memory info: {e}")

        # Try CuPy CUDA if PyTorch not available or as additional check
        if not gpu_available and CUPY_AVAILABLE:
            cupy_available, cupy_count, cupy_memory = GPUDetector.detect_cupy()
            if cupy_available:
                gpu_available = True
                gpu_type = "cuda"
                gpu_count = cupy_count
                gpu_memory_gb = cupy_memory

        # Try Numba CUDA if CuPy not available
        if not gpu_available and NUMBA_CUDA_AVAILABLE:
            numba_available, numba_count = GPUDetector.detect_numba()
            if numba_available:
                gpu_available = True
                gpu_type = "cuda"
                gpu_count = numba_count

        # Try OpenCL if CUDA not available
        if not gpu_available and PYOPENCL_AVAILABLE:
            opencl_available, opencl_count = GPUDetector.detect_opencl()
            if opencl_available:
                gpu_available = True
                gpu_type = "opencl"
                gpu_count = opencl_count

        # Check XGBoost GPU availability (uses nvidia-smi)
        xgboost_available = GPUDetector.detect_xgboost()
        if xgboost_available and not gpu_available:
            # If no other GPU detected, use XGBoost detection
            gpu_available = True
            gpu_type = "xgboost"
            log_info("XGBoost GPU detected via nvidia-smi")

        return GPUInfo(
            available=gpu_available,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            gpu_memory_gb=gpu_memory_gb,
            pytorch_available=pytorch_available,
            pytorch_cuda_version=pytorch_cuda_version,
            pytorch_device_count=pytorch_device_count,
            xgboost_available=xgboost_available,
        )

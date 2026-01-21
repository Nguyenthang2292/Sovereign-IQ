"""
System and hardware management utilities.

This package provides:
- Hardware resource management (CPU, GPU, RAM)
- GPU detection and configuration (PyTorch, XGBoost, CUDA, OpenCL)
- Memory management and monitoring
- System configuration utilities
- Runtime monitoring

Refactored with layered architecture:
- detection/: Hardware detection layer (GPU, CPU, System info)
- managers/: Resource management layer (Hardware, Memory, PyTorch GPU)
- utils/: Shared utilities (Singleton, Monitoring)
"""

# Detection layer
from .detection import CPUDetector, CPUInfo, GPUDetector, GPUInfo, MemoryInfo, SystemInfo

# Managers layer
from .managers.hardware_manager import (
    HardwareManager,
    HardwareResources,
    WorkloadConfig,
    get_hardware_manager,
    reset_hardware_manager,
)
from .managers.memory_manager import (
    MemoryManager,
    MemorySnapshot,
    cleanup_series,
    get_memory_manager,
    reset_memory_manager,
    temp_series,
    track_memory,
)
from .managers.pytorch_gpu_manager import PyTorchGPUManager
from .memory_pool import (
    ArrayPool,
    SeriesPool,
    cleanup_pools,
    get_array_pool,
    get_series_pool,
)

# Shared utilities
from .shared_memory_utils import (
    SHARED_MEMORY_AVAILABLE,
    cleanup_shared_memory,
    reconstruct_dataframe_from_shared_memory,
    setup_shared_memory_for_dataframe,
)
from .system import (
    _check_kmp_duplicate_lib_warning,
    configure_gpu_memory,
    configure_windows_stdio,
    detect_gpu_availability,
    detect_pytorch_cuda_availability,
    detect_pytorch_gpu_availability,
    get_pytorch_env,
)
from .utils.monitoring import RuntimeMonitor, get_runtime_monitor

__all__ = [
    # Detection Layer
    "CPUDetector",
    "CPUInfo",
    "GPUDetector",
    "GPUInfo",
    "SystemInfo",
    "MemoryInfo",
    # Hardware Management
    "HardwareManager",
    "HardwareResources",
    "WorkloadConfig",
    "get_hardware_manager",
    "reset_hardware_manager",
    # Memory Management
    "MemoryManager",
    "MemorySnapshot",
    "get_memory_manager",
    "reset_memory_manager",
    "track_memory",
    "temp_series",
    "cleanup_series",
    # Shared Memory Utilities
    "SHARED_MEMORY_AVAILABLE",
    "setup_shared_memory_for_dataframe",
    "reconstruct_dataframe_from_shared_memory",
    "cleanup_shared_memory",
    # PyTorch GPU Management
    "PyTorchGPUManager",
    "detect_pytorch_cuda_availability",
    "detect_pytorch_gpu_availability",
    "configure_gpu_memory",
    # XGBoost GPU Detection
    "detect_gpu_availability",
    # System Configuration
    "configure_windows_stdio",
    "get_pytorch_env",
    # Runtime Monitoring
    # Runtime Monitoring
    "RuntimeMonitor",
    "get_runtime_monitor",
    "_check_kmp_duplicate_lib_warning",
    # Memory Pooling
    "ArrayPool",
    "SeriesPool",
    "get_array_pool",
    "get_series_pool",
    "cleanup_pools",
]

"""
Resource management layer.

This package provides managers for:
- Hardware resources (CPU, GPU, RAM)
- Memory (monitoring, cleanup, leak detection)
- PyTorch GPU operations
"""

from .hardware_manager import (
    HardwareManager,
    HardwareResources,
    WorkloadConfig,
    get_hardware_manager,
    reset_hardware_manager,
)
from .memory_manager import (
    MemoryManager,
    MemorySnapshot,
    cleanup_series,
    get_memory_manager,
    reset_memory_manager,
    temp_series,
    track_memory,
)
from .pytorch_gpu_manager import PyTorchGPUManager

__all__ = [
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
    # PyTorch GPU Management
    "PyTorchGPUManager",
]

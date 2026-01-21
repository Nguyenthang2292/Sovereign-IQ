"""
Hardware detection layer.

This package provides unified detection for:
- GPU (PyTorch, XGBoost, CuPy, OpenCL, Numba)
- CPU (cores, architecture)
- System information (RAM, psutil wrapper)

Usage:
    from modules.common.system.detection import GPUDetector, SystemInfo, CPUDetector

    # Detect all GPUs
    gpu_info = GPUDetector.detect_all()
    if gpu_info.pytorch_available:
        print(f"PyTorch CUDA {gpu_info.pytorch_cuda_version}")
        print(f"GPU Count: {gpu_info.pytorch_device_count}")

    # Get memory info
    mem_info = SystemInfo.get_memory_info()
    print(f"RAM: {mem_info.available_gb:.2f} GB available")
    print(f"RAM Usage: {mem_info.percent_used:.1f}%")

    # Get CPU info
    cpu_info = CPUDetector.detect()
    print(f"CPU Cores: {cpu_info.cores} (Physical: {cpu_info.cores_physical})")
"""

from .cpu_detector import CPUDetector
from .gpu_detector import GPUDetector, GPUInfo
from .system_info import CPUInfo, MemoryInfo, SystemInfo

__all__ = [
    "CPUDetector",
    "CPUInfo",
    "GPUDetector",
    "GPUInfo",
    "SystemInfo",
    "MemoryInfo",
]

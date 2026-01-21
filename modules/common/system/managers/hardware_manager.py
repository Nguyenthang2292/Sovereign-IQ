"""
Hardware Manager for all modules

Automatically detects and manages CPU, GPU, and RAM resources for optimal performance.
Provides dynamic workload distribution across available hardware.

Supports:
- CUDA (CuPy, Numba, PyTorch)
- OpenCL
- XGBoost GPU
- PyTorch GPU (via PyTorchGPUManager)

Author: Crypto Probability Team
"""

import logging
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

from modules.common.system.detection import CPUDetector, GPUDetector, SystemInfo
from modules.common.system.managers.pytorch_gpu_manager import PyTorchGPUManager

# Type alias for GPU types
GpuType = Literal["cuda", "opencl", "pytorch", "xgboost"]

logger = logging.getLogger(__name__)


@dataclass
class HardwareResources:
    """Hardware resource information"""

    cpu_cores: int
    cpu_cores_physical: int
    total_ram_gb: float
    available_ram_gb: float
    ram_percent_used: float
    gpu_available: bool
    gpu_type: Optional[GpuType] = None
    gpu_memory_gb: Optional[float] = None
    gpu_count: int = 0
    pytorch_gpu_available: bool = False
    xgboost_gpu_available: bool = False


@dataclass
class WorkloadConfig:
    """Workload configuration based on available hardware"""

    use_multiprocessing: bool
    num_processes: int
    use_multithreading: bool
    num_threads: int
    use_gpu: bool
    gpu_backend: Optional[GpuType] = None
    batch_size: int = 1000
    chunk_size: int = 100


class HardwareManager:
    """
    Manages hardware resources for optimal computation.

    Features:
    - Auto-detects CPU cores and RAM
    - Detects GPU availability (CUDA/OpenCL/PyTorch/XGBoost)
    - Provides optimal workload configuration
    - Monitors resource usage
    - Prevents resource exhaustion
    - Integrates with PyTorchGPUManager for PyTorch-specific GPU management
    """

    def __init__(
        self,
        max_cpu_percent: float = 90.0,
        max_ram_percent: float = 85.0,
        min_free_ram_gb: float = 2.0,
        pytorch_gpu_manager: Optional[PyTorchGPUManager] = None,
    ):
        """
        Initialize Hardware Manager.

        Args:
            max_cpu_percent: Maximum CPU utilization (default: 90%)
            max_ram_percent: Maximum RAM utilization (default: 85%)
            min_free_ram_gb: Minimum free RAM to keep (default: 2GB)
            pytorch_gpu_manager: Optional PyTorchGPUManager instance (creates new if None)
        """
        self.max_cpu_percent = max_cpu_percent
        self.max_ram_percent = max_ram_percent
        self.min_free_ram_gb = min_free_ram_gb

        self._resources: Optional[HardwareResources] = None
        self._workload_config: Optional[WorkloadConfig] = None
        self._pytorch_gpu_manager = pytorch_gpu_manager or PyTorchGPUManager()

        logger.info("Hardware Manager initialized")

    def detect_resources(self) -> HardwareResources:
        """
        Detect available hardware resources.

        Uses detection layer for unified hardware detection.

        Returns:
            HardwareResources object with current hardware info
        """
        # CPU detection using detection layer
        cpu_info = CPUDetector.detect()
        cpu_cores = cpu_info.cores
        cpu_cores_physical = cpu_info.cores_physical

        # Memory detection using detection layer
        memory_info = SystemInfo.get_memory_info()
        total_ram_gb = memory_info.total_gb
        available_ram_gb = memory_info.available_gb
        ram_percent_used = memory_info.percent_used

        # GPU detection using unified detector
        gpu_info = GPUDetector.detect_all()

        # Use GPUInfo dataclass fields directly (includes PyTorch info)
        self._resources = HardwareResources(
            cpu_cores=cpu_cores,
            cpu_cores_physical=cpu_cores_physical,
            total_ram_gb=total_ram_gb,
            available_ram_gb=available_ram_gb,
            ram_percent_used=ram_percent_used,
            gpu_available=gpu_info.available,
            gpu_type=gpu_info.gpu_type,
            gpu_memory_gb=gpu_info.gpu_memory_gb,
            gpu_count=gpu_info.gpu_count,
            pytorch_gpu_available=gpu_info.pytorch_available,
            xgboost_gpu_available=gpu_info.xgboost_available,
        )

        logger.info(
            f"Hardware detected: {cpu_cores} cores, "
            f"{total_ram_gb:.2f}GB RAM ({ram_percent_used:.1f}% used), "
            f"GPU: {gpu_info.gpu_type if gpu_info.available else 'None'}"
        )

        return self._resources

    def get_optimal_workload_config(self, workload_size: int, prefer_gpu: bool = True) -> WorkloadConfig:
        """
        Calculate optimal workload configuration based on available resources.

        Args:
            workload_size: Number of items to process
            prefer_gpu: Prefer GPU if available (default: True)

        Returns:
            WorkloadConfig with optimal settings
        """
        if self._resources is None:
            self.detect_resources()

        res = self._resources

        # Determine GPU usage
        use_gpu = res.gpu_available and prefer_gpu
        gpu_backend = res.gpu_type if use_gpu else None

        # Calculate safe RAM-based constraints
        max_parallel_tasks = self._calculate_max_parallel_tasks(workload_size)

        # CPU configuration
        # Reserve 1-2 cores for system
        available_cores = max(1, res.cpu_cores - 2)

        # Multiprocessing: Use physical cores
        use_multiprocessing = res.cpu_cores_physical > 1 and workload_size > 10
        num_processes = (
            min(
                res.cpu_cores_physical - 1,  # Reserve 1 physical core
                max_parallel_tasks,
                workload_size,
            )
            if use_multiprocessing
            else 1
        )

        # Multithreading: Use logical cores per process
        use_multithreading = res.cpu_cores > res.cpu_cores_physical
        threads_per_process = max(1, res.cpu_cores // num_processes) if use_multiprocessing else available_cores
        num_threads = min(threads_per_process, max_parallel_tasks, workload_size)

        # Batch and chunk sizes
        if use_gpu:
            # Larger batches for GPU
            batch_size = min(10000, workload_size)
            chunk_size = min(1000, workload_size // num_processes) if num_processes > 0 else 1000
        else:
            # Smaller batches for CPU
            batch_size = min(1000, workload_size)
            chunk_size = min(100, workload_size // num_processes) if num_processes > 0 else 100

        self._workload_config = WorkloadConfig(
            use_multiprocessing=use_multiprocessing,
            num_processes=num_processes,
            use_multithreading=use_multithreading,
            num_threads=num_threads,
            use_gpu=use_gpu,
            gpu_backend=gpu_backend,
            batch_size=batch_size,
            chunk_size=chunk_size,
        )

        logger.info(
            f"Workload config: MP={use_multiprocessing}({num_processes}), "
            f"MT={use_multithreading}({num_threads}), GPU={gpu_backend}, "
            f"batch={batch_size}, chunk={chunk_size}"
        )

        return self._workload_config

    def _calculate_max_parallel_tasks(self, workload_size: int) -> int:
        """
        Calculate maximum safe parallel tasks based on available RAM.

        Args:
            workload_size: Number of items to process

        Returns:
            Maximum number of parallel tasks
        """
        if not SystemInfo.is_psutil_available() or self._resources is None:
            return 4  # Conservative fallback

        # Conservative estimate: 500MB per task (covers most ML workloads)
        # Users can override via max_ram_percent and min_free_ram_gb settings
        estimated_memory_per_task_gb = 0.5

        # Available memory for computation
        available_for_compute = self._resources.available_ram_gb - self.min_free_ram_gb

        max_tasks = max(1, int(available_for_compute / estimated_memory_per_task_gb))

        return min(max_tasks, workload_size, self._resources.cpu_cores)

    def get_current_usage(self) -> Dict[str, Any]:
        """
        Get current resource usage.

        Returns:
            Dictionary with current CPU, RAM usage
        """
        if not SystemInfo.is_psutil_available():
            return {"cpu_percent": 0.0, "ram_percent": 0.0, "ram_available_gb": 0.0}

        memory_info = SystemInfo.get_memory_info()
        cpu_percent = SystemInfo.get_cpu_percent()

        return {
            "cpu_percent": cpu_percent,
            "ram_percent": memory_info.percent_used,
            "ram_available_gb": memory_info.available_gb,
        }

    def check_resources_available(self) -> Tuple[bool, str]:
        """
        Check if resources are available for new tasks.

        Returns:
            Tuple of (available: bool, reason: str)
        """
        if not SystemInfo.is_psutil_available():
            return True, "Resource monitoring not available"

        memory_info = SystemInfo.get_memory_info()
        cpu_percent = SystemInfo.get_cpu_percent()

        # Check RAM
        if memory_info.percent_used > self.max_ram_percent:
            return False, f"RAM usage too high: {memory_info.percent_used:.1f}%"

        if memory_info.available_gb < self.min_free_ram_gb:
            return False, f"Low RAM: {memory_info.available_gb:.2f}GB available"

        # Check CPU
        if cpu_percent > self.max_cpu_percent:
            return False, f"CPU usage too high: {cpu_percent:.1f}%"

        return True, "Resources available"

    def wait_for_resources(self, timeout: int = 60, check_interval: float = 1.0):
        """
        Wait for resources to become available.

        Args:
            timeout: Maximum wait time in seconds
            check_interval: Time between checks in seconds

        Raises:
            TimeoutError: If resources don't become available within timeout
        """
        import time

        elapsed = 0.0
        while elapsed < timeout:
            available, reason = self.check_resources_available()
            if available:
                return

            logger.debug(f"Waiting for resources: {reason}")
            time.sleep(check_interval)
            elapsed += check_interval

        raise TimeoutError(f"Resources not available after {timeout}s")

    def create_process_pool(self, num_processes: Optional[int] = None) -> mp.Pool:
        """
        Create a multiprocessing Pool with optimal settings.

        Args:
            num_processes: Number of processes (default: from workload config)

        Returns:
            multiprocessing.Pool
        """
        if num_processes is None:
            if self._workload_config is None:
                self.get_optimal_workload_config(workload_size=100)
            num_processes = self._workload_config.num_processes

        return mp.Pool(processes=num_processes)

    def create_thread_pool(self, num_threads: Optional[int] = None) -> ThreadPoolExecutor:
        """
        Create a ThreadPoolExecutor with optimal settings.

        Args:
            num_threads: Number of threads (default: from workload config)

        Returns:
            ThreadPoolExecutor
        """
        if num_threads is None:
            if self._workload_config is None:
                self.get_optimal_workload_config(workload_size=100)
            num_threads = self._workload_config.num_threads

        return ThreadPoolExecutor(max_workers=num_threads)

    def get_resources(self) -> HardwareResources:
        """Get detected resources (detect if not already done)"""
        if self._resources is None:
            self.detect_resources()
        return self._resources

    def get_workload_config(self) -> WorkloadConfig:
        """Get workload config (calculate if not already done)"""
        if self._workload_config is None:
            self.get_optimal_workload_config(workload_size=100)
        return self._workload_config

    def get_pytorch_gpu_manager(self) -> PyTorchGPUManager:
        """Get the PyTorch GPU Manager instance"""
        return self._pytorch_gpu_manager

    def is_pytorch_gpu_available(self) -> bool:
        """Check if PyTorch GPU is available"""
        if self._resources is None:
            self.detect_resources()
        return self._resources.pytorch_gpu_available if self._resources else False

    def is_xgboost_gpu_available(self) -> bool:
        """Check if XGBoost GPU is available"""
        if self._resources is None:
            self.detect_resources()
        return self._resources.xgboost_gpu_available if self._resources else False

    def get_xgboost_gpu_params(self) -> Dict[str, Any]:
        """
        Get XGBoost GPU configuration parameters.

        Returns:
            Dictionary with XGBoost GPU parameters if available, empty dict otherwise
        """
        if not self.is_xgboost_gpu_available():
            return {}

        # XGBoost 2.0+ uses 'hist' with device='cuda' instead of 'gpu_hist'
        # Only return CUDA params if CUDA is actually available (nvidia-smi found GPU but
        # we need CUDA Python bindings for XGBoost to actually use it)
        gpu_info = GPUDetector.detect_all()
        if gpu_info.pytorch_available or gpu_info.gpu_type in ("cuda", "pytorch"):
            return {
                "tree_method": "hist",
                "device": "cuda",
            }
        # nvidia-smi found GPU but no CUDA Python bindings available
        logger.warning(
            "XGBoost GPU detected via nvidia-smi but no CUDA Python bindings found. "
            "Install PyTorch, CuPy, or Numba with CUDA support for GPU acceleration."
        )
        return {}


# Global singleton instance using SingletonMeta
from modules.common.system.utils.singleton import SingletonMeta


class HardwareManagerSingleton(HardwareManager, metaclass=SingletonMeta):
    """HardwareManager with singleton pattern."""
    pass


def get_hardware_manager() -> HardwareManager:
    """Get global HardwareManager instance (singleton)"""
    return HardwareManagerSingleton()


def reset_hardware_manager():
    """Reset global HardwareManager (useful for testing)"""
    from modules.common.system.utils.singleton import reset_singleton
    reset_singleton(HardwareManagerSingleton)

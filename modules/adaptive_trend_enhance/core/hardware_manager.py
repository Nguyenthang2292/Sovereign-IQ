"""
Hardware Manager for Adaptive Trend Enhanced

Automatically detects and manages CPU, GPU, and RAM resources for optimal performance.
Provides dynamic workload distribution across available hardware.

Author: Adaptive Trend Enhanced Team
"""

import logging
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available. Install with: pip install psutil")

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
    gpu_type: Optional[str] = None
    gpu_memory_gb: Optional[float] = None
    gpu_count: int = 0


@dataclass
class WorkloadConfig:
    """Workload configuration based on available hardware"""

    use_multiprocessing: bool
    num_processes: int
    use_multithreading: bool
    num_threads: int
    use_gpu: bool
    gpu_backend: Optional[str] = None  # 'cuda', 'opencl', or None
    batch_size: int = 1000
    chunk_size: int = 100


class HardwareManager:
    """
    Manages hardware resources for optimal computation.

    Features:
    - Auto-detects CPU cores and RAM
    - Detects GPU availability (CUDA/OpenCL)
    - Provides optimal workload configuration
    - Monitors resource usage
    - Prevents resource exhaustion
    """

    def __init__(self, max_cpu_percent: float = 90.0, max_ram_percent: float = 85.0, min_free_ram_gb: float = 2.0):
        """
        Initialize Hardware Manager.

        Args:
            max_cpu_percent: Maximum CPU utilization (default: 90%)
            max_ram_percent: Maximum RAM utilization (default: 85%)
            min_free_ram_gb: Minimum free RAM to keep (default: 2GB)
        """
        self.max_cpu_percent = max_cpu_percent
        self.max_ram_percent = max_ram_percent
        self.min_free_ram_gb = min_free_ram_gb

        self._resources: Optional[HardwareResources] = None
        self._workload_config: Optional[WorkloadConfig] = None

        logger.info("Hardware Manager initialized")

    def detect_resources(self) -> HardwareResources:
        """
        Detect available hardware resources.

        Returns:
            HardwareResources object with current hardware info
        """
        # CPU detection
        cpu_cores = mp.cpu_count()

        if PSUTIL_AVAILABLE:
            cpu_cores_physical = psutil.cpu_count(logical=False) or cpu_cores
            memory = psutil.virtual_memory()
            total_ram_gb = memory.total / (1024**3)
            available_ram_gb = memory.available / (1024**3)
            ram_percent_used = memory.percent
        else:
            cpu_cores_physical = cpu_cores
            total_ram_gb = 8.0  # Fallback estimate
            available_ram_gb = 4.0
            ram_percent_used = 50.0

        # GPU detection
        gpu_available = False
        gpu_type = None
        gpu_memory_gb = None
        gpu_count = 0

        # Try CUDA first
        if CUPY_AVAILABLE:
            try:
                gpu_count = cp.cuda.runtime.getDeviceCount()
                if gpu_count > 0:
                    gpu_available = True
                    gpu_type = "cuda"
                    # Get memory of first GPU
                    gpu_memory_gb = cp.cuda.Device(0).mem_info[1] / (1024**3)
                    logger.info(f"CUDA GPU detected: {gpu_count} device(s), {gpu_memory_gb:.2f} GB")
            except Exception as e:
                logger.warning(f"CUDA detection failed: {e}")

        # Try Numba CUDA if CuPy not available
        if not gpu_available and NUMBA_CUDA_AVAILABLE:
            try:
                gpu_count = len(cuda.gpus)
                if gpu_count > 0:
                    gpu_available = True
                    gpu_type = "cuda"
                    logger.info(f"CUDA GPU detected via Numba: {gpu_count} device(s)")
            except Exception as e:
                logger.warning(f"Numba CUDA detection failed: {e}")

        # Try OpenCL if CUDA not available
        if not gpu_available and PYOPENCL_AVAILABLE:
            try:
                platforms = cl.get_platforms()
                for platform in platforms:
                    devices = platform.get_devices(device_type=cl.device_type.GPU)
                    if devices:
                        gpu_available = True
                        gpu_type = "opencl"
                        gpu_count = len(devices)
                        logger.info(f"OpenCL GPU detected: {gpu_count} device(s)")
                        break
            except Exception as e:
                logger.warning(f"OpenCL detection failed: {e}")

        self._resources = HardwareResources(
            cpu_cores=cpu_cores,
            cpu_cores_physical=cpu_cores_physical,
            total_ram_gb=total_ram_gb,
            available_ram_gb=available_ram_gb,
            ram_percent_used=ram_percent_used,
            gpu_available=gpu_available,
            gpu_type=gpu_type,
            gpu_memory_gb=gpu_memory_gb,
            gpu_count=gpu_count,
        )

        logger.info(
            f"Hardware detected: {cpu_cores} cores, "
            f"{total_ram_gb:.2f}GB RAM ({ram_percent_used:.1f}% used), "
            f"GPU: {gpu_type if gpu_available else 'None'}"
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
        if not PSUTIL_AVAILABLE or self._resources is None:
            return 4  # Conservative fallback

        # Estimate memory per task (conservative: 500MB)
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
        if not PSUTIL_AVAILABLE:
            return {"cpu_percent": 0.0, "ram_percent": 0.0, "ram_available_gb": 0.0}

        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "ram_percent": psutil.virtual_memory().percent,
            "ram_available_gb": psutil.virtual_memory().available / (1024**3),
        }

    def check_resources_available(self) -> Tuple[bool, str]:
        """
        Check if resources are available for new tasks.

        Returns:
            Tuple of (available: bool, reason: str)
        """
        if not PSUTIL_AVAILABLE:
            return True, "Resource monitoring not available"

        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Check RAM
        if memory.percent > self.max_ram_percent:
            return False, f"RAM usage too high: {memory.percent:.1f}%"

        available_gb = memory.available / (1024**3)
        if available_gb < self.min_free_ram_gb:
            return False, f"Low RAM: {available_gb:.2f}GB available"

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


# Global singleton instance
_hardware_manager: Optional[HardwareManager] = None


def get_hardware_manager() -> HardwareManager:
    """Get global HardwareManager instance (singleton)"""
    global _hardware_manager
    if _hardware_manager is None:
        _hardware_manager = HardwareManager()
    return _hardware_manager


def reset_hardware_manager():
    """Reset global HardwareManager (useful for testing)"""
    global _hardware_manager
    _hardware_manager = None

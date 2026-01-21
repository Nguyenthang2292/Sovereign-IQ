"""
Memory Manager for Adaptive Trend Enhanced

Automatic memory monitoring, cleanup, and leak prevention.
Ensures efficient memory usage and prevents memory exhaustion.

Author: Adaptive Trend Enhanced Team
"""

import gc
import logging
import tracemalloc
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

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


logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Memory usage snapshot"""
    timestamp: float
    ram_used_gb: float
    ram_percent: float
    ram_available_gb: float
    gpu_used_gb: Optional[float] = None
    tracemalloc_current_mb: Optional[float] = None
    tracemalloc_peak_mb: Optional[float] = None


class MemoryManager:
    """
    Manages memory usage and prevents memory leaks.

    Features:
    - Real-time memory monitoring
    - Automatic garbage collection
    - Memory threshold alerts
    - GPU memory tracking (if available)
    - Memory leak detection
    - Context managers for safe memory operations
    """

    def __init__(self,
                 warning_threshold_percent: float = 75.0,
                 critical_threshold_percent: float = 85.0,
                 auto_cleanup_threshold_percent: float = 80.0,
                 enable_tracemalloc: bool = False):
        """
        Initialize Memory Manager.

        Args:
            warning_threshold_percent: Warning threshold (default: 75%)
            critical_threshold_percent: Critical threshold (default: 85%)
            auto_cleanup_threshold_percent: Auto-cleanup threshold (default: 80%)
            enable_tracemalloc: Enable tracemalloc for detailed tracking (default: False)
        """
        self.warning_threshold = warning_threshold_percent
        self.critical_threshold = critical_threshold_percent
        self.auto_cleanup_threshold = auto_cleanup_threshold_percent
        self.enable_tracemalloc = enable_tracemalloc

        self._snapshots: List[MemorySnapshot] = []
        self._cleanup_callbacks: List[Callable] = []
        self._weakrefs: List[weakref.ref] = []

        if enable_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start()
            logger.info("tracemalloc enabled for detailed memory tracking")

        logger.info("Memory Manager initialized")

    def get_current_usage(self) -> MemorySnapshot:
        """
        Get current memory usage snapshot.

        Returns:
            MemorySnapshot with current memory info
        """
        import time

        snapshot = MemorySnapshot(
            timestamp=time.time(),
            ram_used_gb=0.0,
            ram_percent=0.0,
            ram_available_gb=0.0
        )

        # RAM usage
        if PSUTIL_AVAILABLE:
            memory = psutil.virtual_memory()
            snapshot.ram_used_gb = memory.used / (1024**3)
            snapshot.ram_percent = memory.percent
            snapshot.ram_available_gb = memory.available / (1024**3)
        else:
            # Fallback estimates
            snapshot.ram_percent = 50.0

        # GPU memory
        if CUPY_AVAILABLE:
            try:
                mempool = cp.get_default_memory_pool()
                snapshot.gpu_used_gb = mempool.used_bytes() / (1024**3)
            except Exception as e:
                logger.debug(f"GPU memory check failed: {e}")

        # tracemalloc
        if self.enable_tracemalloc and tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            snapshot.tracemalloc_current_mb = current / (1024**2)
            snapshot.tracemalloc_peak_mb = peak / (1024**2)

        return snapshot

    def check_memory_status(self) -> tuple[str, MemorySnapshot]:
        """
        Check current memory status.

        Returns:
            Tuple of (status: str, snapshot: MemorySnapshot)
            Status can be: 'ok', 'warning', 'critical'
        """
        snapshot = self.get_current_usage()

        if snapshot.ram_percent >= self.critical_threshold:
            status = 'critical'
        elif snapshot.ram_percent >= self.warning_threshold:
            status = 'warning'
        else:
            status = 'ok'

        # Auto-cleanup if threshold exceeded
        if snapshot.ram_percent >= self.auto_cleanup_threshold:
            logger.warning(f"Auto-cleanup triggered: RAM at {snapshot.ram_percent:.1f}%")
            self.cleanup(aggressive=True)
            # Re-check after cleanup
            snapshot = self.get_current_usage()

        return status, snapshot

    def cleanup(self, aggressive: bool = False):
        """
        Perform memory cleanup.

        Args:
            aggressive: If True, perform more aggressive cleanup
        """
        logger.debug(f"Memory cleanup started (aggressive={aggressive})")

        # Run registered cleanup callbacks
        for callback in self._cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Cleanup callback failed: {e}")

        # Clear weak references to dead objects
        self._weakrefs = [ref for ref in self._weakrefs if ref() is not None]

        # Standard garbage collection
        gc.collect(generation=0)

        if aggressive:
            # Full garbage collection (all generations)
            gc.collect(generation=2)

            # GPU memory cleanup
            if CUPY_AVAILABLE:
                try:
                    mempool = cp.get_default_memory_pool()
                    mempool.free_all_blocks()
                    logger.debug("GPU memory pool cleared")
                except Exception as e:
                    logger.debug(f"GPU memory cleanup failed: {e}")

        # Log result
        snapshot = self.get_current_usage()
        logger.debug(f"Cleanup complete: RAM at {snapshot.ram_percent:.1f}%")

    def register_cleanup_callback(self, callback: Callable):
        """
        Register a callback to be called during cleanup.

        Args:
            callback: Function to call during cleanup (no arguments)
        """
        self._cleanup_callbacks.append(callback)

    def track_object(self, obj: Any):
        """
        Track an object with weak reference for memory management.

        Args:
            obj: Object to track
        """
        self._weakrefs.append(weakref.ref(obj))

    def take_snapshot(self) -> MemorySnapshot:
        """
        Take a memory snapshot and store it.

        Returns:
            MemorySnapshot
        """
        snapshot = self.get_current_usage()
        self._snapshots.append(snapshot)

        # Keep only last 100 snapshots
        if len(self._snapshots) > 100:
            self._snapshots = self._snapshots[-100:]

        return snapshot

    def get_snapshots(self, last_n: Optional[int] = None) -> List[MemorySnapshot]:
        """
        Get stored memory snapshots.

        Args:
            last_n: Number of recent snapshots (None for all)

        Returns:
            List of MemorySnapshot
        """
        if last_n is None:
            return self._snapshots.copy()
        return self._snapshots[-last_n:]

    def detect_memory_leak(self, threshold_mb: float = 100.0) -> bool:
        """
        Detect potential memory leak by comparing snapshots.

        Args:
            threshold_mb: Memory increase threshold in MB

        Returns:
            True if potential leak detected
        """
        if len(self._snapshots) < 10:
            return False

        # Compare first and last snapshots
        first = self._snapshots[0]
        last = self._snapshots[-1]

        ram_increase_gb = last.ram_used_gb - first.ram_used_gb
        ram_increase_mb = ram_increase_gb * 1024

        if ram_increase_mb > threshold_mb:
            logger.warning(f"Potential memory leak detected: "
                         f"{ram_increase_mb:.2f}MB increase")
            return True

        return False

    @contextmanager
    def track_memory(self, operation_name: str = "operation"):
        """
        Context manager to track memory usage of an operation.

        Args:
            operation_name: Name of the operation for logging

        Example:
            with memory_manager.track_memory("my_operation"):
                # Your code here
                process_data()
        """
        # Before
        before = self.get_current_usage()
        logger.debug(f"[{operation_name}] Starting - RAM: {before.ram_percent:.1f}%")

        try:
            yield
        finally:
            # After
            after = self.get_current_usage()
            ram_change = after.ram_used_gb - before.ram_used_gb

            log_msg = (f"[{operation_name}] Completed - "
                      f"RAM: {after.ram_percent:.1f}% "
                      f"(Δ{ram_change:+.3f}GB)")

            if CUPY_AVAILABLE and before.gpu_used_gb is not None and after.gpu_used_gb is not None:
                gpu_change = after.gpu_used_gb - before.gpu_used_gb
                log_msg += f", GPU: {after.gpu_used_gb:.2f}GB (Δ{gpu_change:+.3f}GB)"

            logger.debug(log_msg)

            # Auto-cleanup if needed
            if after.ram_percent >= self.auto_cleanup_threshold:
                self.cleanup(aggressive=False)

    @contextmanager
    def safe_memory_operation(self, operation_name: str = "operation",
                             cleanup_after: bool = True):
        """
        Context manager for safe memory operations with automatic cleanup.

        Args:
            operation_name: Name of the operation
            cleanup_after: Perform cleanup after operation (default: True)

        Example:
            with memory_manager.safe_memory_operation("heavy_computation"):
                result = expensive_function()
        """
        # Check memory before
        status, snapshot = self.check_memory_status()

        if status == 'critical':
            logger.warning(f"[{operation_name}] Starting with critical memory: "
                         f"{snapshot.ram_percent:.1f}%")
            self.cleanup(aggressive=True)

        try:
            with self.track_memory(operation_name):
                yield
        finally:
            if cleanup_after:
                self.cleanup(aggressive=False)

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive memory statistics.

        Returns:
            Dictionary with memory stats
        """
        current = self.get_current_usage()

        stats = {
            'current_ram_gb': current.ram_used_gb,
            'current_ram_percent': current.ram_percent,
            'available_ram_gb': current.ram_available_gb,
            'snapshots_count': len(self._snapshots),
            'tracked_objects': sum(1 for ref in self._weakrefs if ref() is not None),
        }

        if current.gpu_used_gb is not None:
            stats['gpu_used_gb'] = current.gpu_used_gb

        if self.enable_tracemalloc and current.tracemalloc_current_mb is not None:
            stats['tracemalloc_current_mb'] = current.tracemalloc_current_mb
            stats['tracemalloc_peak_mb'] = current.tracemalloc_peak_mb

        # Garbage collector stats
        gc_stats = gc.get_stats()
        if gc_stats:
            stats['gc_collections'] = sum(s.get('collections', 0) for s in gc_stats)

        return stats

    def log_memory_stats(self):
        """Log current memory statistics"""
        stats = self.get_memory_stats()
        logger.info(f"Memory Stats: RAM {stats['current_ram_gb']:.2f}GB "
                   f"({stats['current_ram_percent']:.1f}%), "
                   f"Snapshots: {stats['snapshots_count']}, "
                   f"Tracked objects: {stats['tracked_objects']}")

    def reset(self):
        """Reset memory manager state"""
        self._snapshots.clear()
        self._weakrefs.clear()
        self.cleanup(aggressive=True)

        if self.enable_tracemalloc and tracemalloc.is_tracing():
            tracemalloc.clear_traces()

        logger.info("Memory Manager reset")


# Global singleton instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager(enable_tracemalloc: bool = False) -> MemoryManager:
    """
    Get global MemoryManager instance (singleton).

    Args:
        enable_tracemalloc: Enable tracemalloc (only on first call)

    Returns:
        MemoryManager instance
    """
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager(enable_tracemalloc=enable_tracemalloc)
    return _memory_manager


def reset_memory_manager():
    """Reset global MemoryManager (useful for testing)"""
    global _memory_manager
    if _memory_manager is not None:
        _memory_manager.reset()
    _memory_manager = None


# Convenience context manager
@contextmanager
def track_memory(operation_name: str = "operation"):
    """
    Convenience context manager for tracking memory.

    Example:
        from modules.adaptive_trend_enhance.core.memory_manager import track_memory

        with track_memory("my_operation"):
            # Your code here
            process_data()
    """
    manager = get_memory_manager()
    with manager.track_memory(operation_name):
        yield

import gc
import tracemalloc
from typing import Any, Dict, Optional

from modules.common.ui.logging import log_info, log_warn

"""
Memory management utilities for monitoring and optimizing memory usage.
"""


def get_memory_usage() -> Dict[str, Any]:
    """
    Get current memory usage statistics.

    Returns:
        Dict with memory usage information
    """
    try:
        import psutil

        process = psutil.Process()
        mem_info = process.memory_info()

        vm = psutil.virtual_memory()
        return {
            "rss_mb": mem_info.rss / 1024 / 1024,  # Resident Set Size in MB
            "vms_mb": mem_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
            "percent": process.memory_percent(),
            "available_mb": vm.available / 1024 / 1024,
            "total_mb": vm.total / 1024 / 1024,
        }
    except Exception as e:
        log_warn(f"Unable to get memory usage with psutil: {e}. Using basic memory info fallback.")
        return {
            "rss_mb": None,
            "vms_mb": None,
            "percent": None,
            "available_mb": None,
            "total_mb": None,
        }


def force_garbage_collection(verbose: bool = False) -> Dict[str, Any]:
    """
    Force garbage collection and return statistics.

    Args:
        verbose: If True, log detailed GC statistics

    Returns:
        Dict with GC statistics
    """
    # Get counts before GC
    counts_before = gc.get_count()

    # Run full collection
    collected = gc.collect()

    # Get counts after GC
    counts_after = gc.get_count()

    stats = {
        "collected": collected,
        "counts_before": counts_before,
        "counts_after": counts_after,
    }

    if verbose:
        log_info(f"Garbage collection: collected {collected} objects")
        log_info(f"GC counts before: {counts_before}, after: {counts_after}")

    return stats


def start_memory_tracking() -> Optional[tracemalloc.Snapshot]:
    """
    Start tracking memory allocations using tracemalloc.

    Returns:
        Snapshot ID or None if tracemalloc not available
    """
    try:
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        return tracemalloc.take_snapshot()
    except Exception as e:
        log_warn(f"Could not start memory tracking: {e}")
        return None


def get_memory_snapshot_diff(snapshot1, snapshot2, top_n: int = 10) -> Optional[Dict[str, Any]]:
    """
    Compare two memory snapshots and return top memory consumers.

    Args:
        snapshot1: First snapshot (from start_memory_tracking)
        snapshot2: Second snapshot
        top_n: Number of top consumers to return. Must be a positive integer.
               If None or less than 1, defaults to 10. If not an integer, raises ValueError.

    Returns:
        Dict with top memory consumers or None if tracemalloc not available

    Raises:
        ValueError: If top_n is not an integer
    """
    try:
        if snapshot1 is None or snapshot2 is None:
            return None

        # Validate top_n parameter
        if top_n is None or (isinstance(top_n, int) and top_n < 1):
            top_n = 10
        elif not isinstance(top_n, int):
            raise ValueError(f"top_n must be an integer, got {type(top_n).__name__}")

        top_stats = snapshot2.compare_to(snapshot1, "lineno")

        top_consumers = []
        for index, stat in enumerate(top_stats[:top_n], 1):
            top_consumers.append(
                {
                    "rank": index,
                    "size_diff_mb": stat.size_diff / 1024 / 1024,
                    "size_mb": stat.size / 1024 / 1024,
                    "count_diff": stat.count_diff,
                    "count": stat.count,
                    "traceback": stat.traceback.format()[:3] if stat.traceback else None,
                }
            )

        return {
            "top_consumers": top_consumers,
            "total_size_diff_mb": sum(s.size_diff for s in top_stats) / 1024 / 1024,
        }
    except Exception as e:
        log_warn(f"Could not get memory snapshot diff: {e}")
        return None


def log_memory_usage(context: str = ""):
    """
    Log current memory usage.

    Args:
        context: Context string to include in log message
    """
    mem_info = get_memory_usage()
    if mem_info.get("rss_mb") is not None:
        log_info(
            f"Memory usage {context}: "
            f"RSS={mem_info['rss_mb']:.2f}MB, "
            f"Percent={mem_info['percent']:.2f}%, "
            f"Available={mem_info['available_mb']:.2f}MB"
        )
    else:
        log_info(f"Memory usage {context}: (psutil not available)")


def optimize_memory() -> Dict[str, Any]:
    """
    Run memory optimization routines.
    This includes garbage collection and clearing caches.
    """
    log_info("Starting memory optimization...")

    # Force garbage collection
    gc_stats = force_garbage_collection(verbose=True)

    # Log memory usage after optimization
    log_memory_usage("after optimization")

    return gc_stats

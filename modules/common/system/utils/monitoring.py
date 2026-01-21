"""
Runtime monitoring utilities.

Provides RuntimeMonitor for tracking performance and exceptions.
"""

import os
import time
from functools import wraps
from typing import Any, Callable, Dict

from modules.common.ui.logging import log_info, log_warn


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

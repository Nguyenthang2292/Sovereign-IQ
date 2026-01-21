"""
System information detection with psutil wrapper.

Single source of truth for system information queries with fallback handling.
"""

import logging
from dataclasses import dataclass
from typing import Optional

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available. Install with: pip install psutil")

logger = logging.getLogger(__name__)


@dataclass
class MemoryInfo:
    """Memory information."""

    total_gb: float
    available_gb: float
    used_gb: float
    percent_used: float


@dataclass
class CPUInfo:
    """CPU information."""

    cores: int
    cores_physical: int
    percent_used: float


class SystemInfo:
    """
    Wrapper for system information with fallback.

    Provides a single source of truth for system queries with graceful
    degradation when psutil is not available.
    """

    @staticmethod
    def is_psutil_available() -> bool:
        """
        Check if psutil is available.

        Returns:
            True if psutil is available, False otherwise
        """
        return PSUTIL_AVAILABLE

    @staticmethod
    def get_memory_info() -> MemoryInfo:
        """
        Get RAM information with fallback.

        Returns:
            MemoryInfo with memory statistics, or zeros if psutil unavailable
        """
        if not PSUTIL_AVAILABLE:
            return MemoryInfo(
                total_gb=0.0,
                available_gb=0.0,
                used_gb=0.0,
                percent_used=0.0,
            )

        try:
            mem = psutil.virtual_memory()
            return MemoryInfo(
                total_gb=mem.total / (1024**3),
                available_gb=mem.available / (1024**3),
                used_gb=mem.used / (1024**3),
                percent_used=mem.percent,
            )
        except Exception as e:
            logger.warning(f"Error getting memory info: {e}")
            return MemoryInfo(
                total_gb=0.0,
                available_gb=0.0,
                used_gb=0.0,
                percent_used=0.0,
            )

    @staticmethod
    def get_cpu_info() -> CPUInfo:
        """
        Get CPU information with fallback.

        Returns:
            CPUInfo with CPU statistics, or defaults if psutil unavailable
        """
        if not PSUTIL_AVAILABLE:
            import multiprocessing as mp

            return CPUInfo(
                cores=mp.cpu_count() or 1,
                cores_physical=mp.cpu_count() or 1,
                percent_used=0.0,
            )

        try:
            return CPUInfo(
                cores=psutil.cpu_count(logical=True) or 1,
                cores_physical=psutil.cpu_count(logical=False) or 1,
                percent_used=psutil.cpu_percent(interval=0.1),
            )
        except Exception as e:
            logger.warning(f"Error getting CPU info: {e}")
            import multiprocessing as mp

            return CPUInfo(
                cores=mp.cpu_count() or 1,
                cores_physical=mp.cpu_count() or 1,
                percent_used=0.0,
            )

    @staticmethod
    def get_cpu_percent(interval: float = 0.1) -> float:
        """
        Get current CPU usage percentage.

        Args:
            interval: Sampling interval in seconds

        Returns:
            CPU usage percentage (0-100), or 0.0 if psutil unavailable
        """
        if not PSUTIL_AVAILABLE:
            return 0.0

        try:
            return psutil.cpu_percent(interval=interval)
        except Exception as e:
            logger.warning(f"Error getting CPU percent: {e}")
            return 0.0

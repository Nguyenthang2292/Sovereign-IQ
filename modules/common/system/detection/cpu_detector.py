"""
CPU detection utilities.

Provides CPU information detection.
"""

from .system_info import CPUInfo, SystemInfo


class CPUDetector:
    """
    CPU detection utilities.

    Provides methods to detect and query CPU information.
    """

    @staticmethod
    def detect() -> CPUInfo:
        """
        Detect CPU information.

        Returns:
            CPUInfo with CPU statistics
        """
        return SystemInfo.get_cpu_info()

    @staticmethod
    def get_cores() -> int:
        """
        Get number of CPU cores (logical).

        Returns:
            Number of logical CPU cores
        """
        return SystemInfo.get_cpu_info().cores

    @staticmethod
    def get_physical_cores() -> int:
        """
        Get number of physical CPU cores.

        Returns:
            Number of physical CPU cores
        """
        return SystemInfo.get_cpu_info().cores_physical

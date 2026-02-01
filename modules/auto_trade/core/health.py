"""
Health Check Module.

Provides a registry for system health checks.
"""

import time
from enum import Enum
from typing import Any, Callable, Dict, Tuple, TypedDict


class HealthStatus(Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"


class HealthCheckResult(TypedDict):
    status: str
    details: str
    timestamp: float


class HealthRegistry:
    """
    Registry for health checks.
    """

    def __init__(self) -> None:
        self._checks: Dict[str, Callable[[], Tuple[HealthStatus, str]]] = {}

    def register_check(self, name: str, check_func: Callable[[], Tuple[HealthStatus, str]]) -> None:
        """
        Register a health check function.
        The function should return (HealthStatus, details_string).
        """
        self._checks[name] = check_func

    def check_health(self) -> Dict[str, HealthCheckResult]:
        """
        Run all health checks and return results.
        """
        results = {}
        for name, check_func in self._checks.items():
            try:
                status, details = check_func()
                results[name] = {
                    "status": status.value,
                    "details": details,
                    "timestamp": time.time(),
                }
            except Exception as e:
                results[name] = {
                    "status": HealthStatus.UNHEALTHY.value,
                    "details": f"Check failed: {e}",
                    "timestamp": time.time(),
                }
        return results

    def is_healthy(self) -> bool:
        """
        Returns True if all checks are HEALTHY or DEGRADED (operational).
        Returns False if any check is UNHEALTHY.
        """
        results = self.check_health()
        for res in results.values():
            if res["status"] == HealthStatus.UNHEALTHY.value:
                return False
        return True

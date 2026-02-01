"""
Health Check Module.

Provides a registry for system health checks.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from enum import Enum
from threading import RLock
from typing import Callable, Dict, Literal, Optional, Tuple, TypedDict

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"


class HealthCheckResult(TypedDict):
    status: Literal["HEALTHY", "DEGRADED", "UNHEALTHY"]
    details: str
    timestamp: float


class HealthRegistry:
    """
    Registry for health checks with thread safety and timeout support.
    """

    def __init__(self, default_timeout: Optional[float] = None) -> None:
        """
        Initialize health check registry.

        Args:
            default_timeout: Default timeout in seconds for each health check.
                           If None, checks run without timeout.
        """
        self._checks: Dict[str, Callable[[], Tuple[HealthStatus, str]]] = {}
        self._lock = RLock()
        self._default_timeout = default_timeout

    def register_check(self, name: str, check_func: Callable[[], Tuple[HealthStatus, str]]) -> None:
        """
        Register a health check function.

        The function should return (HealthStatus, details_string).

        Args:
            name: Name of the health check
            check_func: Function that returns (HealthStatus, details_string)
        """
        with self._lock:
            self._checks[name] = check_func

    def unregister_check(self, name: str) -> None:
        """
        Remove a health check from the registry.

        Args:
            name: Name of the health check to remove
        """
        with self._lock:
            self._checks.pop(name, None)

    def check_single(self, name: str) -> HealthCheckResult:
        """
        Run a single health check by name.

        Args:
            name: The name of health check to run

        Returns:
            HealthCheckResult for the specified check

        Raises:
            KeyError: If check name is not registered
        """
        with self._lock:
            check_func = self._checks.get(name)
            if check_func is None:
                raise KeyError(f"Health check '{name}' not found")

        try:
            status, details = check_func()
            return {
                "status": status.value,
                "details": details,
                "timestamp": time.time(),
            }
        except Exception as e:
            logger.error(f"Health check '{name}' failed with exception: {e}", exc_info=True)
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "details": f"Check failed: {e}",
                "timestamp": time.time(),
            }

    def check_health(self, timeout: Optional[float] = None) -> Dict[str, HealthCheckResult]:
        """
        Run all health checks and return results.

        Args:
            timeout: Maximum time (in seconds) to wait for each check.
                    Uses default_timeout if not specified.

        Returns:
            Dictionary mapping check names to their results
        """
        timeout = timeout or self._default_timeout
        results = {}

        with self._lock:
            checks_snapshot = dict(self._checks)

        if not checks_snapshot:
            return results

        if timeout:
            with ThreadPoolExecutor(max_workers=len(checks_snapshot) or 1) as executor:
                for name, check_func in checks_snapshot.items():
                    future = executor.submit(check_func)
                    try:
                        status, details = future.result(timeout=timeout)

                        results[name] = {
                            "status": status.value,
                            "details": details,
                            "timestamp": time.time(),
                        }

                        if status == HealthStatus.UNHEALTHY:
                            logger.warning(f"Health check '{name}' is UNHEALTHY: {details}")
                        elif status == HealthStatus.DEGRADED:
                            logger.info(f"Health check '{name}' is DEGRADED: {details}")

                    except FutureTimeoutError:
                        results[name] = {
                            "status": HealthStatus.UNHEALTHY.value,
                            "details": f"Check timed out after {timeout}s",
                            "timestamp": time.time(),
                        }
                        logger.error(f"Health check '{name}' timed out after {timeout}s")

                    except Exception as e:
                        logger.error(f"Health check '{name}' failed with exception: {e}", exc_info=True)
                        results[name] = {
                            "status": HealthStatus.UNHEALTHY.value,
                            "details": f"Check failed: {e}",
                            "timestamp": time.time(),
                        }
        else:
            for name, check_func in checks_snapshot.items():
                try:
                    status, details = check_func()

                    results[name] = {
                        "status": status.value,
                        "details": details,
                        "timestamp": time.time(),
                    }

                    if status == HealthStatus.UNHEALTHY:
                        logger.warning(f"Health check '{name}' is UNHEALTHY: {details}")
                    elif status == HealthStatus.DEGRADED:
                        logger.info(f"Health check '{name}' is DEGRADED: {details}")

                except Exception as e:
                    logger.error(f"Health check '{name}' failed with exception: {e}", exc_info=True)
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
        Short-circuits on first unhealthy check.
        """
        with self._lock:
            checks_snapshot = dict(self._checks)

        for name, check_func in checks_snapshot.items():
            try:
                status, _ = check_func()
                if status == HealthStatus.UNHEALTHY:
                    return False
            except Exception:
                return False

        return True

    def list_checks(self) -> list[str]:
        """
        Return a list of all registered check names.

        Returns:
            List of check names
        """
        with self._lock:
            return list(self._checks.keys())

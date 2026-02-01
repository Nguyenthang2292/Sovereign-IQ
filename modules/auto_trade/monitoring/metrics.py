"""
Metrics Collection System.

Provides in-memory metrics storage for monitoring system performance.
"""

from enum import Enum
from threading import RLock
from typing import Dict, List, Union


class MetricType(Enum):
    COUNTER = "COUNTER"
    GAUGE = "GAUGE"
    HISTOGRAM = "HISTOGRAM"


class MetricsCollector:
    """
    Collects and stores application metrics in memory.
    """

    def __init__(self) -> None:
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._lock = RLock()

    def increment(self, name: str, value: int = 1) -> None:
        """Increment a counter metric."""
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    def gauge(self, name: str, value: float) -> None:
        """Set a gauge metric value."""
        with self._lock:
            self._gauges[name] = value

    def histogram(self, name: str, value: float) -> None:
        """Record a value in a histogram."""
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = []
            self._histograms[name].append(value)

            # Limit histogram size to prevent memory leaks (keep last 1000 samples)
            if len(self._histograms[name]) > 1000:
                self._histograms[name].pop(0)

    def get_metrics(self) -> Dict[str, Dict[str, Union[int, float, List[float]]]]:
        """Retrieve all metrics."""
        with self._lock:
            return {
                "counters": self._counters.copy(),
                "gauges": self._gauges.copy(),
                "histograms": self._histograms.copy(),
            }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()

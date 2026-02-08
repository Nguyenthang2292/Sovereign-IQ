"""
Metrics Collection System.

Provides in-memory metrics storage for monitoring system performance.
"""

import math
import re
import time
from collections import deque
from enum import Enum
from threading import RLock
from typing import Any, Dict, List, Optional, Union


class MetricType(Enum):
    """Enumeration of metric types."""
    COUNTER = "COUNTER"
    GAUGE = "GAUGE"
    HISTOGRAM = "HISTOGRAM"


class MetricsCollector:
    """
    Collects and stores application metrics in memory.

    Thread-safe implementation supporting counters, gauges, and histograms.
    """

    # Valid metric name pattern: alphanumeric, underscore, dot, hyphen
    METRIC_NAME_PATTERN = re.compile(r'^[a-zA-Z][a-zA-Z0-9_.\-]{0,254}$')
    MAX_METRIC_NAME_LENGTH = 255
    DEFAULT_HISTOGRAM_MAX_SIZE = 1000
    MAX_METRICS_PER_TYPE = 10000  # Prevent unbounded growth

    def __init__(self, histogram_max_size: int = DEFAULT_HISTOGRAM_MAX_SIZE) -> None:
        """
        Initialize metrics collector.

        Args:
            histogram_max_size: Maximum number of samples to keep per histogram (default: 1000)

        Raises:
            ValueError: If histogram_max_size is not positive
        """
        if histogram_max_size <= 0:
            raise ValueError("histogram_max_size must be positive")

        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, deque] = {}
        self._metadata: Dict[str, Dict[str, Union[str, float]]] = {}
        self._histogram_max_size = histogram_max_size
        self._lock = RLock()

    def _validate_metric_name(self, name: str) -> None:
        """
        Validate metric name.

        Args:
            name: Metric name to validate

        Raises:
            ValueError: If name is invalid
        """
        if not name or not isinstance(name, str):
            raise ValueError("Metric name must be a non-empty string")

        if not name.strip():
            raise ValueError("Metric name cannot be whitespace only")

        if len(name) > self.MAX_METRIC_NAME_LENGTH:
            raise ValueError(f"Metric name exceeds maximum length of {self.MAX_METRIC_NAME_LENGTH}")

        if not self.METRIC_NAME_PATTERN.match(name):
            raise ValueError(
                "Metric name must start with a letter and contain only "
                "alphanumeric characters, underscores, dots, or hyphens"
            )

    def _validate_finite_number(self, value: Union[int, float], param_name: str = "value") -> None:
        """
        Validate that a value is a finite number.

        Args:
            value: Value to validate
            param_name: Parameter name for error messages

        Raises:
            ValueError: If value is not a finite number
        """
        if not isinstance(value, (int, float)):
            raise ValueError(f"{param_name} must be a number")

        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            raise ValueError(f"{param_name} must be a finite number (not NaN or Inf)")

    def _update_metadata(self, name: str, metric_type: MetricType) -> None:
        """
        Update metadata for a metric.

        Args:
            name: Metric name
            metric_type: Type of metric
        """
        if name not in self._metadata:
            self._metadata[name] = {
                "type": metric_type.value,
                "created_at": time.time(),
            }
        self._metadata[name]["updated_at"] = time.time()

    def increment(self, name: str, value: int = 1) -> None:
        """
        Increment a counter metric.

        Args:
            name: Metric name (must start with letter, alphanumeric + _.-  allowed)
            value: Increment amount (must be non-negative, default: 1)

        Raises:
            ValueError: If name is invalid or value is negative
            RuntimeError: If max metrics limit exceeded
        """
        try:
            self._validate_metric_name(name)

            if not isinstance(value, int):
                raise ValueError("Increment value must be an integer")

            if value < 0:
                raise ValueError("Increment value must be non-negative")

            with self._lock:
                # Check max metrics limit
                if name not in self._counters and len(self._counters) >= self.MAX_METRICS_PER_TYPE:
                    raise RuntimeError(
                        f"Maximum number of counters ({self.MAX_METRICS_PER_TYPE}) exceeded"
                    )

                self._counters[name] = self._counters.get(name, 0) + value
                self._update_metadata(name, MetricType.COUNTER)

        except Exception:
            # Re-raise validation errors
            raise

    def gauge(self, name: str, value: float) -> None:
        """
        Set a gauge metric value.

        Args:
            name: Metric name (must start with letter, alphanumeric + _.- allowed)
            value: Gauge value (must be finite number)

        Raises:
            ValueError: If name is invalid or value is not finite
            RuntimeError: If max metrics limit exceeded
        """
        try:
            self._validate_metric_name(name)
            self._validate_finite_number(value, "Gauge value")

            with self._lock:
                # Check max metrics limit
                if name not in self._gauges and len(self._gauges) >= self.MAX_METRICS_PER_TYPE:
                    raise RuntimeError(
                        f"Maximum number of gauges ({self.MAX_METRICS_PER_TYPE}) exceeded"
                    )

                self._gauges[name] = float(value)
                self._update_metadata(name, MetricType.GAUGE)

        except Exception:
            # Re-raise validation errors
            raise

    def histogram(self, name: str, value: float) -> None:
        """
        Record a value in a histogram.

        Args:
            name: Metric name (must start with letter, alphanumeric + _.- allowed)
            value: Value to record (must be finite number)

        Raises:
            ValueError: If name is invalid or value is not finite
            RuntimeError: If max metrics limit exceeded
        """
        try:
            self._validate_metric_name(name)
            self._validate_finite_number(value, "Histogram value")

            with self._lock:
                # Check max metrics limit
                if name not in self._histograms and len(self._histograms) >= self.MAX_METRICS_PER_TYPE:
                    raise RuntimeError(
                        f"Maximum number of histograms ({self.MAX_METRICS_PER_TYPE}) exceeded"
                    )

                if name not in self._histograms:
                    self._histograms[name] = deque(maxlen=self._histogram_max_size)

                self._histograms[name].append(float(value))
                self._update_metadata(name, MetricType.HISTOGRAM)

        except Exception:
            # Re-raise validation errors
            raise

    def get_counter(self, name: str) -> int:
        """
        Get counter value.

        Args:
            name: Metric name

        Returns:
            Counter value (0 if not found)
        """
        with self._lock:
            return self._counters.get(name, 0)

    def get_gauge(self, name: str) -> Optional[float]:
        """
        Get gauge value.

        Args:
            name: Metric name

        Returns:
            Gauge value or None if not found
        """
        with self._lock:
            return self._gauges.get(name)

    def get_histogram(self, name: str) -> List[float]:
        """
        Get histogram values.

        Args:
            name: Metric name

        Returns:
            List of histogram values (empty list if not found)
        """
        with self._lock:
            if name in self._histograms:
                return list(self._histograms[name])
            return []

    def get_histogram_stats(self, name: str) -> Dict[str, Union[float, int]]:
        """
        Calculate statistics for a histogram.

        Args:
            name: Metric name

        Returns:
            Dictionary with min, max, avg, p50, p95, p99, count.
            Empty dict if histogram not found or empty.
        """
        with self._lock:
            if name not in self._histograms or not self._histograms[name]:
                return {}

            values = sorted(self._histograms[name])
            count = len(values)

            # Calculate percentiles
            def percentile(sorted_vals: List[float], p: float) -> float:
                """Calculate percentile from sorted values."""
                idx = int(len(sorted_vals) * p)
                return sorted_vals[min(idx, len(sorted_vals) - 1)]

            return {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / count,
                "median": percentile(values, 0.5),
                "p50": percentile(values, 0.5),
                "p95": percentile(values, 0.95),
                "p99": percentile(values, 0.99),
                "count": count,
            }

    def metric_exists(self, name: str, metric_type: Optional[MetricType] = None) -> bool:
        """
        Check if a metric exists.

        Args:
            name: Metric name
            metric_type: Optional metric type to check (checks all types if None)

        Returns:
            True if metric exists, False otherwise
        """
        with self._lock:
            if metric_type is None:
                # Check all types
                return (name in self._counters or
                       name in self._gauges or
                       name in self._histograms)
            elif metric_type == MetricType.COUNTER:
                return name in self._counters
            elif metric_type == MetricType.GAUGE:
                return name in self._gauges
            elif metric_type == MetricType.HISTOGRAM:
                return name in self._histograms
            return False

    def delete_metric(self, name: str, metric_type: Optional[MetricType] = None) -> bool:
        """
        Delete a specific metric.

        Args:
            name: Metric name
            metric_type: Metric type (deletes from all types if None)

        Returns:
            True if at least one metric was deleted, False if not found
        """
        with self._lock:
            deleted = False

            if metric_type is None or metric_type == MetricType.COUNTER:
                if self._counters.pop(name, None) is not None:
                    deleted = True

            if metric_type is None or metric_type == MetricType.GAUGE:
                if self._gauges.pop(name, None) is not None:
                    deleted = True

            if metric_type is None or metric_type == MetricType.HISTOGRAM:
                if self._histograms.pop(name, None) is not None:
                    deleted = True

            if deleted:
                self._metadata.pop(name, None)

            return deleted

    def get_metrics(self) -> Dict[str, Any]:
        """
        Retrieve all metrics.

        Returns:
            Dictionary with counters, gauges, and histograms.
            All values are copies to prevent external mutation.
        """
        with self._lock:
            return {
                "counters": self._counters.copy(),
                "gauges": self._gauges.copy(),
                "histograms": {k: list(v) for k, v in self._histograms.items()},
            }

    def get_metadata(self, name: str) -> Optional[Dict[str, Union[str, float]]]:
        """
        Get metadata for a metric.

        Args:
            name: Metric name

        Returns:
            Metadata dict with type, created_at, updated_at, or None if not found
        """
        with self._lock:
            return self._metadata.get(name, {}).copy() if name in self._metadata else None

    def get_all_metadata(self) -> Dict[str, Dict[str, Union[str, float]]]:
        """
        Get metadata for all metrics.

        Returns:
            Dictionary mapping metric names to metadata
        """
        with self._lock:
            return {k: v.copy() for k, v in self._metadata.items()}

    def get_metric_count(self, metric_type: Optional[MetricType] = None) -> Union[int, Dict[str, int]]:
        """
        Get count of metrics.

        Args:
            metric_type: Optional metric type (returns all counts if None)

        Returns:
            Count of metrics, or dict with counts per type if metric_type is None
        """
        with self._lock:
            if metric_type is None:
                return {
                    "counters": len(self._counters),
                    "gauges": len(self._gauges),
                    "histograms": len(self._histograms),
                    "total": len(self._counters) + len(self._gauges) + len(self._histograms),
                }
            elif metric_type == MetricType.COUNTER:
                return len(self._counters)
            elif metric_type == MetricType.GAUGE:
                return len(self._gauges)
            elif metric_type == MetricType.HISTOGRAM:
                return len(self._histograms)
            return 0

    def reset(self, metric_type: Optional[MetricType] = None) -> None:
        """
        Reset metrics.

        Args:
            metric_type: Optional metric type to reset (resets all if None)
        """
        with self._lock:
            if metric_type is None or metric_type == MetricType.COUNTER:
                self._counters.clear()

            if metric_type is None or metric_type == MetricType.GAUGE:
                self._gauges.clear()

            if metric_type is None or metric_type == MetricType.HISTOGRAM:
                self._histograms.clear()

            if metric_type is None:
                self._metadata.clear()
            else:
                # Remove metadata for the specific type
                names_to_remove = []
                for name, meta in self._metadata.items():
                    if meta.get("type") == metric_type.value:
                        names_to_remove.append(name)
                for name in names_to_remove:
                    self._metadata.pop(name, None)

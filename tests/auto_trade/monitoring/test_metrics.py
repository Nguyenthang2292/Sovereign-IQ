"""
Tests for Metrics Collection System.

Tests metric types, collectors, validation, statistics, and thread safety.
"""

import math
import threading
import time
from collections import deque
from unittest.mock import MagicMock, Mock, patch

import pytest

from modules.auto_trade.monitoring.metrics import MetricType, MetricsCollector


class TestMetricTypeEnum:
    """Test MetricType enum."""

    def test_metric_types_exist(self):
        """Test that all metric types are defined."""
        assert MetricType.COUNTER.value == "COUNTER"
        assert MetricType.GAUGE.value == "GAUGE"
        assert MetricType.HISTOGRAM.value == "HISTOGRAM"

    def test_metric_type_is_string(self):
        """Test that MetricType values are strings."""
        assert isinstance(MetricType.COUNTER.value, str)
        assert isinstance(MetricType.GAUGE.value, str)
        assert isinstance(MetricType.HISTOGRAM.value, str)


class TestMetricsCollectorInitialization:
    """Test MetricsCollector initialization."""

    def test_init_default(self):
        """Test default initialization."""
        collector = MetricsCollector()
        assert collector._histogram_max_size == MetricsCollector.DEFAULT_HISTOGRAM_MAX_SIZE
        assert len(collector._counters) == 0
        assert len(collector._gauges) == 0
        assert len(collector._histograms) == 0

    def test_init_custom_histogram_size(self):
        """Test initialization with custom histogram size."""
        collector = MetricsCollector(histogram_max_size=500)
        assert collector._histogram_max_size == 500

    def test_init_invalid_histogram_size_zero(self):
        """Test that zero histogram size raises error."""
        with pytest.raises(ValueError, match="histogram_max_size must be positive"):
            MetricsCollector(histogram_max_size=0)

    def test_init_invalid_histogram_size_negative(self):
        """Test that negative histogram size raises error."""
        with pytest.raises(ValueError, match="histogram_max_size must be positive"):
            MetricsCollector(histogram_max_size=-1)


class TestCounterMetrics:
    """Test counter metric operations."""

    def test_increment_default(self):
        """Test increment with default value."""
        collector = MetricsCollector()
        collector.increment("requests")
        assert collector.get_counter("requests") == 1

    def test_increment_custom_value(self):
        """Test increment with custom value."""
        collector = MetricsCollector()
        collector.increment("requests", 5)
        assert collector.get_counter("requests") == 5

    def test_increment_multiple_times(self):
        """Test multiple increments accumulate."""
        collector = MetricsCollector()
        collector.increment("requests", 10)
        collector.increment("requests", 20)
        collector.increment("requests", 5)
        assert collector.get_counter("requests") == 35

    def test_increment_empty_name_raises_error(self):
        """Test that empty name raises ValueError."""
        collector = MetricsCollector()
        with pytest.raises(ValueError, match="Metric name must be a non-empty string"):
            collector.increment("")

    def test_increment_whitespace_name_raises_error(self):
        """Test that whitespace-only name raises ValueError."""
        collector = MetricsCollector()
        with pytest.raises(ValueError, match="Metric name cannot be whitespace only"):
            collector.increment("   ")

    def test_increment_negative_value_raises_error(self):
        """Test that negative value raises ValueError."""
        collector = MetricsCollector()
        with pytest.raises(ValueError, match="Increment value must be non-negative"):
            collector.increment("requests", -1)

    def test_increment_float_value_raises_error(self):
        """Test that float value raises ValueError."""
        collector = MetricsCollector()
        with pytest.raises(ValueError, match="Increment value must be an integer"):
            collector.increment("requests", 1.5)  # type: ignore[arg-type]

    def test_increment_invalid_name_format(self):
        """Test that invalid name format raises ValueError."""
        collector = MetricsCollector()
        with pytest.raises(ValueError, match="Metric name must start with a letter"):
            collector.increment("123invalid")

    def test_increment_thread_safety(self):
        """Test concurrent increments are thread-safe."""
        collector = MetricsCollector()

        def increment_many():
            for _ in range(100):
                collector.increment("concurrent_counter")

        threads = [threading.Thread(target=increment_many) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert collector.get_counter("concurrent_counter") == 1000


class TestGaugeMetrics:
    """Test gauge metric operations."""

    def test_gauge_set_value(self):
        """Test setting gauge value."""
        collector = MetricsCollector()
        collector.gauge("memory_usage", 75.5)
        assert collector.get_gauge("memory_usage") == 75.5

    def test_gauge_overwrite_value(self):
        """Test overwriting gauge value."""
        collector = MetricsCollector()
        collector.gauge("cpu_usage", 50.0)
        collector.gauge("cpu_usage", 75.0)
        assert collector.get_gauge("cpu_usage") == 75.0

    def test_gauge_empty_name_raises_error(self):
        """Test that empty name raises ValueError."""
        collector = MetricsCollector()
        with pytest.raises(ValueError, match="Metric name must be a non-empty string"):
            collector.gauge("", 50.0)

    def test_gauge_nan_value_raises_error(self):
        """Test that NaN value raises ValueError."""
        collector = MetricsCollector()
        with pytest.raises(ValueError, match="Gauge value must be a finite number"):
            collector.gauge("invalid_gauge", float('nan'))

    def test_gauge_inf_value_raises_error(self):
        """Test that Inf value raises ValueError."""
        collector = MetricsCollector()
        with pytest.raises(ValueError, match="Gauge value must be a finite number"):
            collector.gauge("invalid_gauge", float('inf'))

    def test_gauge_negative_inf_value_raises_error(self):
        """Test that negative Inf value raises ValueError."""
        collector = MetricsCollector()
        with pytest.raises(ValueError, match="Gauge value must be a finite number"):
            collector.gauge("invalid_gauge", float('-inf'))

    def test_gauge_integer_value(self):
        """Test gauge accepts integer values."""
        collector = MetricsCollector()
        collector.gauge("int_gauge", 100)
        assert collector.get_gauge("int_gauge") == 100.0

    def test_gauge_thread_safety(self):
        """Test concurrent gauge updates are thread-safe."""
        collector = MetricsCollector()

        def set_gauge(value):
            collector.gauge("concurrent_gauge", value)

        threads = [threading.Thread(target=set_gauge, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Final value should be one of the set values
        assert collector.get_gauge("concurrent_gauge") is not None


class TestHistogramMetrics:
    """Test histogram metric operations."""

    def test_histogram_add_single_value(self):
        """Test adding single value to histogram."""
        collector = MetricsCollector()
        collector.histogram("response_time", 150.5)
        values = collector.get_histogram("response_time")
        assert len(values) == 1
        assert values[0] == 150.5

    def test_histogram_add_multiple_values(self):
        """Test adding multiple values to histogram."""
        collector = MetricsCollector()
        collector.histogram("response_time", 100.0)
        collector.histogram("response_time", 200.0)
        collector.histogram("response_time", 150.0)
        values = collector.get_histogram("response_time")
        assert len(values) == 3
        assert 100.0 in values
        assert 200.0 in values
        assert 150.0 in values

    def test_histogram_size_limit(self):
        """Test histogram size limit enforcement."""
        collector = MetricsCollector(histogram_max_size=100)
        for i in range(150):
            collector.histogram("limited_histogram", float(i))

        values = collector.get_histogram("limited_histogram")
        assert len(values) == 100
        # Should keep the most recent values (50-149)
        assert values[0] == 50.0
        assert values[-1] == 149.0

    def test_histogram_empty_name_raises_error(self):
        """Test that empty name raises ValueError."""
        collector = MetricsCollector()
        with pytest.raises(ValueError, match="Metric name must be a non-empty string"):
            collector.histogram("", 100.0)

    def test_histogram_nan_value_raises_error(self):
        """Test that NaN value raises ValueError."""
        collector = MetricsCollector()
        with pytest.raises(ValueError, match="Histogram value must be a finite number"):
            collector.histogram("invalid_histogram", float('nan'))

    def test_histogram_inf_value_raises_error(self):
        """Test that Inf value raises ValueError."""
        collector = MetricsCollector()
        with pytest.raises(ValueError, match="Histogram value must be a finite number"):
            collector.histogram("invalid_histogram", float('inf'))

    def test_histogram_uses_deque(self):
        """Test that histogram uses deque internally."""
        collector = MetricsCollector()
        collector.histogram("test_deque", 1.0)
        assert isinstance(collector._histograms["test_deque"], deque)

    def test_histogram_thread_safety(self):
        """Test concurrent histogram updates are thread-safe."""
        collector = MetricsCollector()

        def add_values():
            for i in range(100):
                collector.histogram("concurrent_histogram", float(i))

        threads = [threading.Thread(target=add_values) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        values = collector.get_histogram("concurrent_histogram")
        assert len(values) == 500


class TestHistogramStatistics:
    """Test histogram statistics calculations."""

    def test_histogram_stats_basic(self):
        """Test basic histogram statistics."""
        collector = MetricsCollector()
        values = [10, 20, 30, 40, 50]
        for v in values:
            collector.histogram("test_stats", float(v))

        stats = collector.get_histogram_stats("test_stats")
        assert stats["min"] == 10.0
        assert stats["max"] == 50.0
        assert stats["avg"] == 30.0
        assert stats["count"] == 5

    def test_histogram_stats_percentiles(self):
        """Test histogram percentile calculations."""
        collector = MetricsCollector()
        values = list(range(1, 101))  # 1 to 100
        for v in values:
            collector.histogram("percentile_test", float(v))

        stats = collector.get_histogram_stats("percentile_test")
        # p50 should be around 50, allow small margin
        assert 49.0 <= stats["p50"] <= 51.0
        assert 94.0 <= stats["p95"] <= 96.0
        assert 98.0 <= stats["p99"] <= 100.0

    def test_histogram_stats_empty(self):
        """Test histogram statistics for non-existent histogram."""
        collector = MetricsCollector()
        stats = collector.get_histogram_stats("nonexistent")
        assert stats == {}

    def test_histogram_stats_empty_histogram(self):
        """Test histogram statistics for empty histogram."""
        collector = MetricsCollector()
        collector._histograms["empty"] = deque(maxlen=100)
        stats = collector.get_histogram_stats("empty")
        assert stats == {}

    def test_histogram_stats_single_value(self):
        """Test histogram statistics with single value."""
        collector = MetricsCollector()
        collector.histogram("single", 42.0)
        stats = collector.get_histogram_stats("single")
        assert stats["min"] == 42.0
        assert stats["max"] == 42.0
        assert stats["avg"] == 42.0
        assert stats["median"] == 42.0


class TestIndividualMetricRetrieval:
    """Test individual metric retrieval methods."""

    def test_get_counter_existing(self):
        """Test getting existing counter."""
        collector = MetricsCollector()
        collector.increment("test_counter", 10)
        assert collector.get_counter("test_counter") == 10

    def test_get_counter_nonexistent(self):
        """Test getting non-existent counter returns 0."""
        collector = MetricsCollector()
        assert collector.get_counter("nonexistent") == 0

    def test_get_gauge_existing(self):
        """Test getting existing gauge."""
        collector = MetricsCollector()
        collector.gauge("test_gauge", 75.5)
        assert collector.get_gauge("test_gauge") == 75.5

    def test_get_gauge_nonexistent(self):
        """Test getting non-existent gauge returns None."""
        collector = MetricsCollector()
        assert collector.get_gauge("nonexistent") is None

    def test_get_histogram_existing(self):
        """Test getting existing histogram."""
        collector = MetricsCollector()
        collector.histogram("test_histogram", 10.0)
        collector.histogram("test_histogram", 20.0)
        values = collector.get_histogram("test_histogram")
        assert len(values) == 2
        assert 10.0 in values
        assert 20.0 in values

    def test_get_histogram_nonexistent(self):
        """Test getting non-existent histogram returns empty list."""
        collector = MetricsCollector()
        assert collector.get_histogram("nonexistent") == []

    def test_get_histogram_returns_copy(self):
        """Test that get_histogram returns a copy."""
        collector = MetricsCollector()
        collector.histogram("test", 10.0)
        values1 = collector.get_histogram("test")
        values2 = collector.get_histogram("test")
        assert values1 is not values2
        assert values1 == values2


class TestMetricExistence:
    """Test metric existence checks."""

    def test_metric_exists_counter(self):
        """Test checking counter existence."""
        collector = MetricsCollector()
        collector.increment("test_counter")
        assert collector.metric_exists("test_counter", MetricType.COUNTER)
        assert not collector.metric_exists("nonexistent", MetricType.COUNTER)

    def test_metric_exists_gauge(self):
        """Test checking gauge existence."""
        collector = MetricsCollector()
        collector.gauge("test_gauge", 50.0)
        assert collector.metric_exists("test_gauge", MetricType.GAUGE)
        assert not collector.metric_exists("nonexistent", MetricType.GAUGE)

    def test_metric_exists_histogram(self):
        """Test checking histogram existence."""
        collector = MetricsCollector()
        collector.histogram("test_histogram", 10.0)
        assert collector.metric_exists("test_histogram", MetricType.HISTOGRAM)
        assert not collector.metric_exists("nonexistent", MetricType.HISTOGRAM)

    def test_metric_exists_any_type(self):
        """Test checking metric existence across all types."""
        collector = MetricsCollector()
        collector.increment("counter")
        collector.gauge("gauge", 50.0)
        collector.histogram("histogram", 10.0)

        assert collector.metric_exists("counter")
        assert collector.metric_exists("gauge")
        assert collector.metric_exists("histogram")
        assert not collector.metric_exists("nonexistent")


class TestMetricDeletion:
    """Test metric deletion functionality."""

    def test_delete_counter(self):
        """Test deleting counter."""
        collector = MetricsCollector()
        collector.increment("test_counter", 10)
        assert collector.delete_metric("test_counter", MetricType.COUNTER)
        assert collector.get_counter("test_counter") == 0

    def test_delete_gauge(self):
        """Test deleting gauge."""
        collector = MetricsCollector()
        collector.gauge("test_gauge", 50.0)
        assert collector.delete_metric("test_gauge", MetricType.GAUGE)
        assert collector.get_gauge("test_gauge") is None

    def test_delete_histogram(self):
        """Test deleting histogram."""
        collector = MetricsCollector()
        collector.histogram("test_histogram", 10.0)
        assert collector.delete_metric("test_histogram", MetricType.HISTOGRAM)
        assert collector.get_histogram("test_histogram") == []

    def test_delete_nonexistent(self):
        """Test deleting non-existent metric returns False."""
        collector = MetricsCollector()
        assert not collector.delete_metric("nonexistent", MetricType.COUNTER)

    def test_delete_all_types(self):
        """Test deleting metric from all types."""
        collector = MetricsCollector()
        # Create metric with same name in multiple types
        collector.increment("metric")
        collector.gauge("metric", 50.0)
        collector.histogram("metric", 10.0)

        # Delete from all types
        assert collector.delete_metric("metric")
        assert not collector.metric_exists("metric")


class TestGetMetrics:
    """Test get_metrics functionality."""

    def test_get_metrics_empty(self):
        """Test getting metrics when empty."""
        collector = MetricsCollector()
        metrics = collector.get_metrics()
        assert metrics["counters"] == {}
        assert metrics["gauges"] == {}
        assert metrics["histograms"] == {}

    def test_get_metrics_all_types(self):
        """Test getting all metric types."""
        collector = MetricsCollector()
        collector.increment("counter", 10)
        collector.gauge("gauge", 50.0)
        collector.histogram("histogram", 100.0)

        metrics = collector.get_metrics()
        assert metrics["counters"]["counter"] == 10
        assert metrics["gauges"]["gauge"] == 50.0
        assert "histogram" in metrics["histograms"]

    def test_get_metrics_returns_copies(self):
        """Test that get_metrics returns copies."""
        collector = MetricsCollector()
        collector.increment("counter", 10)

        metrics1 = collector.get_metrics()
        metrics2 = collector.get_metrics()

        # Should be different dict objects
        assert metrics1 is not metrics2
        assert metrics1["counters"] is not metrics2["counters"]


class TestMetricMetadata:
    """Test metric metadata functionality."""

    def test_metadata_created_on_first_update(self):
        """Test metadata is created on first metric update."""
        collector = MetricsCollector()
        collector.increment("test_counter")

        metadata = collector.get_metadata("test_counter")
        assert metadata is not None
        assert metadata["type"] == "COUNTER"
        assert "created_at" in metadata
        assert "updated_at" in metadata

    def test_metadata_updated_on_subsequent_updates(self):
        """Test metadata is updated on subsequent updates."""
        collector = MetricsCollector()
        collector.increment("test_counter")

        metadata1 = collector.get_metadata("test_counter")
        time.sleep(0.01)  # Small delay
        collector.increment("test_counter")
        metadata2 = collector.get_metadata("test_counter")
        assert metadata1 is not None and metadata2 is not None
        assert metadata2["updated_at"] > metadata1["updated_at"]  # type: ignore[operator]

    def test_metadata_nonexistent(self):
        """Test getting metadata for non-existent metric."""
        collector = MetricsCollector()
        assert collector.get_metadata("nonexistent") is None

    def test_get_all_metadata(self):
        """Test getting metadata for all metrics."""
        collector = MetricsCollector()
        collector.increment("counter")
        collector.gauge("gauge", 50.0)

        all_metadata = collector.get_all_metadata()
        assert all_metadata is not None
        assert "counter" in all_metadata
        assert "gauge" in all_metadata
        assert all_metadata["counter"]["type"] == "COUNTER"
        assert all_metadata["gauge"]["type"] == "GAUGE"


class TestMetricCount:
    """Test metric count functionality."""

    def test_get_metric_count_all(self):
        """Test getting count of all metrics."""
        collector = MetricsCollector()
        collector.increment("c1")
        collector.increment("c2")
        collector.gauge("g1", 50.0)
        collector.histogram("h1", 10.0)

        counts = collector.get_metric_count()
        assert isinstance(counts, dict)
        assert counts["counters"] == 2
        assert counts["gauges"] == 1
        assert counts["histograms"] == 1
        assert counts["total"] == 4

    def test_get_metric_count_by_type(self):
        """Test getting count by metric type."""
        collector = MetricsCollector()
        collector.increment("c1")
        collector.increment("c2")
        collector.gauge("g1", 50.0)

        assert collector.get_metric_count(MetricType.COUNTER) == 2
        assert collector.get_metric_count(MetricType.GAUGE) == 1
        assert collector.get_metric_count(MetricType.HISTOGRAM) == 0


class TestReset:
    """Test reset functionality."""

    def test_reset_all(self):
        """Test resetting all metrics."""
        collector = MetricsCollector()
        collector.increment("counter", 10)
        collector.gauge("gauge", 50.0)
        collector.histogram("histogram", 100.0)

        collector.reset()

        assert collector.get_counter("counter") == 0
        assert collector.get_gauge("gauge") is None
        assert collector.get_histogram("histogram") == []

    def test_reset_counters_only(self):
        """Test resetting counters only."""
        collector = MetricsCollector()
        collector.increment("counter", 10)
        collector.gauge("gauge", 50.0)

        collector.reset(MetricType.COUNTER)

        assert collector.get_counter("counter") == 0
        assert collector.get_gauge("gauge") == 50.0

    def test_reset_gauges_only(self):
        """Test resetting gauges only."""
        collector = MetricsCollector()
        collector.increment("counter", 10)
        collector.gauge("gauge", 50.0)

        collector.reset(MetricType.GAUGE)

        assert collector.get_counter("counter") == 10
        assert collector.get_gauge("gauge") is None

    def test_reset_histograms_only(self):
        """Test resetting histograms only."""
        collector = MetricsCollector()
        collector.increment("counter", 10)
        collector.histogram("histogram", 100.0)

        collector.reset(MetricType.HISTOGRAM)

        assert collector.get_counter("counter") == 10
        assert collector.get_histogram("histogram") == []

    def test_reset_is_idempotent(self):
        """Test that reset can be called multiple times."""
        collector = MetricsCollector()
        collector.increment("counter", 10)

        collector.reset()
        collector.reset()  # Should not raise errors

        assert collector.get_counter("counter") == 0


class TestMetricNameValidation:
    """Test metric name validation."""

    def test_valid_name_simple(self):
        """Test simple valid metric name."""
        collector = MetricsCollector()
        collector.increment("valid_metric")
        assert collector.metric_exists("valid_metric")

    def test_valid_name_with_dots(self):
        """Test valid metric name with dots."""
        collector = MetricsCollector()
        collector.increment("app.requests.total")
        assert collector.metric_exists("app.requests.total")

    def test_valid_name_with_hyphens(self):
        """Test valid metric name with hyphens."""
        collector = MetricsCollector()
        collector.increment("http-requests-total")
        assert collector.metric_exists("http-requests-total")

    def test_invalid_name_starts_with_number(self):
        """Test that name starting with number is invalid."""
        collector = MetricsCollector()
        with pytest.raises(ValueError, match="Metric name must start with a letter"):
            collector.increment("123metric")

    def test_invalid_name_special_characters(self):
        """Test that name with special characters is invalid."""
        collector = MetricsCollector()
        with pytest.raises(ValueError, match="Metric name must start with a letter"):
            collector.increment("metric@invalid")

    def test_invalid_name_too_long(self):
        """Test that overly long name is invalid."""
        collector = MetricsCollector()
        long_name = "a" * 256
        with pytest.raises(ValueError, match="Metric name exceeds maximum length"):
            collector.increment(long_name)


class TestMaxMetricsLimit:
    """Test maximum metrics limit enforcement."""

    def test_max_counters_limit(self):
        """Test maximum counters limit."""
        collector = MetricsCollector()
        collector.MAX_METRICS_PER_TYPE = 10

        # Add up to limit
        for i in range(10):
            collector.increment(f"counter{i}")

        # Adding one more should raise error
        with pytest.raises(RuntimeError, match="Maximum number of counters"):
            collector.increment("counter_overflow")

    def test_max_gauges_limit(self):
        """Test maximum gauges limit."""
        collector = MetricsCollector()
        collector.MAX_METRICS_PER_TYPE = 10

        # Add up to limit
        for i in range(10):
            collector.gauge(f"gauge{i}", float(i))

        # Adding one more should raise error
        with pytest.raises(RuntimeError, match="Maximum number of gauges"):
            collector.gauge("gauge_overflow", 0.0)

    def test_max_histograms_limit(self):
        """Test maximum histograms limit."""
        collector = MetricsCollector()
        collector.MAX_METRICS_PER_TYPE = 10

        # Add up to limit
        for i in range(10):
            collector.histogram(f"histogram{i}", float(i))

        # Adding one more should raise error
        with pytest.raises(RuntimeError, match="Maximum number of histograms"):
            collector.histogram("histogram_overflow", 0.0)


class TestIntegration:
    """Integration tests for MetricsCollector."""

    def test_full_workflow(self):
        """Test complete metrics workflow."""
        collector = MetricsCollector()

        # Add metrics
        collector.increment("requests", 100)
        collector.gauge("memory_usage", 75.5)
        collector.histogram("response_time", 150.0)
        collector.histogram("response_time", 200.0)

        # Verify metrics
        assert collector.get_counter("requests") == 100
        assert collector.get_gauge("memory_usage") == 75.5
        assert len(collector.get_histogram("response_time")) == 2

        # Get all metrics
        metrics = collector.get_metrics()
        assert "requests" in metrics["counters"]
        assert "memory_usage" in metrics["gauges"]
        assert "response_time" in metrics["histograms"]

        # Reset
        collector.reset()
        counts = collector.get_metric_count()
        assert isinstance(counts, dict) and counts["total"] == 0

    def test_concurrent_operations(self):
        """Test concurrent metric operations."""
        collector = MetricsCollector()

        def worker(thread_id):
            for i in range(50):
                collector.increment(f"thread{thread_id}_counter")
                collector.gauge(f"thread{thread_id}_gauge", float(i))
                collector.histogram(f"thread{thread_id}_histogram", float(i))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify all metrics were created
        counts = collector.get_metric_count()
        assert isinstance(counts, dict)
        assert counts["counters"] == 5
        assert counts["gauges"] == 5
        assert counts["histograms"] == 5

    def test_mixed_operations(self):
        """Test mixed metric operations."""
        collector = MetricsCollector()

        # Create metrics
        for i in range(5):
            collector.increment(f"counter{i}")
            collector.gauge(f"gauge{i}", float(i))
            collector.histogram(f"histogram{i}", float(i))

        # Delete some
        collector.delete_metric("counter0", MetricType.COUNTER)
        collector.delete_metric("gauge1", MetricType.GAUGE)

        # Verify counts
        assert collector.get_metric_count(MetricType.COUNTER) == 4
        assert collector.get_metric_count(MetricType.GAUGE) == 4
        assert collector.get_metric_count(MetricType.HISTOGRAM) == 5

        # Reset specific type
        collector.reset(MetricType.HISTOGRAM)
        assert collector.get_metric_count(MetricType.HISTOGRAM) == 0
        assert collector.get_metric_count(MetricType.COUNTER) == 4

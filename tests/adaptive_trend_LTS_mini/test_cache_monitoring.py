"""
Tests for Cache Monitoring System

Tests the cache monitoring features including metrics collection,
dashboard display, API endpoints, and performance insights.
"""

import json
import os
import tempfile
import time

import numpy as np
import pandas as pd
import pytest

from modules.adaptive_trend_LTS_mini.utils.cache_manager import (
    CacheManager,
    reset_cache_manager,
)
from modules.adaptive_trend_LTS_mini.utils.cache_metrics_api import (
    CacheMetricsService,
)
from modules.adaptive_trend_LTS_mini.utils.cache_monitor import (
    CacheMonitor,
)


@pytest.fixture
def cache_manager():
    """Create a fresh cache manager for testing."""
    reset_cache_manager()
    cache = CacheManager(
        max_entries_l1=10,
        max_entries_l2=20,
        max_size_mb_l2=10.0,
        ttl_seconds=3600.0,
        cache_dir=tempfile.mkdtemp(),
    )
    yield cache
    cache.clear()
    reset_cache_manager()


@pytest.fixture
def cache_monitor(cache_manager):
    """Create a cache monitor for testing (uses fixture cache so metrics match)."""
    return CacheMonitor(history_size=50, cache=cache_manager)


@pytest.fixture
def metrics_service(cache_manager):
    """Create a metrics service for testing (uses fixture cache so health/metrics match)."""
    return CacheMetricsService(cache=cache_manager)


@pytest.fixture
def sample_data():
    """Create sample price data for cache operations."""
    np.random.seed(42)
    return pd.Series(np.random.randn(100) * 10 + 100, name="close")


class TestCacheManagerMetrics:
    """Test CacheManager metrics collection."""

    def test_initial_stats(self, cache_manager):
        """Test initial cache statistics are correct."""
        stats = cache_manager.get_stats()

        assert stats["entries"] == 0
        assert stats["entries_l1"] == 0
        assert stats["entries_l2"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["total_requests"] == 0
        assert stats["hit_rate_percent"] == 0.0
        assert stats["evictions"] == 0
        assert stats["promotions"] == 0

    def test_hit_tracking(self, cache_manager, sample_data):
        """Test cache hit tracking."""
        # Store in cache
        cache_manager.put("EMA", 20, sample_data, sample_data.rolling(20).mean())

        # First access - should hit
        result = cache_manager.get("EMA", 20, sample_data)
        assert result is not None

        stats = cache_manager.get_stats()
        assert stats["hits"] == 1
        assert stats["hits_l1"] == 1
        assert stats["misses"] == 0
        assert stats["hit_rate_percent"] == 100.0

    def test_miss_tracking(self, cache_manager, sample_data):
        """Test cache miss tracking."""
        # Try to get non-existent entry
        result = cache_manager.get("EMA", 20, sample_data)
        assert result is None

        stats = cache_manager.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 1
        assert stats["hit_rate_percent"] == 0.0

    def test_l2_hit_tracking(self, cache_manager, sample_data):
        """Test L2 cache hit tracking and promotions."""
        # Fill L1 cache
        for i in range(cache_manager.max_entries_l1 + 5):
            data = sample_data + i
            cache_manager.put("EMA", 20 + i, data, data.rolling(20).mean())

        # Get an entry that should be in L2
        first_data = sample_data
        result = cache_manager.get("EMA", 20, first_data)

        stats = cache_manager.get_stats()
        # Should have L2 hit and promotion
        assert stats["hits_l2"] > 0
        assert stats["promotions"] > 0

    def test_eviction_tracking(self, cache_manager, sample_data):
        """Test eviction tracking."""
        # Fill cache beyond capacity
        entries_to_add = cache_manager.max_entries_l2 + 10

        for i in range(entries_to_add):
            data = sample_data + i
            cache_manager.put("EMA", 20 + i, data, data.rolling(20).mean())

        stats = cache_manager.get_stats()
        # Should have evictions
        assert stats["evictions"] > 0
        # L2 should not exceed max
        assert stats["entries_l2"] <= cache_manager.max_entries_l2

    def test_recent_hit_rate_tracking(self, cache_manager, sample_data):
        """Test recent hit rate calculation."""
        # Add some entries
        for i in range(5):
            data = sample_data + i
            cache_manager.put("EMA", 20 + i, data, data.rolling(20).mean())

        # Generate hits
        for i in range(5):
            data = sample_data + i
            cache_manager.get("EMA", 20 + i, data)

        stats = cache_manager.get_stats()
        # Recent hit rate should be 100% (all hits in last 60s)
        assert stats["recent_hit_rate_percent"] == 100.0
        assert stats["recent_hits"] == 5
        assert stats["recent_misses"] == 0

    def test_detailed_metrics_with_insights(self, cache_manager, sample_data):
        """Test detailed metrics generation with insights."""
        # Generate low hit rate scenario
        for i in range(10):
            # Miss
            cache_manager.get("EMA", 20 + i, sample_data + i)

        # Add some hits
        for i in range(3):
            data = sample_data + i
            cache_manager.put("EMA", 20 + i, data, data.rolling(20).mean())
            cache_manager.get("EMA", 20 + i, data)

        metrics = cache_manager.get_detailed_metrics()

        # Should have insights
        assert "insights" in metrics
        # Low hit rate should trigger insight
        if metrics["hit_rate_percent"] < 50:
            assert any("LOW_HIT_RATE" in insight for insight in metrics["insights"])

    def test_metrics_log_interval(self, cache_manager):
        """Test metrics logging interval configuration."""
        # Set interval
        cache_manager.set_metrics_log_interval(30)
        assert cache_manager._metrics_log_interval == 30

        # Disable
        cache_manager.set_metrics_log_interval(0)
        assert cache_manager._metrics_log_interval == 0


class TestCacheMonitor:
    """Test CacheMonitor functionality."""

    def test_snapshot_creation(self, cache_monitor, cache_manager, sample_data):
        """Test snapshot creation."""
        # Add some cache activity
        cache_manager.put("EMA", 20, sample_data, sample_data.rolling(20).mean())
        cache_manager.get("EMA", 20, sample_data)

        snapshot = cache_monitor.take_snapshot()

        assert "timestamp" in snapshot
        assert "datetime" in snapshot
        assert "hit_rate_percent" in snapshot
        assert "total_requests" in snapshot
        assert len(cache_monitor.history) == 1

    def test_history_limit(self, cache_monitor):
        """Test history size limit."""
        max_size = cache_monitor.history_size

        # Take more snapshots than limit
        for i in range(max_size + 10):
            cache_monitor.take_snapshot()

        # Should not exceed limit
        assert len(cache_monitor.history) == max_size

    def test_trend_analysis(self, cache_monitor, cache_manager, sample_data):
        """Test trend analysis with multiple snapshots."""
        # Generate activity with improving hit rate
        for round_num in range(5):
            # Add entries
            for i in range(3):
                data = sample_data + i
                cache_manager.put("EMA", 20 + i, data, data.rolling(20).mean())

            # Generate more hits than misses
            for i in range(3):
                data = sample_data + i
                cache_manager.get("EMA", 20 + i, data)

            # Take snapshot
            cache_monitor.take_snapshot()

        trends = cache_monitor.get_trends()

        assert "hit_rate" in trends
        assert "current" in trends["hit_rate"]
        assert "min" in trends["hit_rate"]
        assert "max" in trends["hit_rate"]
        assert "avg" in trends["hit_rate"]
        assert "trend" in trends["hit_rate"]
        assert trends["hit_rate"]["trend"] in ["increasing", "decreasing"]
        assert "evictions" in trends
        assert "samples" in trends

    def test_trend_insufficient_history(self, cache_monitor):
        """Test trend analysis with insufficient history."""
        # Only one snapshot
        cache_monitor.take_snapshot()

        trends = cache_monitor.get_trends()
        assert "error" in trends

    def test_export_metrics(self, cache_monitor, cache_manager, sample_data):
        """Test metrics export to JSON."""
        # Generate some activity
        cache_manager.put("EMA", 20, sample_data, sample_data.rolling(20).mean())
        cache_manager.get("EMA", 20, sample_data)

        # Export to temp file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            filepath = f.name

        try:
            cache_monitor.export_metrics(filepath)

            # Verify file exists and contains valid JSON
            assert os.path.exists(filepath)
            with open(filepath, "r") as f:
                data = json.load(f)

            assert "timestamp" in data
            assert "hit_rate_percent" in data
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_export_history(self, cache_monitor):
        """Test history export to JSON."""
        # Take multiple snapshots
        for _ in range(5):
            cache_monitor.take_snapshot()
            time.sleep(0.01)

        # Export to temp file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            filepath = f.name

        try:
            cache_monitor.export_history(filepath)

            # Verify file exists and contains valid JSON
            assert os.path.exists(filepath)
            with open(filepath, "r") as f:
                data = json.load(f)

            assert isinstance(data, list)
            assert len(data) == 5
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_summary_report(self, cache_monitor, cache_manager, sample_data):
        """Test summary report generation."""
        # Generate activity
        cache_manager.put("EMA", 20, sample_data, sample_data.rolling(20).mean())
        cache_manager.get("EMA", 20, sample_data)
        cache_monitor.take_snapshot()

        report = cache_monitor.get_summary_report()

        assert isinstance(report, str)
        assert "CACHE PERFORMANCE SUMMARY" in report
        assert "Hit Rate:" in report
        assert "Total Requests:" in report

    def test_dashboard_display(self, cache_monitor, cache_manager, sample_data, capsys):
        """Test dashboard display output."""
        # Generate activity
        cache_manager.put("EMA", 20, sample_data, sample_data.rolling(20).mean())
        cache_manager.get("EMA", 20, sample_data)

        # Display dashboard
        cache_monitor.display_dashboard(show_trends=False)

        # Capture output
        captured = capsys.readouterr()

        assert "CACHE PERFORMANCE DASHBOARD" in captured.out
        assert "CACHE SIZE:" in captured.out
        assert "HIT/MISS STATISTICS:" in captured.out


class TestCacheMetricsService:
    """Test CacheMetricsService functionality."""

    def test_get_current_metrics(self, metrics_service, cache_manager, sample_data):
        """Test getting current metrics."""
        # Generate activity
        cache_manager.put("EMA", 20, sample_data, sample_data.rolling(20).mean())
        cache_manager.get("EMA", 20, sample_data)

        metrics = metrics_service.get_current_metrics()

        assert "timestamp" in metrics
        assert "metrics" in metrics
        assert "hit_rate_percent" in metrics["metrics"]

    def test_health_status_healthy(self, metrics_service, cache_manager, sample_data):
        """Test health status when cache is performing well."""
        # Generate good hit rate
        for i in range(5):
            data = sample_data + i
            cache_manager.put("EMA", 20 + i, data, data.rolling(20).mean())

        for i in range(5):
            data = sample_data + i
            cache_manager.get("EMA", 20 + i, data)

        health = metrics_service.get_health_status()

        assert "status" in health
        assert "warnings" in health
        # Should be healthy with 100% hit rate
        assert health["status"] == "healthy"

    def test_health_status_degraded(self, metrics_service, cache_manager, sample_data):
        """Test health status when cache performance is degraded."""
        # Generate low hit rate (many misses)
        for i in range(20):
            cache_manager.get("EMA", 20 + i, sample_data + i)

        # Few hits
        for i in range(3):
            data = sample_data + i
            cache_manager.put("EMA", 20 + i, data, data.rolling(20).mean())
            cache_manager.get("EMA", 20 + i, data)

        health = metrics_service.get_health_status()

        # Should be degraded or critical
        assert health["status"] in ["degraded", "critical"]
        assert len(health["warnings"]) > 0

    def test_performance_summary(self, metrics_service, cache_manager, sample_data):
        """Test performance summary generation."""
        # Generate activity
        cache_manager.put("EMA", 20, sample_data, sample_data.rolling(20).mean())
        cache_manager.get("EMA", 20, sample_data)

        summary = metrics_service.get_performance_summary()

        assert "hit_rate" in summary
        assert "overall" in summary["hit_rate"]
        assert "l1" in summary["hit_rate"]
        assert "l2" in summary["hit_rate"]
        assert "recent" in summary["hit_rate"]
        assert "requests" in summary
        assert "cache_size" in summary
        assert "operations" in summary

    def test_snapshot_recording(self, metrics_service, cache_manager, sample_data):
        """Test snapshot recording."""
        # Generate activity
        cache_manager.put("EMA", 20, sample_data, sample_data.rolling(20).mean())

        # Record snapshot
        snapshot = metrics_service.record_snapshot()

        assert "timestamp" in snapshot
        assert "metrics" in snapshot
        assert len(metrics_service.snapshots) == 1

    def test_get_snapshots(self, metrics_service):
        """Test retrieving snapshots."""
        # Record multiple snapshots
        for _ in range(10):
            metrics_service.record_snapshot()
            time.sleep(0.01)

        # Get all
        all_snapshots = metrics_service.get_snapshots()
        assert len(all_snapshots) == 10

        # Get limited
        limited_snapshots = metrics_service.get_snapshots(limit=5)
        assert len(limited_snapshots) == 5

    def test_snapshot_limit(self, metrics_service):
        """Test snapshot storage limit."""
        max_snapshots = metrics_service.max_snapshots

        # Record more than limit
        for _ in range(max_snapshots + 10):
            metrics_service.record_snapshot()

        # Should not exceed limit
        assert len(metrics_service.snapshots) == max_snapshots


class TestCacheMetricsIntegration:
    """Integration tests for cache monitoring system."""

    def test_end_to_end_monitoring(self, cache_manager, cache_monitor, sample_data):
        """Test end-to-end monitoring workflow."""
        # Simulate realistic cache usage
        entries = []
        for i in range(15):
            data = sample_data + i
            ma_result = data.rolling(20).mean()
            cache_manager.put("EMA", 20 + i, data, ma_result)
            entries.append(("EMA", 20 + i, data))

        # Mix of hits and misses
        for i in range(10):
            if i < 5:
                # Hits
                ma_type, length, data = entries[i]
                result = cache_manager.get(ma_type, length, data)
                assert result is not None
            else:
                # Misses
                result = cache_manager.get("EMA", 100 + i, sample_data + 100)
                assert result is None

        # Take snapshot
        snapshot = cache_monitor.take_snapshot()

        # Verify metrics
        assert snapshot["total_requests"] == 10
        assert snapshot["hits"] == 5
        assert snapshot["misses"] == 5
        assert snapshot["hit_rate_percent"] == 50.0

    def test_performance_insights_generation(self, cache_manager, sample_data):
        """Test automatic performance insights generation."""
        # Scenario 1: Low hit rate
        for i in range(20):
            cache_manager.get("EMA", 20 + i, sample_data + i)

        metrics = cache_manager.get_detailed_metrics()
        assert any("LOW_HIT_RATE" in insight for insight in metrics["insights"])

        # Reset
        cache_manager.clear()

        # Scenario 2: High eviction
        for i in range(cache_manager.max_entries_l2 + 50):
            data = sample_data + i
            cache_manager.put("EMA", 20 + i, data, data.rolling(20).mean())

        metrics = cache_manager.get_detailed_metrics()
        # Should have high evictions
        assert metrics["evictions"] > 0

    def test_continuous_monitoring_duration(self, cache_manager, sample_data):
        """Test continuous monitoring with duration limit."""
        # Generate background activity
        def generate_activity():
            for i in range(3):
                data = sample_data + i
                cache_manager.put("EMA", 20 + i, data, data.rolling(20).mean())
                cache_manager.get("EMA", 20 + i, data)

        generate_activity()

        # This would normally run continuously, but we test with very short duration
        # We can't easily test the full function, but we can verify it doesn't crash
        # Note: In real test, we'd use threading or mock time.sleep


class TestCacheMetricsThreadSafety:
    """Test thread safety of cache monitoring."""

    def test_concurrent_metric_access(self, cache_manager, sample_data):
        """Test concurrent access to metrics is thread-safe."""
        import threading

        errors = []

        def worker():
            try:
                for i in range(10):
                    # Put
                    data = sample_data + i
                    cache_manager.put("EMA", 20 + i, data, data.rolling(20).mean())
                    # Get
                    cache_manager.get("EMA", 20 + i, data)
                    # Get stats
                    stats = cache_manager.get_stats()
                    assert isinstance(stats, dict)
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(3):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0

    def test_concurrent_snapshot_recording(self, metrics_service):
        """Test concurrent snapshot recording is thread-safe."""
        import threading

        errors = []

        def worker():
            try:
                for _ in range(5):
                    metrics_service.record_snapshot()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(3):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0
        # Should have recorded snapshots (exact count may vary due to race)
        assert len(metrics_service.snapshots) > 0


class TestCacheMetricsAPI:
    """Test cache metrics API integration."""

    def test_fastapi_router_creation(self):
        """Test FastAPI router creation."""
        pytest.importorskip("fastapi")

        from modules.adaptive_trend_LTS_mini.utils.cache_metrics_api import (
            create_cache_metrics_router,
        )

        router = create_cache_metrics_router()
        assert router is not None
        assert router.prefix == "/cache"

    def test_api_endpoints_registered(self):
        """Test that all expected endpoints are registered."""
        pytest.importorskip("fastapi")

        from modules.adaptive_trend_LTS_mini.utils.cache_metrics_api import (
            create_cache_metrics_router,
        )

        router = create_cache_metrics_router()

        # Collect paths (FastAPI may use .path or .path_format; paths may have prefix)
        route_paths = []
        for route in router.routes:
            p = getattr(route, "path", None) or getattr(route, "path_format", None)
            if p is not None:
                route_paths.append(p)
        all_paths_str = " ".join(route_paths)
        assert "metrics" in all_paths_str
        assert "health" in all_paths_str
        assert "summary" in all_paths_str
        assert "snapshots" in all_paths_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

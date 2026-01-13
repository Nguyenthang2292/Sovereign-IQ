"""
Tests for Memory Utilities (web/utils/memory_utils.py).

Tests cover:
- Memory usage statistics
- Garbage collection
- Memory tracking with tracemalloc
- Memory snapshots comparison
- Memory logging
- Memory optimization
"""

import gc
from unittest.mock import MagicMock, Mock, patch

from web.utils.memory_utils import (
    force_garbage_collection,
    get_memory_snapshot_diff,
    get_memory_usage,
    log_memory_usage,
    optimize_memory,
    start_memory_tracking,
)


class TestGetMemoryUsage:
    """Test get_memory_usage function."""

    def test_get_memory_usage_with_psutil(self):
        """Test getting memory usage when psutil is available."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value = Mock(rss=100 * 1024 * 1024, vms=200 * 1024 * 1024)
        mock_process.memory_percent.return_value = 25.5

        mock_virtual_memory = Mock()
        mock_virtual_memory.available = 500 * 1024 * 1024
        mock_virtual_memory.total = 2000 * 1024 * 1024

        # Patch builtins.__import__ to mock psutil import inside the function
        def mock_import(name, *args, **kwargs):
            if name == "psutil":
                mock_psutil = MagicMock()
                mock_psutil.Process.return_value = mock_process
                mock_psutil.virtual_memory.return_value = mock_virtual_memory
                return mock_psutil
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            mem_info = get_memory_usage()

            assert mem_info["rss_mb"] == 100.0
            assert mem_info["vms_mb"] == 200.0
            assert mem_info["percent"] == 25.5
            assert mem_info["available_mb"] == 500.0
            assert mem_info["total_mb"] == 2000.0

    def test_get_memory_usage_without_psutil(self):
        """Test getting memory usage when psutil is not available."""

        # Patch builtins.__import__ to raise ImportError for psutil
        def mock_import(name, *args, **kwargs):
            if name == "psutil":
                raise ImportError("No module named 'psutil'")
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            mem_info = get_memory_usage()

            assert mem_info["rss_mb"] is None
            assert mem_info["vms_mb"] is None
            assert mem_info["percent"] is None
            assert mem_info["available_mb"] is None
            assert mem_info["total_mb"] is None


class TestForceGarbageCollection:
    """Test force_garbage_collection function."""

    def test_force_garbage_collection(self):
        """Test forcing garbage collection."""
        # Get initial counts
        counts_before = gc.get_count()

        # Force collection
        stats = force_garbage_collection(verbose=False)

        # Verify stats structure
        assert "collected" in stats
        assert "counts_before" in stats
        assert "counts_after" in stats
        assert isinstance(stats["collected"], int)
        assert isinstance(stats["counts_before"], tuple)
        assert isinstance(stats["counts_after"], tuple)
        # Optionally verify counts_before in stats matches earlier result
        assert stats["counts_before"] == counts_before

    def test_force_garbage_collection_verbose(self):
        """Test forcing garbage collection with verbose logging."""
        # log_info uses print(), not Python logging module, so caplog won't capture it
        # Just verify the function runs without error
        stats = force_garbage_collection(verbose=True)

        assert "collected" in stats
        assert "counts_before" in stats
        assert "counts_after" in stats


class TestStartMemoryTracking:
    """Test start_memory_tracking function."""

    def test_start_memory_tracking_success(self):
        """Test starting memory tracking when tracemalloc is available."""
        with patch("web.utils.memory_utils.tracemalloc") as mock_tracemalloc:
            mock_tracemalloc.is_tracing.return_value = False
            mock_snapshot = MagicMock()
            mock_tracemalloc.take_snapshot.return_value = mock_snapshot

            snapshot = start_memory_tracking()

            assert snapshot is not None
            mock_tracemalloc.start.assert_called_once()
            mock_tracemalloc.take_snapshot.assert_called_once()

    def test_start_memory_tracking_already_tracing(self):
        """Test starting memory tracking when already tracing."""
        with patch("web.utils.memory_utils.tracemalloc") as mock_tracemalloc:
            mock_tracemalloc.is_tracing.return_value = True
            mock_snapshot = MagicMock()
            mock_tracemalloc.take_snapshot.return_value = mock_snapshot

            snapshot = start_memory_tracking()

            assert snapshot is not None
            # Should not call start if already tracing
            mock_tracemalloc.start.assert_not_called()
            mock_tracemalloc.take_snapshot.assert_called_once()

    def test_start_memory_tracking_error(self):
        """Test starting memory tracking when tracemalloc raises error."""
        with patch("web.utils.memory_utils.tracemalloc") as mock_tracemalloc:
            mock_tracemalloc.is_tracing.side_effect = Exception("Tracemalloc error")

            snapshot = start_memory_tracking()

            assert snapshot is None

    def test_start_memory_tracking_not_available(self):
        """Test starting memory tracking when tracemalloc is not available."""
        # Mock tracemalloc to raise exception when accessed
        mock_tracemalloc = MagicMock()
        mock_tracemalloc.is_tracing.side_effect = AttributeError("No tracemalloc")

        with patch("web.utils.memory_utils.tracemalloc", mock_tracemalloc):
            snapshot = start_memory_tracking()

            assert snapshot is None


class TestGetMemorySnapshotDiff:
    """Test get_memory_snapshot_diff function."""

    def test_get_memory_snapshot_diff_success(self):
        """Test comparing memory snapshots successfully."""
        # Create mock snapshots
        mock_snapshot1 = MagicMock()
        mock_snapshot2 = MagicMock()

        # Create mock stat objects
        mock_stat1 = MagicMock()
        mock_stat1.size_diff = 10 * 1024 * 1024  # 10 MB
        mock_stat1.size = 20 * 1024 * 1024  # 20 MB
        mock_stat1.count_diff = 100
        mock_stat1.count = 200
        mock_stat1.traceback = None

        mock_stat2 = MagicMock()
        mock_stat2.size_diff = 5 * 1024 * 1024  # 5 MB
        mock_stat2.size = 15 * 1024 * 1024  # 15 MB
        mock_stat2.count_diff = 50
        mock_stat2.count = 150
        mock_stat2.traceback = None

        # Mock compare_to to return list of stats
        mock_snapshot2.compare_to = MagicMock(return_value=[mock_stat1, mock_stat2])

        diff = get_memory_snapshot_diff(mock_snapshot1, mock_snapshot2, top_n=2)

        assert diff is not None
        assert "top_consumers" in diff
        assert "total_size_diff_mb" in diff
        assert len(diff["top_consumers"]) == 2
        assert diff["top_consumers"][0]["rank"] == 1
        assert diff["top_consumers"][0]["size_diff_mb"] == 10.0
        assert diff["top_consumers"][1]["rank"] == 2
        assert diff["top_consumers"][1]["size_diff_mb"] == 5.0
        assert diff["total_size_diff_mb"] == 15.0  # 10 + 5 MB

    def test_get_memory_snapshot_diff_with_traceback(self):
        """Test comparing memory snapshots with traceback."""
        mock_snapshot1 = MagicMock()
        mock_snapshot2 = MagicMock()

        mock_stat = MagicMock()
        mock_stat.size_diff = 10 * 1024 * 1024
        mock_stat.size = 20 * 1024 * 1024
        mock_stat.count_diff = 100
        mock_stat.count = 200
        mock_traceback = MagicMock()
        mock_traceback.format.return_value = ["line1", "line2", "line3"]
        mock_stat.traceback = mock_traceback

        mock_snapshot2.compare_to = MagicMock(return_value=[mock_stat])

        diff = get_memory_snapshot_diff(mock_snapshot1, mock_snapshot2, top_n=1)

        assert diff is not None
        assert diff["top_consumers"][0]["traceback"] == ["line1", "line2", "line3"]

    def test_get_memory_snapshot_diff_none_snapshots(self):
        """Test comparing memory snapshots when one is None."""
        diff = get_memory_snapshot_diff(None, None, top_n=10)
        assert diff is None

    def test_get_memory_snapshot_diff_error(self):
        """Test comparing memory snapshots when error occurs."""
        mock_snapshot1 = MagicMock()
        mock_snapshot2 = MagicMock()
        mock_snapshot2.compare_to.side_effect = Exception("Comparison error")

        diff = get_memory_snapshot_diff(mock_snapshot1, mock_snapshot2, top_n=10)
        assert diff is None


class TestLogMemoryUsage:
    """Test log_memory_usage function."""

    def test_log_memory_usage_with_psutil(self, caplog):
        """Test logging memory usage when psutil is available."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value = Mock(rss=100 * 1024 * 1024, vms=200 * 1024 * 1024)
        mock_process.memory_percent.return_value = 25.5

        mock_virtual_memory = Mock()
        mock_virtual_memory.available = 500 * 1024 * 1024
        mock_virtual_memory.total = 2000 * 1024 * 1024

        # Patch builtins.__import__ to mock psutil import inside the function
        def mock_import(name, *args, **kwargs):
            if name == "psutil":
                mock_psutil = MagicMock()
                mock_psutil.Process.return_value = mock_process
                mock_psutil.virtual_memory.return_value = mock_virtual_memory
                return mock_psutil
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            # log_info uses print(), not Python logging module, so caplog won't capture it
            # Just verify the function runs without error
            log_memory_usage("test context")
            # Function should complete without raising exceptions

    def test_log_memory_usage_without_psutil(self, caplog):
        """Test logging memory usage when psutil is not available."""

        # Patch builtins.__import__ to raise ImportError for psutil
        def mock_import(name, *args, **kwargs):
            if name == "psutil":
                raise ImportError("No module named 'psutil'")
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            # log_info uses print(), not Python logging module, so caplog won't capture it
            # Just verify the function runs without error
            log_memory_usage("test context")
            # Function should complete without raising exceptions


class TestOptimizeMemory:
    """Test optimize_memory function."""

    def test_optimize_memory(self):
        """Test memory optimization routine."""
        # log_info uses print(), not Python logging module, so caplog won't capture it
        # Just verify the function runs without error
        gc_stats = optimize_memory()

        assert "collected" in gc_stats
        # Function should complete without raising exceptions

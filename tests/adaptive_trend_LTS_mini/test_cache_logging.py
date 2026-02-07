
import pytest
import time
import logging
from unittest.mock import MagicMock, patch
from modules.adaptive_trend_LTS_mini.utils.cache_manager import CacheManager, CacheEntry

class TestCachePeriodicLogging:
    """Test periodic logging functionality in CacheManager."""

    def test_init_validation(self):
        """Test validation of periodic log intervals."""
        # Should raise ValueError for negative values
        with pytest.raises(ValueError):
            CacheManager(periodic_log_interval_requests=-1)

        with pytest.raises(ValueError):
            CacheManager(periodic_log_interval_seconds=-5.0)

        # Valid values should pass
        cm = CacheManager(periodic_log_interval_requests=10, periodic_log_interval_seconds=30.0)
        assert cm.periodic_log_interval_requests == 10
        assert cm._metrics_log_interval == 30.0

    @patch('modules.common.ui.logging.log_info')
    def test_request_based_logging(self, mock_log_info):
        """Test logging triggered by request count."""
        # Initialize with logging every 5 requests
        # Set time interval high to avoid time-based logging interference
        cm = CacheManager(
            periodic_log_interval_requests=5,
            periodic_log_interval_seconds=1000.0,
            max_entries_l1=10,
            max_entries_l2=10
        )

        # Reset mock
        mock_log_info.reset_mock()

        # Perform 4 requests - should not log
        for i in range(4):
            cm.get("SMA", 10, f"data_{i}")

        assert mock_log_info.call_count == 0

        # Perform 5th request - should log
        cm.get("SMA", 10, "data_5")

        # Check if log_info was called with Cache Metrics
        assert mock_log_info.call_count >= 1
        args, _ = mock_log_info.call_args
        assert "Cache Metrics:" in args[0]
        assert "Requests=5" in args[0]

    @patch('modules.common.ui.logging.log_info')
    def test_time_based_logging(self, mock_log_info):
        """Test logging triggered by time interval."""
        # Initialize with short time interval
        cm = CacheManager(
            periodic_log_interval_requests=None,
            periodic_log_interval_seconds=0.1,  # 100ms
            max_entries_l1=10
        )

        # Reset mock
        mock_log_info.reset_mock()

        # First request
        cm.get("SMA", 10, "data_1")

        # Should not log yet (unless processing took > 0.1s, unlikely)
        if mock_log_info.call_count > 0:
            mock_log_info.reset_mock()

        # Wait for interval to pass
        time.sleep(0.15)

        # Second request - should trigger logging
        cm.get("SMA", 10, "data_2")

        assert mock_log_info.call_count >= 1
        args, _ = mock_log_info.call_args
        assert "Cache Metrics:" in args[0]

    @patch('modules.common.ui.logging.log_info')
    def test_logging_exception_handling(self, mock_log_info):
        """Test that logging exceptions don't break cache operations."""
        cm = CacheManager(periodic_log_interval_requests=1)

        # Make log_info raise an exception
        mock_log_info.side_effect = Exception("Logging failed")

        # This should not raise an exception
        try:
            cm.get("SMA", 10, "data_1")
        except Exception as e:
            pytest.fail(f"Cache operation failed due to logging exception: {e}")

        # Verify it attempted to log
        assert mock_log_info.call_count == 1

    @patch('modules.common.ui.logging.log_info')
    def test_request_based_logging_with_50_interval(self, mock_log_info):
        """Test logging triggered by 50 request count as specified in task."""
        # Initialize with logging every 50 requests
        cm = CacheManager(
            periodic_log_interval_requests=50,
            periodic_log_interval_seconds=10000.0,  # High value to avoid time-based interference
            max_entries_l1=60,
            max_entries_l2=60
        )

        # Reset mock
        mock_log_info.reset_mock()

        # Perform 49 requests - should not log
        for i in range(49):
            cm.get("SMA", 10, f"data_{i}")

        assert mock_log_info.call_count == 0

        # Perform 50th request - should trigger log
        cm.get("SMA", 10, "data_50")

        # Verify logging was triggered
        assert mock_log_info.call_count >= 1
        args, _ = mock_log_info.call_args
        assert "Cache Metrics:" in args[0]
        assert "Requests=50" in args[0]

    @patch('modules.common.ui.logging.log_info')
    def test_logging_disabled_when_none(self, mock_log_info):
        """Test that periodic logging is disabled when interval is None."""
        # Initialize with both intervals set to None
        cm = CacheManager(
            periodic_log_interval_requests=None,
            periodic_log_interval_seconds=None,
            max_entries_l1=10,
            max_entries_l2=10
        )

        # Reset mock
        mock_log_info.reset_mock()

        # Perform many requests
        for i in range(100):
            cm.get("SMA", 10, f"data_{i}")

        # Should not have triggered any periodic logging
        # (Note: There might be other log calls, but not periodic "Cache Metrics:")
        periodic_log_calls = [
            call for call in mock_log_info.call_args_list
            if call[0] and "Cache Metrics:" in str(call[0][0])
        ]
        assert len(periodic_log_calls) == 0

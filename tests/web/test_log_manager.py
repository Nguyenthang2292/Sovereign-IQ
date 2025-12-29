"""
Tests for LogFileManager (web/utils/log_manager.py).

Tests cover:
- Log file creation
- Writing logs
- Reading logs with offset
- Getting log size
- Deleting logs
- Cleanup old logs
- Thread safety
"""

import os
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import patch

from web.utils.log_manager import LogFileManager, get_log_manager


class TestLogFileManager:
    """Test LogFileManager class."""
    
    def test_init_default_logs_dir(self):
        """Test initialization with default logs directory."""
        manager = LogFileManager()
        assert manager.logs_dir.exists()
        assert manager.logs_dir.name == "logs"
    
    def test_init_custom_logs_dir(self, tmp_path):
        """Test initialization with custom logs directory."""
        custom_dir = tmp_path / "custom_logs"
        manager = LogFileManager(logs_dir=str(custom_dir))
        assert manager.logs_dir == custom_dir
        assert custom_dir.exists()
    
    def test_create_log_file(self, tmp_path):
        """Test creating a log file."""
        manager = LogFileManager(logs_dir=str(tmp_path))
        session_id = "test-session-123"
        command_type = "scan"
        
        log_path = manager.create_log_file(session_id, command_type)
        
        assert log_path.exists()
        assert log_path.name == f"{command_type}_{session_id}.log"
        assert log_path.stat().st_size == 0  # Empty file
    
    def test_write_log(self, tmp_path):
        """Test writing to log file."""
        manager = LogFileManager(logs_dir=str(tmp_path))
        session_id = "test-session-123"
        command_type = "scan"
        
        manager.create_log_file(session_id, command_type)
        manager.write_log(session_id, "Test log message", command_type)
        
        log_path = manager._get_log_path(session_id, command_type)
        content = log_path.read_text(encoding='utf-8')
        assert "Test log message" in content
        assert content.endswith('\n')
    
    def test_write_log_multiple_lines(self, tmp_path):
        """Test writing multiple log messages."""
        manager = LogFileManager(logs_dir=str(tmp_path))
        session_id = "test-session-123"
        command_type = "scan"
        
        manager.create_log_file(session_id, command_type)
        manager.write_log(session_id, "Line 1", command_type)
        manager.write_log(session_id, "Line 2", command_type)
        manager.write_log(session_id, "Line 3", command_type)
        
        log_path = manager._get_log_path(session_id, command_type)
        content = log_path.read_text(encoding='utf-8')
        lines = content.strip().split('\n')
        assert len(lines) == 3
        assert "Line 1" in lines[0]
        assert "Line 2" in lines[1]
        assert "Line 3" in lines[2]
    
    def test_read_log_from_beginning(self, tmp_path):
        """Test reading log file from beginning."""
        manager = LogFileManager(logs_dir=str(tmp_path))
        session_id = "test-session-123"
        command_type = "scan"
        
        manager.create_log_file(session_id, command_type)
        manager.write_log(session_id, "Line 1", command_type)
        manager.write_log(session_id, "Line 2", command_type)
        
        content, offset = manager.read_log(session_id, offset=0, command_type=command_type)
        assert "Line 1" in content
        assert "Line 2" in content
        assert offset > 0
    
    def test_read_log_from_offset(self, tmp_path):
        """Test reading log file from specific offset."""
        manager = LogFileManager(logs_dir=str(tmp_path))
        session_id = "test-session-123"
        command_type = "scan"
        
        manager.create_log_file(session_id, command_type)
        manager.write_log(session_id, "Line 1", command_type)
        
        # Get offset after first write
        _, first_offset = manager.read_log(session_id, offset=0, command_type=command_type)
        
        # Write more
        manager.write_log(session_id, "Line 2", command_type)
        manager.write_log(session_id, "Line 3", command_type)
        
        # Read from offset (should only get new content)
        content, new_offset = manager.read_log(session_id, offset=first_offset, command_type=command_type)
        assert "Line 1" not in content
        assert "Line 2" in content
        assert "Line 3" in content
        assert new_offset > first_offset
    
    def test_read_log_nonexistent_file(self, tmp_path):
        """Test reading from non-existent log file."""
        manager = LogFileManager(logs_dir=str(tmp_path))
        session_id = "nonexistent-session"
        command_type = "scan"
        
        content, offset = manager.read_log(session_id, offset=0, command_type=command_type)
        assert content == ""
        assert offset == 0
    
    def test_get_log_size(self, tmp_path):
        """Test getting log file size."""
        manager = LogFileManager(logs_dir=str(tmp_path))
        session_id = "test-session-123"
        command_type = "scan"
        
        manager.create_log_file(session_id, command_type)
        size_before = manager.get_log_size(session_id, command_type)
        assert size_before == 0
        
        manager.write_log(session_id, "Test message", command_type)
        size_after = manager.get_log_size(session_id, command_type)
        assert size_after > size_before
    
    def test_get_log_size_nonexistent(self, tmp_path):
        """Test getting size of non-existent log file."""
        manager = LogFileManager(logs_dir=str(tmp_path))
        session_id = "nonexistent-session"
        command_type = "scan"
        
        size = manager.get_log_size(session_id, command_type)
        assert size == 0
    
    def test_delete_log(self, tmp_path):
        """Test deleting a log file."""
        manager = LogFileManager(logs_dir=str(tmp_path))
        session_id = "test-session-123"
        command_type = "scan"
        
        log_path = manager.create_log_file(session_id, command_type)
        assert log_path.exists()
        
        result = manager.delete_log(session_id, command_type)
        assert result is True
        assert not log_path.exists()
    
    def test_delete_log_nonexistent(self, tmp_path):
        """Test deleting non-existent log file."""
        manager = LogFileManager(logs_dir=str(tmp_path))
        session_id = "nonexistent-session"
        command_type = "scan"
        
        result = manager.delete_log(session_id, command_type)
        assert result is False
    
    def test_cleanup_old_logs(self, tmp_path):
        """Test cleanup of old log files."""
        manager = LogFileManager(logs_dir=str(tmp_path))
        
        # Create old log file (simulate by setting mtime)
        old_session = "old-session"
        old_log = manager.create_log_file(old_session, "scan")
        manager.write_log(old_session, "Old log", "scan")
        
        # Set mtime to 2 hours ago
        old_time = time.time() - (2 * 3600)
        os.utime(old_log, (old_time, old_time))
        
        # Create new log file
        new_session = "new-session"
        new_log = manager.create_log_file(new_session, "scan")
        manager.write_log(new_session, "New log", "scan")
        
        # Cleanup logs older than 1 hour
        deleted_count = manager.cleanup_old_logs(max_age_hours=1)
        
        assert deleted_count == 1
        assert not old_log.exists()
        assert new_log.exists()
    
    def test_thread_safety_concurrent_writes(self, tmp_path):
        """Test thread safety with concurrent writes."""
        manager = LogFileManager(logs_dir=str(tmp_path))
        session_id = "test-session-123"
        command_type = "scan"
        
        manager.create_log_file(session_id, command_type)
        
        def write_logs(thread_id):
            for i in range(10):
                manager.write_log(session_id, f"Thread {thread_id} message {i}", command_type)
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=write_logs, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify all messages are present
        content, _ = manager.read_log(session_id, offset=0, command_type=command_type)
        assert content.count("message") == 50  # 5 threads * 10 messages
    
    def test_different_command_types(self, tmp_path):
        """Test handling different command types."""
        manager = LogFileManager(logs_dir=str(tmp_path))
        session_id = "test-session-123"
        
        # Create logs for different command types
        scan_log = manager.create_log_file(session_id, "scan")
        analyze_log = manager.create_log_file(session_id, "analyze")
        
        assert scan_log.name == f"scan_{session_id}.log"
        assert analyze_log.name == f"analyze_{session_id}.log"
        
        manager.write_log(session_id, "Scan log", "scan")
        manager.write_log(session_id, "Analyze log", "analyze")
        
        scan_content, _ = manager.read_log(session_id, offset=0, command_type="scan")
        analyze_content, _ = manager.read_log(session_id, offset=0, command_type="analyze")
        
        assert "Scan log" in scan_content
        assert "Analyze log" in analyze_content
        assert "Scan log" not in analyze_content
        assert "Analyze log" not in scan_content


import pytest

class TestGetLogManager:
    """Test get_log_manager function."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        # Reset the singleton instance before and after each test
        global _log_manager_instance
        try:
            prev_instance = _log_manager_instance
        except NameError:
            prev_instance = None
        _log_manager_instance = None
        yield
        _log_manager_instance = None

    def test_get_log_manager_singleton(self):
        """Test that get_log_manager returns singleton instance."""
        manager1 = get_log_manager()
        manager2 = get_log_manager()

        assert manager1 is manager2


class TestLogFileManagerAutoCleanup:
    """Test auto-cleanup functionality in LogFileManager."""
    
    def test_auto_cleanup_before_new_log(self, tmp_path):
        """Test that old logs are cleaned up before creating new log file."""
        manager = LogFileManager(logs_dir=str(tmp_path), auto_cleanup_before_new=True, max_log_age_hours=1)
        
        # Create old log file (simulate by setting mtime)
        old_session = "old-session"
        old_log = manager.create_log_file(old_session, "scan")
        manager.write_log(old_session, "Old log message", "scan")
        
        # Set mtime to 2 hours ago
        old_time = time.time() - (2 * 3600)
        os.utime(old_log, (old_time, old_time))
        
        # Reset last cleanup time to allow cleanup to run again
        manager.reset_cleanup_timer_for_testing()
        
        # Create new log file - should trigger cleanup
        new_session = "new-session"
        new_log = manager.create_log_file(new_session, "scan")
        
        # Verify old log is deleted
        assert not old_log.exists()
        assert new_log.exists()
    
    def test_auto_cleanup_disabled(self, tmp_path):
        """Test that auto-cleanup can be disabled."""
        manager = LogFileManager(logs_dir=str(tmp_path), auto_cleanup_before_new=False, max_log_age_hours=1)
        
        # Create old log file
        old_session = "old-session"
        old_log = manager.create_log_file(old_session, "scan")
        manager.write_log(old_session, "Old log message", "scan")
        
        # Set mtime to 2 hours ago
        old_time = time.time() - (2 * 3600)
        os.utime(old_log, (old_time, old_time))
        
        # Create new log file - should NOT trigger cleanup
        new_session = "new-session"
        new_log = manager.create_log_file(new_session, "scan")
        
        # Verify old log still exists
        assert old_log.exists()
        assert new_log.exists()
    
    def test_auto_cleanup_throttle(self, tmp_path):
        """Test that cleanup is throttled (only runs every 5 minutes)."""
        manager = LogFileManager(logs_dir=str(tmp_path), auto_cleanup_before_new=True, max_log_age_hours=1)
        
        # Create old log file
        old_session = "old-session"
        old_log = manager.create_log_file(old_session, "scan")
        manager.write_log(old_session, "Old log message", "scan")
        
        # Set mtime to 2 hours ago
        old_time = time.time() - (2 * 3600)
        os.utime(old_log, (old_time, old_time))
        
        # Reset last cleanup time to allow first cleanup to run
        manager.reset_cleanup_timer_for_testing()
        
        # Create first new log - cleanup should run
        new_session1 = "new-session-1"
        new_log1 = manager.create_log_file(new_session1, "scan")
        
        # Verify old log is deleted after first creation
        assert not old_log.exists()
        
        # Create another old log
        old_log2 = manager.create_log_file("old-session-2", "scan")
        manager.write_log("old-session-2", "Old log 2", "scan")
        os.utime(old_log2, (old_time, old_time))
        
        # Create second new log immediately - cleanup should NOT run (throttled)
        # Note: _last_cleanup_time is now set from the first cleanup, so this should be throttled
        new_session2 = "new-session-2"
        new_log2 = manager.create_log_file(new_session2, "scan")
        
        # Verify old log 2 still exists (cleanup was throttled)
        assert old_log2.exists()
        assert new_log1.exists()
        assert new_log2.exists()
    
    def test_cleanup_with_env_vars(self, tmp_path, monkeypatch):
        """Test configuration via environment variables."""
        # Set environment variables
        monkeypatch.setenv("LOG_AUTO_CLEANUP", "false")
        monkeypatch.setenv("LOG_MAX_AGE_HOURS", "48")
        
        # Need to create new instance to read env vars
        # Clear the global instance first
        import web.utils.log_manager as log_manager_module
        log_manager_module._log_manager_instance = None
        
        manager = LogFileManager(logs_dir=str(tmp_path))
        
        assert manager.auto_cleanup_before_new is False
        assert manager.max_log_age_hours == 48
        
        # Test with different env values
        monkeypatch.setenv("LOG_AUTO_CLEANUP", "true")
        monkeypatch.setenv("LOG_MAX_AGE_HOURS", "12")
        log_manager_module._log_manager_instance = None
        
        manager2 = LogFileManager(logs_dir=str(tmp_path))
        assert manager2.auto_cleanup_before_new is True
        assert manager2.max_log_age_hours == 12
        
        # Restore
        log_manager_module._log_manager_instance = None


class TestLogFileManagerLocks:
    """Test lock cleanup functionality in LogFileManager."""
    
    def test_cleanup_unused_locks(self, tmp_path):
        """Test cleanup of unused locks."""
        manager = LogFileManager(logs_dir=str(tmp_path))
        
        # Create locks for multiple sessions
        session1 = "session-1"
        session2 = "session-2"
        session3 = "session-3"
        
        # Get locks (creates them)
        _ = manager._get_lock(session1)
        _ = manager._get_lock(session2)
        _ = manager._get_lock(session3)
        
        # Verify locks exist
        assert session1 in manager._locks
        assert session2 in manager._locks
        assert session3 in manager._locks
        
        # Set last_used to past for session1 and session2
        past_time = datetime.now() - timedelta(hours=3)
        with manager._locks_lock:
            manager._lock_last_used[session1] = past_time
            manager._lock_last_used[session2] = past_time
            # session3 remains recent
        
        # Cleanup unused locks (older than 2 hours)
        manager._cleanup_unused_locks(max_age_hours=2)
        
        # Verify old locks are removed
        assert session1 not in manager._locks
        assert session2 not in manager._locks
        assert session1 not in manager._lock_last_used
        assert session2 not in manager._lock_last_used
        
        # Verify recent lock still exists
        assert session3 in manager._locks
        assert session3 in manager._lock_last_used
    
    def test_cleanup_lock_manually(self, tmp_path):
        """Test manual cleanup of a specific lock."""
        manager = LogFileManager(logs_dir=str(tmp_path))
        
        session_id = "test-session"
        
        # Get lock (creates it)
        lock = manager._get_lock(session_id)
        
        # Verify lock exists
        assert session_id in manager._locks
        assert session_id in manager._lock_last_used
        
        # Manually cleanup
        manager.cleanup_lock(session_id)
        
        # Verify lock is removed
        assert session_id not in manager._locks
        assert session_id not in manager._lock_last_used
    
    def test_cleanup_lock_nonexistent(self, tmp_path):
        """Test cleaning up non-existent lock."""
        manager = LogFileManager(logs_dir=str(tmp_path))
        
        # Should not raise error
        manager.cleanup_lock("nonexistent-session")
        
        # Verify no locks exist
        assert len(manager._locks) == 0


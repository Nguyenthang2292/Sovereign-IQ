"""
Log File Manager for managing log files for CLI output streaming.

Each request gets its own log file identified by session_id.
"""

import os
import threading
import time
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime, timedelta

from modules.common.ui.logging import log_info, log_error, log_warn, log_debug


class LogFileManager:
    """Manages log file creation, writing, and reading."""
    
    def __init__(self, logs_dir: Optional[str] = None, auto_cleanup_before_new: Optional[bool] = None, max_log_age_hours: Optional[int] = None, start_cleanup_thread: bool = True):
        """
        Initialize LogFileManager.
        
        Args:
            logs_dir: Directory to store log files. Defaults to 'logs' in project root.
            auto_cleanup_before_new: If True, automatically cleanup old logs before creating new log file.
                                     If None, reads from LOG_AUTO_CLEANUP env var (default: True).
            max_log_age_hours: Maximum age in hours for logs to be kept.
                               If None, reads from LOG_MAX_AGE_HOURS env var (default: 24 hours).
            start_cleanup_thread: If True, start background thread for cleaning up unused locks.
                                  Set to False in tests to avoid spawning threads (default: True).
        """
        if logs_dir is None:
            # Get project root (assuming this file is in web/utils/)
            project_root = Path(__file__).parent.parent.parent
            logs_dir = project_root / "logs"
        
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Cleanup settings - can be configured via environment variables
        if auto_cleanup_before_new is None:
            auto_cleanup_env = os.getenv("LOG_AUTO_CLEANUP", "true").lower()
            self.auto_cleanup_before_new = auto_cleanup_env in ("true", "1", "yes", "on")
        else:
            self.auto_cleanup_before_new = auto_cleanup_before_new
        
        if max_log_age_hours is None:
            self.max_log_age_hours = int(os.getenv("LOG_MAX_AGE_HOURS", "24"))
        else:
            self.max_log_age_hours = max_log_age_hours
        
        self._last_cleanup_time: Optional[datetime] = None
        self._cleanup_lock = threading.Lock()
        
        # Thread lock for file operations
        self._locks: dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()
        self._lock_last_used: dict[str, datetime] = {}
        
        # Start cleanup thread for old locks (only if requested)
        self._cleanup_thread: Optional[threading.Thread] = None
        if start_cleanup_thread:
            self._cleanup_thread = threading.Thread(target=self._cleanup_locks_loop, daemon=True)
            self._cleanup_thread.start()
    
    def _get_lock(self, session_id: str) -> threading.Lock:
        """Get or create a lock for a session_id."""
        with self._locks_lock:
            if session_id not in self._locks:
                self._locks[session_id] = threading.Lock()
            self._lock_last_used[session_id] = datetime.now()
            return self._locks[session_id]
    
    def _get_log_path(self, session_id: str, command_type: str) -> Path:
        """
        Get log file path for a session.
        
        Args:
            session_id: Unique session identifier
            command_type: Type of command ('scan' or 'analyze')
            
        Returns:
            Path to log file
        """
        filename = f"{command_type}_{session_id}.log"
        return self.logs_dir / filename
    
    def create_log_file(self, session_id: str, command_type: str) -> Path:
        """
        Create a new log file for a session.
        Automatically cleans up old logs before creating new file if auto_cleanup_before_new is True.
        
        Args:
            session_id: Unique session identifier
            command_type: Type of command ('scan' or 'analyze')
            
        Returns:
            Path to created log file
        """
        # Cleanup old logs before creating new log file
        if self.auto_cleanup_before_new:
            self._cleanup_old_logs_before_new_request()
        
        log_path = self._get_log_path(session_id, command_type)
        
        with self._get_lock(session_id):
            # Create empty file
            log_path.touch(exist_ok=True)
            log_info(f"Created log file: {log_path}")
        
        return log_path
    
    def write_log(self, session_id: str, message: str, command_type: str = "scan") -> None:
        """
        Write a message to the log file.
        
        Args:
            session_id: Unique session identifier
            message: Message to write
            command_type: Type of command ('scan' or 'analyze')
        """
        log_path = self._get_log_path(session_id, command_type)
        
        with self._get_lock(session_id):
            try:
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(message)
                    if not message.endswith('\n'):
                        f.write('\n')
                    f.flush()  # Ensure immediate write
            except Exception as e:
                log_error(f"Error writing to log file {log_path}: {e}")
    
    def read_log(self, session_id: str, offset: int = 0, command_type: str = "scan") -> Tuple[str, int]:
        """
        Read log file from a specific offset.
        
        Args:
            session_id: Unique session identifier
            offset: Byte offset to start reading from
            command_type: Type of command ('scan' or 'analyze')
            
        Returns:
            Tuple of (log_content, new_offset)
        """
        log_path = self._get_log_path(session_id, command_type)
        
        with self._get_lock(session_id):
            if not log_path.exists():
                return ("", 0)
            
            try:
                with open(log_path, 'rb') as f:
                    f.seek(offset)
                    byte_content = f.read()
                    new_offset = f.tell()
                # Decode bytes to string using utf-8 and handle decode errors gracefully
                content = byte_content.decode('utf-8', errors='replace')
                return (content, new_offset)
            except Exception as e:
                log_error(f"Error reading log file {log_path}: {e}")
                return ("", offset)
    
    def get_log_size(self, session_id: str, command_type: str = "scan") -> int:
        """
        Get the size of the log file in bytes.
        
        Args:
            session_id: Unique session identifier
            command_type: Type of command ('scan' or 'analyze')
            
        Returns:
            File size in bytes
        """
        log_path = self._get_log_path(session_id, command_type)
        
        if not log_path.exists():
            return 0
        
        try:
            return log_path.stat().st_size
        except Exception as e:
            log_error(f"Error getting log file size {log_path}: {e}")
            return 0
    
    def delete_log(self, session_id: str, command_type: str = "scan") -> bool:
        """
        Delete a log file.
        
        Args:
            session_id: Unique session identifier
            command_type: Type of command ('scan' or 'analyze')
            
        Returns:
            True if deleted, False otherwise
        """
        log_path = self._get_log_path(session_id, command_type)
        
        with self._get_lock(session_id):
            try:
                if log_path.exists():
                    log_path.unlink()
                    log_info(f"Deleted log file: {log_path}")
                    return True
                return False
            except Exception as e:
                log_error(f"Error deleting log file {log_path}: {e}")
                return False
    
    def _extract_session_id_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract session_id from log filename.
        
        Filename format: {command_type}_{session_id}.log
        where command_type is 'scan' or 'analyze'.
        
        Args:
            filename: Log filename (e.g., 'scan_abc123.log' or 'analyze_xyz789.log')
            
        Returns:
            Session ID if successfully parsed, None otherwise
        """
        if not filename.endswith('.log'):
            return None
        
        # Remove .log extension
        name_without_ext = filename[:-4]
        
        # Check for known command types and extract session_id
        if name_without_ext.startswith('scan_'):
            return name_without_ext[5:]  # Everything after 'scan_'
        elif name_without_ext.startswith('analyze_'):
            return name_without_ext[8:]  # Everything after 'analyze_'
        
        return None
    
    def cleanup_old_logs(self, max_age_hours: Optional[int] = None) -> int:
        """
        Clean up log files older than specified hours.
        Uses per-session locks to prevent race conditions with concurrent read/write operations.
        
        Args:
            max_age_hours: Maximum age in hours (defaults to self.max_log_age_hours if None)
            
        Returns:
            Number of files deleted
        """
        if max_age_hours is None:
            max_age_hours = self.max_log_age_hours
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        deleted_count = 0
        
        try:
            for log_file in self.logs_dir.glob("*.log"):
                # Extract session_id from filename
                session_id = self._extract_session_id_from_filename(log_file.name)
                
                if session_id is None:
                    # Skip files that don't match expected format
                    log_debug(f"Skipping log file with unexpected format: {log_file.name}")
                    continue
                
                # Acquire session lock before performing stat/unlink
                lock = self._get_lock(session_id)
                with lock:
                    # Perform stat and unlink while lock is held
                    try:
                        file_age = current_time - log_file.stat().st_mtime
                        if file_age > max_age_seconds:
                            try:
                                log_file.unlink()
                                deleted_count += 1
                                log_debug(f"Deleted old log file: {log_file.name}")
                            except FileNotFoundError:
                                # File was already deleted by another process/thread
                                # This is not an error, just skip it
                                pass
                    except FileNotFoundError:
                        # File was deleted between glob and stat, skip it
                        pass
                    except Exception as e:
                        log_error(f"Error cleaning up log file {log_file}: {e}")
        
        except Exception as e:
            log_error(f"Error during log cleanup: {e}")
        
        if deleted_count > 0:
            log_info(f"Cleaned up {deleted_count} old log file(s) (older than {max_age_hours} hours)")
        
        return deleted_count
    
    def _cleanup_old_logs_before_new_request(self, min_interval_minutes: int = 5):
        """
        Cleanup old logs before creating a new log file.
        This method is called automatically when creating a new log file.
        It only runs cleanup if enough time has passed since last cleanup to avoid overhead.
        
        Args:
            min_interval_minutes: Minimum interval in minutes between cleanups (default: 5 minutes)
        """
        with self._cleanup_lock:
            now = datetime.now()
            
            # Only cleanup if enough time has passed since last cleanup
            if self._last_cleanup_time is not None:
                time_since_last_cleanup = now - self._last_cleanup_time
                if time_since_last_cleanup.total_seconds() < min_interval_minutes * 60:
                    # Too soon since last cleanup, skip
                    return
            
            # Perform cleanup
            try:
                deleted_count = self.cleanup_old_logs()
                self._last_cleanup_time = now
                
                if deleted_count > 0:
                    log_debug(f"Auto-cleaned {deleted_count} old log file(s) before creating new log")
            except Exception as e:
                log_warn(f"Error during auto-cleanup before new request: {e}")
                # Don't fail log creation if cleanup fails
    
    def _cleanup_locks_loop(self):
        """Background thread to cleanup unused locks."""
        while True:
            try:
                time.sleep(600)  # Check every 10 minutes
                self._cleanup_unused_locks()
            except Exception as e:
                log_error(f"Error in lock cleanup loop: {e}")
    
    def _cleanup_unused_locks(self, max_age_hours: int = 2):
        """
        Remove locks that haven't been used for a while.
        
        Args:
            max_age_hours: Maximum age in hours before removing lock
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        with self._locks_lock:
            to_remove = []
            for session_id, last_used in self._lock_last_used.items():
                if last_used < cutoff_time:
                    to_remove.append(session_id)
            
            for session_id in to_remove:
                self._locks.pop(session_id, None)
                self._lock_last_used.pop(session_id, None)
            
            if to_remove:
                log_debug(f"Cleaned up {len(to_remove)} unused locks")
    
    def cleanup_lock(self, session_id: str):
        """
        Manually cleanup a specific lock.
        
        Args:
            session_id: Session ID to cleanup
        """
        with self._locks_lock:
            self._locks.pop(session_id, None)
            self._lock_last_used.pop(session_id, None)
    
    def reset_cleanup_timer_for_testing(self):
        """
        Public testing API to reset the cleanup timer.
        
        This method is intended for testing purposes to force cleanup to run
        by resetting the last cleanup time. In normal operation, cleanup is
        throttled to avoid running too frequently (minimum 5 minute interval).
        
        This allows tests to bypass the throttle mechanism and ensure cleanup
        runs when needed for test scenarios.
        
        Note: This method should only be used in tests, not in production code.
        """
        self._reset_cleanup_timer()
    
    def _reset_cleanup_timer(self):
        """
        Internal helper method to reset the cleanup timer.
        
        This method is primarily intended for testing purposes to force cleanup
        to run by resetting the last cleanup time. In normal operation, cleanup
        is throttled to avoid running too frequently.
        
        Note: This is an internal method and should not be used in production code.
        Deprecated: Use reset_cleanup_timer_for_testing() instead for tests.
        """
        with self._cleanup_lock:
            self._last_cleanup_time = None


# Global instance
_log_manager_instance: Optional[LogFileManager] = None


_log_manager_lock = threading.Lock()

def get_log_manager() -> LogFileManager:
    """Get global LogFileManager instance (thread-safe)."""
    global _log_manager_instance
    if _log_manager_instance is None:
        with _log_manager_lock:
            if _log_manager_instance is None:
                _log_manager_instance = LogFileManager()
    return _log_manager_instance


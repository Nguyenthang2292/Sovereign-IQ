
from datetime import datetime, timedelta
from unittest.mock import patch
import threading
import time

from web.utils.task_manager import TaskManager, get_task_manager
from web.utils.task_manager import TaskManager, get_task_manager

"""
Tests for TaskManager (web/utils/task_manager.py).

Tests cover:
- Starting tasks
- Getting task status
- Setting results
- Setting errors
- Getting results
- Cleanup tasks
- Thread safety
"""




class TestTaskManager:
    """Test TaskManager class."""

    def test_init(self):
        """Test TaskManager initialization."""
        manager = TaskManager(cleanup_after_hours=2)
        assert manager.cleanup_after_hours == 2
        assert len(manager._tasks) == 0

    def test_start_task(self):
        """Test starting a background task."""
        manager = TaskManager()
        session_id = "test-session-123"

        def simple_task():
            return {"result": "success"}

        manager.start_task(session_id, simple_task, "scan")

        # Wait a bit for task to complete
        time.sleep(0.1)

        status = manager.get_status(session_id)
        assert status is not None
        assert status["status"] in ["running", "completed"]
        assert status["command_type"] == "scan"

    def test_get_status_running(self):
        """Test getting status of running task."""
        manager = TaskManager()
        session_id = "test-session-123"

        def long_task():
            time.sleep(0.2)
            return {"result": "done"}

        manager.start_task(session_id, long_task, "scan")

        # Check status while running
        status = manager.get_status(session_id)
        assert status is not None
        assert status["status"] == "running"
        assert "started_at" in status

    def test_get_status_completed(self):
        """Test getting status of completed task."""
        manager = TaskManager()
        session_id = "test-session-123"

        def quick_task():
            return {"result": "completed"}

        manager.start_task(session_id, quick_task, "scan")

        # Wait for completion
        time.sleep(0.1)

        status = manager.get_status(session_id)
        assert status is not None
        assert status["status"] == "completed"
        assert "completed_at" in status
        assert "result" in status

    def test_get_status_error(self):
        """Test getting status of failed task."""
        manager = TaskManager()
        session_id = "test-session-123"

        def failing_task():
            raise ValueError("Test error")

        manager.start_task(session_id, failing_task, "scan")

        # Wait for error
        time.sleep(0.1)

        status = manager.get_status(session_id)
        assert status is not None
        assert status["status"] == "error"
        assert "error" in status
        assert "Test error" in status["error"]

    def test_set_result(self):
        """Test manually setting result."""
        manager = TaskManager()
        session_id = "test-session-123"

        # Start task
        manager.start_task(session_id, lambda: None, "scan")

        # Manually set result
        result_data = {"key": "value"}
        manager.set_result(session_id, result_data)

        status = manager.get_status(session_id)
        assert status["status"] == "completed"
        assert status["result"] == result_data

    def test_set_result_not_overwritten_by_run_task(self):
        """Test that set_result() result is not overwritten by run_task() when task_func returns None.

        This test covers the race condition fix where:
        1. set_result() is called to set the result
        2. run_task() completes and task_func() returns None
        3. run_task() should NOT overwrite the result that was already set
        """
        manager = TaskManager()
        session_id = "test-session-123"

        # Start a task that returns None (simulating run_scan() which doesn't return)
        def task_that_returns_none():
            time.sleep(0.05)  # Simulate some work
            return None

        manager.start_task(session_id, task_that_returns_none, "scan")

        # Wait a bit for task to start
        time.sleep(0.01)

        # Manually set result (simulating what batch_scanner does)
        result_data = {"success": True, "summary": {"total": 10}, "long_symbols": ["BTC/USDT"], "short_symbols": []}
        manager.set_result(session_id, result_data)

        # Wait for task to complete
        time.sleep(0.1)

        # Verify result was NOT overwritten by run_task()
        status = manager.get_status(session_id)
        assert status is not None
        assert status["status"] == "completed"
        assert status["result"] == result_data  # Should still be the result from set_result()
        assert status["result"]["summary"]["total"] == 10
        assert "long_symbols" in status["result"]

    def test_run_task_does_not_overwrite_existing_result(self):
        """Test that run_task() does not overwrite result if it was already set by set_result().

        This test simulates the real scenario:
        1. Task is started
        2. During task execution, set_result() is called (e.g., by batch_scanner)
        3. Task completes and task_func() returns None
        4. run_task() should NOT overwrite the result that was already set
        """
        manager = TaskManager()
        session_id = "test-session-123"

        # Start a task that will call set_result() during execution
        original_result = {"original": "data", "count": 5, "summary": {"total": 10}}

        def task_that_sets_result():
            """Task that simulates batch_scanner behavior: calls set_result() then returns None."""
            time.sleep(0.05)  # Simulate some work
            # Simulate what batch_scanner does: call set_result() during execution
            manager.set_result(session_id, original_result)
            return None  # run_scan() doesn't return anything

        manager.start_task(session_id, task_that_sets_result, "scan")

        # Wait for task to complete
        time.sleep(0.1)

        # Verify result from set_result() is still there (not overwritten by None from task_func)
        status = manager.get_status(session_id)
        assert status is not None
        assert status["status"] == "completed"
        # Result should be the one from set_result(), not None
        assert status["result"] == original_result
        assert status["result"]["count"] == 5
        assert status["result"]["summary"]["total"] == 10

    def test_get_result(self):
        """Test getting result of completed task."""
        manager = TaskManager()
        session_id = "test-session-123"

        result_data = {"key": "value"}
        manager.start_task(session_id, lambda: result_data, "scan")

        # Wait for completion
        time.sleep(0.1)

        result = manager.get_result(session_id)
        assert result == result_data

    def test_get_result_not_completed(self):
        """Test getting result of non-completed task."""
        manager = TaskManager()
        session_id = "test-session-123"

        def long_task():
            time.sleep(0.2)
            return {"result": "done"}

        manager.start_task(session_id, long_task, "scan")

        result = manager.get_result(session_id)
        assert result is None

    def test_set_error(self):
        """Test manually setting error."""
        manager = TaskManager()
        session_id = "test-session-123"

        # Start task
        manager.start_task(session_id, lambda: None, "scan")

        # Manually set error
        manager.set_error(session_id, "Custom error message")

        status = manager.get_status(session_id)
        assert status["status"] == "error"
        assert status["error"] == "Custom error message"

    def test_get_status_nonexistent(self):
        """Test getting status of non-existent task."""
        manager = TaskManager()
        status = manager.get_status("nonexistent-session")
        assert status is None

    def test_cleanup_task(self):
        """Test manually cleaning up a task."""
        manager = TaskManager()
        session_id = "test-session-123"

        manager.start_task(session_id, lambda: None, "scan")
        assert manager.get_status(session_id) is not None

        result = manager.cleanup_task(session_id)
        assert result is True
        assert manager.get_status(session_id) is None

    def test_cleanup_task_nonexistent(self):
        """Test cleaning up non-existent task."""
        manager = TaskManager()
        result = manager.cleanup_task("nonexistent-session")
        assert result is False

    def test_cleanup_task_clears_references(self):
        """Test that cleanup_task clears result, error, and thread references."""
        manager = TaskManager()
        session_id = "test-session-123"

        # Start task and set result/error
        manager.start_task(session_id, lambda: {"result": "data"}, "scan")
        time.sleep(0.1)

        # Manually set error to test error clearing
        manager.set_error(session_id, "Test error")

        # Get task before cleanup to verify it has data
        # Note: get_status() removes thread from copy, so check actual task dict
        with manager._lock:
            task_before = manager._tasks.get(session_id)
            assert task_before is not None
            assert task_before.get("error") is not None
            # Thread might be None if task already completed
            # Just verify task exists with error

        # Cleanup task
        result = manager.cleanup_task(session_id)
        assert result is True

        # Verify task is removed
        assert manager.get_status(session_id) is None

    def test_clear_all_results(self):
        """Test clearing all results from completed tasks."""
        manager = TaskManager()

        # Create multiple tasks with different statuses
        completed_id = "completed-session"
        error_id = "error-session"
        cancelled_id = "cancelled-session"
        running_id = "running-session"

        # Completed task with result
        manager.start_task(completed_id, lambda: {"result": "completed_data"}, "scan")
        time.sleep(0.1)
        manager.set_result(completed_id, {"large": "data", "count": 100})

        # Error task with result (task function should raise exception to keep error status)
        def error_task():
            raise RuntimeError("Error occurred")

        manager.start_task(error_id, error_task, "scan")
        time.sleep(0.1)  # Wait for task to fail
        # Manually set result without changing status (simulate error with result data)
        with manager._lock:
            if error_id in manager._tasks:
                manager._tasks[error_id]["result"] = {"error_data": "value"}

        # Cancelled task (simulate by starting then cancelling)
        def cancelled_task():
            time.sleep(0.2)
            return {"result": "done"}

        manager.start_task(cancelled_id, cancelled_task, "scan")
        manager.cancel_task(cancelled_id)  # Cancel it properly
        time.sleep(0.05)  # Wait for cancellation to take effect
        # Manually set result without changing status
        with manager._lock:
            if cancelled_id in manager._tasks:
                manager._tasks[cancelled_id]["result"] = {"cancelled_data": "value"}

        # Running task (should not be cleared)
        def long_task():
            time.sleep(0.3)
            return {"result": "done"}

        manager.start_task(running_id, long_task, "scan")

        # Verify all have results before clearing
        status_completed = manager.get_status(completed_id)
        status_error = manager.get_status(error_id)
        status_cancelled = manager.get_status(cancelled_id)

        assert status_completed["result"] is not None
        assert status_error["result"] is not None
        assert status_cancelled["result"] is not None

        # Clear all results
        with patch("gc.collect") as mock_gc:
            manager.clear_all_results()

            # Verify results are cleared but metadata remains
            status_completed_after = manager.get_status(completed_id)
            status_error_after = manager.get_status(error_id)
            status_cancelled_after = manager.get_status(cancelled_id)
            status_running_after = manager.get_status(running_id)

            # Results should be None
            assert status_completed_after["result"] is None
            assert status_error_after["result"] is None
            assert status_cancelled_after["result"] is None

            # Metadata should still be there
            assert status_completed_after["status"] == "completed"
            assert status_error_after["status"] == "error"
            assert status_cancelled_after["status"] == "cancelled"

            # Running task should not be affected
            assert status_running_after is not None
            assert status_running_after["status"] == "running"

            # GC should be called
            mock_gc.assert_called_once()

    def test_cleanup_old_tasks(self):
        """Test that _cleanup_old_tasks removes old tasks correctly."""
        manager = TaskManager(cleanup_after_hours=1)

        # Create tasks with completed_at in the past
        old_session1 = "old-session-1"
        old_session2 = "old-session-2"
        new_session = "new-session"

        # Create old completed tasks
        manager.start_task(old_session1, lambda: {"result": "old1"}, "scan")
        manager.start_task(old_session2, lambda: {"result": "old2"}, "scan")
        time.sleep(0.1)

        # Manually set completed_at to past
        past_time = datetime.now() - timedelta(hours=2)
        with manager._lock:
            if old_session1 in manager._tasks:
                manager._tasks[old_session1]["completed_at"] = past_time
                manager._tasks[old_session1]["status"] = "completed"
                manager._tasks[old_session1]["result"] = {"large": "data"}
            if old_session2 in manager._tasks:
                manager._tasks[old_session2]["completed_at"] = past_time
                manager._tasks[old_session2]["status"] = "completed"
                manager._tasks[old_session2]["result"] = {"large": "data"}

        # Create new task (should not be cleaned)
        manager.start_task(new_session, lambda: {"result": "new"}, "scan")
        time.sleep(0.1)
        with manager._lock:
            if new_session in manager._tasks:
                manager._tasks[new_session]["completed_at"] = datetime.now()
                manager._tasks[new_session]["status"] = "completed"

        # Verify old tasks exist
        assert manager.get_status(old_session1) is not None
        assert manager.get_status(old_session2) is not None
        assert manager.get_status(new_session) is not None

        # Call cleanup - should remove old tasks but not new ones
        manager._cleanup_old_tasks()

        # Verify old tasks are removed
        assert manager.get_status(old_session1) is None
        assert manager.get_status(old_session2) is None

        # Verify new task is still there
        assert manager.get_status(new_session) is not None

    def test_multiple_tasks(self):
        """Test managing multiple tasks simultaneously."""
        manager = TaskManager()

        session_ids = [f"session-{i}" for i in range(5)]

        for session_id in session_ids:
            manager.start_task(session_id, lambda sid=session_id: {"id": sid}, "scan")

        # Wait for completion
        time.sleep(0.1)

        for session_id in session_ids:
            status = manager.get_status(session_id)
            assert status is not None
            assert status["status"] == "completed"

    def test_thread_safety(self):
        """Test thread safety with concurrent operations."""
        manager = TaskManager()
        session_id = "test-session-123"

        def start_and_check():
            manager.start_task(session_id, lambda: {"result": "done"}, "scan")
            time.sleep(0.05)
            status = manager.get_status(session_id)
            assert status is not None

        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=start_and_check)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Final status should be valid
        status = manager.get_status(session_id)
        assert status is not None


class TestGetTaskManager:
    """Test get_task_manager function."""

    def test_get_task_manager_singleton(self):
        """Test that get_task_manager returns singleton instance."""
        manager1 = get_task_manager()
        manager2 = get_task_manager()

        assert manager1 is manager2


class TestTaskManagerCancel:
    """Test TaskManager cancel functionality."""

    def test_cancel_task(self):
        """Test cancelling a running task."""
        manager = TaskManager()
        session_id = "test-cancel-session"

        # Start a long-running task
        def long_task():
            time.sleep(0.5)
            return {"result": "done"}

        manager.start_task(session_id, long_task, "scan")

        # Wait a bit to ensure task is running
        time.sleep(0.05)

        # Cancel the task
        cancelled = manager.cancel_task(session_id)
        assert cancelled is True

        # Verify task is marked as cancelled
        assert manager.is_cancelled(session_id) is True

        # Wait for task to complete
        time.sleep(0.6)

        # Check status - should be cancelled
        status = manager.get_status(session_id)
        assert status is not None
        assert status["status"] == "cancelled"

    def test_is_cancelled(self):
        """Test checking if a task is cancelled."""
        manager = TaskManager()
        session_id = "test-cancel-check"

        # Task doesn't exist yet
        assert manager.is_cancelled(session_id) is False

        # Start a long-running task to ensure it's still running when we cancel
        def long_task():
            time.sleep(0.2)
            return {"result": "done"}

        manager.start_task(session_id, long_task, "scan")

        # Wait a bit to ensure task is running
        time.sleep(0.05)

        # Not cancelled yet
        assert manager.is_cancelled(session_id) is False

        # Cancel it while it's still running
        cancelled = manager.cancel_task(session_id)
        assert cancelled is True  # Should succeed since task is still running

        # Now it's cancelled
        assert manager.is_cancelled(session_id) is True

    def test_cancel_task_not_found(self):
        """Test cancelling a non-existent task."""
        manager = TaskManager()
        cancelled = manager.cancel_task("nonexistent-session")
        assert cancelled is False

    def test_cancel_task_not_running(self):
        """Test cancelling a task that is not running (should fail)."""
        manager = TaskManager()
        session_id = "test-not-running"

        # Start and complete a task
        manager.start_task(session_id, lambda: {"result": "done"}, "scan")
        time.sleep(0.1)

        # Try to cancel (should fail)
        cancelled = manager.cancel_task(session_id)
        assert cancelled is False

        # Verify task is not cancelled
        assert manager.is_cancelled(session_id) is False

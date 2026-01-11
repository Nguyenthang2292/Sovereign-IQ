
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional
import gc
import threading

from modules.common.ui.logging import log_debug, log_error, log_info, log_warn
from modules.common.ui.logging import log_debug, log_error, log_info, log_warn

"""
Task Manager for managing background tasks and their status.
"""




class TaskManager:
    """Manages background tasks and their status."""

    def __init__(self, cleanup_after_hours: int = 1):
        """
        Initialize TaskManager.

        Args:
            cleanup_after_hours: Hours after which to cleanup completed tasks
        """
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.cleanup_after_hours = cleanup_after_hours

        # Add a shutdown event to allow controlled cleanup of the background thread.
        self._shutdown_event = threading.Event()
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

    def start_task(self, session_id: str, task_func: Callable, command_type: str = "scan") -> None:
        """
        Start a background task.

        Args:
            session_id: Unique session identifier
            task_func: Function to run in background thread
            command_type: Type of command ('scan' or 'analyze')
        """
        with self._lock:
            self._tasks[session_id] = {
                "status": "running",
                "command_type": command_type,
                "started_at": datetime.now(),
                "result": None,
                "error": None,
                "thread": None,
                "cancelled": False,
            }

        def run_task():
            """Run the task function and update status."""
            try:
                log_info(f"Background task started for session {session_id}")

                # Check if cancelled before starting
                with self._lock:
                    if session_id in self._tasks and self._tasks[session_id].get("cancelled", False):
                        log_info(f"Task {session_id} was cancelled before execution")
                        self._tasks[session_id]["status"] = "cancelled"
                        self._tasks[session_id]["completed_at"] = datetime.now()
                        return

                result = task_func()

                with self._lock:
                    if session_id in self._tasks:
                        # Check if cancelled during execution
                        if self._tasks[session_id].get("cancelled", False):
                            log_info(f"Task {session_id} was cancelled during execution")
                            self._tasks[session_id]["status"] = "cancelled"
                            self._tasks[session_id]["completed_at"] = datetime.now()
                            return

                        # Only set status to completed if not already set by set_result()
                        # Only set result if it hasn't been set yet (by set_result())
                        if self._tasks[session_id].get("status") != "completed":
                            self._tasks[session_id]["status"] = "completed"
                        if self._tasks[session_id].get("result") is None and result is not None:
                            self._tasks[session_id]["result"] = result
                        elif self._tasks[session_id].get("result") is not None:
                            # Result already set by set_result(), don't overwrite
                            pass
                        if not self._tasks[session_id].get("completed_at"):
                            self._tasks[session_id]["completed_at"] = datetime.now()

                log_info(f"Background task completed for session {session_id}")
            except Exception as e:
                log_error(f"Background task error for session {session_id}: {e}")

                with self._lock:
                    if session_id in self._tasks:
                        # Don't set error if task was cancelled
                        if not self._tasks[session_id].get("cancelled", False):
                            self._tasks[session_id]["status"] = "error"
                            self._tasks[session_id]["error"] = str(e)
                            self._tasks[session_id]["completed_at"] = datetime.now()

        thread = threading.Thread(target=run_task, daemon=True, name=f"task-{session_id}")
        thread.start()

        with self._lock:
            if session_id in self._tasks:
                self._tasks[session_id]["thread"] = thread

    def get_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a task.

        Args:
            session_id: Unique session identifier

        Returns:
            Dict with status info, or None if not found
        """
        with self._lock:
            if session_id not in self._tasks:
                return None

            task = self._tasks[session_id].copy()
            # Remove thread object from response
            task.pop("thread", None)
            # Shallow copy should be sufficient, as 'task' was already copied and 'thread' removed.
            # If deeply nested mutable results are added to 'task', consider using deepcopy here for safety.
            return task

    def set_result(self, session_id: str, result: Any) -> None:
        """
        Set result for a completed task.

        Args:
            session_id: Unique session identifier
            result: Result data
        """
        with self._lock:
            if session_id in self._tasks:
                self._tasks[session_id]["result"] = result
                self._tasks[session_id]["status"] = "completed"
                self._tasks[session_id]["completed_at"] = datetime.now()

    def get_result(self, session_id: str) -> Optional[Any]:
        """
        Get result of a completed task.

        Args:
            session_id: Unique session identifier

        Returns:
            Result data, or None if not found or not completed
        """
        with self._lock:
            if session_id not in self._tasks:
                return None

            task = self._tasks[session_id]
            if task["status"] == "completed":
                return task.get("result")
            return None

    def set_error(self, session_id: str, error: str) -> None:
        """
        Set error for a failed task.

        Args:
            session_id: Unique session identifier
            error: Error message
        """
        with self._lock:
            if session_id in self._tasks:
                self._tasks[session_id]["error"] = error
                self._tasks[session_id]["status"] = "error"
                self._tasks[session_id]["completed_at"] = datetime.now()

    def _cleanup_loop(self):
        """Background thread to cleanup old tasks."""
        while not self._shutdown_event.is_set():
            try:
                # Wait with timeout to allow checking shutdown event periodically
                if self._shutdown_event.wait(timeout=300):  # Check every 5 minutes or on shutdown
                    break
                self._cleanup_old_tasks()
            except Exception as e:
                log_error(f"Error in cleanup loop: {e}")

    def _remove_task(self, session_id: str) -> None:
        """
        Remove a task and its data structures.
        Assumes lock is already held by caller.

        Args:
            session_id: Unique session identifier
        """
        if session_id in self._tasks:
            del self._tasks[session_id]

    def _cleanup_old_tasks(self):
        """Remove tasks older than cleanup_after_hours and free memory."""
        cutoff_time = datetime.now() - timedelta(hours=self.cleanup_after_hours)

        with self._lock:
            to_remove = []
            for session_id, task in self._tasks.items():
                completed_at = task.get("completed_at")
                if completed_at and completed_at < cutoff_time:
                    to_remove.append(session_id)

            for session_id in to_remove:
                self._remove_task(session_id)
                log_debug(f"Cleaned up task: {session_id}")

        # Explicit garbage collection is generally not needed and Python's GC
        # automatically manages memory efficiently. If memory pressure arises
        # with large tasks in long-running services, consider profiling
        # memory usage to decide if gc.collect() calls are beneficial.
        #
        # Example (commented out):
        # if to_remove:
        #     import gc
        #     gc.collect()
        #     log_debug(f"Garbage collected after cleaning {len(to_remove)} tasks")

    def cancel_task(self, session_id: str) -> bool:
        """
        Cancel a running task.

        Args:
            session_id: Unique session identifier

        Returns:
            True if task was found and cancelled, False otherwise
        """
        with self._lock:
            if session_id not in self._tasks:
                return False

            task = self._tasks[session_id]
            current_status = task.get("status")

            # Only allow cancelling running tasks
            if current_status != "running":
                log_warn(f"Cannot cancel task {session_id}: status is {current_status}")
                return False

            # Set cancelled flag and update status immediately
            task["cancelled"] = True
            task["status"] = "cancelled"
            task["completed_at"] = datetime.now()
            log_info(f"Task {session_id} marked as cancelled")
            return True

    def is_cancelled(self, session_id: str) -> bool:
        """
        Check if a task is cancelled.

        Args:
            session_id: Unique session identifier

        Returns:
            True if task is cancelled, False otherwise
        """
        with self._lock:
            if session_id not in self._tasks:
                return False
            return self._tasks[session_id].get("cancelled", False)

    def cleanup_task(self, session_id: str) -> bool:
        """
        Manually cleanup a specific task and free memory.

        Args:
            session_id: Unique session identifier

        Returns:
            True if task was found and removed
        """
        with self._lock:
            if session_id in self._tasks:
                self._remove_task(session_id)
                return True
            return False

    def clear_all_results(self):
        """
        Clear all result data from completed tasks to free memory.
        Keeps task metadata but removes large result objects.
        """
        with self._lock:
            cleared_count = 0
            for session_id, task in self._tasks.items():
                if task.get("status") in ("completed", "error", "cancelled") and task.get("result") is not None:
                    task["result"] = None
                    cleared_count += 1
            log_info(f"Cleared results from {cleared_count} tasks")

        if cleared_count > 0:
            gc.collect()

    def shutdown(self, task_timeout: float = 10.0):
        """
        Gracefully shutdown the TaskManager.

        Cancels all running tasks, waits for their threads to complete (up to ``task_timeout`` per thread),
        then shuts down the cleanup thread.

        Args:
            task_timeout: Maximum time (in seconds) to wait for each task thread to complete (per-thread timeout).
                          Default is 10 seconds per thread. With multiple running tasks, total shutdown may take
                          up to N * task_timeout. Tasks are daemon threads, so they will be terminated by Python
                          if the process exits before they complete.
        """
        if self._shutdown_event.is_set():
            return  # Already shutting down

        log_info("Shutting down TaskManager...")
        self._shutdown_event.set()

        # Cancel all running tasks and collect their threads
        running_threads = []
        with self._lock:
            for session_id, task in self._tasks.items():
                if task.get("status") == "running":
                    task["cancelled"] = True
                    task["status"] = "cancelled"
                    task["completed_at"] = datetime.now()
                    log_info(f"Task {session_id} marked as cancelled during shutdown")
                    thread = task.get("thread")
                    if thread and thread.is_alive():
                        running_threads.append((session_id, thread))

        # Wait for task threads to complete (with timeout)
        if running_threads:
            log_info(f"Waiting for {len(running_threads)} task thread(s) to complete...")
            for session_id, thread in running_threads:
                thread.join(timeout=task_timeout)
                if thread.is_alive():
                    log_warn(f"Task thread {session_id} did not complete within {task_timeout}s timeout")
                else:
                    log_debug(f"Task thread {session_id} completed")

        # Shutdown cleanup thread
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5.0)
            if self._cleanup_thread.is_alive():
                log_warn("Cleanup thread did not exit within timeout")
            else:
                log_info("TaskManager shutdown complete")


# Global instance
_task_manager_instance: Optional[TaskManager] = None
_task_manager_lock = threading.Lock()


def get_task_manager() -> TaskManager:
    """Get global TaskManager instance."""
    global _task_manager_instance
    with _task_manager_lock:
        if _task_manager_instance is None:
            _task_manager_instance = TaskManager()
        return _task_manager_instance

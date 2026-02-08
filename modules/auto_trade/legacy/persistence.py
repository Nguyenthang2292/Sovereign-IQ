"""
Signal Persistence Module

Handles saving trade signals for historical analysis and accuracy tracking.
"""

import json
import os
import platform
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from modules.auto_trade.core.signal_selector import FinalSignal
from modules.common.ui.logging import log_error, log_info, log_warn

if platform.system() != "Windows":
    import fcntl


class SignalPersistence:
    """
    Manages storage of trading signals.

    Features:
    - Thread-safe concurrent writes
    - Daily file rotation
    - Size-based rotation
    - Data validation
    - Historical query methods
    - Disk space monitoring with caching
    - Multi-process file locking (Unix)
    - Metrics and monitoring
    """

    DISK_SPACE_ERROR_THRESHOLD_MB = 100
    DISK_SPACE_WARN_THRESHOLD_MB = 500
    MAX_FILE_SIZE_BYTES = 100_000_000
    DISK_CHECK_INTERVAL_SECONDS = 60

    def __init__(
        self, storage_dir: str = "data/signals", enable_rotation: bool = True, validate_path: bool = True
    ) -> None:
        """
        Initialize signal persistence.

        Args:
            storage_dir: Directory for signal storage
            enable_rotation: Enable daily file rotation
            validate_path: Validate storage directory is within allowed base (disable for tests)
        """
        # Resolve to absolute path
        base_dir = Path("data").resolve()
        storage_path = Path(storage_dir).resolve()

        if validate_path:
            try:
                storage_path.relative_to(base_dir)
            except ValueError:
                raise ValueError(f"Invalid storage directory: {storage_dir}")

        self.storage_dir = storage_path
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.enable_rotation = enable_rotation
        self._lock = threading.Lock()
        self._last_disk_check: float = 0.0
        self._disk_check_interval = self.DISK_CHECK_INTERVAL_SECONDS
        self.metrics = {
            "total_writes": 0,
            "failed_writes": 0,
            "total_bytes_written": 0,
            "avg_write_time_ms": 0.0,
        }

    def _get_current_filename(self) -> Path:
        """Generate filename with date for rotation."""
        if self.enable_rotation:
            date_str = datetime.now().strftime("%Y-%m-%d")
            filename = self.storage_dir / f"signal_history_{date_str}.jsonl"

            if filename.exists() and filename.stat().st_size > self.MAX_FILE_SIZE_BYTES:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = self.storage_dir / f"signal_history_{timestamp}.jsonl"

            return filename
        else:
            return self.storage_dir / "signal_history.jsonl"

    def _check_disk_space(self) -> bool:
        """Check if sufficient disk space available (cached for 60 seconds)."""
        now = time.time()
        if now - self._last_disk_check < self._disk_check_interval:
            return True

        self._last_disk_check = now
        import shutil

        stat = shutil.disk_usage(self.storage_dir)
        available_mb = stat.free / (1024 * 1024)

        if available_mb < self.DISK_SPACE_ERROR_THRESHOLD_MB:
            log_error(f"Low disk space: {available_mb:.1f}MB available")
            return False

        if available_mb < self.DISK_SPACE_WARN_THRESHOLD_MB:
            log_warn(f"Disk space running low: {available_mb:.1f}MB available")

        return True

    def save_signal(self, signal: FinalSignal) -> bool:
        """
        Append a signal to the history file.

        Args:
            signal: The FinalSignal to save.

        Returns:
            True if successful, False otherwise.
        """
        start_time = time.time()

        try:
            if not self._check_disk_space():
                self.metrics["failed_writes"] += 1
                return False

            if not signal.symbol or not signal.signal_type:
                log_error("Invalid signal: missing symbol or signal_type")
                self.metrics["failed_writes"] += 1
                return False

            try:
                timestamp_str = datetime.fromtimestamp(signal.timestamp).isoformat()
            except (ValueError, OSError, OverflowError) as e:
                log_error(f"Invalid timestamp {signal.timestamp}: {e}")
                self.metrics["failed_writes"] += 1
                return False

            if signal.entry_price <= 0:
                log_error(f"Invalid entry price: {signal.entry_price}")
                self.metrics["failed_writes"] += 1
                return False

            record = {
                "timestamp": timestamp_str,
                "symbol": signal.symbol,
                "type": signal.signal_type,
                "confidence": signal.confidence,
                "entry": signal.entry_price,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "sources": signal.sources,
            }

            with self._lock:
                filename = self._get_current_filename()
                with open(filename, "a", encoding="utf-8") as f:
                    if platform.system() != "Windows":
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # type: ignore
                    try:
                        data = json.dumps(record) + "\n"
                        f.write(data)
                        f.flush()
                        os.fsync(f.fileno())
                        self.metrics["total_bytes_written"] += len(data.encode("utf-8"))
                    finally:
                        if platform.system() != "Windows":
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # type: ignore

            self.metrics["total_writes"] += 1
            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics["avg_write_time_ms"] = (
                self.metrics["avg_write_time_ms"] * (self.metrics["total_writes"] - 1) + elapsed_ms
            ) / self.metrics["total_writes"]

            log_info(f"Saved signal for {signal.symbol} to history.")
            return True

        except Exception as e:
            self.metrics["failed_writes"] += 1
            log_error(f"Failed to save signal history: {e}")
            return False

    def read_signals(
        self, from_date: Optional[date] = None, to_date: Optional[date] = None, symbol: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Read signals from history with optional filtering.

        Args:
            from_date: Start date filter (inclusive)
            to_date: End date filter (inclusive)
            symbol: Filter by specific symbol

        Yields:
            Signal records as dictionaries
        """
        files = sorted(self.storage_dir.glob("signal_history*.jsonl"))

        for filepath in files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue

                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            log_error(f"Corrupted line in {filepath}")
                            continue

                        if from_date or to_date:
                            signal_date = datetime.fromisoformat(record["timestamp"]).date()
                            if from_date and signal_date < from_date:
                                continue
                            if to_date and signal_date > to_date:
                                continue

                        if symbol and record.get("symbol") != symbol:
                            continue

                        yield record

            except Exception as e:
                log_error(f"Error reading {filepath}: {e}")
                continue

    def get_signal_count(self, from_date: Optional[date] = None, to_date: Optional[date] = None) -> int:
        """Get total number of stored signals."""
        return sum(1 for _ in self.read_signals(from_date, to_date))

    def get_signals_by_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """Get all signals for a specific symbol."""
        return list(self.read_signals(symbol=symbol))

    def get_recent_signals(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get signals from the last N days."""
        to_date = date.today()
        from_date = to_date - timedelta(days=days)
        return list(self.read_signals(from_date=from_date, to_date=to_date))

    def get_metrics(self) -> Dict[str, Any]:
        """Get persistence metrics."""
        return self.metrics.copy()

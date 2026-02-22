"""
SQLite-based Signal Persistence Module

Handles saving trade signals with improved query performance and analytics.
"""

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from modules.auto_trade.core.signal_selector import FinalSignal
from modules.common.ui.logging import log_error, log_info, log_warn


class SignalPersistenceSQLite:
    """
    SQLite-based signal persistence with improved query performance.

    Features:
    - ACID transactions for data integrity
    - Efficient indexed queries
    - Advanced filtering and aggregations
    - Thread-safe operations
    - Performance metrics
    - Outcome tracking support
    """

    DISK_SPACE_ERROR_THRESHOLD_MB: int = 100
    DISK_SPACE_WARN_THRESHOLD_MB: int = 500

    db_path: Path
    _lock: threading.Lock
    _last_disk_check: float
    _disk_check_interval: int
    metrics: Dict[str, Any]

    def __init__(self, db_path: str = "data/signals/signals.db", enable_wal: bool = True) -> None:
        """
        Initialize SQLite persistence.

        Args:
            db_path: Path to SQLite database file
            enable_wal: Enable Write-Ahead Logging for better concurrency
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._last_disk_check = 0.0
        self._disk_check_interval = 60
        self.metrics = {
            "total_writes": 0,
            "failed_writes": 0,
            "total_bytes_written": 0,
            "avg_write_time_ms": 0.0,
        }

        # Initialize database
        self._init_database(enable_wal)

    def _init_database(self, enable_wal: bool) -> None:
        """Create tables and indexes if not exist."""
        with self._get_connection() as conn:
            # Enable WAL mode for better concurrency
            if enable_wal:
                conn.execute("PRAGMA journal_mode=WAL")

            # Create signals table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    sources TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_type ON signals(type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol_timestamp ON signals(symbol, timestamp)")

            # Create metrics table for outcome tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signal_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id INTEGER NOT NULL,
                    outcome TEXT,
                    profit_loss REAL,
                    duration_seconds INTEGER,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (signal_id) REFERENCES signals(id) ON DELETE CASCADE
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_signal_id ON signal_metrics(signal_id)")

            conn.commit()
            log_info(f"SQLite database initialized: {self.db_path}")

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _check_disk_space(self) -> bool:
        """Check if sufficient disk space available (cached for 60 seconds)."""
        now = time.time()
        if now - self._last_disk_check < self._disk_check_interval:
            return True

        self._last_disk_check = now
        import shutil

        stat = shutil.disk_usage(self.db_path.parent)
        available_mb = stat.free / (1024 * 1024)

        if available_mb < self.DISK_SPACE_ERROR_THRESHOLD_MB:
            log_error(f"Low disk space: {available_mb:.1f}MB available")
            return False

        if available_mb < self.DISK_SPACE_WARN_THRESHOLD_MB:
            log_warn(f"Disk space running low: {available_mb:.1f}MB available")

        return True

    def save_signal(self, signal: FinalSignal) -> Optional[int]:
        """
        Save a signal to the database.

        Args:
            signal: The FinalSignal to save.

        Returns:
            Signal ID if successful, None otherwise.
        """
        start_time = time.time()

        try:
            if not self._check_disk_space():
                self.metrics["failed_writes"] += 1
                return None

            # Validation
            if not signal.symbol or not signal.signal_type:
                log_error("Invalid signal: missing symbol or signal_type")
                self.metrics["failed_writes"] += 1
                return None

            if signal.entry_price <= 0:
                log_error(f"Invalid entry price: {signal.entry_price}")
                self.metrics["failed_writes"] += 1
                return None

            # Prepare data
            try:
                timestamp_str = datetime.fromtimestamp(signal.timestamp).isoformat()
            except (ValueError, OSError, OverflowError) as e:
                log_error(f"Invalid timestamp {signal.timestamp}: {e}")
                self.metrics["failed_writes"] += 1
                return None

            sources_json = json.dumps(signal.sources)

            with self._lock:
                with self._get_connection() as conn:
                    cursor = conn.execute(
                        """
                        INSERT INTO signals (
                            timestamp, symbol, type, confidence,
                            entry_price, stop_loss, take_profit, sources
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            timestamp_str,
                            signal.symbol,
                            signal.signal_type,
                            signal.confidence,
                            signal.entry_price,
                            signal.stop_loss,
                            signal.take_profit,
                            sources_json,
                        ),
                    )
                    conn.commit()
                    signal_id = cursor.lastrowid

            # Update metrics
            self.metrics["total_writes"] += 1
            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics["avg_write_time_ms"] = (
                self.metrics["avg_write_time_ms"] * (self.metrics["total_writes"] - 1) + elapsed_ms
            ) / self.metrics["total_writes"]

            log_info(f"Saved signal for {signal.symbol} (ID: {signal_id})")
            return signal_id

        except Exception as e:
            self.metrics["failed_writes"] += 1
            log_error(f"Failed to save signal: {e}")
            return None

    def read_signals(
        self,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        symbol: Optional[str] = None,
        signal_type: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Read signals with advanced filtering.

        Args:
            from_date: Start date filter (inclusive)
            to_date: End date filter (inclusive)
            symbol: Filter by symbol
            signal_type: Filter by type ("LONG" or "SHORT")
            limit: Maximum number of results
            offset: Pagination offset

        Returns:
            List of signal records
        """
        query = "SELECT * FROM signals WHERE 1=1"
        params = []

        if from_date:
            query += " AND date(timestamp) >= ?"
            params.append(from_date.isoformat())

        if to_date:
            query += " AND date(timestamp) <= ?"
            params.append(to_date.isoformat())

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        if signal_type:
            query += " AND type = ?"
            params.append(signal_type)

        query += " ORDER BY timestamp DESC"

        if limit:
            query += f" LIMIT {limit} OFFSET {offset}"

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            results = []
            for row in cursor.fetchall():
                record = dict(row)
                # Parse sources JSON
                if record.get("sources"):
                    try:
                        record["sources"] = json.loads(record["sources"])
                    except json.JSONDecodeError:
                        record["sources"] = []
                results.append(record)
            return results

    def get_signal_count(
        self, from_date: Optional[date] = None, to_date: Optional[date] = None, symbol: Optional[str] = None
    ) -> int:
        """Get count of signals matching filters."""
        query = "SELECT COUNT(*) as count FROM signals WHERE 1=1"
        params = []

        if from_date:
            query += " AND date(timestamp) >= ?"
            params.append(from_date.isoformat())

        if to_date:
            query += " AND date(timestamp) <= ?"
            params.append(to_date.isoformat())

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return cursor.fetchone()["count"]

    def get_signals_by_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """Get all signals for a specific symbol."""
        return self.read_signals(symbol=symbol)

    def get_recent_signals(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get signals from the last N days."""
        to_date = date.today()
        from_date = to_date - timedelta(days=days)
        return self.read_signals(from_date=from_date, to_date=to_date)

    def get_statistics(self, from_date: Optional[date] = None, to_date: Optional[date] = None) -> Dict[str, Any]:
        """Get signal statistics for a date range."""
        query = """
            SELECT
                COUNT(*) as total_signals,
                COUNT(DISTINCT symbol) as unique_symbols,
                SUM(CASE WHEN type = 'LONG' THEN 1 ELSE 0 END) as long_signals,
                SUM(CASE WHEN type = 'SHORT' THEN 1 ELSE 0 END) as short_signals,
                AVG(confidence) as avg_confidence,
                MIN(timestamp) as first_signal,
                MAX(timestamp) as last_signal
            FROM signals
            WHERE 1=1
        """
        params = []

        if from_date:
            query += " AND date(timestamp) >= ?"
            params.append(from_date.isoformat())

        if to_date:
            query += " AND date(timestamp) <= ?"
            params.append(to_date.isoformat())

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return dict(cursor.fetchone())

    def update_signal_outcome(
        self, signal_id: int, outcome: str, profit_loss: Optional[float] = None, duration_seconds: Optional[int] = None
    ) -> bool:
        """
        Update outcome tracking for a signal.

        Args:
            signal_id: The signal ID to update
            outcome: Outcome status ("WIN", "LOSS", "PENDING")
            profit_loss: Actual profit/loss value
            duration_seconds: Time to close position

        Returns:
            True if successful, False otherwise
        """
        try:
            with self._lock:
                with self._get_connection() as conn:
                    conn.execute(
                        """
                        INSERT INTO signal_metrics (signal_id, outcome, profit_loss, duration_seconds)
                        VALUES (?, ?, ?, ?)
                    """,
                        (signal_id, outcome, profit_loss, duration_seconds),
                    )
                    conn.commit()
            log_info(f"Updated outcome for signal {signal_id}: {outcome}")
            return True
        except Exception as e:
            log_error(f"Failed to update signal outcome: {e}")
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """Get persistence metrics."""
        return self.metrics.copy()

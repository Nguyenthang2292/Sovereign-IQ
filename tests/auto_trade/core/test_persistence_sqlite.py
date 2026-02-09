"""Tests for SQLite Signal Persistence Module."""

import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from modules.auto_trade.core.persistence_sqlite import SignalPersistenceSQLite
from modules.auto_trade.core.signal_selector import FinalSignal


class TestSignalPersistenceSQLite:
    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create temporary database for testing."""
        db_path = tmp_path / "test_signals.db"
        return str(db_path)

    @pytest.fixture
    def persistence(self, temp_db):
        """Create persistence instance for testing."""
        return SignalPersistenceSQLite(db_path=temp_db)

    @pytest.fixture
    def sample_signal(self):
        """Create sample signal for testing."""
        return FinalSignal(
            symbol="BTCUSDT",
            signal_type="LONG",
            confidence=0.85,
            entry_price=50000.0,
            stop_loss=48000.0,
            take_profit=55000.0,
            sources=["atc", "xgboost"],  # type: ignore[arg-type]
            timestamp=datetime.now().timestamp(),
        )

    def test_database_initialization(self, persistence):
        """Test that database is initialized with proper schema."""
        with persistence._get_connection() as conn:
            # Check signals table exists
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='signals'")
            assert cursor.fetchone() is not None

            # Check signal_metrics table exists
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='signal_metrics'")
            assert cursor.fetchone() is not None

            # Check indexes exist
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_signals_timestamp'")
            assert cursor.fetchone() is not None

    def test_save_signal_basic(self, persistence, sample_signal):
        """Test basic signal saving."""
        signal_id = persistence.save_signal(sample_signal)

        assert signal_id is not None
        assert signal_id > 0
        assert persistence.metrics["total_writes"] == 1
        assert persistence.metrics["failed_writes"] == 0

    def test_save_signal_validation(self, persistence):
        """Test signal validation on save."""
        # Missing symbol
        signal = FinalSignal(
            symbol="",
            signal_type="LONG",
            confidence=0.5,
            entry_price=100.0,
            stop_loss=90.0,
            take_profit=110.0,
            timestamp=time.time(),
        )
        assert persistence.save_signal(signal) is None
        assert persistence.metrics["failed_writes"] == 1

        assert persistence.metrics["failed_writes"] == 1

    def test_read_signals_basic(self, persistence, sample_signal):
        """Test basic signal reading."""
        persistence.save_signal(sample_signal)

        signals = persistence.read_signals()
        assert len(signals) == 1
        assert signals[0]["symbol"] == "BTCUSDT"
        assert signals[0]["type"] == "LONG"
        assert signals[0]["confidence"] == 0.85
        assert signals[0]["entry_price"] == 50000.0

    def test_read_signals_filtering(self, persistence):
        """Test signal filtering."""
        # Create signals with different attributes
        signals_data = [
            ("BTCUSDT", "LONG", datetime.now()),
            ("ETHUSDT", "SHORT", datetime.now()),
            ("BTCUSDT", "SHORT", datetime.now() - timedelta(days=1)),
        ]

        for symbol, signal_type, timestamp in signals_data:
            signal = FinalSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=0.5,
                entry_price=1000.0,
                stop_loss=900.0 if signal_type == "LONG" else 1100.0,
                take_profit=1100.0 if signal_type == "LONG" else 900.0,
                timestamp=timestamp.timestamp(),
            )
            persistence.save_signal(signal)

        # Test symbol filtering
        btc_signals = persistence.read_signals(symbol="BTCUSDT")
        assert len(btc_signals) == 2

        # Test type filtering
        long_signals = persistence.read_signals(signal_type="LONG")
        assert len(long_signals) == 1

        # Test date filtering
        today = date.today()
        today_signals = persistence.read_signals(from_date=today, to_date=today)
        assert len(today_signals) == 2

    def test_read_signals_pagination(self, persistence):
        """Test pagination support."""
        # Create 10 signals
        for i in range(10):
            signal = FinalSignal(
                symbol="BTC" + str(i),
                signal_type="LONG",
                confidence=0.5,
                entry_price=1000.0,
                stop_loss=900.0,
                take_profit=1100.0,
                timestamp=time.time(),
            )
            persistence.save_signal(signal)

        # Test limit
        page1 = persistence.read_signals(limit=5)
        assert len(page1) == 5

        # Test offset
        page2 = persistence.read_signals(limit=5, offset=5)
        assert len(page2) == 5
        assert page1[0]["id"] != page2[0]["id"]

    def test_get_signal_count(self, persistence, sample_signal):
        """Test signal counting."""
        assert persistence.get_signal_count() == 0

        persistence.save_signal(sample_signal)
        assert persistence.get_signal_count() == 1

        persistence.save_signal(sample_signal)
        assert persistence.get_signal_count() == 2

    def test_get_signals_by_symbol(self, persistence):
        """Test getting signals by symbol."""
        for symbol in ["BTCUSDT", "ETHUSDT", "BTCUSDT"]:
            signal = FinalSignal(
                symbol=symbol,
                signal_type="LONG",
                confidence=0.5,
                entry_price=1000.0,
                stop_loss=900.0,
                take_profit=1100.0,
                timestamp=time.time(),
            )
            persistence.save_signal(signal)

        btc_signals = persistence.get_signals_by_symbol("BTCUSDT")
        assert len(btc_signals) == 2
        assert all(s["symbol"] == "BTCUSDT" for s in btc_signals)

    def test_get_recent_signals(self, persistence):
        """Test getting recent signals."""
        # Create signals at different times
        now = datetime.now()
        for days_ago in [0, 3, 10]:
            signal = FinalSignal(
                symbol="BTCUSDT",
                signal_type="LONG",
                confidence=0.5,
                entry_price=1000.0,
                stop_loss=900.0,
                take_profit=1100.0,
                timestamp=(now - timedelta(days=days_ago)).timestamp(),
            )
            persistence.save_signal(signal)

        # Get last 7 days
        recent = persistence.get_recent_signals(days=7)
        assert len(recent) == 2

    def test_get_statistics(self, persistence):
        """Test statistics aggregation."""
        # Create diverse signals
        signals_data = [
            ("BTCUSDT", "LONG", 0.8),
            ("ETHUSDT", "SHORT", 0.6),
            ("BTCUSDT", "LONG", 0.9),
        ]

        for symbol, signal_type, confidence in signals_data:
            signal = FinalSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                entry_price=1000.0,
                stop_loss=900.0 if signal_type == "LONG" else 1100.0,
                take_profit=1100.0 if signal_type == "LONG" else 900.0,
                timestamp=time.time(),
            )
            persistence.save_signal(signal)

        stats = persistence.get_statistics()
        assert stats["total_signals"] == 3
        assert stats["unique_symbols"] == 2
        assert stats["long_signals"] == 2
        assert stats["short_signals"] == 1
        assert 0.7 < stats["avg_confidence"] < 0.8

    def test_update_signal_outcome(self, persistence, sample_signal):
        """Test outcome tracking."""
        signal_id = persistence.save_signal(sample_signal)

        success = persistence.update_signal_outcome(
            signal_id=signal_id, outcome="WIN", profit_loss=500.0, duration_seconds=3600
        )
        assert success is True

        # Verify outcome was saved
        with persistence._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM signal_metrics WHERE signal_id = ?", (signal_id,))
            result = cursor.fetchone()
            assert result is not None
            assert result["outcome"] == "WIN"
            assert result["profit_loss"] == 500.0

    def test_concurrent_writes(self, persistence):
        """Test thread-safe concurrent writes."""
        num_threads = 10
        signals_per_thread = 10

        def write_signals(thread_id):
            for i in range(signals_per_thread):
                signal = FinalSignal(
                    symbol=f"THREAD{thread_id}_{i}",
                    signal_type="LONG",
                    confidence=0.5,
                    entry_price=1000.0,
                    stop_loss=900.0,
                    take_profit=1100.0,
                    timestamp=time.time(),
                )
                persistence.save_signal(signal)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            list(executor.map(write_signals, range(num_threads)))

        assert persistence.get_signal_count() == num_threads * signals_per_thread

    def test_concurrent_reads(self, persistence, sample_signal):
        """Test concurrent read operations."""
        # Pre-populate with data
        for i in range(100):
            persistence.save_signal(sample_signal)

        def read_signals():
            return persistence.read_signals(limit=10)

        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(lambda _: read_signals(), range(10)))

        # All reads should succeed
        assert all(len(r) == 10 for r in results)

    def test_sources_json_serialization(self, persistence):
        """Test that sources are properly serialized/deserialized."""
        signal = FinalSignal(
            symbol="BTCUSDT",
            signal_type="LONG",
            confidence=0.5,
            entry_price=1000.0,
            stop_loss=900.0,
            take_profit=1100.0,
            sources=["atc", "xgboost", "gemini"],  # type: ignore[arg-type]
            timestamp=time.time(),
        )
        persistence.save_signal(signal)

        signals = persistence.read_signals()
        assert len(signals) == 1
        assert signals[0]["sources"] == ["atc", "xgboost", "gemini"]

    def test_metrics_tracking(self, persistence, sample_signal):
        """Test that metrics are properly tracked."""
        assert persistence.metrics["total_writes"] == 0
        assert persistence.metrics["failed_writes"] == 0

        persistence.save_signal(sample_signal)

        assert persistence.metrics["total_writes"] == 1
        assert persistence.metrics["failed_writes"] == 0
        assert persistence.metrics["avg_write_time_ms"] > 0

        metrics = persistence.get_metrics()
        assert metrics["total_writes"] == 1

"""Tests for SignalPersistence."""

import json
import shutil
import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from modules.auto_trade.legacy.persistence import SignalPersistence
from modules.auto_trade.core.signal_selector import FinalSignal


class TestSignalPersistence:
    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Provide temporary storage directory."""
        return str(tmp_path / "test_signals")

    @pytest.fixture
    def persistence(self, temp_storage):
        """Create SignalPersistence instance."""
        return SignalPersistence(storage_dir=temp_storage, enable_rotation=False, validate_path=False)

    def test_save_signal_success(self, persistence, temp_storage):
        """Test successful signal save."""
        signal = FinalSignal(
            symbol="BTC/USDT",
            signal_type="LONG",
            entry_price=50000,
            stop_loss=49000,
            take_profit=52000,
            timestamp=time.time(),
            confidence=0.85,
            sources={"xgboost_score": 0.8, "gemini_score": 0.9},
        )

        result = persistence.save_signal(signal)

        assert result is True

        history_file = Path(temp_storage) / "signal_history.jsonl"
        assert history_file.exists()

        with open(history_file, "r") as f:
            line = f.readline()
            record = json.loads(line)
            assert record["symbol"] == "BTC/USDT"
            assert record["type"] == "LONG"
            assert record["confidence"] == 0.85
            assert record["entry"] == 50000
            assert record["stop_loss"] == 49000
            assert record["take_profit"] == 52000

    def test_save_signal_creates_directory(self, tmp_path):
        """Test that storage directory is created if missing."""
        storage_dir = str(tmp_path / "nested" / "storage")
        persistence = SignalPersistence(storage_dir=storage_dir, validate_path=False)

        assert Path(storage_dir).exists()

        signal = FinalSignal("BTC/USDT", "LONG", 50000, 49000, 52000)
        assert persistence.save_signal(signal) is True

    def test_save_signal_multiple_writes(self, persistence, temp_storage):
        """Test multiple signals can be appended."""
        signals = [
            FinalSignal("BTC/USDT", "LONG", 50000, 49000, 52000, timestamp=time.time()),
            FinalSignal("ETH/USDT", "SHORT", 3000, 3100, 2900, timestamp=time.time()),
            FinalSignal("BNB/USDT", "LONG", 400, 390, 420, timestamp=time.time()),
        ]

        for signal in signals:
            assert persistence.save_signal(signal) is True

        history_file = Path(temp_storage) / "signal_history.jsonl"
        with open(history_file, "r") as f:
            lines = f.readlines()
            assert len(lines) == 3

            for i, line in enumerate(lines):
                record = json.loads(line)
                assert record["symbol"] == signals[i].symbol
                assert record["type"] == signals[i].signal_type

    def test_save_signal_invalid_timestamp(self, persistence):
        """Test handling of invalid timestamp."""
        signal = FinalSignal(
            symbol="BTC/USDT",
            signal_type="LONG",
            entry_price=50000,
            stop_loss=49000,
            take_profit=52000,
            timestamp=-1,
        )

        result = persistence.save_signal(signal)

        assert isinstance(result, bool)

    def test_save_signal_concurrent_writes(self, persistence, temp_storage):
        """Test concurrent writes are thread-safe."""
        signals = [
            FinalSignal(f"SYMBOL{i}/USDT", "LONG", 1000 + i, 900 + i, 1100 + i, timestamp=time.time())
            for i in range(10)
        ]

        threads = []
        for signal in signals:
            thread = threading.Thread(target=persistence.save_signal, args=(signal,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        history_file = Path(temp_storage) / "signal_history.jsonl"
        with open(history_file, "r") as f:
            lines = f.readlines()
            assert len(lines) == 10

            symbols = set()
            for line in lines:
                record = json.loads(line)
                symbols.add(record["symbol"])

            assert len(symbols) == 10

    def test_save_signal_file_write_error(self, persistence):
        """Test handling of file write errors."""
        signal = FinalSignal("BTC/USDT", "LONG", 50000, 49000, 52000)

        with patch("builtins.open", side_effect=IOError("Permission denied")):
            result = persistence.save_signal(signal)
            assert result is False

    def test_signal_json_format(self, persistence, temp_storage):
        """Test that saved JSON has correct format."""
        signal = FinalSignal(
            symbol="BTC/USDT",
            signal_type="LONG",
            entry_price=50000.5,
            stop_loss=49000.0,
            take_profit=52000.0,
            timestamp=1704067200.0,
            confidence=0.85,
            sources={"xgboost_score": 0.8, "gemini_score": 0.9, "gemini_reasoning": "Test"},
        )

        persistence.save_signal(signal)

        history_file = Path(temp_storage) / "signal_history.jsonl"
        with open(history_file, "r") as f:
            record = json.loads(f.readline())

        assert "timestamp" in record
        assert "symbol" in record
        assert "type" in record
        assert "confidence" in record
        assert "entry" in record
        assert "stop_loss" in record
        assert "take_profit" in record
        assert "sources" in record

        assert isinstance(record["timestamp"], str)
        assert isinstance(record["symbol"], str)
        assert isinstance(record["confidence"], float)
        assert isinstance(record["sources"], dict)

    def test_save_signal_empty_sources(self, persistence):
        """Test signal with empty sources dict."""
        signal = FinalSignal(
            symbol="BTC/USDT", signal_type="LONG", entry_price=50000, stop_loss=49000, take_profit=52000, sources={}
        )

        result = persistence.save_signal(signal)
        assert result is True

    def test_read_signals(self, persistence, temp_storage):
        """Test reading signals from history."""
        signal1 = FinalSignal("BTC/USDT", "LONG", 50000, 49000, 52000, timestamp=time.time())
        signal2 = FinalSignal("ETH/USDT", "SHORT", 3000, 3100, 2900, timestamp=time.time())

        persistence.save_signal(signal1)
        persistence.save_signal(signal2)

        signals = list(persistence.read_signals())
        assert len(signals) == 2

    def test_get_signal_count(self, persistence):
        """Test getting signal count."""
        for i in range(3):
            signal = FinalSignal(f"SYMBOL{i}/USDT", "LONG", 1000 + i, 900 + i, 1100 + i)
            persistence.save_signal(signal)

        count = persistence.get_signal_count()
        assert count == 3

    def test_get_signals_by_symbol(self, persistence):
        """Test filtering signals by symbol."""
        signal1 = FinalSignal("BTC/USDT", "LONG", 50000, 49000, 52000, timestamp=time.time())
        signal2 = FinalSignal("ETH/USDT", "SHORT", 3000, 3100, 2900, timestamp=time.time())
        signal3 = FinalSignal("BTC/USDT", "SHORT", 51000, 52000, 49000, timestamp=time.time())

        persistence.save_signal(signal1)
        persistence.save_signal(signal2)
        persistence.save_signal(signal3)

        btc_signals = persistence.get_signals_by_symbol("BTC/USDT")
        assert len(btc_signals) == 2

    def test_get_recent_signals(self, persistence):
        """Test getting recent signals."""
        for i in range(3):
            signal = FinalSignal(f"SYMBOL{i}/USDT", "LONG", 1000 + i, 900 + i, 1100 + i)
            persistence.save_signal(signal)

        recent = persistence.get_recent_signals(days=7)
        assert len(recent) == 3

    def test_file_rotation(self, tmp_path):
        """Test daily file rotation."""
        persistence = SignalPersistence(
            storage_dir=str(tmp_path / "test_signals"), enable_rotation=True, validate_path=False
        )

        signal = FinalSignal("BTC/USDT", "LONG", 50000, 49000, 52000, timestamp=time.time())
        persistence.save_signal(signal)

        storage_path = Path(tmp_path) / "test_signals"
        files = list(storage_path.glob("signal_history_*.jsonl"))
        assert len(files) == 1

    def test_path_traversal_protection(self, tmp_path):
        """Test path traversal validation."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        with pytest.raises(ValueError, match="Invalid storage directory"):
            SignalPersistence(storage_dir="../../../etc/passwords")

    def test_validate_storage_dir(self, persistence):
        """Test storage directory validation."""
        assert persistence.storage_dir.exists()
        assert persistence.storage_dir.is_dir()

    def test_disk_full_scenario(self, persistence, monkeypatch):
        """Test handling of low disk space."""
        signal = FinalSignal("BTC/USDT", "LONG", 50000, 49000, 52000)

        mock_usage = MagicMock()
        mock_usage.free = 50 * 1024 * 1024

        monkeypatch.setattr(shutil, "disk_usage", lambda x: mock_usage)

        result = persistence.save_signal(signal)
        assert result is False
        assert persistence.metrics["failed_writes"] > 0

    def test_corrupted_line_handling(self, persistence, temp_storage):
        """Test that corrupted JSON lines are skipped."""
        history_file = Path(temp_storage) / "signal_history.jsonl"

        with open(history_file, "w") as f:
            f.write('{"symbol":"BTC/USDT","type":"LONG","entry":50000}\n')
            f.write('{"invalid json line\n')
            f.write('{"symbol":"ETH/USDT","type":"SHORT","entry":3000}\n')

        signals = list(persistence.read_signals())
        assert len(signals) == 2
        assert signals[0]["symbol"] == "BTC/USDT"
        assert signals[1]["symbol"] == "ETH/USDT"

    def test_metrics_tracking(self, persistence):
        """Test metrics tracking in save_signal."""
        signal = FinalSignal("BTC/USDT", "LONG", 50000, 49000, 52000)

        persistence.save_signal(signal)
        persistence.save_signal(signal)

        metrics = persistence.get_metrics()
        assert metrics["total_writes"] == 2
        assert metrics["failed_writes"] == 0
        assert metrics["total_bytes_written"] > 0
        assert metrics["avg_write_time_ms"] > 0

    def test_disk_space_check_caching(self, persistence, monkeypatch):
        """Test that disk space check is cached."""
        call_count = [0]

        mock_usage = MagicMock()
        mock_usage.free = 1024 * 1024 * 1024

        def mock_disk_usage(path):
            call_count[0] += 1
            return mock_usage

        monkeypatch.setattr(shutil, "disk_usage", mock_disk_usage)

        signal = FinalSignal("BTC/USDT", "LONG", 50000, 49000, 52000)

        persistence.save_signal(signal)
        first_calls = call_count[0]

        time.sleep(0.1)

        persistence.save_signal(signal)

        assert call_count[0] == first_calls

    def test_file_size_based_rotation(self, tmp_path, monkeypatch):
        """Test file rotation based on size."""
        persistence = SignalPersistence(
            storage_dir=str(tmp_path / "test_signals"), enable_rotation=True, validate_path=False
        )

        monkeypatch.setattr(SignalPersistence, "MAX_FILE_SIZE_BYTES", 100)

        signal = FinalSignal("BTC/USDT", "LONG", 50000, 49000, 52000)

        persistence.save_signal(signal)
        persistence.save_signal(signal)

        storage_path = Path(tmp_path) / "test_signals"
        files = list(storage_path.glob("signal_history_*.jsonl"))
        assert len(files) >= 2

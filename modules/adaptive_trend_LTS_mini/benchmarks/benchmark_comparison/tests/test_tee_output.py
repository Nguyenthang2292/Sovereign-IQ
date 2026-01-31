"""Unit tests for TeeOutput class."""

import io
import sys

import pytest

from modules.adaptive_trend_LTS_mini.benchmarks.benchmark_comparison.main import TeeOutput


def test_tee_output_write():
    """Test TeeOutput writes to both stdout and file."""
    mock_file = io.StringIO()
    mock_stdout = io.StringIO()
    mock_stderr = io.StringIO()

    tee = TeeOutput(mock_file)
    tee.stdout = mock_stdout
    tee.stderr = mock_stderr

    tee.write("Test message\n")

    # Check that message was written to both
    mock_stdout.seek(0)
    mock_file.seek(0)

    assert "Test message" in mock_stdout.read()
    assert "Test message" in mock_file.read()


class MockFlushable:
    """Simple mock to track flush calls."""

    def __init__(self):
        self.flush_called = 0

    def flush(self):
        self.flush_called += 1


def test_tee_output_flush():
    """Test TeeOutput flush method."""
    mock_file = MockFlushable()
    mock_stdout = MockFlushable()
    mock_stderr = MockFlushable()

    tee = TeeOutput(mock_file)
    tee.stdout = mock_stdout
    tee.stderr = mock_stderr

    tee.flush()

    # Both should have been flushed
    assert mock_stdout.flush_called == 1
    assert mock_file.flush_called == 1


class MockStream:
    """Simple mock for isatty check."""

    def __init__(self, is_atty_val):
        self._is_atty = is_atty_val

    def isatty(self):
        return self._is_atty


def test_tee_output_isatty():
    """Test TeeOutput isatty method."""
    mock_file = MockFlushable()
    mock_stdout_true = MockStream(True)
    mock_stderr = MockFlushable()

    tee = TeeOutput(mock_file)
    tee.stdout = mock_stdout_true
    tee.stderr = mock_stderr

    assert tee.isatty() is True

    mock_stdout_false = MockStream(False)
    tee.stdout = mock_stdout_false
    assert tee.isatty() is False


def test_tee_output_integration(tmp_path):
    """Integration test with actual file."""
    test_file = tmp_path / "test_output.txt"

    with open(str(test_file), "w", encoding="utf-8") as f:
        tee = TeeOutput(f)

        # Redirect stdout
        original_stdout = sys.stdout
        sys.stdout = tee

        try:
            print("Line 1")
            print("Line 2")
            print("Line 3")
        finally:
            sys.stdout = original_stdout

    # Verify file content
    content = test_file.read_text()
    assert "Line 1" in content
    assert "Line 2" in content
    assert "Line 3" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

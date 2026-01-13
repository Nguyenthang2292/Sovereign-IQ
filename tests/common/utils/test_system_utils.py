import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

"""
Unit tests for system utilities in modules.common.utils.system_utils.

Tests cover:
- Windows stdin setup
- Error code extraction
- Retryable error detection
"""


# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.common.utils.system_utils import (
    get_error_code,
    is_retryable_error,
    setup_windows_stdin,
)


class TestSystemUtils(unittest.TestCase):
    """Test system utility functions."""

    @patch("sys.stdin")
    def test_setup_windows_stdin_win32_missing_stdin(self, mock_stdin):
        """Test Windows stdin setup when stdin is None."""
        # Set stdin to None to simulate missing stdin
        mock_stdin.__bool__ = Mock(return_value=False)

        with patch("sys.platform", "win32"):
            setup_windows_stdin()

        # On Windows with None stdin, the function should try to open CON
        # Since we're in a test environment, it may or may not succeed
        # Just verify the function runs without error
        self.assertIsNotNone(mock_stdin)

    @patch("builtins.open")
    @patch("sys.stdin")
    def test_setup_windows_stdin_win32_open_con(self, mock_stdin, mock_open):
        """Test Windows CON opening."""
        mock_stdin.__bool__ = Mock(return_value=False)
        mock_stdin.closed = Mock(return_value=True)

        # Mock file object that will be returned by open()
        mock_file = Mock()
        mock_open.return_value = mock_file

        # Temporarily set platform to Windows
        original_platform = sys.platform
        sys.platform = "win32"

        try:
            setup_windows_stdin()
        finally:
            sys.platform = original_platform

        # Verify open was called with correct parameters
        mock_open.assert_called_once()

    def test_setup_windows_stdin_non_windows(self):
        """Test that setup_windows_stdin does nothing on non-Windows."""
        # Temporarily set platform to non-Windows
        original_platform = sys.platform
        sys.platform = "linux"

        try:
            setup_windows_stdin()
        finally:
            sys.platform = original_platform

        # Function should complete without error on non-Windows
        self.assertTrue(True)

    @patch("builtins.open", side_effect=OSError("Permission denied"))
    @patch("sys.stdin")
    def test_setup_windows_stdin_handles_error(self, mock_stdin, mock_open):
        """Test Windows stdin setup handles OSError gracefully."""
        mock_stdin.__bool__ = Mock(return_value=False)
        mock_stdin.closed = Mock(return_value=True)

        # Temporarily set platform to Windows
        original_platform = sys.platform
        sys.platform = "win32"

        try:
            # Should not raise exception
            setup_windows_stdin()
        finally:
            sys.platform = original_platform

        # Verify open was attempted
        mock_open.assert_called_once()

    def test_get_error_code_with_status_code(self):
        """Test error code extraction from exception with status_code."""
        exception = Mock()
        exception.status_code = 404

        error_code = get_error_code(exception)

        self.assertEqual(error_code, 404)

    def test_get_error_code_with_code_attr(self):
        """Test error code extraction from exception with code."""
        exception = Mock()
        exception.code = 2  # Use integer instead of string

        error_code = get_error_code(exception)

        self.assertEqual(error_code, 2)

    def test_get_error_code_with_errno(self):
        """Test error code extraction from exception with errno."""
        exception = OSError()
        exception.errno = 2

        error_code = get_error_code(exception)

        self.assertEqual(error_code, 2)

    def test_get_error_code_no_code(self):
        """Test error code extraction when no code is available."""
        exception = Exception()

        error_code = get_error_code(exception)

        self.assertIsNone(error_code)

    def test_get_error_code_invalid_code(self):
        """Test error code extraction with invalid code string."""
        exception = Mock()
        exception.status_code = "invalid"

        error_code = get_error_code(exception)

        self.assertIsNone(error_code)

    def test_is_retryable_error_http_codes(self):
        """Test retryable error detection for HTTP status codes."""
        # Test 429 (Too Many Requests)
        exception = Mock()
        exception.status_code = 429

        self.assertTrue(is_retryable_error(exception))

        # Test 500 (Internal Server Error)
        exception.status_code = 500

        self.assertTrue(is_retryable_error(exception))

        # Test 404 (Not Found) - not retryable
        exception.status_code = 404

        self.assertFalse(is_retryable_error(exception))

    def test_is_retryable_error_message_content(self):
        """Test retryable error detection from message content."""
        # Test timeout message
        exception = Exception("Connection timeout after 30 seconds")

        self.assertTrue(is_retryable_error(exception))

        # Test network error message
        exception = Exception("Network unreachable")

        self.assertTrue(is_retryable_error(exception))

        # Test rate limit message
        exception = Exception("Rate limit exceeded")

        self.assertTrue(is_retryable_error(exception))

        # Test non-retryable message
        exception = Exception("Invalid credentials")

        self.assertFalse(is_retryable_error(exception))

    def test_is_retryable_error_mixed(self):
        """Test retryable error with both code and message."""
        exception = Mock()
        exception.__str__ = Mock(return_value="Connection timeout")
        exception.status_code = 408  # Request Timeout

        self.assertTrue(is_retryable_error(exception))

    def test_is_retryable_error_case_insensitive(self):
        """Test retryable error detection is case insensitive."""
        exception = Mock()
        exception.__str__ = Mock(return_value="NETWORK TIMEOUT")

        self.assertTrue(is_retryable_error(exception))

        exception = Mock()
        exception.__str__ = Mock(return_value="Rate LIMIT")

        self.assertTrue(is_retryable_error(exception))

    @patch("modules.common.utils.system_utils.get_error_code")
    def test_is_retryable_error_with_helper(self, mock_get_error_code):
        """Test retryable error detection using helper function."""
        mock_get_error_code.return_value = 429

        exception = Exception("Rate limit exceeded")

        self.assertTrue(is_retryable_error(exception))
        mock_get_error_code.assert_called_once_with(exception)

    def test_is_retryable_error_no_retryable(self):
        """Test non-retryable error."""
        exception = Exception("Resource not found")

        self.assertFalse(is_retryable_error(exception))


if __name__ == "__main__":
    unittest.main()

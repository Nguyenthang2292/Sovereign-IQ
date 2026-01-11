
from unittest.mock import patch

import pytest

from modules.common.utils import safe_input

from modules.common.utils import safe_input

"""
Test suite for safe_input() function.

Tests the safe_input() utility for proper error handling,
default value behavior, and Windows compatibility.
"""





class TestSafeInput:
    """Test cases for safe_input() function."""

    def test_normal_input(self):
        """Test normal input without errors."""
        with patch("builtins.input", return_value="test input"):
            result = safe_input("Enter value: ")
            assert result == "test input"

    def test_input_with_whitespace(self):
        """Test that whitespace is stripped from input."""
        with patch("builtins.input", return_value="  test input  "):
            result = safe_input("Enter value: ")
            assert result == "test input"

    def test_default_value_on_empty_input(self):
        """Test default value is returned when input is empty string."""
        with patch("builtins.input", return_value=""):
            result = safe_input("Enter value: ", default="default")
            assert result == "default"

    def test_default_value_on_os_error(self):
        """Test default value is returned on OSError."""
        with patch("builtins.input", side_effect=OSError("Input closed")):
            result = safe_input("Enter value: ", default="default")
            assert result == "default"

    def test_default_value_on_io_error(self):
        """Test default value is returned on IOError."""
        with patch("builtins.input", side_effect=IOError("I/O error")):
            result = safe_input("Enter value: ", default="default")
            assert result == "default"

    def test_default_value_on_eof_error(self):
        """Test default value is returned on EOFError."""
        with patch("builtins.input", side_effect=EOFError()):
            result = safe_input("Enter value: ", default="default")
            assert result == "default"

    def test_default_value_on_attribute_error(self):
        """Test default value is returned on AttributeError."""
        with patch("builtins.input", side_effect=AttributeError("No stdin")):
            result = safe_input("Enter value: ", default="default")
            assert result == "default"

    def test_default_value_on_value_error(self):
        """Test default value is returned on ValueError."""
        with patch("builtins.input", side_effect=ValueError("Invalid value")):
            result = safe_input("Enter value: ", default="default")
            assert result == "default"

    def test_exception_raised_when_no_default(self):
        """Test exception is raised when no default is provided."""
        with patch("modules.common.utils.input", side_effect=OSError("Input closed")):
            with pytest.raises(OSError):
                safe_input("Enter value: ")

    def test_empty_string_default(self):
        """Test that empty string can be used as default."""
        with patch("builtins.input", return_value=""):
            result = safe_input("Enter value: ", default="")
            assert result == ""

    def test_prompt_displayed(self):
        """Test that prompt is correctly displayed."""
        with patch("builtins.input", return_value="test") as mock_input:
            result = safe_input("Enter your name: ")
            mock_input.assert_called_once_with("Enter your name: ")
            assert result == "test"

    def test_multiple_exceptions_fallthrough(self):
        """Test that all exception types are caught."""
        exceptions = [OSError(), IOError(), EOFError(), AttributeError(), ValueError()]

        for exc in exceptions:
            with patch("builtins.input", side_effect=exc):
                result = safe_input("Enter: ", default="fallback")
                assert result == "fallback"


class TestSafeInputWindowsCompatibility:
    """Test Windows-specific scenarios for safe_input()."""

    @patch("sys.platform", "win32")
    def test_windows_platform_handling(self):
        """Test that function works on Windows platform."""
        with patch("builtins.input", return_value="windows input"):
            result = safe_input("Enter: ", default="default")
            assert result == "windows input"

    @patch("sys.platform", "linux")
    def test_linux_platform_handling(self):
        """Test that function works on Linux platform."""
        with patch("builtins.input", return_value="linux input"):
            result = safe_input("Enter: ", default="default")
            assert result == "linux input"


class TestSafeInputIntegration:
    """Integration tests for safe_input() with realistic scenarios."""

    def test_menu_selection_with_default(self):
        """Test menu selection with default option."""
        with patch("builtins.input", return_value=""):
            result = safe_input("Select option [1-3]: ", default="1")
            assert result == "1"

    def test_yes_no_confirmation(self):
        """Test yes/no confirmation input."""
        with patch("builtins.input", return_value="yes"):
            result = safe_input("Continue? (y/n): ")
            assert result == "yes"

    def test_numeric_input(self):
        """Test numeric input."""
        with patch("builtins.input", return_value="123"):
            result = safe_input("Enter number: ")
            assert result == "123"

    def test_file_path_input(self):
        """Test file path input."""
        with patch("builtins.input", return_value="/path/to/file.txt"):
            result = safe_input("Enter file path: ")
            assert result == "/path/to/file.txt"

    def test_special_characters_input(self):
        """Test input with special characters."""
        with patch("builtins.input", return_value="test@#$%^&*()"):
            result = safe_input("Enter special chars: ")
            assert result == "test@#$%^&*()"

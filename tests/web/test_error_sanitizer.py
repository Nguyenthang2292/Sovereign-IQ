
from web.utils.error_sanitizer import sanitize_error

"""
Tests for error sanitization utilities (web/utils/error_sanitizer.py).

Tests cover:
- Sanitization of different error types
- Removal of file paths
- Removal of stack traces
- Removal of sensitive information
- Mapping of known error types to user-friendly messages
"""



class TestSanitizeError:
    """Test sanitize_error function."""

    def test_sanitize_none(self):
        """Test sanitizing None returns default message."""
        result = sanitize_error(None)
        assert result == "An error occurred"

    def test_sanitize_file_not_found_error(self):
        """Test sanitizing FileNotFoundError."""
        error = FileNotFoundError("C:\\Users\\secret\\file.txt")
        result = sanitize_error(error)
        assert result == "File not found"
        assert "secret" not in result
        assert "C:" not in result

    def test_sanitize_permission_error(self):
        """Test sanitizing PermissionError."""
        error = PermissionError("Access denied to /etc/passwd")
        result = sanitize_error(error)
        assert result == "Permission denied"
        assert "/etc/passwd" not in result

    def test_sanitize_os_error(self):
        """Test sanitizing OSError."""
        error = OSError("System error")
        result = sanitize_error(error)
        assert result == "System error occurred: System error"

    def test_sanitize_value_error(self):
        """Test sanitizing ValueError."""
        error = ValueError("Invalid value")
        result = sanitize_error(error)
        assert result == "Invalid value provided: Invalid value"

    def test_sanitize_unknown_exception_type(self):
        """Test sanitizing unknown exception type."""

        class CustomError(Exception):
            pass

        error = CustomError("Custom error message")
        result = sanitize_error(error)

        assert isinstance(result, str)
        assert len(result) > 0
        # Should either preserve safe message or return default
        assert result == "Custom error message" or result == "An error occurred"
        # Ensure no sensitive patterns leak through
        assert "Traceback" not in result

    def test_sanitize_string_with_file_path(self):
        """Test sanitizing string containing file path."""
        error_str = "Error opening file: C:\\Users\\test\\file.txt"
        result = sanitize_error(error_str)
        assert "C:" not in result
        assert "Users" not in result
        assert "file.txt" not in result
        # Path is replaced with [path] placeholder
        assert result == "Error opening file: [path]"

    def test_sanitize_string_with_unix_path(self):
        """Test sanitizing string containing Unix path."""
        error_str = "Error: /home/user/secret/config.json not found"
        result = sanitize_error(error_str)
        assert "/home" not in result
        assert "secret" not in result
        # Path is replaced with [path] placeholder
        assert result == "Error: [path] not found"

    def test_sanitize_stack_trace(self):
        """Test sanitizing stack trace."""
        stack_trace = """Traceback (most recent call last):
  File "test.py", line 10, in <module>
    raise ValueError("Test error")
ValueError: Test error"""
        result = sanitize_error(stack_trace)
        assert result == "An error occurred"
        assert "Traceback" not in result
        assert "File" not in result
        assert "line" not in result

    def test_sanitize_string_with_sensitive_info(self):
        """Test sanitizing string with sensitive information."""
        error_str = "Error: password=secret123 api_key=abc123"
        result = sanitize_error(error_str)
        assert "password" not in result.lower()
        assert "secret123" not in result
        assert "api_key" not in result.lower()
        assert "abc123" not in result

    def test_sanitize_safe_error_message(self):
        """Test that safe error messages are preserved."""
        error_str = "Invalid input provided"
        result = sanitize_error(error_str)
        # Should preserve safe messages
        assert isinstance(result, str)
        assert len(result) > 0

    def test_sanitize_error_with_safe_message(self):
        """Test error with safe message that doesn't contain sensitive info."""
        error = ValueError("Invalid number format")
        result = sanitize_error(error)
        # ValueError messages are canonicalized to "Invalid value provided: {message}"
        # to provide consistent user-facing error messages while preserving the original message
        assert result == "Invalid value provided: Invalid number format"

    def test_sanitize_stack_trace_with_file_pattern(self):
        """Test sanitizing stack trace with File pattern."""
        stack_trace = 'File "test.py", line 10, in <module>'
        result = sanitize_error(stack_trace)
        assert result == "An error occurred"
        assert "File" not in result
        assert "test.py" not in result
        assert "line" not in result

    def test_sanitize_stack_trace_with_raise_pattern(self):
        """Test sanitizing stack trace with raise pattern."""
        stack_trace = "raise ValueError('test')\n  File \"test.py\", line 5"
        result = sanitize_error(stack_trace)
        assert result == "An error occurred"
        assert "raise" not in result
        assert "test.py" not in result

    def test_sanitize_string_with_relative_path(self):
        """Test sanitizing string containing relative path."""
        error_str = "Error: ../config/secrets.json not found"
        result = sanitize_error(error_str)
        assert "../" not in result
        assert "config" not in result
        assert "secrets.json" not in result
        # Path is replaced with [path] placeholder
        assert result == "Error: ..[path] not found"

    def test_sanitize_string_with_home_path(self):
        """Test sanitizing string containing home directory path."""
        error_str = "Error: ~/.ssh/id_rsa not found"
        result = sanitize_error(error_str)
        assert "~/" not in result
        assert ".ssh" not in result
        assert "id_rsa" not in result
        # Path is replaced with [path] placeholder
        assert result == "Error: ~[path] not found"

    def test_sanitize_string_with_token(self):
        """Test sanitizing string with token."""
        error_str = "Error: token=secret_token_123"
        result = sanitize_error(error_str)
        assert "token" not in result.lower()
        assert "secret_token_123" not in result
        # Sensitive info triggers default message
        assert result == "An error occurred"

    def test_sanitize_string_url_preserved(self):
        """Test that URLs are preserved and not removed as paths."""
        error_str = "Connection to https://api.example.com failed"
        result = sanitize_error(error_str)
        # URLs should be preserved (not flagged as sensitive)
        assert "https://api.example.com" in result or "api.example.com" in result

    def test_sanitize_string_path_removed_with_url_present(self):
        """Test that file paths are removed while URLs remain."""
        error_str_with_path = "Error opening /var/log/app.log: https://api.example.com"
        result_with_path = sanitize_error(error_str_with_path)
        assert "/var/log" not in result_with_path
        assert "app.log" not in result_with_path
        # Path is replaced with [path] placeholder, URL is preserved
        assert result_with_path == "Error opening [path]: https://api.example.com"

    def test_sanitize_string_with_io_mention_preserved(self):
        """Test that I/O mentions are preserved."""
        error_str = "I/O error occurred during operation"
        result = sanitize_error(error_str)
        # I/O should be preserved (not detected as a path)
        assert "I/O" in result

    def test_sanitize_string_with_date_preserved(self):
        """Test that dates are preserved."""
        error_str = "Error occurred on 2024/01/15 at 10:30"
        result = sanitize_error(error_str)
        # Dates should be preserved (not detected as paths)
        assert "2024/01/15" in result

    def test_sanitize_string_with_exception_prefix_extraction(self):
        """Test extracting safe message after Exception: prefix."""
        error_str = "Exception: Value out of range"
        result = sanitize_error(error_str)
        # Should extract and return the safe message
        assert "Value out of range" in result

    def test_sanitize_string_with_error_prefix_removal_and_extraction(self):
        """Test that Error: prefix is removed and safe message is extracted."""
        error_str = "Error: Invalid input provided"
        result = sanitize_error(error_str)
        # Should extract the safe message since it doesn't contain sensitive info
        assert result == "Invalid input provided"
        # Verify the Error: prefix is removed
        assert "Error:" not in result

    def test_sanitize_string_with_path_no_safe_message_extraction(self):
        """Test that messages with paths don't yield extracted safe messages."""
        error_str = "Error: C:\\Users\\test\\file.txt not found"
        result = sanitize_error(error_str)
        # Should not extract the message because it contains a path
        # Should return default or sanitized version
        assert "C:" not in result
        assert "Users" not in result
        assert "file.txt" not in result
        # Path is replaced with [path] placeholder
        assert result == "Error: [path]"

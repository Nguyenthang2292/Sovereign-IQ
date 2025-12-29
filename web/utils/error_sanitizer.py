"""
Error sanitization utilities for safe error message display.

Sanitizes error messages to prevent exposing sensitive information
like file paths, stack traces, or internal details to users.
"""

import re
import time
import uuid
from typing import Union, Optional


# Common error type mappings to user-friendly messages
ERROR_MESSAGE_MAP = {
    "FileNotFoundError": "File not found",
    "PermissionError": "Permission denied",
    "IsADirectoryError": "Expected a file, but found a directory",
    "NotADirectoryError": "Expected a directory, but found a file",
    "OSError": "System error occurred",
    "IOError": "Input/output error occurred",
    "ValueError": "Invalid value provided",
    "KeyError": "Required key not found",
    "TypeError": "Invalid type provided",
    "AttributeError": "Attribute not available",
    "ConnectionError": "Connection error occurred",
    "TimeoutError": "Operation timed out",
    "MemoryError": "Insufficient memory",
}

# URL pattern for detecting URLs in error messages (used to avoid flagging URLs as file paths)
URL_PATTERN = r'(https?|ftp|file|ws|wss)://[^\s<>"|?*\n]+'
URL_PATTERN_COMPILED = re.compile(URL_PATTERN, re.IGNORECASE)


def sanitize_error(error: Union[Exception, str, None]) -> str:
    """
    Sanitize an error to a user-friendly message.
    
    Removes:
    - Stack traces
    - File paths
    - Line numbers
    - Internal implementation details
    
    Maps known error types to user-friendly messages.
    
    Args:
        error: Exception object, error string, or None
        
    Returns:
        Sanitized error message safe for user display
        
    Examples:
        >>> sanitize_error(FileNotFoundError("C:\\Users\\secret\\file.txt"))
        "File not found"
        
        >>> sanitize_error("Traceback (most recent call last):\\n  File...")
        "An error occurred"
        
        >>> sanitize_error(None)
        "An error occurred"
    """
    if error is None:
        return "An error occurred"
    
    # Convert exception to string if needed
    if isinstance(error, Exception):
        error_type = type(error).__name__
        error_str = str(error)
        
        # First, try to map known error types
        if error_type in ERROR_MESSAGE_MAP:
            base_message = ERROR_MESSAGE_MAP[error_type]
            
            # For some errors, we can include safe parts of the message
            # but strip out file paths and sensitive info
            if error_str and not _contains_sensitive_info(error_str):
                # Only include error message if it doesn't contain sensitive info
                # and is reasonably short (not a stack trace)
                if len(error_str) < 200 and "\n" not in error_str:
                    sanitized_str = _remove_file_paths(error_str)
                    return f"{base_message}: {sanitized_str}"
            
            return base_message    
        
        # Check for stack trace patterns
        if _is_stack_trace(error_str):
            return "An error occurred"
        
        # Check for sensitive information
        if _contains_sensitive_info(error_str):
            # Try to extract a safe message
            safe_message = _extract_safe_message(error_str)
            if safe_message:
                return safe_message
            return "An error occurred"
        
        # If it's a reasonable error message (short, no newlines), use it
        if len(error_str) < 300 and "\n" not in error_str:
            # Still sanitize any remaining paths
            sanitized = _remove_file_paths(error_str)
            return sanitized if sanitized else "An error occurred"
    
    # Fallback
    return "An error occurred"


def _is_stack_trace(text: str) -> bool:
    """Check if text looks like a stack trace."""
    # Strong indicators that are very specific to stack traces
    strong_indicators = [
        "Traceback (most recent call last)",
        "  File \"",  # Python traceback with indentation
        "  at ",  # JavaScript/TypeScript traceback with indentation
    ]
    
    # Medium indicators that need context (more specific patterns)
    medium_indicators = [
        r',\s+line\s+\d+',  # Python: ", line 42" (comma + line + number)
        r'\s+in\s+<module>',  # Python: " in <module>"
        r'\s+in\s+\w+\s*$',  # Python: " in function_name" at end of line
        r'raise\s+\w+',  # Python: "raise ValueError"
    ]
    
    # Check for strong indicators first (single match is enough)
    if any(indicator in text for indicator in strong_indicators):
        return True
    
    # Check for medium indicators (require at least 2 matches to avoid false positives)
    medium_matches = sum(1 for pattern in medium_indicators if re.search(pattern, text))
    if medium_matches >= 2:
        return True
    
    # Also check for "File \"" without indentation (less common but still valid)
    if 'File "' in text and ('line' in text.lower() or 'in ' in text):
        return True
    
    return False


def _contains_sensitive_info(text: str) -> bool:
    """Check if text contains sensitive information like file paths."""
    # First check for URLs - we don't want to flag URLs as sensitive
    if URL_PATTERN_COMPILED.search(text):
        # If text contains URLs, temporarily remove them before checking paths
        text_without_urls = URL_PATTERN_COMPILED.sub('', text)
    else:
        text_without_urls = text
    
    # Patterns for file paths (Windows and Unix)
    # URLs are already removed from text_without_urls, so we can safely match paths
    path_patterns = [
        r'[A-Za-z]:\\[^:<>"|?*\n]*',  # Windows absolute paths (C:\...)
        r'\\[^:<>"|?*\n]*',  # Windows relative paths (\...)
        # Unix paths: require at least one letter to avoid matching dates like "2024/01/15" or "I/O"
        # Use negative lookbehind to ensure we're not matching part of a word
        r'(?<![a-zA-Z0-9])/[a-zA-Z0-9_./-]*[a-zA-Z][a-zA-Z0-9_./-]*',
        r'~/[^\s<>"|?*\n]+',  # Home directory paths (~/...)
        r'\.\./[^\s<>"|?*\n]*',  # Relative paths (../...)
    ]
    
    for pattern in path_patterns:
        if re.search(pattern, text_without_urls):
            return True
    
    # Check for common sensitive patterns
    sensitive_patterns = [
        r'password\s*[:=]\s*\S+',
        r'api[_-]?key\s*[:=]\s*\S+',
        r'token\s*[:=]\s*\S+',
        r'secret\s*[:=]\s*\S+',
    ]
    
    for pattern in sensitive_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    return False


def _remove_file_paths(text: str) -> str:
    """Remove file paths from text, preserving URLs."""
    # First, protect URLs by temporarily replacing them with placeholders
    url_placeholders = {}
    
    def url_replacer(match):
        # Generate a guaranteed-unique placeholder using UUID v4
        # Keep retrying until we find a placeholder that doesn't exist in the text
        # and hasn't been used by us already
        max_attempts = 10
        for _ in range(max_attempts):
            unique_suffix = uuid.uuid4().hex
            placeholder = f'__URL_PLACEHOLDER_{unique_suffix}__'
            
            # Check if this placeholder already exists in the text or has been used
            if placeholder not in text and placeholder not in url_placeholders:
                url_placeholders[placeholder] = match.group(0)
                return placeholder
        
        # Fallback: if all attempts fail (extremely unlikely), use a more complex pattern
        # with timestamp and random components
        unique_suffix = f"{uuid.uuid4().hex}_{int(time.time() * 1000000)}"
        placeholder = f'__URL_PLACEHOLDER_{unique_suffix}__'
        # Even in fallback, check and retry if needed
        while placeholder in text or placeholder in url_placeholders:
            unique_suffix = f"{uuid.uuid4().hex}_{int(time.time() * 1000000)}"
            placeholder = f'__URL_PLACEHOLDER_{unique_suffix}__'
        url_placeholders[placeholder] = match.group(0)
        return placeholder
    
    # Replace URLs with placeholders
    text = URL_PATTERN_COMPILED.sub(url_replacer, text)
    
    # Remove Windows paths (C:\... or D:\...)
    # Match Windows paths including spaces, but stop at sentence boundaries
    text = re.sub(r'[A-Za-z]:\\(?:[^:<>"|?*\n]|\\ )+', '[path]', text)
    
    # Remove Unix paths - URLs are already protected by placeholders
    # Match / followed by path characters (at least one letter to avoid dates like "2024/01/15")
    # Use word boundary or whitespace before / to ensure we're matching a path, not part of text
    text = re.sub(r'(?<![a-zA-Z0-9])/(?:[a-zA-Z0-9_./\-]|\\ )+[a-zA-Z](?:[a-zA-Z0-9_./\-]|\\ )*', '[path]', text)
    
    # Remove relative paths
    text = re.sub(r'\.\./[^\s<>"|?*\n]*', '[path]', text)
    text = re.sub(r'~/[^\s<>"|?*\n]*', '[path]', text)    
    
    # Restore URLs
    for placeholder, url in url_placeholders.items():
        text = text.replace(placeholder, url)
    
    return text.strip()


def _extract_safe_message(text: str) -> Optional[str]:
    """
    Try to extract a safe message from error text.
    
    Looks for common error message patterns that don't contain sensitive info.
    """
    # Look for error messages after common prefixes
    patterns = [
        r'Error:\s*([^\n]+)',
        r'Exception:\s*([^\n]+)',
        r'([A-Z][^:\n]{10,100})',  # Capitalized sentences
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            message = match.group(1) if match.groups() else match.group(0)
            message = message.strip()
            # Only return if it doesn't contain sensitive info
            if message and not _contains_sensitive_info(message) and len(message) < 200:
                return message
    
    return None


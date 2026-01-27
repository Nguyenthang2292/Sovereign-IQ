"""Exceptions for data fetching operations."""

from typing import Optional


class SymbolFetchError(Exception):
    """Custom exception for symbol fetching errors."""

    def __init__(self, message: str, original_exception: Optional[Exception] = None, is_retryable: bool = False):
        super().__init__(message)
        self.original_exception = original_exception
        self.is_retryable = is_retryable

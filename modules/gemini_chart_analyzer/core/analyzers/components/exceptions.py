"""Custom Gemini analyzer exceptions."""


class GeminiAPIError(Exception):
    """Base exception for Gemini API errors."""


class GeminiModelNotFoundError(GeminiAPIError):
    """Raised when requested model is not found."""


class GeminiQuotaExceededError(GeminiAPIError):
    """Raised when API quota is exceeded."""


class GeminiRateLimitError(GeminiAPIError):
    """Raised when rate limit is exceeded."""


class GeminiInvalidRequestError(GeminiAPIError):
    """Raised when request is invalid."""


class GeminiAuthenticationError(GeminiAPIError):
    """Raised when authentication fails."""


class GeminiResponseParseError(GeminiAPIError):
    """Raised when response cannot be parsed."""


class GeminiImageValidationError(GeminiAPIError):
    """Raised when image validation fails."""

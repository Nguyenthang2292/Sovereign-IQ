"""Custom exceptions for the gemini_chart_analyzer module."""


class GeminiAnalyzerError(Exception):
    """Base exception for all errors in the gemini_chart_analyzer module."""

    pass


class ScanConfigurationError(GeminiAnalyzerError):
    """Exception raised when scan configuration is invalid."""

    pass


class DataFetchError(GeminiAnalyzerError):
    """Exception raised when OHLCV or symbol data fetching fails."""

    pass


class ChartGenerationError(GeminiAnalyzerError):
    """Exception raised when chart image generation fails."""

    pass


class GeminiAnalysisError(GeminiAnalyzerError):
    """Exception raised when Gemini analysis or parsing fails."""

    pass


class ReportGenerationError(GeminiAnalyzerError):
    """Exception raised when report generation (HTML/CSV/JSON) fails."""

    pass

"""Type definitions for gemini_chart_analyzer module."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SignalResult:
    """Result for a single signal analysis."""

    signal: str  # "LONG", "SHORT", or "NONE"
    confidence: float  # 0.0 to 1.0


@dataclass
class SymbolScanResult:
    """Results for a single symbol across multiple timeframes."""

    timeframes: Dict[str, SignalResult]
    aggregated: Optional[SignalResult] = None


@dataclass
class BatchScanResult:
    """Overall results for a batch scan operation."""

    long_symbols: List[str] = field(default_factory=list)
    short_symbols: List[str] = field(default_factory=list)
    none_symbols: List[str] = field(default_factory=list)
    long_symbols_with_confidence: List[Tuple[str, float]] = field(default_factory=list)
    short_symbols_with_confidence: List[Tuple[str, float]] = field(default_factory=list)
    none_symbols_with_confidence: List[Tuple[str, float]] = field(default_factory=list)
    all_results: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    results_file: str = ""
    html_report_path: Optional[str] = None
    status: str = "completed"
    batches_processed: int = 0
    total_batches: int = 0

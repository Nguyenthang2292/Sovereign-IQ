"""
Result Manager Component

Handles categorization, sorting, and persistence of batch scan results.
"""

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from modules.common.ui.logging import log_error, log_info, log_success, log_warn
from modules.gemini_chart_analyzer.core.exceptions import ReportGenerationError
from modules.gemini_chart_analyzer.core.utils.chart_paths import get_analysis_results_dir


class ResultManager:
    """
    Manages batch scan results including categorization, sorting, and persistence.

    Handles:
    - Categorizing results into LONG/SHORT/NONE signals
    - Sorting by confidence scores
    - Saving results to JSON files
    - Generating HTML reports
    """

    def categorize_and_sort_results(
        self, all_results: Dict[str, Any]
    ) -> Tuple[
        List[str], List[str], List[str], List[Tuple[str, float]], List[Tuple[str, float]], List[Tuple[str, float]]
    ]:
        """
        Categorize results into LONG/SHORT/NONE and sort by confidence.

        Args:
            all_results: Dictionary mapping symbol to result (dict with 'signal' and 'confidence' or legacy string)

        Returns:
            Tuple of 6 lists:
            - long_symbols: List of LONG symbols (sorted by confidence, high to low)
            - short_symbols: List of SHORT symbols (sorted by confidence, high to low)
            - none_symbols: List of NONE symbols (sorted by confidence, high to low)
            - long_symbols_with_confidence: List of (symbol, confidence) tuples for LONG
            - short_symbols_with_confidence: List of (symbol, confidence) tuples for SHORT
            - none_symbols_with_confidence: List of (symbol, confidence) tuples for NONE
        """
        long_symbols_with_confidence = []
        short_symbols_with_confidence = []
        none_symbols_with_confidence = []
        legacy_format_symbols = []

        for symbol, result in all_results.items():
            if hasattr(result, "signal"):
                signal = result.signal
                confidence = result.confidence
            elif isinstance(result, dict):
                signal = result.get("signal", "NONE")
                confidence = result.get("confidence", 0.0)
            else:
                # Backward compatibility: if result is string
                # Use 0.0 confidence to denote fallback/untrusted confidence for legacy string results
                # This ensures real confidence scores remain distinguishable from fallbacks
                signal = result if isinstance(result, str) else "NONE"
                confidence = 0.0
                legacy_format_symbols.append(symbol)

            if signal == "LONG":
                long_symbols_with_confidence.append((symbol, confidence))
            elif signal == "SHORT":
                short_symbols_with_confidence.append((symbol, confidence))
            else:
                none_symbols_with_confidence.append((symbol, confidence))

        # Log legacy format warning as summary if any found
        if legacy_format_symbols:
            count = len(legacy_format_symbols)
            examples = legacy_format_symbols[:5]  # Show first 5 as examples
            examples_str = ", ".join(examples)
            if count > 5:
                examples_str = f"{examples_str}, ..."
            log_warn(
                f"Legacy format detected for {count} symbol(s) ({'e.g., ' + examples_str if count > 5 else examples_str}): "
                f"using fallback confidence 0.0"
            )

        # Sort by confidence (descending)
        long_symbols_with_confidence.sort(key=lambda x: x[1], reverse=True)
        short_symbols_with_confidence.sort(key=lambda x: x[1], reverse=True)
        none_symbols_with_confidence.sort(key=lambda x: x[1], reverse=True)

        # Extract just symbols (sorted by confidence)
        long_symbols = [s for s, _ in long_symbols_with_confidence]
        short_symbols = [s for s, _ in short_symbols_with_confidence]
        none_symbols = [s for s, _ in none_symbols_with_confidence]

        return (
            long_symbols,
            short_symbols,
            none_symbols,
            long_symbols_with_confidence,
            short_symbols_with_confidence,
            none_symbols_with_confidence,
        )

    def save_results(
        self,
        all_results: Dict[str, Any],
        long_symbols: List[str],
        short_symbols: List[str],
        summary: Dict,
        timeframe: str,
        long_with_confidence: Optional[List[Tuple[str, float]]] = None,
        short_with_confidence: Optional[List[Tuple[str, float]]] = None,
        timeframes: Optional[List[str]] = None,
    ) -> str:
        """
        Save scan results to JSON file and generate HTML report.

        Args:
            all_results: Full results dictionary
            long_symbols: List of LONG symbols (sorted by confidence)
            short_symbols: List of SHORT symbols (sorted by confidence)
            summary: Summary statistics
            timeframe: Timeframe string (or primary timeframe for multi-TF)
            long_with_confidence: List of (symbol, confidence) tuples for LONG
            short_with_confidence: List of (symbol, confidence) tuples for SHORT
            timeframes: Optional list of timeframes (for multi-TF mode)

        Returns:
            Path to saved results JSON file

        Raises:
            ReportGenerationError: If file save or report generation fails
        """
        results_base_dir = get_analysis_results_dir()
        output_dir = os.path.join(results_base_dir, "batch_scan")
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if timeframes:
            tf_str = "_".join(timeframes)
            results_file = os.path.join(output_dir, f"batch_scan_multi_tf_{tf_str}_{timestamp}.json")
        else:
            results_file = os.path.join(output_dir, f"batch_scan_{timeframe}_{timestamp}.json")

        # Convert dataclasses to dicts for JSON serialization
        serializable_results = {}
        for k, v in all_results.items():
            if hasattr(v, "__dataclass_fields__"):
                serializable_results[k] = asdict(v)
            else:
                serializable_results[k] = v

        results_data = {
            "timestamp": datetime.now().isoformat(),
            "timeframe": timeframe,
            "timeframes": timeframes if timeframes else [timeframe],
            "summary": summary,
            "long_symbols": long_symbols,  # Sorted by confidence
            "short_symbols": short_symbols,  # Sorted by confidence
            "long_symbols_with_confidence": long_with_confidence or [],
            "short_symbols_with_confidence": short_with_confidence or [],
            "all_results": serializable_results,
        }

        try:
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
        except OSError as e:
            log_error(f"IO Error saving results file {results_file}: {e}")
            raise ReportGenerationError(f"Failed to save results file: {e}") from e
        except Exception as e:
            log_error(f"Unexpected error saving results file: {e}")
            raise ReportGenerationError(f"Unexpected error saving results file: {e}") from e

        # Generate HTML report
        self._generate_html_report(results_data, output_dir)

        return results_file

    def _generate_html_report(self, results_data: Dict[str, Any], output_dir: str):
        """
        Generate HTML report and attempt to open in browser.

        Args:
            results_data: Results dictionary to generate report from
            output_dir: Directory to save HTML report

        Note:
            Failures in HTML generation are logged as warnings but don't raise exceptions
            to avoid breaking the main scan workflow.
        """
        try:
            from modules.gemini_chart_analyzer.core.reporting.html_report import generate_batch_html_report

            html_path = generate_batch_html_report(results_data, output_dir)
            log_success(f"Generated HTML report: {html_path}")

            # Open HTML report in browser
            try:
                import webbrowser

                html_uri = Path(html_path).resolve().as_uri()
                webbrowser.open(html_uri)
                log_success("Opened HTML report in browser")
            except Exception as e:
                log_warn(f"Could not open browser automatically: {e}")
                log_info(f"Please open file manually: {html_path}")
        except Exception as e:
            log_warn(f"Could not generate HTML report: {e}")
            # Don't fail the whole operation if HTML generation fails

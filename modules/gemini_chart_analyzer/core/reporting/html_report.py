"""
HTML Report Generation for Batch Scan Results.

This module provides a backward-compatible interface for batch scan reports,
delegating to the centralized html_report_generator.
"""

from typing import Dict
from modules.gemini_chart_analyzer.core.reporting.html_report_generator import generate_html_report


def generate_batch_html_report(results_data: Dict, output_dir: str) -> str:
    """
    Tạo HTML report cho batch scan results.
    Delegates to centralized generate_html_report.
    """
    return generate_html_report(analysis_data=results_data, output_dir=output_dir, report_type="batch")

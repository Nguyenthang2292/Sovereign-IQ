"""
Centralized HTML Report Generator for Chart Analysis.

This module provides a unified interface for generating different types of HTML reports:
- Single timeframe analysis
- Multi-timeframe analysis
- Batch scan results

This is the main entry point that delegates to specialized sub-modules in the
generators package.
"""

from datetime import datetime
from typing import Any, Dict

from modules.gemini_chart_analyzer.core.reporting.generators import (
    generate_batch_report,
    generate_multi_tf_report,
    generate_single_report,
)


def generate_html_report(
    analysis_data: Dict[str, Any],
    output_dir: str,
    report_type: str = "single",
    **kwargs
) -> str:
    """
    Generate an HTML report based on the report type.

    This is the main entry point for HTML report generation. It delegates to
    specialized generators based on the report_type parameter.

    Args:
        analysis_data: Data for the report. Structure depends on report_type:
            - single: Dict with 'symbol', 'timeframe', 'analysis' keys
            - multi: Dict with 'symbol', 'timeframes', 'aggregated' keys
            - batch: Dict with 'timestamp', 'summary', results keys
        output_dir: Directory to save the HTML file
        report_type: Type of report ('single', 'multi', 'batch')
        **kwargs: Additional parameters specific to each report type:
            - single: chart_path, report_datetime
            - multi: timeframes_list, report_datetime
            - batch: (no additional params needed)

    Returns:
        Path to the generated HTML file

    Raises:
        ValueError: If report_type is not recognized
        ReportGenerationError: If report generation fails

    Examples:
        Single timeframe report:
        >>> generate_html_report(
        ...     analysis_data={'symbol': 'BTC/USDT', 'timeframe': '1h', 'analysis': '...'},
        ...     output_dir='./outputs',
        ...     report_type='single',
        ...     chart_path='./chart.png',
        ...     report_datetime=datetime.now()
        ... )

        Multi-timeframe report:
        >>> generate_html_report(
        ...     analysis_data={'symbol': 'BTC/USDT', 'timeframes': {...}, 'aggregated': {...}},
        ...     output_dir='./outputs',
        ...     report_type='multi',
        ...     timeframes_list=['1h', '4h', '1d'],
        ...     report_datetime=datetime.now()
        ... )

        Batch scan report:
        >>> generate_html_report(
        ...     analysis_data={'timestamp': '...', 'summary': {...}, 'all_results': {...}},
        ...     output_dir='./outputs',
        ...     report_type='batch'
        ... )
    """
    if report_type == "batch":
        return generate_batch_report(analysis_data, output_dir)

    elif report_type == "multi":
        return generate_multi_tf_report(
            symbol=analysis_data.get("symbol", "Unknown"),
            timeframes_list=kwargs.get("timeframes_list", []),
            results=analysis_data,
            report_datetime=kwargs.get("report_datetime", datetime.now()),
            output_dir=output_dir,
        )

    elif report_type == "single":
        return generate_single_report(
            symbol=analysis_data.get("symbol", "Unknown"),
            timeframe=analysis_data.get("timeframe", "1h"),
            chart_path=kwargs.get("chart_path", ""),
            analysis_result=analysis_data.get("analysis", ""),
            report_datetime=kwargs.get("report_datetime", datetime.now()),
            output_dir=output_dir,
        )

    else:
        raise ValueError(f"Unknown report_type: {report_type}. Must be 'single', 'multi', or 'batch'.")


# Maintain backward compatibility by exposing internal functions
# These are deprecated and should not be used in new code
def _generate_single_report(*args, **kwargs):
    """Deprecated: Use generate_single_report from generators.single_report instead."""
    return generate_single_report(*args, **kwargs)


def _generate_multi_tf_report(*args, **kwargs):
    """Deprecated: Use generate_multi_tf_report from generators.multi_tf_report instead."""
    return generate_multi_tf_report(*args, **kwargs)


def _generate_batch_report(*args, **kwargs):
    """Deprecated: Use generate_batch_report from generators.batch_report instead."""
    return generate_batch_report(*args, **kwargs)

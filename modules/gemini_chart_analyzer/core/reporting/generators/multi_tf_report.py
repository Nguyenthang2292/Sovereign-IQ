"""
Multi-timeframe HTML report generator.

This module generates HTML reports for multi-timeframe analysis with accordion layout.
"""

import os
from datetime import datetime
from typing import Dict, List

from modules.gemini_chart_analyzer.core.exceptions import ReportGenerationError
from modules.gemini_chart_analyzer.core.reporting.generators.chart_utils import (
    find_chart_paths_for_timeframes,
    sanitize_chart_path,
    sanitize_symbol_for_filename,
)
from modules.gemini_chart_analyzer.core.reporting.generators.formatters import (
    escape_html,
    format_text_to_html,
    get_signal_color,
)
from modules.gemini_chart_analyzer.core.reporting.generators.styles import (
    get_multi_tf_report_styles,
)
from modules.gemini_chart_analyzer.core.utils.chart_paths import get_charts_dir


def generate_multi_tf_report(
    symbol: str,
    timeframes_list: List[str],
    results: Dict,
    report_datetime: datetime,
    output_dir: str,
) -> str:
    """
    Generate HTML report for multi-timeframe analysis.

    Args:
        symbol: Trading symbol (e.g., "BTC/USDT")
        timeframes_list: List of timeframes analyzed
        results: Analysis results dictionary containing:
            - timeframes: Dict[str, Dict] with per-timeframe results
            - aggregated: Dict with aggregated signal and confidence
        report_datetime: Datetime of the report
        output_dir: Directory to save the HTML file

    Returns:
        Path to the generated HTML file

    Raises:
        ReportGenerationError: If report generation fails
    """
    # Format datetime
    datetime_str = report_datetime.strftime("%d/%m/%Y %H:%M:%S")

    # Escape user-derived text for HTML
    symbol_escaped = escape_html(symbol)
    datetime_str_escaped = escape_html(datetime_str)

    # Find chart paths for each timeframe
    charts_dir = get_charts_dir()
    chart_paths = find_chart_paths_for_timeframes(symbol, timeframes_list, str(charts_dir))

    # Get aggregated results
    aggregated = results.get("aggregated", {})
    agg_signal = aggregated.get("signal", "NONE")
    agg_confidence = aggregated.get("confidence", 0.0)

    # Escape aggregated signal
    agg_signal_escaped = escape_html(str(agg_signal))

    # Generate timeframe sections
    timeframe_sections = []
    for idx, tf in enumerate(timeframes_list):
        tf_result = results.get("timeframes", {}).get(tf, {})
        signal = tf_result.get("signal", "NONE")
        confidence = tf_result.get("confidence", 0.0)
        analysis_text = tf_result.get("analysis", "")
        chart_path = chart_paths.get(tf, "")

        tf_escaped = escape_html(tf)
        signal_escaped = escape_html(str(signal))
        confidence_str = f"{confidence:.2f}"
        analysis_html = (
            format_text_to_html(analysis_text)
            if analysis_text
            else "<p>Không có phân tích</p>"
        )

        if chart_path:
            chart_src = sanitize_chart_path(chart_path, output_dir)
            chart_html = f'<div class="chart-container"><img src="{chart_src}" alt="Chart {symbol_escaped} {tf_escaped}"></div>'
        else:
            chart_html = '<div class="chart-placeholder"><p>⚠️ Không tìm thấy biểu đồ</p></div>'

        signal_color = get_signal_color(signal)
        accordion_id = f"accordion-{idx}"

        timeframe_sections.append(f"""
            <div class="accordion-item">
                <input type="checkbox" id="{accordion_id}" class="accordion-checkbox">
                <label for="{accordion_id}" class="accordion-header">
                    <span class="timeframe-label">{tf_escaped}</span>
                    <span class="signal-badge" style="background-color: {signal_color}">{signal_escaped} ({confidence_str})</span>
                    <span class="accordion-toggle">▼</span>
                </label>
                <div class="accordion-content">
                    <div class="timeframe-info"><strong>Signal:</strong> <span style="color: {signal_color}">{signal_escaped}</span> | <strong>Confidence:</strong> {confidence_str}</div>
                    <div class="timeframe-chart">{chart_html}</div>
                    <div class="timeframe-analysis"><h3>Phân Tích Chi Tiết</h3><div class="analysis-content">{analysis_html}</div></div>
                </div>
            </div>
        """)

    timeframe_sections_html = "\n".join(timeframe_sections)
    agg_signal_color = get_signal_color(agg_signal)

    # Get CSS styles
    css_styles = get_multi_tf_report_styles()

    # HTML template
    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-TF Report - {symbol_escaped}</title>
    <style>{css_styles}</style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Multi-Timeframe Analysis: {symbol_escaped}</h1>
            <div class="datetime">📅 {datetime_str_escaped}</div>
            <div class="agg-signal">Aggregated Signal: <span style="color: {agg_signal_color}">{agg_signal_escaped} ({agg_confidence:.2f})</span></div>
        </div>
        <div class="content">{timeframe_sections_html}</div>
    </div>
</body>
</html>"""

    # Save HTML file
    timestamp = report_datetime.strftime("%Y%m%d_%H%M%S")
    safe_symbol = sanitize_symbol_for_filename(symbol)
    html_filename = f"multi_tf_{safe_symbol}_{timestamp}.html"
    html_path = os.path.join(output_dir, html_filename)

    os.makedirs(output_dir, exist_ok=True)
    try:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
    except OSError as e:
        raise ReportGenerationError(f"Failed to write multi-TF HTML report: {e}") from e

    return html_path

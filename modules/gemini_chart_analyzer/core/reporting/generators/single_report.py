"""
Single timeframe HTML report generator.

This module generates HTML reports for single timeframe analysis.
"""

import os
from datetime import datetime

from modules.gemini_chart_analyzer.core.exceptions import ReportGenerationError
from modules.gemini_chart_analyzer.core.reporting.generators.chart_utils import (
    embed_chart_as_base64,
    sanitize_symbol_for_filename,
)
from modules.gemini_chart_analyzer.core.reporting.generators.formatters import (
    escape_html,
    format_text_to_html,
)
from modules.gemini_chart_analyzer.core.reporting.generators.styles import (
    get_single_report_styles,
)


def generate_single_report(
    symbol: str,
    timeframe: str,
    chart_path: str,
    analysis_result: str,
    report_datetime: datetime,
    output_dir: str,
) -> str:
    """
    Generate HTML report for single timeframe analysis.

    Args:
        symbol: Trading symbol (e.g., "BTC/USDT")
        timeframe: Timeframe (e.g., "1h", "4h")
        chart_path: Path to the chart image
        analysis_result: Analysis text from Gemini AI
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
    timeframe_escaped = escape_html(timeframe)
    datetime_str_escaped = escape_html(datetime_str)

    # Convert analysis text to HTML
    analysis_html = format_text_to_html(analysis_result)

    # Embed image as base64 for standalone HTML file
    image_src = embed_chart_as_base64(chart_path) or ""

    # Get CSS styles
    css_styles = get_single_report_styles()

    # HTML template
    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo Cáo Phân Tích - {symbol_escaped} {timeframe_escaped}</title>
    <style>{css_styles}</style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Báo Cáo Phân Tích Kỹ Thuật - {symbol_escaped} {timeframe_escaped}</h1>
            <div class="subtitle">Gemini AI Chart Analysis Report</div>
            <div class="datetime">📅 Ngày xuất báo cáo: {datetime_str_escaped}</div>
        </div>
        <div class="content">
            <div class="info-section">
                <h2>📈 Thông Tin Giao Dịch</h2>
                <div class="info-grid">
                    <div class="info-item"><label>Symbol</label><span>{symbol_escaped}</span></div>
                    <div class="info-item"><label>Timeframe</label><span>{timeframe_escaped}</span></div>
                    <div class="info-item"><label>Ngày Phân Tích</label><span>{datetime_str_escaped}</span></div>
                </div>
            </div>
            <div class="chart-section">
                <h2>📉 Biểu Đồ Kỹ Thuật</h2>
                <div class="chart-container">
                    {f'<img src="{image_src}" alt="Chart {symbol_escaped} {timeframe_escaped}">' if image_src else "<p>No chart available</p>"}
                </div>
            </div>
            <div class="analysis-section">
                <h2>🤖 Phân Tích Từ Gemini AI</h2>
                <div class="analysis-content">{analysis_html}</div>
            </div>
        </div>
    </div>
</body>
</html>"""

    # Save HTML file
    timestamp = report_datetime.strftime("%Y%m%d_%H%M%S")
    safe_symbol = sanitize_symbol_for_filename(symbol)
    html_filename = f"{safe_symbol}_{timeframe}_{timestamp}.html"
    html_path = os.path.join(output_dir, html_filename)

    os.makedirs(output_dir, exist_ok=True)
    try:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
    except OSError as e:
        raise ReportGenerationError(f"Failed to write single HTML report: {e}") from e

    return html_path

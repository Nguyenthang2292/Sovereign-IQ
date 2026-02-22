"""
Batch scan HTML report generator.

This module generates HTML reports for batch scan results with sortable tables
and accordion layout.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from modules.gemini_chart_analyzer.core.exceptions import ReportGenerationError
from modules.gemini_chart_analyzer.core.reporting.generators.formatters import (
    escape_html,
    get_signal_color,
)
from modules.gemini_chart_analyzer.core.reporting.generators.styles import (
    get_batch_report_styles,
)
from modules.gemini_chart_analyzer.core.utils import find_project_root


def generate_batch_report(results_data: Dict[str, Any], output_dir: str) -> str:
    """
    Generate HTML report for batch scan results.

    Args:
        results_data: Batch scan results dictionary containing:
            - timestamp: ISO format timestamp
            - timeframes: List of timeframes scanned
            - summary: Summary statistics (total, long, short, none counts)
            - long_symbols_with_confidence: List of (symbol, confidence) tuples
            - short_symbols_with_confidence: List of (symbol, confidence) tuples
            - all_results: Dict of all symbol results
        output_dir: Directory to save the HTML file

    Returns:
        Path to the generated HTML file

    Raises:
        ReportGenerationError: If report generation fails
    """
    # Parse data
    timestamp_str = results_data.get("timestamp", datetime.now().isoformat())
    try:
        report_datetime = datetime.fromisoformat(timestamp_str)
    except (ValueError, TypeError):
        report_datetime = datetime.now()

    datetime_str = report_datetime.strftime("%d/%m/%Y %H:%M:%S")
    timeframes = results_data.get("timeframes", [])
    summary = results_data.get("summary", {})
    long_with_conf = results_data.get("long_symbols_with_confidence", [])
    short_with_conf = results_data.get("short_symbols_with_confidence", [])
    all_results = results_data.get("all_results", {})

    # Extract NONE symbols
    none_with_conf = _extract_none_symbols(all_results)

    # Summary statistics
    total_symbols = summary.get("total_symbols", 0)
    scanned_symbols = summary.get("scanned_symbols", 0)
    long_count = summary.get("long_count", 0)
    short_count = summary.get("short_count", 0)
    none_count = summary.get("none_count", 0)
    long_percentage = summary.get("long_percentage", 0.0)
    short_percentage = summary.get("short_percentage", 0.0)

    # Calculate none_percentage explicitly
    total_count = scanned_symbols if scanned_symbols > 0 else (long_count + short_count + none_count)
    none_percentage = (none_count / total_count * 100.0) if total_count > 0 else 0.0

    # Generate symbol rows for each category
    is_multi_tf = len(timeframes) > 1 if timeframes else False
    primary_tf = timeframes[0] if timeframes else results_data.get("timeframe", "1h")

    long_rows = _generate_symbol_rows(long_with_conf, "LONG", all_results, timeframes, is_multi_tf, primary_tf)
    short_rows = _generate_symbol_rows(short_with_conf, "SHORT", all_results, timeframes, is_multi_tf, primary_tf)
    none_rows = _generate_symbol_rows(none_with_conf, "NONE", all_results, timeframes, is_multi_tf, primary_tf)

    timeframes_str = ", ".join(timeframes) if timeframes else "N/A"
    datetime_str_escaped = escape_html(datetime_str)
    timeframes_str_escaped = escape_html(timeframes_str)

    # Find main script path for detail command
    html_dir = Path(output_dir).resolve()
    project_root = find_project_root(html_dir)
    main_script_path = project_root / "main_gemini_chart_analyzer.py"
    main_script_absolute = str(main_script_path.resolve())

    # Get CSS styles
    css_styles = get_batch_report_styles()

    # Generate JavaScript
    javascript = _generate_batch_report_javascript(main_script_absolute)

    # Full HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Batch Scan Report - {datetime_str_escaped}</title>
    <style>{css_styles}</style>
    {javascript}
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Batch Scan Report</h1>
            <div>📅 {datetime_str_escaped} | Timeframes: {timeframes_str_escaped}</div>
        </div>
        <div class="content">
            <div class="summary-section">
                <div class="summary-grid">
                    <div class="summary-card"><div class="label">Total</div><div class="value">{total_symbols}</div><div class="subvalue">Scanned: {scanned_symbols}</div></div>
                    <div class="summary-card long"><div class="label">LONG</div><div class="value">{long_count}</div><div class="subvalue">{long_percentage:.2f}%</div></div>
                    <div class="summary-card short"><div class="label">SHORT</div><div class="value">{short_count}</div><div class="subvalue">{short_percentage:.2f}%</div></div>
                    <div class="summary-card none"><div class="label">NONE</div><div class="value">{none_count}</div><div class="subvalue">{none_percentage:.2f}%</div></div>
                </div>
            </div>
            <div class="accordion-container">
                <div class="accordion-item">
                    <input type="checkbox" id="acc-long" class="accordion-checkbox" checked>
                    <label for="acc-long" class="accordion-header"><strong>LONG ({long_count})</strong></label>
                    <div class="accordion-content"><table class="symbols-table"><tbody>{long_rows}</tbody></table></div>
                </div>
                <div class="accordion-item">
                    <input type="checkbox" id="acc-short" class="accordion-checkbox" checked>
                    <label for="acc-short" class="accordion-header"><strong>SHORT ({short_count})</strong></label>
                    <div class="accordion-content"><table class="symbols-table"><tbody>{short_rows}</tbody></table></div>
                </div>
                <div class="accordion-item">
                    <input type="checkbox" id="acc-none" class="accordion-checkbox">
                    <label for="acc-none" class="accordion-header"><strong>NONE ({none_count})</strong></label>
                    <div class="accordion-content"><table class="symbols-table"><tbody>{none_rows}</tbody></table></div>
                </div>
            </div>
        </div>
    </div>
    <div id="detailModal" class="modal-overlay">
        <div class="modal-content">
            <h2>📊 Detail: <span id="modalSymbolName"></span></h2>
            <div class="command-container"><p class="command-text" id="modalCommandText"></p></div>
            <button onclick="copyCommand()">Copy</button><button onclick="closeModal()">Close</button>
        </div>
    </div>
</body></html>"""

    # Save HTML file
    timestamp = report_datetime.strftime("%Y%m%d_%H%M%S")
    html_filename = f"batch_scan_{timestamp}.html"
    html_path = os.path.join(output_dir, html_filename)
    os.makedirs(output_dir, exist_ok=True)
    try:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
    except OSError as e:
        raise ReportGenerationError(f"Failed to write batch HTML report: {e}") from e

    return html_path


def _extract_none_symbols(all_results: Dict[str, Any]) -> List[tuple]:
    """Extract NONE symbols from all results."""
    none_with_conf = []
    for symbol, result in all_results.items():
        if isinstance(result, dict):
            signal = result.get("signal", "NONE")
            if signal.upper() == "NONE":
                confidence = result.get("confidence", 0.0)
                none_with_conf.append((symbol, confidence))

    # Sort by confidence (descending)
    none_with_conf.sort(key=lambda x: x[1], reverse=True)
    return none_with_conf


def _generate_symbol_rows(
    symbols_with_conf: List[tuple],
    signal_type: str,
    all_results: Dict,
    timeframes: List[str],
    is_multi_tf: bool,
    primary_tf: str,
) -> str:
    """Generate HTML table rows for symbols."""
    rows = []
    for symbol, confidence in symbols_with_conf:
        result = all_results.get(symbol, {})
        timeframe_breakdown = result.get("timeframe_breakdown", {}) if isinstance(result, dict) else {}

        # Generate timeframe breakdown badges
        breakdown_badges = []
        if timeframe_breakdown:
            for tf, tf_data in timeframe_breakdown.items():
                if isinstance(tf_data, dict):
                    tf_signal = tf_data.get("signal", "NONE")
                    tf_conf = tf_data.get("confidence", 0.0)
                    tf_color = get_signal_color(tf_signal)
                    badge_style = f"background-color: {tf_color}20; color: {tf_color}; border: 1px solid {tf_color}"
                    badge_content = f"{escape_html(str(tf))}: {escape_html(str(tf_signal))} ({tf_conf:.2f})"
                    breakdown_badges.append(f'<span class="tf-badge" style="{badge_style}">{badge_content}</span>')

        breakdown_html = " ".join(breakdown_badges) if breakdown_badges else '<span class="no-breakdown">N/A</span>'
        signal_color = get_signal_color(signal_type)
        symbol_json = json.dumps(symbol)
        timeframes_json = json.dumps(timeframes) if timeframes else "[]"
        symbol_escaped = escape_html(str(symbol))
        confidence_escaped = escape_html(str(confidence))
        signal_type_escaped = escape_html(str(signal_type))
        width_pct = max(0, min(confidence, 1)) * 100

        rows.append(f"""
                <tr data-symbol="{symbol_escaped}" data-confidence="{confidence_escaped}">
                    <td class="symbol-cell">{symbol_escaped}</td>
                    <td class="signal-cell"><span class="signal-badge" style="background-color: {signal_color}">{signal_type_escaped}</span></td>
                    <td class="confidence-cell">
                        <div class="confidence-bar-container">
                            <span class="confidence-value">{confidence:.2f}</span>
                            <div class="confidence-bar"><div class="confidence-fill" style="width: {width_pct}%; background-color: {signal_color}"></div></div>
                        </div>
                    </td>
                    <td class="breakdown-cell">{breakdown_html}</td>
                    <td class="action-cell">
                        <button class="detail-btn" onclick='showDetailModal({symbol_json}, {timeframes_json}, {str(is_multi_tf).lower()}, {json.dumps(primary_tf)})'>Xem chi tiết</button>
                    </td>
                </tr>""")
    return "\n".join(rows)


def _generate_batch_report_javascript(main_script_path: str) -> str:
    """Generate JavaScript for batch report interactivity."""
    return f"""
    <script>
        const MAIN_SCRIPT_PATH = {json.dumps(main_script_path)};
        function showDetailModal(symbol, timeframes, isMultiTF, primaryTF) {{
            const modal = document.getElementById('detailModal');
            document.getElementById('modalSymbolName').textContent = symbol;
            let command;
            if (isMultiTF && timeframes && timeframes.length > 1) {{
                command = `python "${{MAIN_SCRIPT_PATH}}" --symbol "${{symbol}}" --timeframes "${{timeframes.join(',')}}"`;
            }} else {{
                command = `python "${{MAIN_SCRIPT_PATH}}" --symbol "${{symbol}}" --timeframe "${{primaryTF || '1h'}}"`;
            }}
            document.getElementById('modalCommandText').textContent = command;
            modal.classList.add('show');
        }}
        function closeModal() {{ document.getElementById('detailModal').classList.remove('show'); }}
        function copyCommand() {{
            const cmd = document.getElementById('modalCommandText').textContent;
            navigator.clipboard.writeText(cmd).then(() => alert('Copied!'));
        }}
    </script>
    """

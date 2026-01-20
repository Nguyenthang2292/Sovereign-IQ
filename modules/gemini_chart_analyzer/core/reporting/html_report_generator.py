"""
Centralized HTML Report Generator for Chart Analysis.

This module provides a unified interface for generating different types of HTML reports:
- Single timeframe analysis
- Multi-timeframe analysis
- Batch scan results
"""

import base64
import json
import os
from datetime import datetime
from html import escape as html_escape
from pathlib import Path
from typing import Any, Dict, List

from modules.common.ui.logging import log_warn
from modules.gemini_chart_analyzer.core.exceptions import ReportGenerationError
from modules.gemini_chart_analyzer.core.utils import find_project_root
from modules.gemini_chart_analyzer.core.utils.chart_paths import get_charts_dir


def generate_html_report(analysis_data: Dict[str, Any], output_dir: str, report_type: str = "single", **kwargs) -> str:
    """
    Generate an HTML report based on the report type.

    Args:
        analysis_data: Data for the report
        output_dir: Directory to save the HTML file
        report_type: Type of report ('single', 'multi', 'batch')
        **kwargs: Additional parameters specific to each report type

    Returns:
        Path to the generated HTML file
    """
    if report_type == "batch":
        return _generate_batch_report(analysis_data, output_dir)
    elif report_type == "multi":
        return _generate_multi_tf_report(
            symbol=analysis_data.get("symbol", "Unknown"),
            timeframes_list=kwargs.get("timeframes_list", []),
            results=analysis_data,
            report_datetime=kwargs.get("report_datetime", datetime.now()),
            output_dir=output_dir,
        )
    else:  # single
        return _generate_single_report(
            symbol=analysis_data.get("symbol", "Unknown"),
            timeframe=analysis_data.get("timeframe", "1h"),
            chart_path=kwargs.get("chart_path", ""),
            analysis_result=analysis_data.get("analysis", ""),
            report_datetime=kwargs.get("report_datetime", datetime.now()),
            output_dir=output_dir,
        )


def _generate_single_report(
    symbol: str, timeframe: str, chart_path: str, analysis_result: str, report_datetime: datetime, output_dir: str
) -> str:
    """Generate HTML report for single timeframe analysis."""
    # Format datetime
    datetime_str = report_datetime.strftime("%d/%m/%Y %H:%M:%S")

    # Escape user-derived text for HTML
    symbol_escaped = html_escape(symbol)
    timeframe_escaped = html_escape(timeframe)
    datetime_str_escaped = html_escape(datetime_str)

    # Convert analysis text to HTML
    analysis_html = _format_text_to_html(analysis_result)

    # Embed image as base64 for standalone HTML file
    try:
        if chart_path and os.path.exists(chart_path):
            with open(chart_path, "rb") as f:
                image_data = f.read()
                image_base64 = base64.b64encode(image_data).decode("utf-8")
                # Detect image format
                if chart_path.lower().endswith(".png"):
                    image_mime = "image/png"
                elif chart_path.lower().endswith(".jpg") or chart_path.lower().endswith(".jpeg"):
                    image_mime = "image/jpeg"
                else:
                    image_mime = "image/png"
                image_src = f"data:{image_mime};base64,{image_base64}"
        else:
            image_src = ""
            log_warn(f"Chart path not found: {chart_path}")
    except Exception as e:
        log_warn(f"Cannot embed image: {e}, using placeholder")
        image_src = ""

    # HTML template
    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo Cáo Phân Tích - {symbol_escaped} {timeframe_escaped}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%); color: #e0e0e0; line-height: 1.6; padding: 20px; min-height: 100vh; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: #2a2a2a; border-radius: 12px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5); overflow: hidden; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center; color: white; }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); }}
        .header .subtitle {{ font-size: 1.1em; opacity: 0.9; }}
        .header .datetime {{ margin-top: 15px; font-size: 0.95em; opacity: 0.85; font-style: italic; }}
        .content {{ padding: 30px; }}
        .info-section {{ background: #333; padding: 20px; border-radius: 8px; margin-bottom: 30px; border-left: 4px solid #667eea; }}
        .info-section h2 {{ color: #667eea; margin-bottom: 15px; font-size: 1.5em; }}
        .info-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px; }}
        .info-item {{ background: #3a3a3a; padding: 12px; border-radius: 6px; }}
        .info-item label {{ color: #aaa; font-size: 0.9em; display: block; margin-bottom: 5px; }}
        .info-item span {{ color: #fff; font-size: 1.1em; font-weight: bold; }}
        .chart-section {{ background: #333; padding: 20px; border-radius: 8px; margin-bottom: 30px; border-left: 4px solid #48bb78; }}
        .chart-section h2 {{ color: #48bb78; margin-bottom: 15px; font-size: 1.5em; }}
        .chart-container {{ text-align: center; background: #1a1a1a; padding: 15px; border-radius: 8px; margin-top: 15px; }}
        .chart-container img {{ max-width: 100%; height: auto; border-radius: 6px; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4); }}
        .analysis-section {{ background: #333; padding: 20px; border-radius: 8px; border-left: 4px solid #f6ad55; }}
        .analysis-section h2 {{ color: #f6ad55; margin-bottom: 15px; font-size: 1.5em; }}
        .analysis-content {{ background: #2a2a2a; padding: 20px; border-radius: 6px; margin-top: 15px; line-height: 1.8; }}
        .analysis-content p {{ margin-bottom: 15px; }}
        .analysis-content strong {{ color: #f6ad55; }}
        .analysis-content em {{ color: #a0a0a0; font-style: italic; }}
    </style>
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
    safe_symbol = symbol.replace("/", "_").replace(":", "_")
    html_filename = f"{safe_symbol}_{timeframe}_{timestamp}.html"
    html_path = os.path.join(output_dir, html_filename)

    os.makedirs(output_dir, exist_ok=True)
    try:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
    except OSError as e:
        raise ReportGenerationError(f"Failed to write single HTML report: {e}") from e

    return html_path


def _generate_multi_tf_report(
    symbol: str, timeframes_list: List[str], results: Dict, report_datetime: datetime, output_dir: str
) -> str:
    """Generate HTML report for multi-timeframe analysis."""
    # Format datetime
    datetime_str = report_datetime.strftime("%d/%m/%Y %H:%M:%S")

    # Escape user-derived text for HTML
    symbol_escaped = html_escape(symbol)
    datetime_str_escaped = html_escape(datetime_str)

    # Find chart paths for each timeframe
    charts_dir = get_charts_dir()
    chart_paths = _find_chart_paths_for_timeframes(symbol, timeframes_list, str(charts_dir))

    # Get aggregated results
    aggregated = results.get("aggregated", {})
    agg_signal = aggregated.get("signal", "NONE")
    agg_confidence = aggregated.get("confidence", 0.0)

    # Escape aggregated signal
    agg_signal_escaped = html_escape(str(agg_signal))

    # Generate timeframe sections
    timeframe_sections = []
    for idx, tf in enumerate(timeframes_list):
        tf_result = results.get("timeframes", {}).get(tf, {})
        signal = tf_result.get("signal", "NONE")
        confidence = tf_result.get("confidence", 0.0)
        analysis_text = tf_result.get("analysis", "")
        chart_path = chart_paths.get(tf, "")

        tf_escaped = html_escape(tf)
        signal_escaped = html_escape(str(signal))
        confidence_str = f"{confidence:.2f}"
        analysis_html = _format_text_to_html(analysis_text) if analysis_text else "<p>Không có phân tích</p>"

        if chart_path:
            chart_src = _sanitize_chart_path(chart_path, output_dir)
            chart_html = (
                f'<div class="chart-container"><img src="{chart_src}" alt="Chart {symbol_escaped} {tf_escaped}"></div>'
            )
        else:
            chart_html = '<div class="chart-placeholder"><p>⚠️ Không tìm thấy biểu đồ</p></div>'

        signal_color = _get_signal_color(signal)
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
    agg_signal_color = _get_signal_color(agg_signal)

    # HTML template
    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-TF Report - {symbol_escaped}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #1a1a1a; color: #e0e0e0; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: #2a2a2a; border-radius: 12px; overflow: hidden; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center; color: white; }}
        .agg-signal {{ font-size: 2em; margin-top: 20px; padding: 10px; border-radius: 8px; background: rgba(0,0,0,0.2); }}
        .accordion-item {{ border-bottom: 1px solid #444; }}
        .accordion-checkbox {{ display: none; }}
        .accordion-header {{ display: flex; justify-content: space-between; padding: 20px; background: #333; cursor: pointer; }}
        .accordion-content {{ max-height: 0; overflow: hidden; transition: max-height 0.3s; background: #222; }}
        .accordion-checkbox:checked ~ .accordion-content {{ max-height: 2000px; padding: 20px; }}
        .signal-badge {{ padding: 5px 12px; border-radius: 6px; font-weight: bold; color: white; }}
        .chart-container img {{ max-width: 100%; border-radius: 6px; }}
        .analysis-content {{ line-height: 1.8; margin-top: 10px; }}
    </style>
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
    safe_symbol = symbol.replace("/", "_").replace(":", "_")
    html_filename = f"multi_tf_{safe_symbol}_{timestamp}.html"
    html_path = os.path.join(output_dir, html_filename)

    os.makedirs(output_dir, exist_ok=True)
    try:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
    except OSError as e:
        raise ReportGenerationError(f"Failed to write multi-TF HTML report: {e}") from e

    return html_path


def _generate_batch_report(results_data: Dict, output_dir: str) -> str:
    """
    Tạo HTML report cho batch scan results với accordion layout và sortable tables.
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
    none_with_conf = []
    for symbol, result in all_results.items():
        if isinstance(result, dict):
            signal = result.get("signal", "NONE")
            if signal.upper() == "NONE":
                confidence = result.get("confidence", 0.0)
                none_with_conf.append((symbol, confidence))

    # Sort by confidence (descending)
    none_with_conf.sort(key=lambda x: x[1], reverse=True)

    # Summary statistics
    total_symbols = summary.get("total_symbols", 0)
    scanned_symbols = summary.get("scanned_symbols", 0)
    long_count = summary.get("long_count", 0)
    short_count = summary.get("short_count", 0)
    none_count = summary.get("none_count", 0)
    long_percentage = summary.get("long_percentage", 0.0)
    short_percentage = summary.get("short_percentage", 0.0)
    avg_long_conf = summary.get("avg_long_confidence", 0.0)
    avg_short_conf = summary.get("avg_short_confidence", 0.0)

    # Calculate none_percentage explicitly
    total_count = scanned_symbols if scanned_symbols > 0 else (long_count + short_count + none_count)
    none_percentage = (none_count / total_count * 100.0) if total_count > 0 else 0.0

    # Inner helper for rows
    def generate_symbol_rows(symbols_with_conf: List, signal_type: str) -> str:
        rows = []
        is_multi_tf = len(timeframes) > 1 if timeframes else False
        for symbol, confidence in symbols_with_conf:
            result = all_results.get(symbol, {})
            timeframe_breakdown = result.get("timeframe_breakdown", {}) if isinstance(result, dict) else {}

            breakdown_badges = []
            if timeframe_breakdown:
                for tf, tf_data in timeframe_breakdown.items():
                    if isinstance(tf_data, dict):
                        tf_signal = tf_data.get("signal", "NONE")
                        tf_conf = tf_data.get("confidence", 0.0)
                        tf_color = _get_signal_color(tf_signal)
                        badge_style = f"background-color: {tf_color}20; color: {tf_color}; border: 1px solid {tf_color}"
                        badge_content = f"{html_escape(str(tf))}: {html_escape(str(tf_signal))} ({tf_conf:.2f})"
                        breakdown_badges.append(f'<span class="tf-badge" style="{badge_style}">{badge_content}</span>')

            breakdown_html = " ".join(breakdown_badges) if breakdown_badges else '<span class="no-breakdown">N/A</span>'
            signal_color = _get_signal_color(signal_type)
            symbol_json = json.dumps(symbol)
            timeframes_json = json.dumps(timeframes) if timeframes else "[]"
            primary_tf = timeframes[0] if timeframes else results_data.get("timeframe", "1h")
            symbol_escaped = html_escape(str(symbol))
            confidence_escaped = html_escape(str(confidence))
            signal_type_escaped = html_escape(str(signal_type))
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

    long_rows = generate_symbol_rows(long_with_conf, "LONG")
    short_rows = generate_symbol_rows(short_with_conf, "SHORT")
    none_rows = generate_symbol_rows(none_with_conf, "NONE")
    timeframes_str = ", ".join(timeframes) if timeframes else "N/A"
    datetime_str_escaped = html_escape(datetime_str)
    timeframes_str_escaped = html_escape(timeframes_str)

    html_dir = Path(output_dir).resolve()
    project_root = find_project_root(html_dir)
    main_script_path = project_root / "main_gemini_chart_analyzer.py"
    main_script_absolute = str(main_script_path.resolve())

    # Full HTML content (omitted style for brevity in replacement, but I should keep it high quality)
    # I'll use the one I wrote before but with the logic integrated.

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Batch Scan Report - {datetime_str_escaped}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #1e1e1e; color: #e0e0e0; line-height: 1.6; padding: 20px; }}
        .container {{ max-width: 1600px; margin: 0 auto; background: #2a2a2a; border-radius: 12px; overflow: hidden; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center; color: white; }}
        .content {{ padding: 30px; }}
        .summary-section {{ background: #333; padding: 25px; border-radius: 8px; margin-bottom: 30px; border-left: 4px solid #667eea; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-top: 20px; }}
        .summary-card {{ background: #3a3a3a; padding: 20px; border-radius: 8px; text-align: center; }}
        .summary-card.long .value {{ color: #48bb78; }}
        .summary-card.short .value {{ color: #f56565; }}
        .summary-card .value {{ font-size: 2em; font-weight: bold; }}
        .accordion-item {{ background: #333; border-radius: 8px; margin-bottom: 15px; border: 1px solid #444; overflow: hidden; }}
        .accordion-checkbox {{ display: none; }}
        .accordion-header {{ display: flex; align-items: center; justify-content: space-between; padding: 20px 25px; background: #3a3a3a; cursor: pointer; }}
        .accordion-content {{ max-height: 0; overflow: hidden; transition: max-height 0.3s; }}
        .accordion-checkbox:checked ~ .accordion-content {{ max-height: 10000px; }}
        .symbols-table {{ width: 100%; border-collapse: collapse; background: #333; }}
        .symbols-table th {{ background: #3a3a3a; padding: 15px; text-align: left; color: #667eea; cursor: pointer; position: sticky; top: 0; }}
        .symbols-table td {{ padding: 12px 15px; border-bottom: 1px solid #444; }}
        .confidence-bar-container {{ display: flex; align-items: center; gap: 10px; }}
        .confidence-bar {{ flex: 1; height: 15px; background: #1a1a1a; border-radius: 10px; overflow: hidden; }}
        .confidence-fill {{ height: 100%; border-radius: 10px; }}
        .detail-btn {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; }}
        .tf-badge {{ display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 0.8em; margin: 2px; }}
        /* Modal */
        .modal-overlay {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); z-index: 1000; justify-content: center; align-items: center; }}
        .modal-overlay.show {{ display: flex; }}
        .modal-content {{ background: #2a2a2a; border-radius: 12px; padding: 30px; width: 90%; max-width: 700px; }}
        .command-container {{ background: #1a1a1a; padding: 15px; border-radius: 6px; margin: 15px 0; }}
        .command-text {{ color: #48bb78; font-family: monospace; word-break: break-all; }}
    </style>
    <script>
        const MAIN_SCRIPT_PATH = {json.dumps(main_script_absolute)};
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


def _format_text_to_html(text: str) -> str:
    """Convert analysis text (with simple markdown-like syntax) to HTML."""
    if not text:
        return ""

    # Escape HTML first
    html_text = html_escape(text)

    # Replace markdown-like syntax
    # Bold: **text**
    import re

    html_text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html_text)

    # Italic: *text*
    html_text = re.sub(r"\*(.*?)\*", r"<em>\1</em>", html_text)

    # Newlines to <p> tags
    paragraphs = html_text.split("\n\n")
    formatted_paragraphs = []
    for p in paragraphs:
        if p.strip():
            # Convert single newlines within paragraphs to <br>
            p_with_br = p.replace("\n", "<br>")
            formatted_paragraphs.append(f"<p>{p_with_br}</p>")

    return "\n".join(formatted_paragraphs)


def _get_signal_color(signal: str) -> str:
    """Get color for signal type."""
    signal_upper = str(signal).upper()
    if signal_upper == "LONG":
        return "#48bb78"  # green
    elif signal_upper == "SHORT":
        return "#f56565"  # red
    else:
        return "#a0a0a0"  # gray


def _sanitize_chart_path(chart_path: str, output_dir: str) -> str:
    """Convert absolute chart path to a relative path for HTML use."""
    try:
        rel_path = os.path.relpath(chart_path, output_dir)
        return rel_path.replace("\\", "/")
    except Exception:
        return Path(chart_path).name


def _find_chart_paths_for_timeframes(symbol: str, timeframes: List[str], charts_dir: str) -> Dict[str, str]:
    """Find chart image files for each timeframe of a symbol."""
    results = {}
    if not os.path.exists(charts_dir):
        return results

    safe_symbol = symbol.replace("/", "_").replace(":", "_")
    all_files = os.listdir(charts_dir)

    for tf in timeframes:
        # Match pattern: {safe_symbol}_{tf}_{timestamp}.png
        matches = [f for f in all_files if f.startswith(f"{safe_symbol}_{tf}_") and f.endswith(".png")]
        if matches:
            # Get latest match
            latest = sorted(matches, reverse=True)[0]
            results[tf] = os.path.join(charts_dir, latest)

    return results

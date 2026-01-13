"""
CLI Main Program for Gemini Chart Analyzer

Workflow:
1. Enter symbol names and timeframe
2. Generate chart images based on the entered information (can add indicators such as MA, RSI, etc.)
3. Save the image
4. Access Google Gemini using the API configured in config/config_api.py
5. Upload the image for Google Gemini to analyze the chart image (LONG/SHORT - TP/SL)
"""

import base64
import fnmatch
import glob
import html
import os
import re
import sys
import urllib.parse
import warnings
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import markdown

# Add project root to sys.path to ensure modules can be imported
# This is needed when running the file directly from subdirectories
if "__file__" in globals():
    project_root = Path(__file__).parent.parent.parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

# Fix encoding issues on Windows
from modules.common.utils import configure_windows_stdio

configure_windows_stdio()

import json

from colorama import Fore, Style
from colorama import init as colorama_init

from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager
from modules.common.utils import color_text, log_error, log_info, log_success, log_warn, normalize_timeframe
from modules.gemini_chart_analyzer.cli.argument_parser import parse_args
from modules.gemini_chart_analyzer.cli.interactive_menu import interactive_config_menu
from modules.gemini_chart_analyzer.core.analyzers.gemini_chart_analyzer import GeminiChartAnalyzer
from modules.gemini_chart_analyzer.core.analyzers.multi_timeframe_coordinator import MultiTimeframeCoordinator
from modules.gemini_chart_analyzer.core.generators.chart_generator import ChartGenerator
from modules.gemini_chart_analyzer.core.utils.chart_paths import get_analysis_results_dir, get_charts_dir

# Suppress specific warnings from third-party libraries while preserving important warnings
# Suppress DeprecationWarning from pandas, numpy, matplotlib (common in data science libraries)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pandas")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="matplotlib")
# Suppress FutureWarning from pandas (version compatibility warnings)
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
# Suppress specific benign UserWarning from matplotlib (font/backend issues that don't affect functionality)
# Only suppress known harmless warnings: font-related and backend-related messages
warnings.filterwarnings(
    "ignore", category=UserWarning, module="matplotlib", message=r"(?i).*(findfont|font family|glyph|backend).*"
)
colorama_init(autoreset=True)


def _convert_args_to_config(args):
    """Convert parsed arguments to configuration format."""
    # Extract symbol and timeframe
    symbol = args.symbol
    timeframe = getattr(args, "timeframe", None)
    timeframes_list = getattr(args, "timeframes_list", None)

    # Build indicators dict
    indicators = {}

    if not args.no_ma:
        if args.ma_periods_list:
            indicators["MA"] = {"periods": args.ma_periods_list}
        else:
            indicators["MA"] = {"periods": [20, 50, 200]}

    if not args.no_rsi:
        indicators["RSI"] = {"period": args.rsi_period}

    if not args.no_macd:
        indicators["MACD"] = {"fast": 12, "slow": 26, "signal": 9}

    if args.enable_bb:
        indicators["BB"] = {"period": args.bb_period, "std": 2}

    # Prompt configuration
    prompt_type = args.prompt_type
    custom_prompt = getattr(args, "custom_prompt", None)

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "timeframes_list": timeframes_list,
        "indicators": indicators,
        "prompt_type": prompt_type,
        "custom_prompt": custom_prompt,
        "limit": args.limit,
        "chart_figsize": args.chart_figsize_tuple,
        "chart_dpi": args.chart_dpi,
        "no_cleanup": args.no_cleanup,
    }


def _convert_menu_to_config(config):
    """Convert interactive menu config to format used by main()."""
    return {
        "symbol": config.symbol,
        "timeframe": getattr(config, "timeframe", None),
        "timeframes_list": getattr(config, "timeframes_list", None),
        "indicators": getattr(config, "indicators", {}),
        "prompt_type": config.prompt_type,
        "custom_prompt": getattr(config, "custom_prompt", None),
        "limit": getattr(config, "limit", 500),
        "chart_figsize": getattr(config, "chart_figsize_tuple", (16, 10)),
        "chart_dpi": getattr(config, "chart_dpi", 150),
        "no_cleanup": getattr(config, "no_cleanup", False),
    }


def cleanup_old_files(directory: str, exclude_patterns: Optional[list] = None) -> int:
    """
    Xóa tất cả files cũ trong thư mục.

    Args:
        directory: Đường dẫn thư mục cần cleanup
        exclude_patterns: List các patterns để giữ lại (optional)

    Returns:
        Số lượng files đã xóa
    """
    if exclude_patterns is None:
        exclude_patterns = []

    if not os.path.exists(directory):
        return 0

    deleted_count = 0
    errors = []

    try:
        # Liệt kê tất cả files trong thư mục (không bao gồm subdirectories)
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)

            # Chỉ xóa files, không xóa directories
            if not os.path.isfile(file_path):
                continue

            # Kiểm tra exclude patterns
            should_exclude = False
            for pattern in exclude_patterns:
                if fnmatch.fnmatch(filename, pattern):
                    should_exclude = True
                    break
            if should_exclude:
                continue

            # Xóa file
            try:
                os.remove(file_path)
                deleted_count += 1
            except Exception as e:
                errors.append(f"{filename}: {str(e)}")

        if deleted_count > 0:
            log_info(f"Đã xóa {deleted_count} file(s) cũ trong {directory}")

        if errors:
            log_warn(f"Có {len(errors)} file(s) không thể xóa: {', '.join(errors[:3])}")

    except Exception as e:
        log_warn(f"Lỗi khi cleanup thư mục {directory}: {e}")

    return deleted_count


def _sanitize_chart_path(chart_path: str, output_dir: str) -> str:
    """
    Sanitize chart path để sử dụng an toàn trong HTML.

    Args:
        chart_path: Đường dẫn tuyệt đối đến file chart
        output_dir: Thư mục output để tính relative path

    Returns:
        Đường dẫn đã được sanitize và URL-encode
    """
    try:
        # Convert to relative path
        chart_rel_path = os.path.relpath(chart_path, output_dir)
        # Replace backslashes with slashes
        chart_rel_path = chart_rel_path.replace("\\", "/")
        # URL-encode the path components
        # Split path and encode each component separately
        path_parts = chart_rel_path.split("/")
        encoded_parts = [urllib.parse.quote(part, safe="") for part in path_parts]
        return "/".join(encoded_parts)
    except Exception:
        # Fallback: normalize and URL-encode the path for use in HTML src attributes
        normalized_path = chart_path.replace("\\", "/")
        path_parts = normalized_path.split("/")

        # Preserve drive letters (e.g., 'C:') and empty parts from leading slashes
        encoded_parts = []
        for i, part in enumerate(path_parts):
            if i == 0 and re.fullmatch(r"^[A-Za-z]:$", part):  # Drive letter on Windows
                encoded_parts.append(part)
            elif part == "":  # Leading slash or empty part
                encoded_parts.append(part)
            else:
                encoded_parts.append(urllib.parse.quote(part, safe=""))
        return "/".join(encoded_parts)


def format_text_to_html(text: str) -> str:
    """
    Convert text từ Gemini thành HTML format sử dụng markdown library.

    Hàm này sử dụng thư viện markdown để xử lý markdown một cách an toàn,
    hỗ trợ nested formatting (ví dụ: **bold *italic* bold**) và tránh
    các vấn đề với placeholder collisions.

    Markdown library tự động xử lý:
    - Nested formatting (bold chứa italic và ngược lại)
    - HTML escaping cho text content
    - Paragraphs và line breaks

    Args:
        text: Text markdown từ Gemini

    Returns:
        HTML string với formatting đã được convert
    """

    # Sử dụng markdown library để convert markdown thành HTML
    # Extensions được sử dụng:
    # - 'fenced_code': Hỗ trợ code blocks với ```
    # - 'tables': Hỗ trợ markdown tables
    html_output = markdown.markdown(text, extensions=["fenced_code", "tables"], output_format="html")

    # Nếu output rỗng hoặc chỉ có whitespace, wrap trong <p> tag
    if not html_output.strip():
        return "<p></p>"

    # Xử lý line breaks: convert single newlines trong paragraphs thành <br>
    # Markdown library tự động wrap paragraphs trong <p> tags
    # Nhưng single newlines trong paragraphs cần được convert thành <br>
    # Chỉ xử lý newlines bên trong <p> tags, không ảnh hưởng đến code blocks hoặc pre tags
    html_output = re.sub(
        r"(<p[^>]*>)(.*?)(</p>)",
        lambda m: m.group(1) + m.group(2).replace("\n", "<br>") + m.group(3),
        html_output,
        flags=re.DOTALL,
    )

    return html_output


def _find_chart_paths_for_timeframes(symbol: str, timeframes_list: List[str], charts_dir: str) -> Dict[str, str]:
    """
    Tìm chart paths cho mỗi timeframe từ charts directory.

    Tìm các file chart mới nhất match pattern {symbol}_{timeframe}_*.png
    cho mỗi timeframe trong danh sách.

    Args:
        symbol: Trading symbol (e.g., 'BTC/USDT')
        timeframes_list: List of timeframes (e.g., ['1h', '30m', '15m'])
        charts_dir: Directory chứa chart files

    Returns:
        Dict mapping timeframe -> chart_path (empty string nếu không tìm thấy)
    """
    safe_symbol = symbol.replace("/", "_").replace(":", "_")
    chart_paths = {}

    for tf in timeframes_list:
        # Pattern để tìm chart files: {symbol}_{timeframe}_*.png
        pattern = os.path.join(charts_dir, f"{safe_symbol}_{tf}_*.png")
        matching_files = glob.glob(pattern)

        if matching_files:
            # Sort theo modification time, lấy file mới nhất
            latest_file = max(matching_files, key=os.path.getmtime)
            chart_paths[tf] = latest_file
        else:
            chart_paths[tf] = ""

    return chart_paths


def generate_html_report(
    symbol: str, timeframe: str, chart_path: str, analysis_result: str, report_datetime: datetime, output_dir: str
) -> str:
    """
    Tạo HTML report với thông tin phân tích.

    Args:
        symbol: Tên symbol (e.g., 'BTC/USDT')
        timeframe: Timeframe (e.g., '1h')
        chart_path: Đường dẫn đến file ảnh biểu đồ
        analysis_result: Kết quả phân tích từ Gemini
        report_datetime: Ngày giờ tạo báo cáo
        output_dir: Thư mục lưu HTML file

    Returns:
        Đường dẫn đến file HTML đã tạo
    """
    # Format datetime
    datetime_str = report_datetime.strftime("%d/%m/%Y %H:%M:%S")

    # Escape user-derived text for HTML
    symbol_escaped = html.escape(symbol)
    timeframe_escaped = html.escape(timeframe)
    datetime_str_escaped = html.escape(datetime_str)

    # Convert analysis text to HTML
    analysis_html = format_text_to_html(analysis_result)

    # Embed image as base64 for standalone HTML file
    try:
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
    except Exception as e:
        log_warn(f"Không thể embed ảnh: {e}, sử dụng relative path")
        # Fallback to relative path - sanitize it
        image_src = _sanitize_chart_path(chart_path, output_dir)

    # HTML template
    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo Cáo Phân Tích - {symbol_escaped} {timeframe_escaped}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
            color: #e0e0e0;
            line-height: 1.6;
            padding: 20px;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: #2a2a2a;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            text-align: center;
            color: white;
        }}

        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }}

        .header .subtitle {{
            font-size: 1.1em;
            opacity: 0.9;
        }}

        .header .datetime {{
            margin-top: 15px;
            font-size: 0.95em;
            opacity: 0.85;
            font-style: italic;
        }}

        .content {{
            padding: 30px;
        }}

        .info-section {{
            background: #333;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            border-left: 4px solid #667eea;
        }}

        .info-section h2 {{
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.5em;
        }}

        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}

        .info-item {{
            background: #3a3a3a;
            padding: 12px;
            border-radius: 6px;
        }}

        .info-item label {{
            color: #aaa;
            font-size: 0.9em;
            display: block;
            margin-bottom: 5px;
        }}

        .info-item span {{
            color: #fff;
            font-size: 1.1em;
            font-weight: bold;
        }}

        .chart-section {{
            background: #333;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            border-left: 4px solid #48bb78;
        }}

        .chart-section h2 {{
            color: #48bb78;
            margin-bottom: 15px;
            font-size: 1.5em;
        }}

        .chart-container {{
            text-align: center;
            background: #1a1a1a;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
        }}

        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
        }}

        .analysis-section {{
            background: #333;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #f6ad55;
        }}

        .analysis-section h2 {{
            color: #f6ad55;
            margin-bottom: 15px;
            font-size: 1.5em;
        }}

        .analysis-content {{
            background: #2a2a2a;
            padding: 20px;
            border-radius: 6px;
            margin-top: 15px;
            line-height: 1.8;
        }}

        .analysis-content p {{
            margin-bottom: 15px;
        }}

        .analysis-content strong {{
            color: #f6ad55;
        }}

        .analysis-content em {{
            color: #a0a0a0;
            font-style: italic;
        }}

        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 1.8em;
            }}

            .content {{
                padding: 20px;
            }}

            .info-grid {{
                grid-template-columns: 1fr;
            }}
        }}
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
                    <div class="info-item">
                        <label>Symbol</label>
                        <span>{symbol_escaped}</span>
                    </div>
                    <div class="info-item">
                        <label>Timeframe</label>
                        <span>{timeframe_escaped}</span>
                    </div>
                    <div class="info-item">
                        <label>Ngày Phân Tích</label>
                        <span>{datetime_str_escaped}</span>
                    </div>
                </div>
            </div>

            <div class="chart-section">
                <h2>📉 Biểu Đồ Kỹ Thuật</h2>
                <div class="chart-container">
                    <img src="{image_src}" alt="Chart {symbol_escaped} {timeframe_escaped}">
                </div>
            </div>

            <div class="analysis-section">
                <h2>🤖 Phân Tích Từ Gemini AI</h2>
                <div class="analysis-content">
                    {analysis_html}
                </div>
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

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return html_path


def generate_multi_tf_html_report(
    symbol: str, timeframes_list: List[str], results: Dict, report_datetime: datetime, output_dir: str
) -> str:
    """
    Tạo HTML report cho multi-timeframe analysis với accordion layout.

    Args:
        symbol: Tên symbol (e.g., 'BTC/USDT')
        timeframes_list: List of timeframes (e.g., ['1h', '30m', '15m'])
        results: Dict từ multi-timeframe analysis với structure:
            {
                'symbol': str,
                'timeframes': {
                    '1h': {'signal': 'LONG', 'confidence': 0.7, 'analysis': '...'},
                    ...
                },
                'aggregated': {...}
            }
        report_datetime: Ngày giờ tạo báo cáo
        output_dir: Thư mục lưu HTML file

    Returns:
        Đường dẫn đến file HTML đã tạo
    """
    # Format datetime
    datetime_str = report_datetime.strftime("%d/%m/%Y %H:%M:%S")

    # Escape user-derived text for HTML
    symbol_escaped = html.escape(symbol)
    datetime_str_escaped = html.escape(datetime_str)

    # Find chart paths for each timeframe
    charts_dir = get_charts_dir()
    chart_paths = _find_chart_paths_for_timeframes(symbol, timeframes_list, str(charts_dir))

    # Get aggregated results
    aggregated = results.get("aggregated", {})
    agg_signal = aggregated.get("signal", "NONE")
    agg_confidence = aggregated.get("confidence", 0.0)
    timeframe_breakdown = aggregated.get("timeframe_breakdown", {})

    # Escape aggregated signal
    agg_signal_escaped = html.escape(str(agg_signal))

    # Signal color mapping
    def get_signal_color(signal: str) -> str:
        signal_upper = signal.upper()
        if signal_upper == "LONG":
            return "#48bb78"  # green
        elif signal_upper == "SHORT":
            return "#f56565"  # red
        else:
            return "#a0a0a0"  # gray

    # Generate timeframe accordion sections
    timeframe_sections = []
    for idx, tf in enumerate(timeframes_list):
        tf_result = results.get("timeframes", {}).get(tf, {})
        signal = tf_result.get("signal", "NONE")
        confidence = tf_result.get("confidence", 0.0)
        analysis_text = tf_result.get("analysis", "")
        chart_path = chart_paths.get(tf, "")

        # Escape user-derived text for HTML
        tf_escaped = html.escape(tf)
        signal_escaped = html.escape(str(signal))
        confidence_str = f"{confidence:.2f}"

        # Format analysis text to HTML
        analysis_html = format_text_to_html(analysis_text) if analysis_text else "<p>Không có phân tích</p>"

        # Chart image section
        # NOTE: For multi-timeframe report, images are referenced by relative file paths
        # (not embedded as base64). This is intentional to reduce file size when many
        # charts are included. If you want full self-contained HTML portability,
        # consider embedding images as base64 (see generate_html_report).
        if chart_path:
            # Sanitize chart path for relative HTML use
            chart_src = _sanitize_chart_path(chart_path, output_dir)
            chart_html = (
                f'<div class="chart-container"><img src="{chart_src}" alt="Chart {symbol_escaped} {tf_escaped}"></div>'
            )
        else:
            chart_html = '<div class="chart-placeholder"><p>⚠️ Không tìm thấy biểu đồ cho timeframe này</p></div>'

        signal_color = get_signal_color(signal)
        accordion_id = f"accordion-{idx}"

        timeframe_sections.append(f"""
            <div class="accordion-item">
                <input type="checkbox" id="{accordion_id}" class="accordion-checkbox">
                <label for="{accordion_id}" class="accordion-header">
                    <span class="timeframe-label">{tf_escaped}</span>
                    <span class="signal-badge" style="background-color: {signal_color}">
                        {signal_escaped} ({confidence_str})
                    </span>
                    <span class="accordion-toggle">▼</span>
                </label>
                <div class="accordion-content">
                    <div class="timeframe-info">
                        <div class="timeframe-summary">
                            <strong>Signal:</strong> <span style="color: {signal_color}">{signal_escaped}</span> |
                            <strong>Confidence:</strong> {confidence_str}
                        </div>
                    </div>
                    <div class="timeframe-chart">
                        {chart_html}
                    </div>
                    <div class="timeframe-analysis">
                        <h3>Phân Tích Chi Tiết</h3>
                        <div class="analysis-content">
                            {analysis_html}
                        </div>
                    </div>
                </div>
            </div>
        """)

    timeframe_sections_html = "\n".join(timeframe_sections)

    # Generate aggregated breakdown HTML
    breakdown_items = []
    for tf in timeframes_list:
        if tf in timeframe_breakdown:
            tf_breakdown = timeframe_breakdown[tf]
            tf_signal = tf_breakdown.get("signal", "NONE")
            tf_conf = tf_breakdown.get("confidence", 0.0)
            tf_weight = tf_breakdown.get("weight", 0.0)
            tf_color = get_signal_color(tf_signal)
            # Escape user-derived text
            tf_escaped = html.escape(tf)
            tf_signal_escaped = html.escape(str(tf_signal))
            tf_conf_str = f"{tf_conf:.2f}"
            tf_weight_str = f"{tf_weight:.2%}"
            breakdown_items.append(f"""
                <div class="breakdown-item">
                    <span class="breakdown-tf">{tf_escaped}</span>
                    <span class="breakdown-signal" style="color: {tf_color}">{tf_signal_escaped}</span>
                    <span class="breakdown-conf">Conf: {tf_conf_str}</span>
                    <span class="breakdown-weight">Weight: {tf_weight_str}</span>
                </div>
            """)

    breakdown_html = "\n".join(breakdown_items) if breakdown_items else "<p>Không có dữ liệu breakdown</p>"

    agg_signal_color = get_signal_color(agg_signal)

    # HTML template
    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo Cáo Phân Tích Multi-Timeframe - {symbol_escaped}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
            color: #e0e0e0;
            line-height: 1.6;
            padding: 20px;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: #2a2a2a;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            text-align: center;
            color: white;
        }}

        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }}

        .header .subtitle {{
            font-size: 1.1em;
            opacity: 0.9;
        }}

        .header .datetime {{
            margin-top: 15px;
            font-size: 0.95em;
            opacity: 0.85;
            font-style: italic;
        }}

        .header .timeframes-list {{
            margin-top: 10px;
            font-size: 0.9em;
            opacity: 0.9;
        }}

        .content {{
            padding: 30px;
        }}

        .aggregated-section {{
            background: #333;
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 30px;
            border-left: 4px solid #667eea;
        }}

        .aggregated-section h2 {{
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.8em;
        }}

        .aggregated-summary {{
            display: flex;
            align-items: center;
            gap: 20px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}

        .aggregated-signal {{
            font-size: 2em;
            font-weight: bold;
            padding: 15px 30px;
            border-radius: 8px;
            text-align: center;
            min-width: 200px;
        }}

        .aggregated-confidence {{
            font-size: 1.5em;
            color: #f6ad55;
        }}

        .breakdown-container {{
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #444;
        }}

        .breakdown-title {{
            color: #aaa;
            font-size: 1.1em;
            margin-bottom: 15px;
        }}

        .breakdown-items {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}

        .breakdown-item {{
            background: #3a3a3a;
            padding: 12px 15px;
            border-radius: 6px;
            display: flex;
            align-items: center;
            gap: 15px;
        }}

        .breakdown-tf {{
            font-weight: bold;
            min-width: 50px;
            color: #667eea;
        }}

        .breakdown-signal {{
            font-weight: bold;
            min-width: 80px;
        }}

        .breakdown-conf {{
            color: #aaa;
            min-width: 100px;
        }}

        .breakdown-weight {{
            color: #888;
            margin-left: auto;
        }}

        .accordion-container {{
            margin-top: 30px;
        }}

        .accordion-item {{
            background: #333;
            border-radius: 8px;
            margin-bottom: 15px;
            overflow: hidden;
            border: 1px solid #444;
        }}

        .accordion-checkbox {{
            display: none;
        }}

        .accordion-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 20px 25px;
            background: #3a3a3a;
            cursor: pointer;
            user-select: none;
            transition: background-color 0.3s ease;
        }}

        .accordion-header:hover {{
            background: #444;
        }}

        .timeframe-label {{
            font-size: 1.3em;
            font-weight: bold;
            color: #667eea;
            min-width: 80px;
        }}

        .signal-badge {{
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: bold;
            color: white;
            font-size: 1em;
        }}

        .accordion-toggle {{
            font-size: 0.8em;
            color: #aaa;
            transition: transform 0.3s ease;
        }}

        .accordion-checkbox:checked ~ .accordion-header .accordion-toggle {{
            transform: rotate(180deg);
        }}

        .accordion-content {{
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease;
        }}

        .accordion-checkbox:checked ~ .accordion-content {{
            max-height: 10000px;
        }}

        .timeframe-info {{
            padding: 20px 25px;
            background: #2a2a2a;
            border-bottom: 1px solid #444;
        }}

        .timeframe-summary {{
            color: #ccc;
        }}

        .timeframe-chart {{
            padding: 20px 25px;
            background: #2a2a2a;
            border-bottom: 1px solid #444;
        }}

        .chart-container {{
            text-align: center;
            background: #1a1a1a;
            padding: 15px;
            border-radius: 8px;
        }}

        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
        }}

        .chart-placeholder {{
            text-align: center;
            padding: 40px;
            color: #888;
            background: #1a1a1a;
            border-radius: 8px;
        }}

        .timeframe-analysis {{
            padding: 20px 25px;
            background: #2a2a2a;
        }}

        .timeframe-analysis h3 {{
            color: #f6ad55;
            margin-bottom: 15px;
            font-size: 1.3em;
        }}

        .analysis-content {{
            background: #1a1a1a;
            padding: 20px;
            border-radius: 6px;
            line-height: 1.8;
        }}

        .analysis-content p {{
            margin-bottom: 15px;
        }}

        .analysis-content strong {{
            color: #f6ad55;
        }}

        .analysis-content em {{
            color: #a0a0a0;
            font-style: italic;
        }}

        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 1.8em;
            }}

            .content {{
                padding: 20px;
            }}

            .aggregated-summary {{
                flex-direction: column;
                align-items: flex-start;
            }}

            .aggregated-signal {{
                font-size: 1.5em;
                min-width: auto;
                width: 100%;
            }}

            .breakdown-item {{
                flex-wrap: wrap;
            }}

            .breakdown-weight {{
                margin-left: 0;
            }}

            .accordion-header {{
                flex-wrap: wrap;
                gap: 10px;
            }}

            .timeframe-label {{
                min-width: auto;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Báo Cáo Phân Tích Multi-Timeframe - {symbol_escaped}</h1>
            <div class="subtitle">Gemini AI Chart Analysis Report</div>
            <div class="datetime">📅 Ngày xuất báo cáo: {datetime_str_escaped}</div>
            <div class="timeframes-list">Timeframes: {", ".join(html.escape(tf) for tf in timeframes_list)}</div>
        </div>

        <div class="content">
            <div class="aggregated-section">
                <h2>📈 Kết Quả Tổng Hợp</h2>
                <div class="aggregated-summary">
                    <div class="aggregated-signal" style="background-color: {agg_signal_color}">
                        {agg_signal_escaped}
                    </div>
                    <div class="aggregated-confidence">
                        Confidence: {agg_confidence:.2f}
                    </div>
                </div>
                <div class="breakdown-container">
                    <div class="breakdown-title">📊 Chi Tiết Theo Timeframe:</div>
                    <div class="breakdown-items">
                        {breakdown_html}
                    </div>
                </div>
            </div>

            <div class="accordion-container">
                <h2 style="color: #667eea; margin-bottom: 20px; font-size: 1.6em;">
                    📋 Phân Tích Chi Tiết Theo Timeframe
                </h2>
                {timeframe_sections_html}
            </div>
        </div>
    </div>
</body>
</html>"""

    # Save HTML file
    timestamp = report_datetime.strftime("%Y%m%d_%H%M%S")
    safe_symbol = symbol.replace("/", "_").replace(":", "_")
    html_filename = f"{safe_symbol}_multi_tf_{timestamp}.html"
    html_path = os.path.join(output_dir, html_filename)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return html_path


# generate_batch_html_report has been moved to modules.gemini_chart_analyzer.core.html_report
# to avoid circular dependencies. Import it from there if needed.


def parse_and_build_config(args):
    """
    Parse arguments or show interactive menu and build configuration.

    Args:
        args: Parsed arguments from parse_args(), or None if no args

    Returns:
        dict: Configuration dictionary with keys: symbol, timeframe, timeframes_list,
              indicators, prompt_type, custom_prompt, limit, chart_figsize, chart_dpi, no_cleanup
    """
    if args is None:
        # No arguments provided, use interactive menu
        config = interactive_config_menu()
        cfg = _convert_menu_to_config(config)
    else:
        # Arguments provided, use CLI mode
        cfg = _convert_args_to_config(args)

    return cfg


def init_components():
    """
    Initialize ExchangeManager and DataFetcher.

    Returns:
        tuple: (exchange_manager, data_fetcher)
    """
    log_info("Initializing ExchangeManager and DataFetcher...")
    exchange_manager = ExchangeManager()
    data_fetcher = DataFetcher(exchange_manager)
    return exchange_manager, data_fetcher


def run_multi_tf_analysis(
    symbol: str,
    timeframes_list: List[str],
    data_fetcher: DataFetcher,
    indicators: Dict,
    prompt_type: str,
    custom_prompt: Optional[str],
    limit: int,
    chart_figsize: tuple,
    chart_dpi: int,
    no_cleanup: bool,
):
    """
    Run multi-timeframe analysis.

    Args:
        symbol: Trading symbol
        timeframes_list: List of timeframes to analyze
        data_fetcher: DataFetcher instance
        indicators: Dictionary containing indicators configuration
        prompt_type: Prompt type for Gemini
        custom_prompt: Custom prompt (optional)
        limit: Number of candles to fetch
        chart_figsize: Chart size (width, height)
        chart_dpi: Chart DPI
        no_cleanup: Whether to skip cleaning up old charts

    Returns:
        tuple: (results, primary_timeframe)
            - results: Dictionary containing results of multi-timeframe analysis
            - primary_timeframe: Primary timeframe (first timeframe in the list)
    """
    log_info(f"Multi-timeframe analysis mode: {', '.join(timeframes_list)}")

    # Clean up old charts if needed
    if not no_cleanup:
        charts_dir = get_charts_dir()
        if os.path.exists(str(charts_dir)):
            cleanup_old_files(str(charts_dir))

    # Initialize multi-timeframe analyzer
    mtf_analyzer = MultiTimeframeCoordinator()

    # Initialize chart generator and analyzer instances once (to avoid repeated allocation)
    chart_gen = ChartGenerator(figsize=chart_figsize, style="dark_background", dpi=chart_dpi)
    gemini_analyzer = GeminiChartAnalyzer()

    # Define helper functions for multi-timeframe analysis
    def fetch_data_func(sym, tf):
        df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
            symbol=sym, timeframe=tf, limit=limit, check_freshness=False
        )
        return df

    def generate_chart_func(df, sym, tf):
        return chart_gen.create_chart(
            df=df, symbol=sym, timeframe=tf, indicators=indicators or None, show_volume=True, show_grid=True
        )

    def analyze_chart_func(chart_path, sym, tf):
        return gemini_analyzer.analyze_chart(
            image_path=chart_path, symbol=sym, timeframe=tf, prompt_type=prompt_type, custom_prompt=custom_prompt
        )

    # Run multi-timeframe analysis
    results = mtf_analyzer.analyze_deep(
        symbol=symbol,
        timeframes=timeframes_list,
        fetch_data_func=fetch_data_func,
        generate_chart_func=generate_chart_func,
        analyze_chart_func=analyze_chart_func,
    )

    # Display results
    print()
    print(color_text("=" * 60, Fore.GREEN))
    print(color_text("MULTI-TIMEFRAME ANALYSIS RESULTS", Fore.GREEN))
    print(color_text("=" * 60, Fore.GREEN))
    print()
    print(f"Symbol: {symbol}")
    print()

    # Display timeframe breakdown
    for tf in timeframes_list:
        if tf in results["timeframes"]:
            tf_result = results["timeframes"][tf]
            signal = tf_result.get("signal", "NONE")
            confidence = tf_result.get("confidence", 0.0)
            conf_bars = "█" * int(confidence * 10)
            print(f"{tf:>4}: {signal:>6} (confidence: {confidence:.2f}) {conf_bars}")

    print()
    # Display aggregated result
    aggregated = results["aggregated"]
    agg_signal = aggregated.get("signal", "NONE")
    agg_conf = aggregated.get("confidence", 0.0)
    agg_bars = "█" * int(agg_conf * 10)
    print(color_text(f"AGGREGATED: {agg_signal} (confidence: {agg_conf:.2f}) {agg_bars}", Fore.CYAN, Style.BRIGHT))
    print()
    print(color_text("=" * 60, Fore.GREEN))

    primary_timeframe = timeframes_list[0] if timeframes_list else "1h"
    return results, primary_timeframe


def run_single_tf_analysis(
    symbol: str,
    timeframe: str,
    data_fetcher: DataFetcher,
    indicators: Dict,
    prompt_type: str,
    custom_prompt: Optional[str],
    limit: int,
    chart_figsize: tuple,
    chart_dpi: int,
    no_cleanup: bool,
):
    """
    Run single timeframe analysis.

    Args:
        symbol: Trading symbol
        timeframe: Timeframe to analyze
        data_fetcher: DataFetcher instance
        indicators: Dictionary containing indicators config
        prompt_type: Prompt type for Gemini
        custom_prompt: Custom prompt (optional)
        limit: Number of candles to fetch
        chart_figsize: Chart size (width, height)
        chart_dpi: Chart DPI
        no_cleanup: Whether to skip cleaning up old charts

    Returns:
        tuple: (analysis_result, chart_path, primary_timeframe)
            - analysis_result: Analysis result from Gemini
            - chart_path: Path to the generated chart file
            - primary_timeframe: Timeframe used
    """
    # Fetch OHLCV data
    log_info(f"Fetching OHLCV data for {symbol} ({timeframe})...")
    df, exchange_id = data_fetcher.fetch_ohlcv_with_fallback_exchange(
        symbol=symbol, timeframe=timeframe, limit=limit, check_freshness=False
    )

    if df is None or df.empty:
        log_error("Failed to fetch OHLCV data. Please check the symbol and timeframe.")
        return None, None, None

    log_success(f"Fetched {len(df)} candles from {exchange_id}")

    # Clean up old charts before generating a new one (unless disabled)
    if not no_cleanup:
        charts_dir = get_charts_dir()
        if os.path.exists(str(charts_dir)):
            cleanup_old_files(str(charts_dir))

    # Generate chart
    log_info("Generating chart...")
    chart_generator = ChartGenerator(figsize=chart_figsize, style="dark_background", dpi=chart_dpi)

    chart_path = chart_generator.create_chart(
        df=df, symbol=symbol, timeframe=timeframe, indicators=indicators or None, show_volume=True, show_grid=True
    )

    log_success(f"Chart generated: {chart_path}")

    # Analyze with Gemini
    log_info("Initializing Gemini Analyzer...")
    gemini_analyzer = GeminiChartAnalyzer()

    log_info("Sending image to Google Gemini for analysis...")
    analysis_result = gemini_analyzer.analyze_chart(
        image_path=chart_path, symbol=symbol, timeframe=timeframe, prompt_type=prompt_type, custom_prompt=custom_prompt
    )

    # Display result
    print()
    print(color_text("=" * 60, Fore.GREEN))
    print(color_text("GEMINI ANALYSIS RESULT", Fore.GREEN))
    print(color_text("=" * 60, Fore.GREEN))
    print()
    print(analysis_result)
    print()
    print(color_text("=" * 60, Fore.GREEN))

    return analysis_result, chart_path, timeframe


def save_and_open_reports(
    symbol: str,
    primary_timeframe: str,
    results_or_analysis,
    chart_path: Optional[str],
    output_dir: str,
    report_datetime: datetime,
    is_multi_tf: bool,
    timeframes_list: Optional[List[str]],
    prompt_type: str,
    no_cleanup: bool,
):
    """
    Save analysis results to files and open HTML report in the browser.

    Args:
        symbol: Trading symbol
        primary_timeframe: Primary timeframe
        results_or_analysis: Analysis result (dict for multi-tf, str for single-tf)
        chart_path: Path to chart (None for multi-tf)
        output_dir: Output directory
        report_datetime: Report creation datetime
        is_multi_tf: True if multi-timeframe analysis
        timeframes_list: List of timeframes (used for multi-tf only)
        prompt_type: Prompt type used
        no_cleanup: Whether to skip cleaning up old results
    """
    # Clean up old results before saving new ones (unless disabled)
    os.makedirs(output_dir, exist_ok=True)
    if not no_cleanup:
        cleanup_old_files(output_dir)

    timestamp = report_datetime.strftime("%Y%m%d_%H%M%S")
    safe_symbol = symbol.replace("/", "_").replace(":", "_")

    if is_multi_tf:
        # Multi-timeframe: Save JSON and summary text
        results = results_or_analysis

        # Save JSON with full results
        json_file = os.path.join(output_dir, f"{safe_symbol}_multi_tf_{timestamp}.json")
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "symbol": symbol,
                    "timestamp": report_datetime.isoformat(),
                    "timeframes_list": timeframes_list,
                    "timeframes": results["timeframes"],
                    "aggregated": results["aggregated"],
                    "prompt_type": prompt_type,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        log_success(f"Saved JSON results: {json_file}")

        # Save summary text file
        result_file = os.path.join(output_dir, f"{safe_symbol}_multi_tf_{timestamp}.txt")
        with open(result_file, "w", encoding="utf-8") as f:
            f.write(f"Symbol: {symbol}\n")
            f.write(f"Timeframes: {', '.join(timeframes_list)}\n")
            f.write(f"Prompt Type: {prompt_type}\n")
            f.write(f"\n{'=' * 60}\n")
            f.write("MULTI-TIMEFRAME ANALYSIS RESULTS\n")
            f.write(f"{'=' * 60}\n\n")

            # Timeframe breakdown
            for tf in timeframes_list:
                if tf in results["timeframes"]:
                    tf_result = results["timeframes"][tf]
                    signal = tf_result.get("signal", "NONE")
                    confidence = tf_result.get("confidence", 0.0)
                    f.write(f"{tf}: {signal} (confidence: {confidence:.2f})\n")
                    if "analysis" in tf_result:
                        f.write(f"  Analysis: {tf_result['analysis'][:200]}...\n")
                    f.write("\n")

            # Aggregated result
            aggregated = results["aggregated"]
            agg_signal = aggregated.get("signal", "NONE")
            agg_conf = aggregated.get("confidence", 0.0)
            f.write(f"\nAGGREGATED: {agg_signal} (confidence: {agg_conf:.2f})\n")

        log_success(f"Saved summary: {result_file}")

        # Generate HTML report for multi-timeframe
        log_info("Generating HTML report for multi-timeframe...")
        html_path = generate_multi_tf_html_report(
            symbol=symbol,
            timeframes_list=timeframes_list,
            results=results,
            report_datetime=report_datetime,
            output_dir=output_dir,
        )
        log_success(f"HTML report generated: {html_path}")

        # Open browser
        try:
            html_uri = Path(html_path).resolve().as_uri()
            webbrowser.open(html_uri)
            log_success("Opened HTML report in browser")
        except Exception as e:
            log_warn(f"Could not open browser automatically: {e}")

    else:
        # Single timeframe: Original logic
        analysis_result = results_or_analysis

        # Save .txt file (backward compatibility)
        result_file = os.path.join(output_dir, f"{safe_symbol}_{primary_timeframe}_{timestamp}.txt")
        with open(result_file, "w", encoding="utf-8") as f:
            f.write(f"Symbol: {symbol}\n")
            f.write(f"Timeframe: {primary_timeframe}\n")
            f.write(f"Chart Path: {chart_path}\n")
            f.write(f"Prompt Type: {prompt_type}\n")
            f.write(f"\n{'=' * 60}\n")
            f.write("ANALYSIS RESULTS\n")
            f.write(f"{'=' * 60}\n\n")
            f.write(analysis_result if isinstance(analysis_result, str) else str(analysis_result))

        log_success(f"Analysis result saved: {result_file}")

        # Generate HTML report and open browser
        log_info("Generating HTML report...")
        html_path = generate_html_report(
            symbol=symbol,
            timeframe=primary_timeframe,
            chart_path=chart_path,
            analysis_result=analysis_result if isinstance(analysis_result, str) else str(analysis_result),
            report_datetime=report_datetime,
            output_dir=output_dir,
        )

        log_success(f"HTML report generated: {html_path}")

        # Open HTML in browser
        try:
            html_uri = Path(html_path).resolve().as_uri()
            webbrowser.open(html_uri)
            log_success("Opened HTML report in browser")
        except Exception as e:
            log_warn(f"Could not open browser automatically: {e}")
            log_info(f"Please open the file manually: {html_path}")


def main():
    """Main function for Gemini Chart Analyzer."""
    print()
    print(color_text("=" * 60, Fore.CYAN))
    print(color_text("GEMINI CHART ANALYZER", Fore.CYAN))
    print(color_text("=" * 60, Fore.CYAN))

    try:
        # 1. Parse command line arguments or show interactive menu and build config
        args = parse_args()
        cfg = parse_and_build_config(args)

        # Extract configuration
        symbol = cfg["symbol"]
        timeframe = cfg.get("timeframe")
        timeframes_list = cfg.get("timeframes_list")
        indicators = cfg["indicators"]
        prompt_type = cfg["prompt_type"]
        custom_prompt = cfg.get("custom_prompt")
        limit = cfg.get("limit", 500)
        chart_figsize = cfg.get("chart_figsize", (16, 10))
        chart_dpi = cfg.get("chart_dpi", 150)
        no_cleanup = cfg.get("no_cleanup", False)

        # Validate symbol
        if not symbol:
            log_error("Symbol is required. Please provide --symbol or use interactive menu.")
            return

        # Determine if multi-timeframe mode
        is_multi_tf = timeframes_list is not None and len(timeframes_list) > 0
        if not is_multi_tf:
            # Single timeframe mode (backward compatible)
            if timeframe:
                timeframe = normalize_timeframe(timeframe)
            else:
                timeframe = "1h"  # Default

        # 2. Initialize components
        exchange_manager, data_fetcher = init_components()

        # 3. Multi-timeframe or single timeframe analysis
        if is_multi_tf:
            results, primary_timeframe = run_multi_tf_analysis(
                symbol=symbol,
                timeframes_list=timeframes_list,
                data_fetcher=data_fetcher,
                indicators=indicators,
                prompt_type=prompt_type,
                custom_prompt=custom_prompt,
                limit=limit,
                chart_figsize=chart_figsize,
                chart_dpi=chart_dpi,
                no_cleanup=no_cleanup,
            )
            analysis_result = results
            chart_path = None
        else:
            analysis_result, chart_path, primary_timeframe = run_single_tf_analysis(
                symbol=symbol,
                timeframe=timeframe,
                data_fetcher=data_fetcher,
                indicators=indicators,
                prompt_type=prompt_type,
                custom_prompt=custom_prompt,
                limit=limit,
                chart_figsize=chart_figsize,
                chart_dpi=chart_dpi,
                no_cleanup=no_cleanup,
            )

            if analysis_result is None:
                # Error occurred during analysis
                return

        # 4. Save results and open reports
        output_dir = str(get_analysis_results_dir())
        report_datetime = datetime.now()

        save_and_open_reports(
            symbol=symbol,
            primary_timeframe=primary_timeframe,
            results_or_analysis=analysis_result,
            chart_path=chart_path,
            output_dir=output_dir,
            report_datetime=report_datetime,
            is_multi_tf=is_multi_tf,
            timeframes_list=timeframes_list,
            prompt_type=prompt_type,
            no_cleanup=no_cleanup,
        )

    except KeyboardInterrupt:
        log_warn("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        log_error(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

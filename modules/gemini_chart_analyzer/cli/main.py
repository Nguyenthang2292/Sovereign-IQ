"""
CLI Main Program for Gemini Chart Analyzer

Workflow:
1. Nhập tên symbols và timeframe
2. Tạo ảnh biểu đồ từ thông tin nhập vào (có thể thêm indicators như MA, RSI, etc.)
3. Lưu ảnh
4. Truy cập Google Gemini với API được config trong config/config_api.py
5. Tải ảnh lên để Google Gemini phân tích hình ảnh biểu đồ (LONG/SHORT - TP/SL)
"""

import warnings
import os
import sys
import webbrowser
import base64
import fnmatch
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to sys.path to ensure modules can be imported
# This is needed when running the file directly from subdirectories
if '__file__' in globals():
    project_root = Path(__file__).parent.parent.parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

# Fix encoding issues on Windows
from modules.common.utils import configure_windows_stdio
configure_windows_stdio()

from colorama import Fore, init as colorama_init
from modules.common.utils import color_text, log_info, log_error, log_success, log_warn, normalize_timeframe
from modules.common.core.exchange_manager import ExchangeManager
from modules.common.core.data_fetcher import DataFetcher
from modules.gemini_chart_analyzer.core.chart_generator import ChartGenerator
from modules.gemini_chart_analyzer.core.gemini_analyzer import GeminiAnalyzer
from modules.gemini_chart_analyzer.cli.argument_parser import parse_args
from modules.gemini_chart_analyzer.cli.interactive_menu import interactive_config_menu

warnings.filterwarnings("ignore")
colorama_init(autoreset=True)


def _convert_args_to_config(args):
    """Convert parsed arguments to configuration format."""
    # Extract symbol and timeframe
    symbol = args.symbol
    timeframe = args.timeframe
    
    # Build indicators dict
    indicators = {}
    
    if not args.no_ma:
        if args.ma_periods_list:
            indicators['MA'] = {'periods': args.ma_periods_list}
        else:
            indicators['MA'] = {'periods': [20, 50, 200]}
    
    if not args.no_rsi:
        indicators['RSI'] = {'period': args.rsi_period}
    
    if not args.no_macd:
        indicators['MACD'] = {'fast': 12, 'slow': 26, 'signal': 9}
    
    if args.enable_bb:
        indicators['BB'] = {'period': args.bb_period, 'std': 2}
    
    # Prompt configuration
    prompt_type = args.prompt_type
    custom_prompt = getattr(args, 'custom_prompt', None)
    
    return {
        'symbol': symbol,
        'timeframe': timeframe,
        'indicators': indicators,
        'prompt_type': prompt_type,
        'custom_prompt': custom_prompt,
        'limit': args.limit,
        'chart_figsize': args.chart_figsize_tuple,
        'chart_dpi': args.chart_dpi,
        'no_cleanup': args.no_cleanup,
    }


def _convert_menu_to_config(config):
    """Convert interactive menu config to format used by main()."""
    return {
        'symbol': config.symbol,
        'timeframe': config.timeframe,
        'indicators': getattr(config, 'indicators', {}),
        'prompt_type': config.prompt_type,
        'custom_prompt': getattr(config, 'custom_prompt', None),
        'limit': getattr(config, 'limit', 500),
        'chart_figsize': getattr(config, 'chart_figsize_tuple', (16, 10)),
        'chart_dpi': getattr(config, 'chart_dpi', 150),
        'no_cleanup': getattr(config, 'no_cleanup', False),
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
    import markdown
    import re
    
    # Sử dụng markdown library để convert markdown thành HTML
    # Extensions được sử dụng:
    # - 'fenced_code': Hỗ trợ code blocks với ```
    # - 'tables': Hỗ trợ markdown tables
    html_output = markdown.markdown(
        text,
        extensions=['fenced_code', 'tables'],
        output_format='html'
    )
    
    # Nếu output rỗng hoặc chỉ có whitespace, wrap trong <p> tag
    if not html_output.strip():
        return '<p></p>'
    
    # Xử lý line breaks: convert single newlines trong paragraphs thành <br>
    # Markdown library tự động wrap paragraphs trong <p> tags
    # Nhưng single newlines trong paragraphs cần được convert thành <br>
    # Chỉ xử lý newlines bên trong <p> tags, không ảnh hưởng đến code blocks hoặc pre tags
    html_output = re.sub(
        r'(<p[^>]*>)(.*?)(</p>)',
        lambda m: m.group(1) + m.group(2).replace('\n', '<br>') + m.group(3),
        html_output,
        flags=re.DOTALL
    )
    
    return html_output


def generate_html_report(
    symbol: str,
    timeframe: str,
    chart_path: str,
    analysis_result: str,
    report_datetime: datetime,
    output_dir: str
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
    
    # Convert analysis text to HTML
    analysis_html = format_text_to_html(analysis_result)
    
    # Embed image as base64 for standalone HTML file
    try:
        with open(chart_path, 'rb') as f:
            image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            # Detect image format
            if chart_path.lower().endswith('.png'):
                image_mime = 'image/png'
            elif chart_path.lower().endswith('.jpg') or chart_path.lower().endswith('.jpeg'):
                image_mime = 'image/jpeg'
            else:
                image_mime = 'image/png'
            image_src = f"data:{image_mime};base64,{image_base64}"
    except Exception as e:
        log_warn(f"Không thể embed ảnh: {e}, sử dụng relative path")
        # Fallback to relative path
        chart_rel_path = os.path.relpath(chart_path, output_dir)
        image_src = chart_rel_path.replace('\\', '/')
    
    # HTML template
    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo Cáo Phân Tích - {symbol} {timeframe}</title>
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
            <h1>📊 Báo Cáo Phân Tích Kỹ Thuật</h1>
            <div class="subtitle">Gemini AI Chart Analysis Report</div>
            <div class="datetime">📅 Ngày xuất báo cáo: {datetime_str}</div>
        </div>
        
        <div class="content">
            <div class="info-section">
                <h2>📈 Thông Tin Giao Dịch</h2>
                <div class="info-grid">
                    <div class="info-item">
                        <label>Symbol</label>
                        <span>{symbol}</span>
                    </div>
                    <div class="info-item">
                        <label>Timeframe</label>
                        <span>{timeframe}</span>
                    </div>
                    <div class="info-item">
                        <label>Ngày Phân Tích</label>
                        <span>{datetime_str}</span>
                    </div>
                </div>
            </div>
            
            <div class="chart-section">
                <h2>📉 Biểu Đồ Kỹ Thuật</h2>
                <div class="chart-container">
                    <img src="{image_src}" alt="Chart {symbol} {timeframe}">
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
    safe_symbol = symbol.replace('/', '_').replace(':', '_')
    html_filename = f"{safe_symbol}_{timeframe}_{timestamp}.html"
    html_path = os.path.join(output_dir, html_filename)
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return html_path


def main():
    """Main function cho Gemini Chart Analyzer."""
    print()
    print(color_text("=" * 60, Fore.CYAN))
    print(color_text("GEMINI CHART ANALYZER", Fore.CYAN))
    print(color_text("=" * 60, Fore.CYAN))
    
    try:
        # 1. Parse arguments or show interactive menu
        args = parse_args()
        
        if args is None:
            # No arguments provided, use interactive menu
            config = interactive_config_menu()
            cfg = _convert_menu_to_config(config)
        else:
            # Arguments provided, use CLI mode
            cfg = _convert_args_to_config(args)
        
        # Extract configuration
        symbol = cfg['symbol']
        timeframe = normalize_timeframe(cfg['timeframe'])  # Normalize timeframe (accept both '15m' and 'm15', etc.)
        indicators = cfg['indicators']
        prompt_type = cfg['prompt_type']
        custom_prompt = cfg.get('custom_prompt')
        limit = cfg.get('limit', 500)
        chart_figsize = cfg.get('chart_figsize', (16, 10))
        chart_dpi = cfg.get('chart_dpi', 150)
        no_cleanup = cfg.get('no_cleanup', False)
        
        # Validate symbol
        if not symbol:
            log_error("Symbol is required. Please provide --symbol or use interactive menu.")
            return
        
        # 2. Khởi tạo components
        log_info("Đang khởi tạo ExchangeManager và DataFetcher...")
        exchange_manager = ExchangeManager()
        data_fetcher = DataFetcher(exchange_manager)
        
        # 3. Fetch dữ liệu OHLCV
        log_info(f"Đang lấy dữ liệu OHLCV cho {symbol} ({timeframe})...")
        df, exchange_id = data_fetcher.fetch_ohlcv_with_fallback_exchange(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
            check_freshness=False
        )
        
        if df is None or df.empty:
            log_error("Không thể lấy dữ liệu OHLCV. Vui lòng kiểm tra lại symbol và timeframe.")
            return
        
        log_success(f"Đã lấy {len(df)} nến từ {exchange_id}")
        
        # 4. Cleanup charts cũ trước khi tạo biểu đồ mới (nếu không bị disable)
        if not no_cleanup:
            # Get charts directory from module
            from modules.gemini_chart_analyzer.core.chart_generator import _get_charts_dir
            charts_dir = _get_charts_dir()
            if os.path.exists(charts_dir):
                cleanup_old_files(charts_dir)
        
        # 5. Tạo biểu đồ
        log_info("Đang tạo biểu đồ...")
        chart_generator = ChartGenerator(figsize=chart_figsize, style='dark_background', dpi=chart_dpi)
        
        chart_path = chart_generator.create_chart(
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            indicators=indicators or None,  # or just: indicators
            show_volume=True,
            show_grid=True
        )
        
        log_success(f"Đã tạo biểu đồ: {chart_path}")
        
        # 6. Phân tích bằng Gemini
        log_info("Đang khởi tạo Gemini Analyzer...")
        gemini_analyzer = GeminiAnalyzer()
        
        log_info("Đang gửi ảnh lên Google Gemini để phân tích...")
        analysis_result = gemini_analyzer.analyze_chart(
            image_path=chart_path,
            symbol=symbol,
            timeframe=timeframe,
            prompt_type=prompt_type,
            custom_prompt=custom_prompt
        )
        
        # 7. Hiển thị kết quả
        print()
        print(color_text("=" * 60, Fore.GREEN))
        print(color_text("KẾT QUẢ PHÂN TÍCH TỪ GEMINI", Fore.GREEN))
        print(color_text("=" * 60, Fore.GREEN))
        print()
        print(analysis_result)
        print()
        print(color_text("=" * 60, Fore.GREEN))
        
        # 8. Cleanup results cũ trước khi lưu kết quả mới (nếu không bị disable)
        # Get analysis results directory from module
        from modules.gemini_chart_analyzer.core.chart_generator import _get_analysis_results_dir
        output_dir = _get_analysis_results_dir()
        os.makedirs(output_dir, exist_ok=True)
        if not no_cleanup:
            cleanup_old_files(output_dir)
        
        # 9. Lưu kết quả vào file
        
        report_datetime = datetime.now()
        timestamp = report_datetime.strftime("%Y%m%d_%H%M%S")
        safe_symbol = symbol.replace('/', '_').replace(':', '_')
        
        # Lưu file .txt (backward compatibility)
        result_file = os.path.join(output_dir, f"{safe_symbol}_{timeframe}_{timestamp}.txt")
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"Symbol: {symbol}\n")
            f.write(f"Timeframe: {timeframe}\n")
            f.write(f"Chart Path: {chart_path}\n")
            f.write(f"Prompt Type: {prompt_type}\n")
            f.write(f"\n{'='*60}\n")
            f.write("KẾT QUẢ PHÂN TÍCH\n")
            f.write(f"{'='*60}\n\n")
            f.write(analysis_result)
        
        log_success(f"Đã lưu kết quả phân tích: {result_file}")
        
        # 10. Tạo HTML report và mở browser
        log_info("Đang tạo HTML report...")
        html_path = generate_html_report(
            symbol=symbol,
            timeframe=timeframe,
            chart_path=chart_path,
            analysis_result=analysis_result,
            report_datetime=report_datetime,
            output_dir=output_dir
        )
        
        log_success(f"Đã tạo HTML report: {html_path}")
        
        # Mở HTML trên browser
        try:
            html_uri = Path(html_path).resolve().as_uri()
            webbrowser.open(html_uri)
            log_success("Đã mở HTML report trên browser")
        except Exception as e:
            log_warn(f"Không thể mở browser tự động: {e}")
            log_info(f"Vui lòng mở file thủ công: {html_path}")
        
    except KeyboardInterrupt:
        log_warn("\nĐã hủy bởi người dùng")
        sys.exit(0)
    except Exception as e:
        log_error(f"Lỗi: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


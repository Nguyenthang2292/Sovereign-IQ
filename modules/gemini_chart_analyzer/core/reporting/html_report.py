"""
HTML Report Generation for Batch Scan Results.

This module provides functionality to generate HTML reports for batch scan results,
avoiding circular dependencies by being in the core module.
"""

import os
import json
from datetime import datetime
from html import escape as html_escape
from pathlib import Path
from typing import Dict, List, Tuple

from modules.gemini_chart_analyzer.core.utils import find_project_root

def generate_batch_html_report(
    results_data: Dict,
    output_dir: str
) -> str:
    """
    Tạo HTML report cho batch scan results với accordion layout và sortable tables.
    
    Args:
        results_data: Dict từ batch scan với structure:
            {
                'timestamp': str,
                'timeframes': List[str],
                'summary': {
                    'total_symbols': int,
                    'scanned_symbols': int,
                    'long_count': int,
                    'short_count': int,
                    'none_count': int,
                    'long_percentage': float,
                    'short_percentage': float,
                    'avg_long_confidence': float,
                    'avg_short_confidence': float
                },
                'long_symbols_with_confidence': List[Tuple[str, float]],
                'short_symbols_with_confidence': List[Tuple[str, float]],
                'all_results': Dict[str, Dict]
            }
        output_dir: Thư mục lưu HTML file
        
    Returns:
        Đường dẫn đến file HTML đã tạo
    """
    # Parse data
    timestamp_str = results_data.get('timestamp', datetime.now().isoformat())
    try:
        report_datetime = datetime.fromisoformat(timestamp_str)
    except (ValueError, TypeError):
        report_datetime = datetime.now()    
    
    datetime_str = report_datetime.strftime("%d/%m/%Y %H:%M:%S")
    timeframes = results_data.get('timeframes', [])
    summary = results_data.get('summary', {})
    long_with_conf = results_data.get('long_symbols_with_confidence', [])
    short_with_conf = results_data.get('short_symbols_with_confidence', [])
    all_results = results_data.get('all_results', {})
    
    # Signal color mapping
    def get_signal_color(signal: str) -> str:
        signal_upper = signal.upper()
        if signal_upper == 'LONG':
            return '#48bb78'  # green
        elif signal_upper == 'SHORT':
            return '#f56565'  # red
        else:
            return '#a0a0a0'  # gray
    
    # Extract NONE symbols
    none_with_conf = []
    for symbol, result in all_results.items():
        if isinstance(result, dict):
            signal = result.get('signal', 'NONE')
            if signal.upper() == 'NONE':
                confidence = result.get('confidence', 0.0)
                none_with_conf.append((symbol, confidence))
    
    # Sort by confidence (descending)
    none_with_conf.sort(key=lambda x: x[1], reverse=True)
    
    # Summary statistics
    total_symbols = summary.get('total_symbols', 0)
    scanned_symbols = summary.get('scanned_symbols', 0)
    long_count = summary.get('long_count', 0)
    short_count = summary.get('short_count', 0)
    none_count = summary.get('none_count', 0)
    long_percentage = summary.get('long_percentage', 0.0)
    short_percentage = summary.get('short_percentage', 0.0)
    avg_long_conf = summary.get('avg_long_confidence', 0.0)
    avg_short_conf = summary.get('avg_short_confidence', 0.0)
    
    # Calculate none_percentage explicitly from counts to avoid rounding errors
    total_count = scanned_symbols if scanned_symbols > 0 else (long_count + short_count + none_count)
    if total_count > 0:
        none_percentage = (none_count / total_count) * 100.0
        none_percentage = max(0.0, none_percentage)  # Clamp to ensure non-negative
    else:
        none_percentage = 0.0
    
    # Generate symbol table rows
    def generate_symbol_rows(symbols_with_conf: List, signal_type: str) -> str:
        rows = []
        is_multi_tf = len(timeframes) > 1 if timeframes else False
        for symbol, confidence in symbols_with_conf:
            result = all_results.get(symbol, {})
            timeframe_breakdown = result.get('timeframe_breakdown', {}) if isinstance(result, dict) else {}
            
            # Generate timeframe breakdown badges
            breakdown_badges = []
            if timeframe_breakdown:
                for tf, tf_data in timeframe_breakdown.items():
                    if isinstance(tf_data, dict):
                        tf_signal = tf_data.get('signal', 'NONE')
                        tf_conf = tf_data.get('confidence', 0.0)
                        tf_color = get_signal_color(tf_signal)
                        breakdown_badges.append(
                            f'<span class="tf-badge" style="background-color: {tf_color}20; color: {tf_color}; border: 1px solid {tf_color}">'
                            f'{html_escape(str(tf))}: {html_escape(str(tf_signal))} ({tf_conf:.2f})</span>'
                        )
            
            breakdown_html = ' '.join(breakdown_badges) if breakdown_badges else '<span class="no-breakdown">N/A</span>'
            signal_color = get_signal_color(signal_type)
            
            # Escape symbol for use in JavaScript using json.dumps for proper escaping
            symbol_json = json.dumps(symbol)
            timeframes_json = json.dumps(timeframes) if timeframes else '[]'
            primary_tf = timeframes[0] if timeframes else results_data.get('timeframe', '1h')
            
            # Escape values for safe HTML insertion
            symbol_escaped = html_escape(str(symbol))
            confidence_escaped = html_escape(str(confidence))
            signal_type_escaped = html_escape(str(signal_type))
            primary_tf_escaped = html_escape(str(primary_tf))
            
            # Clamp confidence to 0-1 range for bar width calculation
            # while keeping displayed text unchanged
            width_pct = max(0, min(confidence, 1)) * 100
            
            rows.append(f"""
                <tr data-symbol="{symbol_escaped}" data-confidence="{confidence_escaped}">
                    <td class="symbol-cell">{symbol_escaped}</td>
                    <td class="signal-cell">
                        <span class="signal-badge" style="background-color: {signal_color}">
                            {signal_type_escaped}
                        </span>
                    </td>
                    <td class="confidence-cell">
                        <div class="confidence-bar-container">
                            <span class="confidence-value">{confidence:.2f}</span>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: {width_pct}%; background-color: {signal_color}"></div>
                            </div>
                        </div>
                    </td>
                    <td class="breakdown-cell">{breakdown_html}</td>
                    <td class="action-cell">
                        <button class="detail-btn" onclick='showDetailModal({symbol_json}, {timeframes_json}, {str(is_multi_tf).lower()}, {json.dumps(primary_tf)})'>
                            Xem chi tiết
                        </button>
                    </td>
                </tr>
            """)
        return '\n'.join(rows)
    
    long_rows = generate_symbol_rows(long_with_conf, 'LONG')
    short_rows = generate_symbol_rows(short_with_conf, 'SHORT')
    none_rows = generate_symbol_rows(none_with_conf, 'NONE')
    
    timeframes_str = ', '.join(timeframes) if timeframes else 'N/A'
    
    # Escape values for safe HTML insertion
    datetime_str_escaped = html_escape(datetime_str)
    timeframes_str_escaped = html_escape(timeframes_str)
    
    # Get absolute path to main_gemini_chart_analyzer.py
    # HTML is in analysis_results/batch_scan/, main script is at project root
    # Calculate absolute path using robust project root discovery
    html_dir = Path(output_dir).resolve()
    project_root = find_project_root(html_dir)
    main_script_path = project_root / 'main_gemini_chart_analyzer.py'
    main_script_absolute = str(main_script_path.resolve())
    
    # HTML template
    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Batch Scan Report - {datetime_str_escaped}</title>
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
            max-width: 1600px;
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
        
        .summary-section {{
            background: #333;
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 30px;
            border-left: 4px solid #667eea;
        }}
        
        .summary-section h2 {{
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.8em;
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .summary-card {{
            background: #3a3a3a;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        
        .summary-card .label {{
            color: #aaa;
            font-size: 0.9em;
            margin-bottom: 10px;
        }}
        
        .summary-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #fff;
        }}
        
        .summary-card.long .value {{
            color: #48bb78;
        }}
        
        .summary-card.short .value {{
            color: #f56565;
        }}
        
        .summary-card.none .value {{
            color: #a0a0a0;
        }}
        
        .summary-card .subvalue {{
            font-size: 0.9em;
            color: #888;
            margin-top: 5px;
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
        
        .accordion-title {{
            display: flex;
            align-items: center;
            gap: 15px;
            font-size: 1.3em;
            font-weight: bold;
        }}
        
        .signal-type-badge {{
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: bold;
            color: white;
            font-size: 0.9em;
        }}
        
        .count-badge {{
            background: #667eea;
            padding: 5px 12px;
            border-radius: 12px;
            font-size: 0.9em;
            font-weight: bold;
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
        
        .table-container {{
            padding: 25px;
            background: #2a2a2a;
            overflow-x: auto;
        }}
        
        .symbols-table {{
            width: 100%;
            border-collapse: collapse;
            background: #333;
        }}
        
        .symbols-table th {{
            background: #3a3a3a;
            padding: 15px;
            text-align: left;
            font-weight: bold;
            color: #667eea;
            cursor: pointer;
            user-select: none;
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        
        .symbols-table th:hover {{
            background: #444;
        }}
        
        .symbols-table th.sortable::after {{
            content: ' ↕';
            opacity: 0.5;
            font-size: 0.8em;
        }}
        
        .symbols-table th.sort-asc::after {{
            content: ' ↑';
            opacity: 1;
        }}
        
        .symbols-table th.sort-desc::after {{
            content: ' ↓';
            opacity: 1;
        }}
        
        .symbols-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #444;
        }}
        
        .symbols-table tr:hover {{
            background: #3a3a3a;
        }}
        
        .symbol-cell {{
            font-weight: bold;
            color: #fff;
        }}
        
        .signal-badge {{
            padding: 5px 12px;
            border-radius: 6px;
            font-weight: bold;
            color: white;
            font-size: 0.9em;
            display: inline-block;
        }}
        
        .confidence-cell {{
            min-width: 150px;
        }}
        
        .confidence-bar-container {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .confidence-value {{
            min-width: 45px;
            font-weight: bold;
            color: #f6ad55;
        }}
        
        .confidence-bar {{
            flex: 1;
            height: 20px;
            background: #1a1a1a;
            border-radius: 10px;
            overflow: hidden;
        }}
        
        .confidence-fill {{
            height: 100%;
            border-radius: 10px;
            transition: width 0.3s ease;
        }}
        
        .breakdown-cell {{
            min-width: 300px;
        }}
        
        .tf-badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            margin-right: 5px;
            margin-bottom: 5px;
        }}
        
        .no-breakdown {{
            color: #888;
            font-style: italic;
        }}
        
        .action-cell {{
            text-align: center;
        }}
        
        .detail-btn {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: bold;
            font-size: 0.9em;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        
        .detail-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }}
        
        .detail-btn:active {{
            transform: translateY(0);
        }}
        
        /* Modal styles */
        .modal-overlay {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }}
        
        .modal-overlay.show {{
            display: flex;
        }}
        
        .modal-content {{
            background: #2a2a2a;
            border-radius: 12px;
            padding: 30px;
            max-width: 700px;
            width: 90%;
            max-height: 90vh;
            overflow-y: auto;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
            position: relative;
        }}
        
        .modal-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #444;
        }}
        
        .modal-header h2 {{
            color: #667eea;
            font-size: 1.8em;
            margin: 0;
        }}
        
        .modal-close {{
            background: #f56565;
            color: white;
            border: none;
            width: 32px;
            height: 32px;
            border-radius: 50%;
            cursor: pointer;
            font-size: 1.2em;
            font-weight: bold;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.2s ease;
        }}
        
        .modal-close:hover {{
            background: #e53e3e;
        }}
        
        .modal-body {{
            margin-bottom: 20px;
        }}
        
        .modal-body p {{
            color: #ccc;
            margin-bottom: 15px;
            line-height: 1.6;
        }}
        
        .command-container {{
            background: #1a1a1a;
            border: 1px solid #444;
            border-radius: 6px;
            padding: 15px;
            margin: 15px 0;
            position: relative;
        }}
        
        .command-text {{
            color: #48bb78;
            font-family: 'Courier New', monospace;
            font-size: 0.95em;
            word-break: break-all;
            user-select: all;
            margin: 0;
        }}
        
        .command-actions {{
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }}
        
        .copy-btn {{
            background: #48bb78;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: bold;
            flex: 1;
            transition: background 0.2s ease;
        }}
        
        .copy-btn:hover {{
            background: #38a169;
        }}
        
        .copy-btn.copied {{
            background: #667eea;
        }}
        
        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 1.8em;
            }}
            
            .content {{
                padding: 20px;
            }}
            
            .summary-grid {{
                grid-template-columns: 1fr;
            }}
            
            .table-container {{
                padding: 15px;
            }}
            
            .symbols-table {{
                font-size: 0.9em;
            }}
            
            .symbols-table th,
            .symbols-table td {{
                padding: 10px;
            }}
        }}
    </style>
    <script>
        function sortTable(tableId, columnIndex, isNumeric) {{
            const table = document.getElementById(tableId);
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const header = table.querySelectorAll('th')[columnIndex];
            const isAscending = header.classList.contains('sort-asc');
            
            // Remove sort classes from all headers
            table.querySelectorAll('th').forEach(th => {{
                th.classList.remove('sort-asc', 'sort-desc', 'sortable');
            }});
            
            // Add sortable class to all headers
            table.querySelectorAll('th').forEach(th => {{
                th.classList.add('sortable');
            }});
            
            // Sort rows
            rows.sort((a, b) => {{
                let aVal, bVal;
                if (columnIndex === 0) {{
                    // Symbol column
                    aVal = a.querySelector('.symbol-cell').textContent.trim();
                    bVal = b.querySelector('.symbol-cell').textContent.trim();
                }} else if (columnIndex === 2) {{
                    // Confidence column
                    aVal = parseFloat(a.getAttribute('data-confidence'));
                    bVal = parseFloat(b.getAttribute('data-confidence'));
                    isNumeric = true;
                }} else {{
                    return 0;
                }}
                
                if (isNumeric) {{
                    return isAscending ? aVal - bVal : bVal - aVal;
                }} else {{
                    return isAscending ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
                }}
            }});
            
            // Clear and re-append sorted rows
            tbody.innerHTML = '';
            rows.forEach(row => tbody.appendChild(row));
            
            // Update header class
            header.classList.remove('sortable');
            header.classList.add(isAscending ? 'sort-desc' : 'sort-asc');
        }}
        
        // Make headers clickable
        document.addEventListener('DOMContentLoaded', function() {{
            document.querySelectorAll('.symbols-table th').forEach((header, index) => {{
                if (index === 0 || index === 2) {{
                    header.classList.add('sortable');
                    header.style.cursor = 'pointer';
                    header.addEventListener('click', () => {{
                        sortTable(header.closest('table').id, index, index === 2);
                    }});
                }}
            }});
        }});
        
        // Script absolute path (embedded from Python)
        const MAIN_SCRIPT_PATH = {json.dumps(main_script_absolute)};
        
        function getScriptPath() {{
            return MAIN_SCRIPT_PATH;
        }}
        
        // Show detail modal with command
        function showDetailModal(symbol, timeframes, isMultiTF, primaryTF) {{
            const modal = document.getElementById('detailModal');
            const symbolNameEl = document.getElementById('modalSymbolName');
            const commandTextEl = document.getElementById('modalCommandText');
            
            symbolNameEl.textContent = symbol;
            
            // Generate command
            const scriptPath = getScriptPath();
            let command;
            if (isMultiTF && timeframes && timeframes.length > 1) {{
                const timeframesStr = timeframes.join(',');
                command = `python "${{scriptPath}}" --symbol "${{symbol}}" --timeframes "${{timeframesStr}}"`;
            }} else {{
                const tf = primaryTF || (timeframes && timeframes.length > 0 ? timeframes[0] : '1h');
                command = `python "${{scriptPath}}" --symbol "${{symbol}}" --timeframe "${{tf}}"`;
            }}
            
            commandTextEl.textContent = command;
            modal.classList.add('show');
        }}
        
        // Close modal
        function closeModal() {{
            const modal = document.getElementById('detailModal');
            modal.classList.remove('show');
            const copyBtn = document.getElementById('copyCommandBtn');
            copyBtn.classList.remove('copied');
            copyBtn.textContent = 'Copy Command';
        }}
        
        // Copy command to clipboard
        function copyCommand() {{
            const commandTextEl = document.getElementById('modalCommandText');
            const command = commandTextEl.textContent;
            const copyBtn = document.getElementById('copyCommandBtn');
            
            if (navigator.clipboard && navigator.clipboard.writeText) {{
                navigator.clipboard.writeText(command).then(() => {{
                    copyBtn.classList.add('copied');
                    copyBtn.textContent = 'Copied!';
                    setTimeout(() => {{
                        copyBtn.classList.remove('copied');
                        copyBtn.textContent = 'Copy Command';
                    }}, 2000);
                }}).catch(err => {{
                    console.error('Failed to copy:', err);
                    fallbackCopyCommand(command, copyBtn);
                }});
            }} else {{
                fallbackCopyCommand(command, copyBtn);
            }}
        }}
        
        // Fallback copy method for older browsers
        function fallbackCopyCommand(text, button) {{
            const textarea = document.createElement('textarea');
            textarea.value = text;
            textarea.style.position = 'fixed';
            textarea.style.opacity = '0';
            document.body.appendChild(textarea);
            textarea.select();
            try {{
                document.execCommand('copy');
                button.classList.add('copied');
                button.textContent = 'Copied!';
                setTimeout(() => {{
                    button.classList.remove('copied');
                    button.textContent = 'Copy Command';
                }}, 2000);
            }} catch (err) {{
                console.error('Fallback copy failed:', err);
                alert('Failed to copy command. Please select and copy manually.');
            }}
            document.body.removeChild(textarea);
        }}
        
        // Close modal when clicking overlay
        document.addEventListener('DOMContentLoaded', function() {{
            const modal = document.getElementById('detailModal');
            if (modal) {{
                modal.addEventListener('click', function(e) {{
                    if (e.target === modal) {{
                        closeModal();
                    }}
                }});
            }}
        }});
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Batch Scan Report</h1>
            <div class="subtitle">Market-Wide Analysis Report</div>
            <div class="datetime">📅 Ngày xuất báo cáo: {datetime_str_escaped}</div>
            <div class="timeframes-list">Timeframes: {timeframes_str_escaped}</div>
        </div>
        
        <div class="content">
            <div class="summary-section">
                <h2>📈 Tổng Quan Thống Kê</h2>
                <div class="summary-grid">
                    <div class="summary-card">
                        <div class="label">Total Symbols</div>
                        <div class="value">{total_symbols}</div>
                        <div class="subvalue">Scanned: {scanned_symbols}</div>
                    </div>
                    <div class="summary-card long">
                        <div class="label">LONG Signals</div>
                        <div class="value">{long_count}</div>
                        <div class="subvalue">{long_percentage:.2f}% | Avg Conf: {avg_long_conf:.2f}</div>
                    </div>
                    <div class="summary-card short">
                        <div class="label">SHORT Signals</div>
                        <div class="value">{short_count}</div>
                        <div class="subvalue">{short_percentage:.2f}% | Avg Conf: {avg_short_conf:.2f}</div>
                    </div>
                    <div class="summary-card none">
                        <div class="label">NONE Signals</div>
                        <div class="value">{none_count}</div>
                        <div class="subvalue">{none_percentage:.2f}%</div>
                    </div>
                </div>
            </div>
            
            <div class="accordion-container">
                <h2 style="color: #667eea; margin-bottom: 20px; font-size: 1.8em;">📋 Chi Tiết Symbols</h2>
                
                <!-- LONG Section -->
                <div class="accordion-item">
                    <input type="checkbox" id="accordion-long" class="accordion-checkbox" checked>
                    <label for="accordion-long" class="accordion-header">
                        <div class="accordion-title">
                            <span class="signal-type-badge" style="background-color: #48bb78;">LONG</span>
                            <span class="count-badge">{long_count} symbols</span>
                        </div>
                        <span class="accordion-toggle">▼</span>
                    </label>
                    <div class="accordion-content">
                        <div class="table-container">
                            <table class="symbols-table" id="table-long">
                                <thead>
                                    <tr>
                                        <th>Symbol</th>
                                        <th>Signal</th>
                                        <th>Confidence</th>
                                        <th>Timeframe Breakdown</th>
                                        <th>Action</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {long_rows}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
                
                <!-- SHORT Section -->
                <div class="accordion-item">
                    <input type="checkbox" id="accordion-short" class="accordion-checkbox" checked>
                    <label for="accordion-short" class="accordion-header">
                        <div class="accordion-title">
                            <span class="signal-type-badge" style="background-color: #f56565;">SHORT</span>
                            <span class="count-badge">{short_count} symbols</span>
                        </div>
                        <span class="accordion-toggle">▼</span>
                    </label>
                    <div class="accordion-content">
                        <div class="table-container">
                            <table class="symbols-table" id="table-short">
                                <thead>
                                    <tr>
                                        <th>Symbol</th>
                                        <th>Signal</th>
                                        <th>Confidence</th>
                                        <th>Timeframe Breakdown</th>
                                        <th>Action</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {short_rows}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
                
                <!-- NONE Section -->
                <div class="accordion-item">
                    <input type="checkbox" id="accordion-none" class="accordion-checkbox">
                    <label for="accordion-none" class="accordion-header">
                        <div class="accordion-title">
                            <span class="signal-type-badge" style="background-color: #a0a0a0;">NONE</span>
                            <span class="count-badge">{none_count} symbols</span>
                        </div>
                        <span class="accordion-toggle">▼</span>
                    </label>
                    <div class="accordion-content">
                        <div class="table-container">
                            <table class="symbols-table" id="table-none">
                                <thead>
                                    <tr>
                                        <th>Symbol</th>
                                        <th>Signal</th>
                                        <th>Confidence</th>
                                        <th>Timeframe Breakdown</th>
                                        <th>Action</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {none_rows}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Detail Modal -->
    <div id="detailModal" class="modal-overlay">
        <div class="modal-content">
            <div class="modal-header">
                <h2>📊 Xem Chi Tiết Symbol</h2>
                <button class="modal-close" onclick="closeModal()">×</button>
            </div>
            <div class="modal-body">
                <p><strong>Symbol:</strong> <span id="modalSymbolName"></span></p>
                <p>Để xem phân tích chi tiết cho symbol này, chạy command sau trong terminal:</p>
                <div class="command-container">
                    <p class="command-text" id="modalCommandText"></p>
                </div>
                <div class="command-actions">
                    <button class="copy-btn" id="copyCommandBtn" onclick="copyCommand()">Copy Command</button>
                    <button class="modal-close" onclick="closeModal()" style="padding: 10px 20px; border-radius: 6px; cursor: pointer;">Đóng</button>
                </div>
            </div>
        </div>
    </div>
</body>
</html>"""
    
    # Save HTML file
    timestamp = report_datetime.strftime("%Y%m%d_%H%M%S")
    if timeframes:
        tf_str = "_".join(timeframes)
        html_filename = f"batch_scan_multi_tf_{tf_str}_{timestamp}.html"
    else:
        timeframe = results_data.get('timeframe', '1h')
        html_filename = f"batch_scan_{timeframe}_{timestamp}.html"
    
    html_path = os.path.join(output_dir, html_filename)
    
    # Đảm bảo thư mục output tồn tại trước khi ghi file
    os.makedirs(output_dir, exist_ok=True)
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return html_path



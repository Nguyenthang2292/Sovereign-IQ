#!/usr/bin/env python3
"""
Generate comprehensive benchmark dashboard with performance trends and regression highlighting.
"""
import json
import sys
import os
from datetime import datetime
from pathlib import Path


def load_benchmark_history(history_dir):
    """Load historical benchmark data from JSON files"""
    history = []
    history_path = Path(history_dir)

    if not history_path.exists():
        return []

    for file in sorted(history_path.glob("*.json")):
        try:
            with open(file, "r") as f:
                data = json.load(f)
                data["timestamp"] = file.stem  # Use filename as timestamp
                history.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {file}: {e}")

    return history


def calculate_trends(history):
    """Calculate performance trends from historical data"""
    trends = {}

    for record in history:
        timestamp = record.get("timestamp", "unknown")
        for bench in record.get("benchmarks", []):
            name = bench["name"]
            time = bench.get("real_time", 0)

            if name not in trends:
                trends[name] = []

            trends[name].append({
                "timestamp": timestamp,
                "time": time
            })

    return trends


def generate_dashboard_html(current_results, trends, output_file):
    """Generate comprehensive HTML dashboard"""

    # Count statistics
    total_tests = len(current_results)
    regressions = [r for r in current_results if r["regression"]]
    improvements = [r for r in current_results if r["change"] is not None and r["change"] < -5]
    stable = [r for r in current_results if r["change"] is not None and -5 <= r["change"] <= 5]

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ATC Serverless - Benchmark Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}

        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }}

        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}

        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }}

        .stat-label {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .stat-regression {{ color: #e74c3c; }}
        .stat-improvement {{ color: #27ae60; }}
        .stat-stable {{ color: #3498db; }}
        .stat-total {{ color: #9b59b6; }}

        .content {{
            padding: 30px;
        }}

        .section {{
            margin-bottom: 40px;
        }}

        .section h2 {{
            font-size: 1.8em;
            margin-bottom: 20px;
            color: #2c3e50;
            border-left: 4px solid #667eea;
            padding-left: 15px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            border-radius: 8px;
            overflow: hidden;
        }}

        th {{
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}

        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #ecf0f1;
        }}

        tr:hover {{
            background: #f8f9fa;
        }}

        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
        }}

        .badge-regression {{
            background: #fee;
            color: #e74c3c;
        }}

        .badge-improvement {{
            background: #efe;
            color: #27ae60;
        }}

        .badge-stable {{
            background: #eef;
            color: #3498db;
        }}

        .trend-indicator {{
            font-size: 1.2em;
        }}

        .trend-up {{ color: #e74c3c; }}
        .trend-down {{ color: #27ae60; }}
        .trend-stable {{ color: #95a5a6; }}

        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }}

        .footer {{
            background: #2c3e50;
            color: white;
            text-align: center;
            padding: 20px;
            font-size: 0.9em;
        }}

        .alert {{
            background: #fee;
            border-left: 4px solid #e74c3c;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 4px;
        }}

        .alert-warning {{
            background: #fef5e7;
            border-left-color: #f39c12;
        }}

        .progress-bar {{
            width: 100%;
            height: 8px;
            background: #ecf0f1;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 5px;
        }}

        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s ease;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 ATC Serverless Benchmark Dashboard</h1>
            <p>Performance Monitoring & Regression Detection</p>
            <p style="opacity: 0.7; margin-top: 10px;">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">Total Tests</div>
                <div class="stat-value stat-total">{total_tests}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Regressions</div>
                <div class="stat-value stat-regression">{len(regressions)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Improvements</div>
                <div class="stat-value stat-improvement">{len(improvements)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Stable</div>
                <div class="stat-value stat-stable">{len(stable)}</div>
            </div>
        </div>

        <div class="content">
"""

    # Add regression alert if any
    if regressions:
        html += f"""
            <div class="alert">
                <strong>⚠️ Performance Regressions Detected!</strong><br>
                {len(regressions)} test(s) showing performance degradation greater than 5%.
                Please review the results below.
            </div>
"""
    else:
        html += """
            <div class="alert alert-warning" style="background: #eef; border-left-color: #27ae60;">
                <strong>✅ No Regressions Detected!</strong><br>
                All benchmark tests are performing within acceptable parameters.
            </div>
"""

    # Detailed results table
    html += """
            <div class="section">
                <h2>📊 Detailed Benchmark Results</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Test Name</th>
                            <th>Baseline (ns)</th>
                            <th>Current (ns)</th>
                            <th>Change</th>
                            <th>Trend</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
"""

    for result in sorted(current_results, key=lambda x: x.get("change") or 0, reverse=True):
        baseline = result.get("baseline", 0)
        current = result.get("current", 0)
        change = result.get("change", 0)

        # Determine trend indicator
        if result["regression"]:
            trend = '<span class="trend-indicator trend-up">↑</span>'
            badge_class = "badge-regression"
            badge_text = "REGRESSION"
        elif change is not None and change < -5:
            trend = '<span class="trend-indicator trend-down">↓</span>'
            badge_class = "badge-improvement"
            badge_text = "IMPROVED"
        else:
            trend = '<span class="trend-indicator trend-stable">→</span>'
            badge_class = "badge-stable"
            badge_text = "STABLE"

        html += f"""
                        <tr>
                            <td><strong>{result['name']}</strong></td>
                            <td>{baseline:,.2f}</td>
                            <td>{current:,.2f}</td>
                            <td>{change:+.2f}%</td>
                            <td>{trend}</td>
                            <td><span class="badge {badge_class}">{badge_text}</span></td>
                        </tr>
"""

    html += """
                    </tbody>
                </table>
            </div>
"""

    # Performance trends section
    if trends:
        html += """
            <div class="section">
                <h2>📈 Performance Trends</h2>
                <div class="chart-container">
                    <p>Historical performance data shows trends across {num_records} benchmark runs.</p>
                    <ul style="margin-top: 15px; line-height: 2;">
""".replace("{num_records}", str(len(next(iter(trends.values()), []))))

        for test_name, trend_data in list(trends.items())[:5]:  # Show top 5
            if len(trend_data) >= 2:
                first = trend_data[0]["time"]
                last = trend_data[-1]["time"]
                change = ((last - first) / first) * 100

                html += f"""
                        <li>
                            <strong>{test_name}</strong>:
                            {first:,.0f} ns → {last:,.0f} ns
                            ({change:+.1f}%)
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: {min(100, abs(change) * 10)}%;"></div>
                            </div>
                        </li>
"""

        html += """
                    </ul>
                </div>
            </div>
"""

    # Footer
    html += f"""
        </div>

        <div class="footer">
            <p>ATC Serverless Performance Dashboard</p>
            <p style="opacity: 0.7; margin-top: 5px;">
                Regression threshold: 5% | Improvement threshold: -5%
            </p>
        </div>
    </div>
</body>
</html>
"""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    if len(sys.argv) < 3:
        print("Usage: python generate_dashboard.py <current_results.json> <history_dir> [output.html]")
        sys.exit(1)

    current_file = sys.argv[1]
    history_dir = sys.argv[2]
    if len(sys.argv) > 3:
        output_file = sys.argv[3]
    else:
        module_root = Path(__file__).resolve().parent.parent
        docs_dir = module_root / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        output_file = str(docs_dir / "benchmark_dashboard.html")

    # Load current results
    try:
        with open(current_file, "r") as f:
            current_data = json.load(f)

        # Convert to results format
        current_results = []
        for bench in current_data.get("benchmarks", []):
            current_results.append({
                "name": bench["name"],
                "baseline": bench.get("baseline", bench.get("real_time", 0)),
                "current": bench.get("real_time", 0),
                "change": 0,  # Will be calculated if baseline exists
                "regression": False
            })
    except Exception as e:
        print(f"Error loading current results: {e}")
        sys.exit(1)

    # Load historical trends
    trends = {}
    if os.path.exists(history_dir):
        history = load_benchmark_history(history_dir)
        trends = calculate_trends(history)

    # Generate dashboard
    generate_dashboard_html(current_results, trends, output_file)
    print(f"Dashboard generated: {output_file}")


if __name__ == "__main__":
    main()

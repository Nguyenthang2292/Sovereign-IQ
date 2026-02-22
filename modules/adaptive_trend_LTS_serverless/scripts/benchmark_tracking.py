#!/usr/bin/env python3
import json
import sys
from datetime import datetime
from pathlib import Path


def load_benchmarks(file_path):
    """Load benchmark data from JSON file"""
    with open(file_path, "r") as f:
        return json.load(f)


def save_benchmarks(data, file_path):
    """Save benchmark data to JSON file"""
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def compare_benchmarks(baseline, current):
    """Compare baseline and current benchmarks"""
    results = []

    for current_test in current["benchmarks"]:
        test_name = current_test["name"]

        # Find matching baseline test
        baseline_test = next((t for t in baseline["benchmarks"] if t["name"] == test_name), None)

        if baseline_test:
            current_time = current_test["real_time"]
            baseline_time = baseline_test["real_time"]

            # Calculate percentage change
            change = ((current_time - baseline_time) / baseline_time) * 100

            results.append(
                {
                    "name": test_name,
                    "baseline": baseline_time,
                    "current": current_time,
                    "change": change,
                    "regression": change > 5,  # 5% threshold for regression
                }
            )
        else:
            results.append(
                {
                    "name": test_name,
                    "baseline": None,
                    "current": current_test["real_time"],
                    "change": None,
                    "regression": False,
                }
            )

    return results


def generate_regression_report(results, output_file):
    """Generate detailed regression report"""
    regressions = [r for r in results if r["regression"]]

    if not regressions:
        report = "No performance regressions detected.\n"
    else:
        report = f"Performance Regression Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += "=" * 80 + "\n\n"
        report += f"Total Regressions Detected: {len(regressions)}\n"
        report += f"Regression Threshold: 5%\n\n"

        for reg in regressions:
            report += f"Test: {reg['name']}\n"
            report += f"  Baseline: {reg['baseline']:.2f} ns\n"
            report += f"  Current:  {reg['current']:.2f} ns\n"
            report += f"  Change:   {reg['change']:+.2f}%\n"
            report += f"  Status:   REGRESSION (>5% slower)\n"
            report += "-" * 80 + "\n"

    with open(output_file, "w") as f:
        f.write(report)

    return len(regressions)


def generate_html_report(results, output_file):
    """Generate HTML report for benchmark results"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build HTML with proper escaping for CSS braces
    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html>
<head>
    <title>Benchmark Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .regression { color: red; font-weight: bold; }
        .improvement { color: green; }
    </style>
</head>
<body>
    <h1>Benchmark Results</h1>
    <p>Generated on: """ + timestamp + """</p>
    <table>
        <tr>
            <th>Test Name</th>
            <th>Baseline (ns)</th>
            <th>Current (ns)</th>
            <th>Change (%)</th>
            <th>Status</th>
        </tr>
""")

    for result in results:
        status_class = (
            "regression"
            if result["regression"]
            else "improvement"
            if result["change"] is not None and result["change"] < 0
            else ""
        )
        status_text = (
            "REGRESSION"
            if result["regression"]
            else "IMPROVEMENT"
            if result["change"] is not None and result["change"] < 0
            else "NO CHANGE"
        )

        baseline = result.get('baseline')
        current = result.get('current')
        change = result.get('change')

        baseline_str = f"{baseline:.2f}" if baseline is not None else "N/A"
        current_str = f"{current:.2f}" if current is not None else "N/A"
        change_str = f"{change:.2f}" if change is not None else "N/A"

        html_parts.append(f"""        <tr>
            <td>{result['name']}</td>
            <td>{baseline_str}</td>
            <td>{current_str}</td>
            <td>{change_str}%</td>
            <td class="{status_class}">{status_text}</td>
        </tr>
""")

    html_parts.append("""    </table>
</body>
</html>
""")

    with open(output_file, "w") as f:
        f.write("".join(html_parts))


def main():
    if len(sys.argv) != 3:
        print("Usage: python benchmark_tracking.py <baseline_file> <current_file>")
        sys.exit(1)

    baseline_file = sys.argv[1]
    current_file = sys.argv[2]

    # Load benchmark data
    baseline = load_benchmarks(baseline_file)
    current = load_benchmarks(current_file)

    # Compare benchmarks
    results = compare_benchmarks(baseline, current)

    module_root = Path(__file__).resolve().parent.parent
    docs_dir = module_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Generate HTML report
    output_file = str(docs_dir / "benchmark_report.html")
    generate_html_report(results, output_file)
    print(f"Benchmark report generated: {output_file}")

    # Generate regression report
    regression_file = str(docs_dir / "regression_report.txt")
    num_regressions = generate_regression_report(results, regression_file)
    print(f"Regression report generated: {regression_file}")

    # Check for regressions
    if num_regressions > 0:
        print(f"WARNING: {num_regressions} performance regressions detected!")
        sys.exit(1)
    else:
        print("No performance regressions detected.")


if __name__ == "__main__":
    main()

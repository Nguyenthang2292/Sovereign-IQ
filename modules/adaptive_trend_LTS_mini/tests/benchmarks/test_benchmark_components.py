import sys
import unittest
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.adaptive_trend_LTS_mini.benchmarks.benchmark_comparison.comparison import compare_signals
from modules.adaptive_trend_LTS_mini.benchmarks.benchmark_comparison.html_formatter import ansi_to_html
from modules.adaptive_trend_LTS_mini.benchmarks.benchmark_comparison.main import TeeOutput


class TestBenchmarkComparison(unittest.TestCase):
    def test_compare_signals_perfect_match(self):
        """Test signal comparison with identical data."""
        # Create dummy data
        dates = pd.date_range("2023-01-01", periods=100, freq="h")
        signal = pd.Series(np.random.randn(100), index=dates)
        equity = pd.Series(np.random.randn(100), index=dates)

        results = {"BTC/USDT": {"Average_Signal": signal, "Average_Signal_S": equity}}

        # Compare with itself
        comparison = compare_signals(results, results, results, results, results, results, results)

        # Check metrics
        self.assertEqual(comparison["orig_rust"]["match_rate_percent"], 100.0)
        self.assertEqual(comparison["orig_approx"]["match_rate_percent"], 100.0)
        self.assertEqual(comparison["orig_rust"]["max_difference"], 0.0)

    def test_compare_signals_edge_cases(self):
        """Test comparison with edge cases: empty data, NaN values, different lengths."""
        dates = pd.date_range("2023-01-01", periods=10, freq="h")

        # Case 1: NaN values
        res_nan = {
            "BTC/USDT": {
                "Average_Signal": pd.Series([np.nan] * 10, index=dates),
                "Average_Signal_S": pd.Series([0.0] * 10, index=dates),
            }
        }
        # Ensure NaNs don't cause crashes
        comp_nan = compare_signals(res_nan, res_nan, res_nan, res_nan, res_nan, res_nan, res_nan)
        self.assertIsNotNone(comp_nan)

        # Case 2: Empty results
        res_empty = {}
        comp_empty = compare_signals(res_empty, res_empty, res_empty, res_empty, res_empty, res_empty, res_empty)
        self.assertEqual(comp_empty["total_symbols"], 0)

        # Case 3: Different lengths (handled by index intersection)
        dates_short = dates[:5]
        res_short = {
            "BTC/USDT": {
                "Average_Signal": pd.Series(np.ones(5), index=dates_short),
                "Average_Signal_S": pd.Series(np.zeros(5), index=dates_short),
            }
        }

        res_long = {
            "BTC/USDT": {
                "Average_Signal": pd.Series(np.ones(10), index=dates),
                "Average_Signal_S": pd.Series(np.zeros(10), index=dates),
            }
        }

        # Compare short vs long
        # Note: compare_signals logic expects dicts of dicts
        # It handles missing keys gracefully (logs error/returns None stats)
        # But for valid keys with different lengths, it uses intersection

        comp = compare_signals(res_long, res_short, res_short, res_short, res_short, res_short, res_short)

        # Should match perfectly on the intersection (first 5 points are 1.0)
        # Should match perfectly on the intersection (first 5 points are 1.0)
        self.assertEqual(comp["orig_rust"]["match_rate_percent"], 100.0)


class TestTeeOutput(unittest.TestCase):
    def test_write_and_flush(self):
        """Test writing to both stdout and file."""
        # Mock stdout
        captured_stdout = StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured_stdout

        # Mock file
        captured_file = StringIO()

        try:
            tee = TeeOutput(captured_file)

            test_str = "Test log entry\n"
            tee.write(test_str)
            tee.flush()

            # Check stdout
            self.assertEqual(captured_stdout.getvalue(), test_str)
            # Check file
            self.assertEqual(captured_file.getvalue(), test_str)

        finally:
            sys.stdout = original_stdout

    def test_isatty(self):
        """Test isatty delegation."""
        # Mock file
        captured_file = StringIO()
        tee = TeeOutput(captured_file)

        # Should return regular stdout isatty (usually False in tests, True in terminal)
        # Just ensure it doesn't crash
        result = tee.isatty()
        self.assertIsInstance(result, bool)


class TestHTMLFormatter(unittest.TestCase):
    def test_ansi_to_html(self):
        """Test conversion of ANSI color codes to HTML."""
        # Simple string
        simple = "Hello World"
        self.assertIn("Hello World", ansi_to_html(simple))

        # ANSI color string (Red text) - code 31 maps to #f48771 in html_formatter.py
        ansi_str = "\033[31mError\033[0m"
        html = ansi_to_html(ansi_str)

        # Check for span with color style
        self.assertIn('<span style="color: #f48771;">Error</span>', html)

        # Bold text - implementation uses <strong> tag
        bold_str = "\033[1mBold\033[0m"
        html_bold = ansi_to_html(bold_str)
        self.assertIn("<strong>Bold</strong>", html_bold)


if __name__ == "__main__":
    unittest.main()

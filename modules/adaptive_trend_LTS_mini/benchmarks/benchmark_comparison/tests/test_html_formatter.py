"""Tests for HTML formatter."""

import pytest

from modules.adaptive_trend_LTS_mini.benchmarks.benchmark_comparison.html_formatter import ansi_to_html


def test_ansi_to_html_basic():
    """Test basic ANSI to HTML conversion."""
    text = "This is a test"
    html = ansi_to_html(text)

    # Should be wrapped in HTML structure
    assert "<html>" in html
    assert "</html>" in html
    assert "test" in html


def test_ansi_to_html_with_colors():
    """Test ANSI color codes conversion."""
    # ANSI color codes
    text = "\033[32mSuccess\033[0m \033[31mError\033[0m \033[33mWarning\033[0m"
    html = ansi_to_html(text)

    # Should contain color spans
    assert "color:" in html
    # Should not contain raw ANSI codes
    assert "\033[" not in html


def test_ansi_to_html_empty():
    """Test empty string conversion."""
    text = ""
    html = ansi_to_html(text)

    assert "<html>" in html
    assert "</html>" in html


def test_ansi_to_html_multiline():
    """Test multiline text conversion."""
    text = "Line 1\nLine 2\nLine 3"
    html = ansi_to_html(text)

    assert "Line 1" in html
    assert "Line 2" in html
    assert "Line 3" in html


def test_ansi_to_html_special_chars():
    """Test special characters are escaped."""
    text = "<script>alert('test')</script> & <div>test</div>"
    html = ansi_to_html(text)

    # Special HTML characters should be escaped
    assert "<script>" not in html
    assert "&lt;script&gt;" in html or "<" not in html.replace("<html>", "").replace("</html>", "").replace(
        "<pre>", ""
    ).replace("</pre>", "").replace("<body>", "").replace("</body>", "").replace("<head>", "").replace("</head>", "")


def test_ansi_to_html_bold():
    """Test bold ANSI code."""
    text = "\033[1mBold Text\033[0m"
    html = ansi_to_html(text)

    # Should handle bold formatting
    assert "\033[1m" not in html
    assert "Bold Text" in html


def test_ansi_to_html_background_color():
    """Test background color codes."""
    text = "\033[42mGreen Background\033[0m"
    html = ansi_to_html(text)

    # Should handle background colors
    assert "\033[42m" not in html
    assert "Green Background" in html


def test_ansi_to_html_bright_colors():
    """Test bright color codes."""
    text = "\033[92mBright Green\033[0m"
    html = ansi_to_html(text)

    # Should handle bright colors
    assert "\033[92m" not in html
    assert "Bright Green" in html


def test_ansi_to_html_multiple_codes():
    """Test multiple ANSI codes on same line."""
    text = "\033[1;32;40mBold Green on Black\033[0m"
    html = ansi_to_html(text)

    # Should handle combined codes
    assert "\033[" not in html
    assert "Bold Green on Black" in html


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

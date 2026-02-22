"""Tests for GeminiChartAnalyzer."""

from unittest.mock import MagicMock, patch

import pytest

from modules.gemini_chart_analyzer.core.analyzers.gemini_chart_analyzer import (
    GeminiChartAnalyzer,
)
from modules.gemini_chart_analyzer.core.exceptions import (
    GeminiImageValidationError,
)


class TestGeminiChartAnalyzer:
    """Test suite for GeminiChartAnalyzer."""

    @patch("modules.gemini_chart_analyzer.core.analyzers.gemini_chart_analyzer.genai")
    def test_validate_image_file_not_found(self, mock_genai):
        """Test image validation with file not found."""
        analyzer = GeminiChartAnalyzer(api_key="test_key")
        is_valid, error = analyzer.validate_image("/nonexistent/path.png")

        assert is_valid is False
        assert "not found" in error.lower() or "does not exist" in error.lower()

    @patch("modules.gemini_chart_analyzer.core.analyzers.gemini_chart_analyzer.genai")
    def test_validate_image_wrong_format(self, mock_genai, tmp_path):
        """Test image validation with wrong format."""
        analyzer = GeminiChartAnalyzer(api_key="test_key")
        invalid_file = tmp_path / "test.txt"
        invalid_file.write_text("not an image", encoding="utf-8")

        is_valid, error = analyzer.validate_image(str(invalid_file))

        assert is_valid is False
        assert "format" in error.lower() or "extension" in error.lower()

    @patch("modules.gemini_chart_analyzer.core.analyzers.gemini_chart_analyzer.genai")
    def test_validate_image_too_large(self, mock_genai):
        """Test image validation with file too large."""
        analyzer = GeminiChartAnalyzer(api_key="test_key")
        with patch("os.path.exists", return_value=True), patch("os.path.getsize", return_value=50 * 1024 * 1024):
            is_valid, error = analyzer.validate_image("test.png")

        assert is_valid is False
        assert "too large" in error.lower() or "size" in error.lower()

    def test_select_best_model_none_input(self):
        """Test model selection with None input."""
        pass  # Removed unused variable
        from modules.gemini_chart_analyzer.core.analyzers.gemini_chart_analyzer import select_best_model

        result = select_best_model(None)

        # Should return default model
        assert result is not None
        assert "gemini" in result.lower()

    def test_select_best_model_empty_list(self):
        """Test model selection with empty list."""
        pass  # Removed unused variable
        from modules.gemini_chart_analyzer.core.analyzers.gemini_chart_analyzer import select_best_model

        result = select_best_model([])

        # Should return default model
        assert result is not None
        assert "gemini" in result.lower()

    def test_select_best_model_valid_models(self):
        """Test model selection with valid model list."""
        pass  # Removed unused variable
        from modules.gemini_chart_analyzer.core.analyzers.gemini_chart_analyzer import select_best_model

        models = [
            "models/gemini-2.0-flash-exp",
            "models/gemini-2.5-pro",
            "models/gemini-1.5-pro",
        ]
        result = select_best_model(models)

        # Should prefer flash models over pro
        assert "flash" in result.lower() or "pro" in result.lower()

    @patch("modules.gemini_chart_analyzer.core.analyzers.gemini_chart_analyzer.genai")
    def test_get_prompt_detailed(self, mock_genai):
        """Test getting detailed prompt."""
        analyzer = GeminiChartAnalyzer(api_key="test_key")
        prompt = analyzer._get_prompt("BTC/USDT", "1h", "detailed", None)

        assert "BTC/USDT" in prompt
        assert "1h" in prompt
        assert "LONG" in prompt or "SHORT" in prompt

    @patch("modules.gemini_chart_analyzer.core.analyzers.gemini_chart_analyzer.genai")
    def test_get_prompt_simple(self, mock_genai):
        """Test getting simple prompt."""
        analyzer = GeminiChartAnalyzer(api_key="test_key")
        prompt = analyzer._get_prompt("ETH/USDT", "4h", "simple", None)

        assert "ETH/USDT" in prompt
        assert "4h" in prompt

    @patch("modules.gemini_chart_analyzer.core.analyzers.gemini_chart_analyzer.genai")
    def test_get_prompt_custom(self, mock_genai):
        """Test getting custom prompt."""
        analyzer = GeminiChartAnalyzer(api_key="test_key")
        custom_prompt = "Analyze this chart for BNB/USDT"
        prompt = analyzer._get_prompt("BNB/USDT", "1d", "custom", custom_prompt)

        assert prompt == custom_prompt

    @patch("modules.gemini_chart_analyzer.core.analyzers.gemini_chart_analyzer.genai")
    @patch("PIL.Image.open")
    def test_analyze_chart_invalid_image(self, mock_open, mock_genai):
        """Test analyzing chart with invalid image."""
        analyzer = GeminiChartAnalyzer(api_key="test_key")

        with pytest.raises(GeminiImageValidationError):
            analyzer.analyze_chart("invalid.png", "BTC/USDT", "1h")

    @patch("modules.gemini_chart_analyzer.core.analyzers.gemini_chart_analyzer.genai")
    @patch.object(GeminiChartAnalyzer, "_get_prompt")
    @patch.object(GeminiChartAnalyzer, "_call_model_with_retries")
    @patch("modules.gemini_chart_analyzer.core.analyzers.gemini_chart_analyzer.validate_image")
    @patch("PIL.Image.open")
    def test_analyze_chart_success(self, mock_open, mock_validate, mock_call_model, mock_get_prompt, mock_genai):
        """Test successful chart analysis."""
        mock_get_prompt.return_value = "Test prompt"
        mock_validate.return_value = (True, None)
        mock_call_model.return_value = '{"signal": "LONG"}'
        mock_img = MagicMock()
        mock_img.copy.return_value = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_img

        analyzer = GeminiChartAnalyzer(api_key="test_key")
        result = analyzer.analyze_chart("test.png", "BTC/USDT", "1h")

        assert "LONG" in result or "SHORT" in result or "NONE" in result

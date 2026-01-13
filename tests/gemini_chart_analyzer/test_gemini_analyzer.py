"""
Tests for GeminiChartAnalyzer class.

Tests cover:
- Initialization with API key
- Prompt generation (detailed, simple, custom)
- Chart analysis (mocked API calls)
- Error handling
"""

from unittest.mock import Mock, patch

import pytest

from modules.gemini_chart_analyzer.core.analyzers.gemini_chart_analyzer import GeminiChartAnalyzer


@pytest.fixture
def mock_api_key():
    """Mock API key for testing."""
    return "test-api-key-12345"


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a dummy image file for testing."""
    image_path = tmp_path / "test_chart.png"
    # Create a minimal PNG file (1x1 pixel)
    # This is a valid minimal PNG file header
    png_header = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xdb\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    image_path.write_bytes(png_header)
    return str(image_path)


@pytest.fixture
def mock_genai_old_api():
    """
    Fixture that creates and configures a mock genai object for old API testing.

    Returns a tuple of (mock_genai, mock_model) where:
    - mock_genai: Mock with spec=['configure', 'GenerativeModel']
    - mock_model: Mock model returned by GenerativeModel
    """
    mock_genai = Mock(spec=["configure", "GenerativeModel"])
    mock_genai.configure = Mock()
    mock_model = Mock()
    mock_genai.GenerativeModel = Mock(return_value=mock_model)
    return (mock_genai, mock_model)


@pytest.fixture
def mock_analyzer_new_api(mock_api_key):
    """Fixture that sets up mocked genai with new API and returns analyzer, mock_client, mock_genai."""
    with patch("modules.gemini_chart_analyzer.core.analyzers.gemini_chart_analyzer.genai") as mock_genai:
        mock_client = Mock()
        mock_genai.Client = Mock(return_value=mock_client)
        mock_model = Mock()
        mock_model.name = "models/gemini-2.5-flash"
        mock_client.models.list = Mock(return_value=[mock_model])

        analyzer = GeminiChartAnalyzer(api_key=mock_api_key)

        yield (analyzer, mock_client, mock_genai)


class TestGeminiChartAnalyzerInit:
    """Test GeminiChartAnalyzer initialization."""

    @patch("modules.gemini_chart_analyzer.core.analyzers.gemini_chart_analyzer.genai")
    def test_init_with_api_key_new_api(self, mock_genai, mock_api_key):
        """Test initialization with API key using new API (Client)."""
        # Mock new API with Client
        mock_client = Mock()
        mock_genai.Client = Mock(return_value=mock_client)

        # Mock model listing
        mock_model = Mock()
        mock_model.name = "models/gemini-2.5-flash"
        mock_client.models.list = Mock(return_value=[mock_model])

        analyzer = GeminiChartAnalyzer(api_key=mock_api_key)

        assert analyzer.use_new_api is True
        assert analyzer.client == mock_client

    def test_init_with_api_key_old_api(self, mock_genai_old_api, mock_api_key, monkeypatch):
        """Test initialization with API key using old API (configure)."""
        mock_genai, mock_model = mock_genai_old_api
        # Patch the genai module with our fixture mock
        monkeypatch.setattr("modules.gemini_chart_analyzer.core.analyzers.gemini_chart_analyzer.genai", mock_genai)

        analyzer = GeminiChartAnalyzer(api_key=mock_api_key)

        assert analyzer.use_new_api is False
        assert analyzer.model == mock_model
        mock_genai.configure.assert_called_once_with(api_key=mock_api_key)

    @patch("modules.gemini_chart_analyzer.core.analyzers.gemini_chart_analyzer.genai")
    def test_init_from_config(self, mock_genai):
        """Test initialization with API key from config."""
        # Mock new API
        mock_client = Mock()
        mock_genai.Client = Mock(return_value=mock_client)

        mock_model = Mock()
        mock_model.name = "models/gemini-2.5-flash"
        mock_client.models.list = Mock(return_value=[mock_model])

        with patch("config.config_api.GEMINI_API_KEY", "config-key-123"):
            analyzer = GeminiChartAnalyzer()

            assert analyzer.use_new_api is True

    @patch("modules.gemini_chart_analyzer.core.analyzers.gemini_chart_analyzer.genai")
    def test_init_no_api_key_raises_error(self, mock_genai):
        """Test initialization without API key raises error."""
        with patch("config.config_api.GEMINI_API_KEY", None):
            with pytest.raises(ValueError, match="GEMINI_API_KEY not provided"):
                GeminiChartAnalyzer()

    @patch("modules.gemini_chart_analyzer.core.analyzers.gemini_chart_analyzer.genai")
    def test_init_empty_api_key_raises_error(self, mock_genai):
        """Test initialization with empty API key raises error."""
        with pytest.raises(ValueError, match="GEMINI_API_KEY not provided"):
            GeminiChartAnalyzer(api_key="")


class TestGeminiChartAnalyzerPrompts:
    """Test prompt generation."""

    def test_get_prompt_detailed(self, mock_analyzer_new_api):
        """Test detailed prompt generation."""
        analyzer, *_ = mock_analyzer_new_api

        prompt = analyzer._get_prompt("BTC/USDT", "1h", "detailed", None)

        assert "BTC/USDT" in prompt
        assert "1h" in prompt
        assert "Xu hướng chính" in prompt
        assert "mẫu hình" in prompt
        assert "Indicators" in prompt
        assert "LONG" in prompt
        assert "SHORT" in prompt

    def test_get_prompt_simple(self, mock_analyzer_new_api):
        """Test simple prompt generation."""
        analyzer, *_ = mock_analyzer_new_api

        prompt = analyzer._get_prompt("ETH/USDT", "4h", "simple", None)

        assert "ETH/USDT" in prompt
        assert "4h" in prompt
        assert "mô hình" in prompt or "mẫu hình" in prompt

    def test_get_prompt_custom(self, mock_analyzer_new_api):
        """Test custom prompt generation."""
        analyzer, *_ = mock_analyzer_new_api

        custom_prompt = "Analyze this chart and tell me if it's bullish or bearish"
        prompt = analyzer._get_prompt("BTC/USDT", "1h", "custom", custom_prompt)

        assert prompt == custom_prompt

    def test_get_prompt_default(self, mock_analyzer_new_api):
        """Test default prompt generation for unknown type."""
        analyzer, *_ = mock_analyzer_new_api

        prompt = analyzer._get_prompt("BTC/USDT", "1h", "unknown", None)

        assert "BTC/USDT" in prompt
        assert "1h" in prompt


class TestGeminiChartAnalyzerAnalyzeChart:
    """Test chart analysis functionality."""

    @patch("modules.gemini_chart_analyzer.core.analyzers.gemini_chart_analyzer.validate_image")
    def test_analyze_chart_new_api(self, mock_validate_image, mock_analyzer_new_api, sample_image_path):
        """Test chart analysis using new API."""
        analyzer, mock_client, mock_genai = mock_analyzer_new_api

        # Mock validation to pass
        mock_validate_image.return_value = (True, None)

        # Mock API response
        mock_response = Mock()
        mock_response.text = "This is a bullish chart with strong upward momentum."
        mock_client.models.generate_content = Mock(return_value=mock_response)
        result = analyzer.analyze_chart(
            image_path=sample_image_path, symbol="BTC/USDT", timeframe="1h", prompt_type="detailed"
        )

        assert "bullish" in result.lower()
        mock_client.models.generate_content.assert_called_once()

    @patch("modules.gemini_chart_analyzer.core.analyzers.gemini_chart_analyzer.validate_image")
    def test_analyze_chart_old_api(
        self, mock_validate_image, mock_genai_old_api, mock_api_key, sample_image_path, monkeypatch
    ):
        """Test chart analysis using old API."""
        mock_genai, mock_model = mock_genai_old_api
        # Patch the genai module with our fixture mock
        monkeypatch.setattr("modules.gemini_chart_analyzer.core.analyzers.gemini_chart_analyzer.genai", mock_genai)

        # Mock validation to pass
        mock_validate_image.return_value = (True, None)

        # Mock API response
        mock_response = Mock()
        mock_response.text = "Bearish signal detected."
        mock_model.generate_content = Mock(return_value=mock_response)

        analyzer = GeminiChartAnalyzer(api_key=mock_api_key)
        result = analyzer.analyze_chart(image_path=sample_image_path, symbol="ETH/USDT", timeframe="4h")

        assert isinstance(result, str)
        assert "bearish" in result.lower()
        mock_model.generate_content.assert_called_once()

    @patch("modules.gemini_chart_analyzer.core.analyzers.gemini_chart_analyzer.validate_image")
    def test_analyze_chart_response_with_candidates(
        self, mock_validate_image, mock_analyzer_new_api, sample_image_path
    ):
        """Test chart analysis with response containing candidates."""
        analyzer, mock_client, mock_genai = mock_analyzer_new_api

        # Mock validation to pass
        mock_validate_image.return_value = (True, None)

        # Mock response with candidates structure
        mock_part = Mock()
        mock_part.text = "Analysis result"
        mock_content = Mock()
        mock_content.parts = [mock_part]
        mock_candidate = Mock()
        mock_candidate.content = mock_content
        # Create mock without .text attribute to trigger candidates path
        mock_response = Mock(spec=["candidates"])
        mock_response.candidates = [mock_candidate]
        mock_client.models.generate_content = Mock(return_value=mock_response)
        result = analyzer.analyze_chart(image_path=sample_image_path, symbol="BTC/USDT", timeframe="1h")

        assert isinstance(result, str)
        assert "Analysis result" in result

    def test_analyze_chart_file_not_found(self, mock_analyzer_new_api):
        """Test chart analysis with non-existent file raises error."""
        from modules.gemini_chart_analyzer.core.analyzers.components.exceptions import GeminiImageValidationError

        analyzer, mock_client, mock_genai = mock_analyzer_new_api

        with pytest.raises(GeminiImageValidationError, match="Image file not found"):
            analyzer.analyze_chart(image_path="/nonexistent/path/chart.png", symbol="BTC/USDT", timeframe="1h")

    @patch("modules.gemini_chart_analyzer.core.analyzers.gemini_chart_analyzer.validate_image")
    def test_analyze_chart_api_error(self, mock_validate_image, mock_analyzer_new_api, sample_image_path):
        """Test chart analysis handles API errors."""
        analyzer, mock_client, mock_genai = mock_analyzer_new_api

        # Mock validation to pass
        mock_validate_image.return_value = (True, None)

        # Mock API error
        mock_client.models.generate_content = Mock(side_effect=Exception("API Error"))

        with pytest.raises(Exception, match="API Error"):
            analyzer.analyze_chart(image_path=sample_image_path, symbol="BTC/USDT", timeframe="1h")

"""
Tests for BatchGeminiAnalyzer class.
Tests cover:
- Initialization with cooldown
- Batch chart analysis with JSON parsing
- Cooldown mechanism
- Error handling
- JSON response parsing (various formats)
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock

from modules.gemini_chart_analyzer.core.batch_gemini_analyzer import BatchGeminiAnalyzer


@pytest.fixture
def mock_api_key():
    """Mock API key for testing."""
    return "test-api-key-12345"


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a dummy image file for testing."""
    image_path = tmp_path / "test_batch_chart.png"
    png_header = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xdb\x00\x00\x00\x00IEND\xaeB`\x82'
    image_path.write_bytes(png_header)
    return str(image_path)


@pytest.fixture
def sample_symbols():
    """Sample symbols for batch analysis."""
    return ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"]


def _setup_genai_mocks(mock_genai):
    """Helper function to set up genai client and model mocks.
    
    Args:
        mock_genai: The mocked genai module
        
    Returns:
        tuple: (mock_client, mock_model)
    """
    mock_client = Mock()
    mock_genai.Client = Mock(return_value=mock_client)
    mock_model = Mock()
    mock_model.name = 'models/gemini-2.5-flash'
    mock_client.models.list = Mock(return_value=[mock_model])
    return mock_client, mock_model


@pytest.fixture
def mock_batch_analyzer(mock_api_key):
    """Fixture that sets up mocked genai and returns analyzer."""
    with patch('modules.gemini_chart_analyzer.core.gemini_analyzer.genai') as mock_genai:
        mock_client, mock_model = _setup_genai_mocks(mock_genai)
        
        analyzer = BatchGeminiAnalyzer(api_key=mock_api_key, cooldown_seconds=0.1)
        
        yield (analyzer, mock_client, mock_genai)


class TestBatchGeminiAnalyzerInit:
    """Test BatchGeminiAnalyzer initialization."""
    
    @patch('modules.gemini_chart_analyzer.core.gemini_analyzer.genai')
    def test_init_with_cooldown(self, mock_genai, mock_api_key):
        """Test initialization with custom cooldown."""
        _setup_genai_mocks(mock_genai)
        
        analyzer = BatchGeminiAnalyzer(api_key=mock_api_key, cooldown_seconds=5.0)
        
        assert analyzer.cooldown_seconds == 5.0
        assert analyzer.last_request_time == 0.0
    
    @patch('modules.gemini_chart_analyzer.core.gemini_analyzer.genai')
    def test_init_default_cooldown(self, mock_genai, mock_api_key):
        """Test initialization with default cooldown."""
        _setup_genai_mocks(mock_genai)
        
        analyzer = BatchGeminiAnalyzer(api_key=mock_api_key)
        
        assert analyzer.cooldown_seconds == 2.5


class TestBatchGeminiAnalyzerCooldown:
    """Test cooldown mechanism."""
    
    def test_apply_cooldown_first_request(self, mock_batch_analyzer):
        """Test cooldown on first request (no wait)."""
        analyzer, _, _ = mock_batch_analyzer
        
        import time
        start_time = time.time()
        analyzer._apply_cooldown()
        elapsed = time.time() - start_time
        
        # First request should not wait (last_request_time is 0.0, so no cooldown needed)
        assert elapsed < 0.1
        # _apply_cooldown doesn't update last_request_time (it only waits if needed)
        # The timestamp is updated in analyze_batch_chart() after successful API call
        assert analyzer.last_request_time == 0.0
    
    @patch('modules.gemini_chart_analyzer.core.batch_gemini_analyzer.time.sleep')
    @patch('modules.gemini_chart_analyzer.core.batch_gemini_analyzer.time.time')
    def test_apply_cooldown_second_request(self, mock_time, mock_sleep, mock_batch_analyzer):
        """Test cooldown on second request (should wait)."""
        analyzer, _, _ = mock_batch_analyzer
        analyzer.cooldown_seconds = 0.1
        
        # First request - no wait needed (last_request_time is 0.0)
        mock_time.return_value = 100.0
        analyzer._apply_cooldown()
        # _apply_cooldown doesn't update last_request_time, so we simulate it manually
        first_time = 100.0
        analyzer.last_request_time = first_time
        
        # Second request - only 0.05s elapsed, needs to wait 0.05s more
        mock_time.return_value = 100.05  # Only 0.05s after first request
        analyzer._apply_cooldown()
        
        # Verify time.sleep was called with the expected wait time
        # wait_time = cooldown_seconds - time_since_last = 0.1 - 0.05 = 0.05
        mock_sleep.assert_called_once()
        # Check the argument with tolerance for floating point precision
        call_args = mock_sleep.call_args[0][0]
        assert abs(call_args - 0.05) < 0.001, f"Expected sleep time ~0.05, got {call_args}"


class TestBatchGeminiAnalyzerCreatePrompt:
    """Test batch prompt creation."""
    
    def test_create_batch_prompt(self, mock_batch_analyzer, sample_symbols):
        """Test batch prompt creation."""
        analyzer, _, _ = mock_batch_analyzer
        
        prompt = analyzer._create_batch_prompt(sample_symbols)
        
        assert "5" in prompt  # Number of symbols
        # Explicitly check that every symbol from sample_symbols appears in the prompt
        for symbol in sample_symbols:
            assert symbol in prompt, f"Symbol {symbol} not found in generated prompt"
        assert "LONG" in prompt
        assert "SHORT" in prompt
        assert "NONE" in prompt
        # Case-insensitive JSON format check
        prompt_lower = prompt.lower()
        assert "json" in prompt_lower, "JSON format requirement not found in prompt (case-insensitive)"
    
    def test_create_batch_prompt_many_symbols(self, mock_batch_analyzer):
        """Test batch prompt with many symbols (should truncate in prompt)."""
        analyzer, _, _ = mock_batch_analyzer
        many_symbols = [f"SYM{i}/USDT" for i in range(150)]
        
        prompt = analyzer._create_batch_prompt(many_symbols)
        
        assert "150" in prompt
        assert "..." in prompt or "total" in prompt.lower()


class TestBatchGeminiAnalyzerParseJSON:
    """Test JSON response parsing."""
    
    def test_parse_json_response_valid(self, mock_batch_analyzer, sample_symbols):
        """Test parsing valid JSON response."""
        analyzer, _, _ = mock_batch_analyzer
        
        json_response = {
            "BTC/USDT": {"signal": "LONG", "confidence": 0.85},
            "ETH/USDT": {"signal": "SHORT", "confidence": 0.70},
            "BNB/USDT": {"signal": "NONE", "confidence": 0.50},
            "SOL/USDT": {"signal": "LONG", "confidence": 0.90},
            "ADA/USDT": {"signal": "NONE", "confidence": 0.30}
        }
        response_text = json.dumps(json_response)
        
        result = analyzer._parse_json_response(response_text, sample_symbols)
        
        assert len(result) == 5
        assert result["BTC/USDT"]["signal"] == "LONG"
        assert result["BTC/USDT"]["confidence"] == 0.85
        assert result["ETH/USDT"]["signal"] == "SHORT"
        assert result["BNB/USDT"]["signal"] == "NONE"
    
    def test_parse_json_response_markdown_wrapped(self, mock_batch_analyzer, sample_symbols):
        """Test parsing JSON wrapped in markdown code blocks."""
        analyzer, _, _ = mock_batch_analyzer
        
        json_data = {
            "BTC/USDT": {"signal": "LONG", "confidence": 0.85},
            "ETH/USDT": {"signal": "SHORT", "confidence": 0.70}
        }
        response_text = f"```json\n{json.dumps(json_data)}\n```"
        
        result = analyzer._parse_json_response(response_text, sample_symbols[:2])
        
        assert len(result) == 2
        assert result["BTC/USDT"]["signal"] == "LONG"
    
    def test_parse_json_response_missing_symbols(self, mock_batch_analyzer, sample_symbols):
        """Test parsing JSON with missing symbols (should default to NONE)."""
        analyzer, _, _ = mock_batch_analyzer
        
        json_response = {
            "BTC/USDT": {"signal": "LONG", "confidence": 0.85}
            # Missing other symbols
        }
        response_text = json.dumps(json_response)
        
        result = analyzer._parse_json_response(response_text, sample_symbols)
        
        assert len(result) == 5
        assert result["BTC/USDT"]["signal"] == "LONG"
        # Missing symbols should default to NONE
        assert result["ETH/USDT"]["signal"] == "NONE"
        assert result["ETH/USDT"]["confidence"] == 0.0
    
    def test_parse_json_response_invalid_json(self, mock_batch_analyzer, sample_symbols):
        """Test parsing invalid JSON (should return all NONE)."""
        analyzer, _, _ = mock_batch_analyzer
        
        response_text = "This is not valid JSON {invalid}"
        
        result = analyzer._parse_json_response(response_text, sample_symbols)
        
        # All should be NONE with 0 confidence
        for symbol in sample_symbols:
            assert result[symbol]["signal"] == "NONE"
            assert result[symbol]["confidence"] == 0.0
    
    def test_parse_json_response_old_format_string(self, mock_batch_analyzer, sample_symbols):
        """Test parsing old format (string signals instead of dict)."""
        analyzer, _, _ = mock_batch_analyzer
        
        json_response = {
            "BTC/USDT": "LONG",
            "ETH/USDT": "SHORT",
            "BNB/USDT": "NONE"
        }
        response_text = json.dumps(json_response)
        
        result = analyzer._parse_json_response(response_text, sample_symbols[:3])
        
        assert result["BTC/USDT"]["signal"] == "LONG"
        assert result["ETH/USDT"]["signal"] == "SHORT"
        assert result["BNB/USDT"]["signal"] == "NONE"
        # Should have default confidence
        assert result["BTC/USDT"]["confidence"] > 0


class TestBatchGeminiAnalyzerAnalyzeBatchChart:
    """Test batch chart analysis."""
    
    @patch('PIL.Image.open')
    def test_analyze_batch_chart_success(self, mock_image_open, mock_batch_analyzer, 
                                         sample_image_path, sample_symbols):
        """Test successful batch chart analysis."""
        analyzer, mock_client, _ = mock_batch_analyzer
        
        # Mock image
        mock_img = Mock()
        mock_img.copy.return_value = mock_img
        mock_image_open.return_value.__enter__ = Mock(return_value=mock_img)
        mock_image_open.return_value.__exit__ = Mock(return_value=None)
        
        # Mock API response with JSON
        json_response = {
            "BTC/USDT": {"signal": "LONG", "confidence": 0.85},
            "ETH/USDT": {"signal": "SHORT", "confidence": 0.70},
            "BNB/USDT": {"signal": "NONE", "confidence": 0.50}
        }
        mock_response = Mock()
        mock_response.text = json.dumps(json_response)
        mock_client.models.generate_content = Mock(return_value=mock_response)
        
        result = analyzer.analyze_batch_chart(
            image_path=sample_image_path,
            batch_id=1,
            total_batches=5,
            symbols=sample_symbols[:3]
        )
        
        assert len(result) == 3
        assert result["BTC/USDT"]["signal"] == "LONG"
        assert result["ETH/USDT"]["signal"] == "SHORT"
        mock_client.models.generate_content.assert_called_once()
    
    @patch('PIL.Image.open')
    def test_analyze_batch_chart_api_error(self, mock_image_open, mock_batch_analyzer,
                                          sample_image_path, sample_symbols):
        """Test batch chart analysis handles API errors."""
        analyzer, mock_client, _ = mock_batch_analyzer
        
        # Mock image
        mock_img = Mock()
        mock_img.copy.return_value = mock_img
        mock_image_open.return_value.__enter__ = Mock(return_value=mock_img)
        mock_image_open.return_value.__exit__ = Mock(return_value=None)
        
        # Mock API error
        mock_client.models.generate_content = Mock(side_effect=Exception("API Error"))
        
        result = analyzer.analyze_batch_chart(
            image_path=sample_image_path,
            batch_id=1,
            total_batches=5,
            symbols=sample_symbols[:3]
        )
        
        # Should return all NONE on error
        for symbol in sample_symbols[:3]:
            assert result[symbol]["signal"] == "NONE"
            assert result[symbol]["confidence"] == 0.0
    
    def test_analyze_batch_chart_file_not_found(self, mock_batch_analyzer, sample_symbols):
        """Test batch chart analysis with non-existent file."""
        analyzer, _, _ = mock_batch_analyzer
        
        result = analyzer.analyze_batch_chart(
            image_path="/nonexistent/path/chart.png",
            batch_id=1,
            total_batches=5,
            symbols=sample_symbols[:3]
        )
        
        # Should return all NONE on file error
        for symbol in sample_symbols[:3]:
            assert result[symbol]["signal"] == "NONE"
            assert result[symbol]["confidence"] == 0.0


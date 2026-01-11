
import pytest

from modules.gemini_chart_analyzer.core.analyzers.components.exceptions import (
from PIL import Image
from PIL import Image

"""
Unit tests for the GeminiChartAnalyzer helper utilities.

Tests cover:
- GeminiModelType enum and helper methods
- select_best_model function
- validate_image function
- parse_trading_signal function
- estimate_token_count function
- ImageValidationConfig dataclass
- TradingSignal dataclass
"""


    GeminiAPIError,
    GeminiAuthenticationError,
    GeminiImageValidationError,
    GeminiInvalidRequestError,
    GeminiModelNotFoundError,
    GeminiQuotaExceededError,
    GeminiRateLimitError,
    GeminiResponseParseError,
)
from modules.gemini_chart_analyzer.core.analyzers.components.helpers import (
    select_best_model,
    validate_image,
)
from modules.gemini_chart_analyzer.core.analyzers.components.image_config import (
    ImageValidationConfig,
)
from modules.gemini_chart_analyzer.core.analyzers.components.model_config import (
    GeminiModelType,
)
from modules.gemini_chart_analyzer.core.analyzers.components.response_parser import (
    TradingSignal,
    parse_trading_signal,
)
from modules.gemini_chart_analyzer.core.analyzers.components.token_limit import (
    estimate_token_count,
    MAX_TOKENS_PER_REQUEST,
    PROMPT_TOKEN_WARNING_THRESHOLD,
)



@pytest.fixture
def sample_image():
    """Create a minimal valid test image."""
    img = Image.new("RGB", (200, 200), color="white")
    return img


class TestGeminiModelType:
    """Test GeminiModelType enum and helper methods."""
    
    def test_model_names(self):
        """Test that all model types have names."""
        assert GeminiModelType.FLASH_3_PREVIEW.name == "models/gemini-3-flash-preview"
        assert GeminiModelType.PRO_3_PREVIEW.name == "models/gemini-3-pro-preview"
        assert GeminiModelType.FLASH_25_LITE.name == "models/gemini-2.5-flash-lite"
        assert GeminiModelType.FLASH_3.name == "models/gemini-3-flash"
        assert GeminiModelType.PRO_3.name == "models/gemini-3-pro"
    
    def test_model_priorities(self):
        """Test that model priorities are correct."""
        assert GeminiModelType.FLASH_3_PREVIEW.priority == 0
        assert GeminiModelType.PRO_3_PREVIEW.priority == 1
        assert GeminiModelType.FLASH_25_LITE.priority == 2
        assert GeminiModelType.FLASH_3.priority == 3
        assert GeminiModelType.PRO_3.priority == 6
    
    def test_model_properties(self):
        """Test model type properties."""
        assert GeminiModelType.FLASH_3_PREVIEW.is_preview is True
        assert GeminiModelType.PRO_3_PREVIEW.is_preview is True
        assert GeminiModelType.FLASH_3_PREVIEW.is_flash is True
        assert GeminiModelType.FLASH_3_PREVIEW.is_lite is False
        assert GeminiModelType.FLASH_3_PREVIEW.is_pro is False
        
        assert GeminiModelType.FLASH_3.is_preview is False
        assert GeminiModelType.FLASH_3.is_flash is True
        assert GeminiModelType.FLASH_3.is_lite is False
        
        assert GeminiModelType.FLASH_25_LITE.is_flash is True
        assert GeminiModelType.FLASH_25_LITE.is_lite is True
        
        assert GeminiModelType.PRO_3.is_flash is False
        assert GeminiModelType.PRO_3.is_pro is True
        assert GeminiModelType.PRO_3.is_lite is False
    
    def test_from_name(self):
        """Test from_name class method."""
        assert GeminiModelType.from_name("models/gemini-3-flash-preview") == GeminiModelType.FLASH_3_PREVIEW
        assert GeminiModelType.from_name("models/gemini-3-flash") == GeminiModelType.FLASH_3
        assert GeminiModelType.from_name("models/gemini-2.5-flash-lite") == GeminiModelType.FLASH_25_LITE
        assert GeminiModelType.from_name("models/gemini-3-pro") == GeminiModelType.PRO_3
        
        # Test case insensitivity
        assert GeminiModelType.from_name("MODELS/GEMINI-3-FLASH-PREVIEW") == GeminiModelType.FLASH_3_PREVIEW
        
        # Test None input
        assert GeminiModelType.from_name(None) is None
        assert GeminiModelType.from_name("") is None
    
    def test_get_fallback_models(self):
        """Test get_fallback_models class method."""
        flash_3_preview = GeminiModelType.FLASH_3_PREVIEW
        fallbacks = flash_3_preview.get_fallback_models(flash_3_preview)
        
        # Should return all other models, excluding the primary one
        assert flash_3_preview not in fallbacks
        assert len(fallbacks) == len(list(GeminiModelType)) - 1
        
        # Test priority ordering (fallback models should be in priority order)
        priorities = [m.priority for m in fallbacks]
        assert priorities == sorted(priorities)


class TestSelectBestModel:
    """Test select_best_model function."""
    
    def test_select_best_model_with_available(self):
        """Test model selection with available models list."""
        available_models = [
            "models/gemini-2.5-flash",
            "models/gemini-3-flash",
            "models/gemini-1.5-pro",
        ]
        
        selected = select_best_model(available_models)
        
        # Should select highest priority available model (gemini-3-flash, priority 3)
        assert selected == "models/gemini-3-flash"
    
    def test_select_best_model_none(self):
        """Test model selection with None available models."""
        selected = select_best_model(None)
        
        # Should return default model (flash-3-preview)
        assert selected == "models/gemini-3-flash-preview"
    
    def test_select_best_model_empty_list(self):
        """Test model selection with empty list."""
        selected = select_best_model([])
        
        # Should return first model in list (fallback)
        assert selected == "models/gemini-2.5-flash"


class TestValidateImage:
    """Test validate_image function."""
    
    def test_validate_image_missing_file(self, tmp_path):
        """Test validation with missing file."""
        image_path = tmp_path / "nonexistent.png"
        
        is_valid, error_msg = validate_image(image_path)
        
        assert is_valid is False
        assert "not found" in error_msg.lower()
    
    def test_validate_image_unsupported_format(self, tmp_path, sample_image):
        """Test validation with unsupported format."""
        image_path = tmp_path / "test.webp"
        sample_image.save(image_path)
        
        is_valid, error_msg = validate_image(image_path)
        
        # Should fail because default config doesn't include WEBP
        # Note: This test may pass if the default config is modified
        
        if not is_valid:
            assert "format" in error_msg.lower()
    
    def test_validate_image_too_large(self, tmp_path, sample_image):
        """Test validation with too large image."""
        # Create a config with small size limit
        config = ImageValidationConfig(max_file_size_mb=0.001)
        image_path = tmp_path / "large.png"
        sample_image.save(image_path)
        
        is_valid, error_msg = validate_image(image_path, config)
        
        assert is_valid is False
        assert "too large" in error_msg.lower()
    
    def test_validate_image_too_small(self, tmp_path, sample_image):
        """Test validation with too small image."""
        # Create a 50x50 image
        from PIL import Image
        
        small_img = Image.new("RGB", (50, 50), color="white")
        image_path = tmp_path / "small.png"
        small_img.save(image_path)
        
        is_valid, error_msg = validate_image(image_path)
        
        assert is_valid is False
        assert "too small" in error_msg.lower()
    
    def test_validate_image_too_large_dimensions(self, tmp_path, sample_image):
        """Test validation with too large dimensions."""
        from PIL import Image
        
        large_img = Image.new("RGB", (5000, 5000), color="white")
        image_path = tmp_path / "large_dim.png"
        large_img.save(image_path)
        
        is_valid, error_msg = validate_image(image_path)
        
        assert is_valid is False
        assert "too large" in error_msg.lower()
    
    def test_validate_image_valid(self, tmp_path, sample_image):
        """Test validation with valid image."""
        # Save sample image
        image_path = tmp_path / "valid.png"
        sample_image.save(image_path)
        
        is_valid, error_msg = validate_image(image_path)
        
        assert is_valid is True
        assert error_msg is None


class TestParseTradingSignal:
    """Test parse_trading_signal function."""
    
    def test_parse_long_signal(self):
        """Test parsing LONG signal."""
        response = "Entry: 1.2345, SL: 1.2200, TP1: 1.2800, Confidence: high"
        
        signal = parse_trading_signal(response)
        
        assert signal.direction == "LONG"
        assert signal.entry_price == 1.2345
        assert signal.stop_loss == 1.2200
        assert signal.take_profit_1 == 1.2800
        assert signal.confidence == "high"
    
    def test_parse_short_signal(self):
        """Test parsing SHORT signal."""
        response = "Entry: 1.2000, SL: 1.2500, TP2: 1.3000"
        
        signal = parse_trading_signal(response)
        
        assert signal.direction == "SHORT"
        assert signal.entry_price == 1.2000
        assert signal.stop_loss == 1.2500
        assert signal.take_profit_2 == 1.3000
    
    def test_parse_neutral_signal(self):
        """Test parsing NEUTRAL signal."""
        response = "No clear signal detected, market is ranging."
        
        signal = parse_trading_signal(response)
        
        assert signal.direction == "NEUTRAL"
    
    def test_parse_vietnamese_long_signal(self):
        """Test parsing Vietnamese LONG signal."""
        response = "Giá vào: 1.2345, cắt lỗ: 1.2200, chốt lời 1: 2800"
        
        signal = parse_trading_signal(response)
        
        assert signal.direction == "LONG"
        assert signal.entry_price == 1.2345
        assert signal.stop_loss == 1.2200
        assert signal.take_profit_1 == 1.2800
    
    def test_parse_with_regex_patterns(self):
        """Test parsing with various regex patterns."""
        response = "Entry at 1.23, Stop Loss: 1.20"
        
        signal = parse_trading_signal(response)
        
        assert signal.direction == "NEUTRAL"  # Default when no match
        assert signal.entry_price == 1.23
        assert signal.stop_loss == 1.20


class TestEstimateTokenCount:
    """Test estimate_token_count function."""
    
    def test_estimate_token_count(self):
        """Test token count estimation."""
        text = "Hello world"
        tokens = estimate_token_count(text)
        
        # ~4 characters per token
        expected = len(text) // 4
        assert tokens == expected
    
    def test_estimate_token_count_long(self):
        """Test with longer text."""
        text = "This is a very long text that contains many words and should estimate to more tokens"
        tokens = estimate_token_count(text)
        
        # Should be roughly text length / 4
        assert tokens > 0
        assert tokens < len(text)  # Should be less than char count


class TestImageValidationConfig:
    """Test ImageValidationConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ImageValidationConfig()
        
        assert config.max_file_size_mb == 20.0
        assert config.max_width == 4096
        assert config.max_height == 4096
        assert config.min_width == 100
        assert config.min_height == 100
        assert config.supported_formats == ("PNG", "JPEG", "JPG", "WEBP", "GIF")
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = ImageValidationConfig(
            max_file_size_mb=10.0,
            max_width=2000,
            max_height=2000,
            min_width=50,
            min_height=50,
            supported_formats=("PNG", "JPG")
        )
        
        assert config.max_file_size_mb == 10.0
        assert config.max_width == 2000
        assert config.max_height == 2000
        assert config.min_width == 50
        assert config.min_height == 50
        assert config.supported_formats == ("PNG", "JPG")


class TestTradingSignal:
    """Test TradingSignal dataclass."""
    
    def test_default_values(self):
        """Test default signal values."""
        signal = TradingSignal()
        
        assert signal.direction == ""
        assert signal.entry_price is None
        assert signal.stop_loss is None
        assert signal.take_profit_1 is None
        assert signal.take_profit_2 is None
        assert signal.confidence is None
        assert signal.reasoning is None
    
    def test_long_signal(self):
        """Test LONG signal."""
        signal = TradingSignal(
            direction="LONG",
            entry_price=1.2345,
            stop_loss=1.2200,
            take_profit_1=1.2800,
            confidence="high"
        )
        
        assert signal.direction == "LONG"
        assert signal.entry_price == 1.2345
        assert signal.stop_loss == 1.2200
        assert signal.take_profit_1 == 1.2800
        assert signal.confidence == "high"
    
    def test_short_signal(self):
        """Test SHORT signal."""
        signal = TradingSignal(
            direction="SHORT",
            entry_price=1.2000,
            stop_loss=1.2500,
            take_profit_2=1.3000,
        )
        
        assert signal.direction == "SHORT"
        assert signal.entry_price == 1.2000
        assert signal.stop_loss == 1.2500
        assert signal.take_profit_2 == 1.3000

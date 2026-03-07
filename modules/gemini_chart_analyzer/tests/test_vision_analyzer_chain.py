"""Tests for VisionAnalyzerChain."""

from unittest.mock import MagicMock, patch

import pytest

from modules.gemini_chart_analyzer.core.analyzers.vision_analyzer_chain import (
    VisionAnalyzerChain,
    VisionChainExhaustedError,
)


@pytest.fixture
def mock_gemini():
    """Mock Gemini provider."""
    with patch("modules.gemini_chart_analyzer.core.analyzers.vision_analyzer_chain.GeminiVisionProvider") as mock:
        provider = MagicMock()
        provider.is_available.return_value = True
        provider.provider_name = "gemini"
        mock.return_value = provider
        yield provider


@pytest.fixture
def mock_qwen():
    """Mock Qwen provider."""
    with patch("modules.gemini_chart_analyzer.core.analyzers.vision_analyzer_chain.QwenVisionProvider") as mock:
        provider = MagicMock()
        provider.is_available.return_value = True
        provider.provider_name = "qwen"
        mock.return_value = provider
        yield provider


def test_chain_uses_gemini_when_available(mock_gemini, mock_qwen):
    """Test that chain uses Gemini first when both are available."""
    chain = VisionAnalyzerChain(gemini_api_key="gemini_key", qwen_api_key="qwen_key")

    mock_gemini.analyze_chart.return_value = "Gemini Response"

    result = chain.analyze_chart(
        image_path="test.png",
        symbol="BTC/USDT",
        timeframe="1h",
    )

    assert result == "Gemini Response"
    mock_gemini.analyze_chart.assert_called_once()
    mock_qwen.analyze_chart.assert_not_called()


def test_chain_falls_back_to_qwen_on_gemini_failure(mock_gemini, mock_qwen):
    """Test that chain falls back to Qwen if Gemini fails."""
    chain = VisionAnalyzerChain(gemini_api_key="gemini_key", qwen_api_key="qwen_key")

    mock_gemini.analyze_chart.side_effect = Exception("Gemini API Error")
    mock_qwen.analyze_chart.return_value = "Qwen Response"

    result = chain.analyze_chart(
        image_path="test.png",
        symbol="BTC/USDT",
        timeframe="1h",
    )

    assert result == "Qwen Response"
    mock_gemini.analyze_chart.assert_called_once()
    mock_qwen.analyze_chart.assert_called_once()


def test_chain_raises_when_all_fail(mock_gemini, mock_qwen):
    """Test that chain raises VisionChainExhaustedError when all providers fail."""
    chain = VisionAnalyzerChain(gemini_api_key="gemini_key", qwen_api_key="qwen_key")

    mock_gemini.analyze_chart.side_effect = Exception("Gemini Error")
    mock_qwen.analyze_chart.side_effect = Exception("Qwen Error")

    with pytest.raises(VisionChainExhaustedError) as exc_info:
        chain.analyze_chart(
            image_path="test.png",
            symbol="BTC/USDT",
            timeframe="1h",
        )

    assert "All vision providers exhausted" in str(exc_info.value)
    mock_gemini.analyze_chart.assert_called_once()
    mock_qwen.analyze_chart.assert_called_once()


def test_chain_skips_unavailable_provider(mock_gemini, mock_qwen):
    """Test that chain skips unavailable providers."""
    mock_qwen.is_available.return_value = False

    chain = VisionAnalyzerChain(gemini_api_key="gemini_key")

    assert len(chain._providers) == 1
    assert chain._providers[0] == mock_gemini


def test_qwen_provider_is_available_with_key():
    """Test chain availability logic."""
    with (
        patch("modules.gemini_chart_analyzer.core.analyzers.vision_analyzer_chain.QwenVisionProvider") as mock_qwen_cls,
        patch(
            "modules.gemini_chart_analyzer.core.analyzers.vision_analyzer_chain.GeminiVisionProvider"
        ) as mock_gemini_cls,
    ):
        qwen_inst = MagicMock()
        qwen_inst.is_available.return_value = True
        mock_qwen_cls.return_value = qwen_inst

        gemini_inst = MagicMock()
        gemini_inst.is_available.return_value = False
        mock_gemini_cls.return_value = gemini_inst

        chain = VisionAnalyzerChain(qwen_api_key="qwen_key")

        assert chain.is_available() is True

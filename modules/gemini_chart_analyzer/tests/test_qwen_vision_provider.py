"""Tests for QwenVisionProvider."""

from unittest.mock import MagicMock, patch

import pytest

from modules.gemini_chart_analyzer.core.analyzers.qwen_vision_provider import (
    QwenVisionProvider,
)


@pytest.fixture
def mock_openai_client():
    with patch("openai.OpenAI") as mock_cls:
        client = MagicMock()
        mock_cls.return_value = client
        yield client


@pytest.fixture
def mock_requests_get():
    with patch("requests.get") as mock_get:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "data": [{"id": "qwen-vl-plus"}, {"id": "qwen-vl-max"}, {"id": "qwen-vl-edit"}, {"id": "wan-vl"}]
        }
        mock_get.return_value = response
        yield mock_get


def test_qwen_provider_loads_models(mock_openai_client, mock_requests_get):
    """Test that Qwen provider fetches and filters models correctly."""
    provider = QwenVisionProvider(api_key="test_key")

    assert provider.is_available() is True
    assert "qwen-vl-max" in provider._models
    assert "qwen-vl-plus" in provider._models
    assert "qwen-vl-edit" not in provider._models
    assert "wan-vl" not in provider._models


def test_qwen_provider_base64_encode():
    """Test base64 encoding formatting."""
    provider = QwenVisionProvider(api_key="test_key", models=["qwen-vl-max"])

    with patch("builtins.open"), patch("base64.b64encode") as mock_b64:
        mock_b64.return_value = b"test_base64"

        result = provider._encode_image("test_image.png")

        assert result == "data:image/png;base64,test_base64"

        result = provider._encode_image("test_image.jpg")

        assert result == "data:image/jpeg;base64,test_base64"


def test_qwen_provider_success(mock_openai_client):
    """Test successful chart analysis."""
    provider = QwenVisionProvider(api_key="test_key", models=["qwen-vl-max"])

    response_msg = MagicMock()
    response_msg.message.content = '{"signal": "LONG"}'

    response = MagicMock()
    response.choices = [response_msg]
    mock_openai_client.chat.completions.create.return_value = response

    with (
        patch.object(provider, "_encode_image", return_value="data:image/png;base64,123"),
        patch.object(provider, "_get_prompt", return_value="Test prompt"),
    ):
        result = provider.analyze_chart("test.png", "BTC/USDT", "1h")

        assert result == '{"signal": "LONG"}'
        mock_openai_client.chat.completions.create.assert_called_once()
        args, kwargs = mock_openai_client.chat.completions.create.call_args
        assert kwargs["model"] == "qwen-vl-max"


def test_qwen_provider_fallback(mock_openai_client):
    """Test fallback to next model on failure."""
    provider = QwenVisionProvider(api_key="test_key", models=["qwen-vl-max", "qwen-vl-plus"])

    class MockError(Exception):
        status_code = 429

    # First model fails with 429, second succeeds
    fail_response = MagicMock()
    fail_response.side_effect = MockError("Quota Exceeded")

    success_response_msg = MagicMock()
    success_response_msg.message.content = '{"signal": "LONG"}'
    success_response = MagicMock()
    success_response.choices = [success_response_msg]

    mock_openai_client.chat.completions.create.side_effect = [MockError("Quota Exceeded"), success_response]

    with (
        patch.object(provider, "_encode_image", return_value="data:image/png;base64,123"),
        patch.object(provider, "_get_prompt", return_value="Test prompt"),
    ):
        result = provider.analyze_chart("test.png", "BTC/USDT", "1h")

        assert result == '{"signal": "LONG"}'
        assert mock_openai_client.chat.completions.create.call_count == 2

        calls = mock_openai_client.chat.completions.create.call_args_list
        assert calls[0][1]["model"] == "qwen-vl-max"
        assert calls[1][1]["model"] == "qwen-vl-plus"


@patch("time.sleep")
def test_qwen_provider_retry(mock_sleep, mock_openai_client):
    """Test retry on 503 error."""
    provider = QwenVisionProvider(api_key="test_key", models=["qwen-vl-max"])
    provider.MAX_RETRIES = 2

    class MockError(Exception):
        status_code = 503

    success_response_msg = MagicMock()
    success_response_msg.message.content = '{"signal": "LONG"}'
    success_response = MagicMock()
    success_response.choices = [success_response_msg]

    mock_openai_client.chat.completions.create.side_effect = [MockError("Overloaded"), success_response]

    with (
        patch.object(provider, "_encode_image", return_value="data:image/png;base64,123"),
        patch.object(provider, "_get_prompt", return_value="Test prompt"),
    ):
        result = provider.analyze_chart("test.png", "BTC/USDT", "1h")

        assert result == '{"signal": "LONG"}'
        assert mock_sleep.call_count == 1
        assert mock_openai_client.chat.completions.create.call_count == 2

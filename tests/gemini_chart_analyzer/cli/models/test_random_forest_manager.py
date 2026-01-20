"""
Tests for random forest model manager.

Tests cover:
- Model status checking for non-existent model files
- Model validation
- Display model status
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from modules.gemini_chart_analyzer.cli.models.random_forest_manager import (
    check_random_forest_model_status,
    delete_old_model,
    display_model_status,
)


@pytest.fixture(autouse=True)
def mock_config():
    """Mock config to use temporary directories."""
    with (
        patch("modules.gemini_chart_analyzer.cli.models.random_forest_manager.MODELS_DIR") as mock_models_dir,
        patch(
            "modules.gemini_chart_analyzer.cli.models.random_forest_manager.RANDOM_FOREST_MODEL_FILENAME",
            "random_forest_model.pkl",
        ),
    ):
        mock_models_dir.__truediv__ = lambda self, other: Path(tempfile.gettempdir()) / "test_models" / other
        mock_models_dir / "random_forest_model.pkl"  # This will create a Path object
        yield


@pytest.fixture
def mock_validate_model_custom():
    """Custom mock for validate_model that can be overridden per test."""

    def _mock(return_value=(False, "Model validation mocked")):
        return patch(
            "modules.gemini_chart_analyzer.cli.models.random_forest_manager.validate_model", return_value=return_value
        )

    return _mock


@pytest.fixture(autouse=True)
def mock_validate_model_default(request):
    """Mock validate_model to prevent real model validation by default."""
    # Skip autouse for tests that need custom mocking
    if "deprecated_features" in request.function.__name__:
        yield
        return

    with patch("modules.gemini_chart_analyzer.cli.models.random_forest_manager.validate_model") as mock:
        mock.return_value = (False, "Model validation mocked")
        yield mock


@pytest.fixture
def temp_models_dir():
    """Create a temporary directory for test model files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        models_dir = Path(tmpdir) / "models"
        models_dir.mkdir()
        yield models_dir


@pytest.fixture
def valid_model_file(temp_models_dir):
    """Create a valid mock model file."""
    model_path = temp_models_dir / "random_forest_model.pkl"
    model_path.write_text("mock model content")
    return model_path


@pytest.fixture
def mock_validate_model():
    """Mock the validate_model function."""
    with patch("modules.gemini_chart_analyzer.cli.models.random_forest_manager.validate_model") as mock:
        mock.return_value = (True, "Model is valid")
        yield mock


def test_check_model_status_missing_file(mock_validate_model_default):
    """Test status check for non-existent model file."""
    # Ensure no model file exists in temp directory
    temp_model_path = Path(tempfile.gettempdir()) / "test_models" / "random_forest_model.pkl"
    if temp_model_path.exists():
        temp_model_path.unlink()

    status = check_random_forest_model_status(model_path=None)

    assert status["exists"] is False
    assert status["compatible"] is False
    assert status["model_path"] != ""
    assert status["error_message"] is not None
    assert "not found" in status["error_message"].lower()


def test_check_model_status_with_path(mock_validate_model_default):
    """Test status check with custom model path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.pkl"
        model_path.write_text("test")

        status = check_random_forest_model_status(model_path=str(model_path))

        assert status["exists"] is True
        assert status["model_path"] == str(model_path)
        # Error message will be from mocked validation, not None
        assert status["error_message"] == "Model validation mocked"


def test_check_model_status_valid_model(temp_models_dir):
    """Test status check with valid model file."""
    # Create the default model file path
    model_path = Path(tempfile.gettempdir()) / "test_models" / "random_forest_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text("model data")

    # Mock validation to return success
    with patch("modules.gemini_chart_analyzer.cli.models.random_forest_manager.validate_model") as mock_validate:
        mock_validate.return_value = (True, "Model is valid")

        status = check_random_forest_model_status()

        assert status["exists"] is True
        assert status["compatible"] is True
        assert status["model_path"] == str(model_path)
        mock_validate.assert_called_once()


def test_check_model_status_invalid_model(temp_models_dir):
    """Test status check with invalid model file."""
    # Create the default model file path
    model_path = Path(tempfile.gettempdir()) / "test_models" / "random_forest_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text("corrupt data")

    status = check_random_forest_model_status()

    assert status["exists"] is True
    assert status["compatible"] is False
    # The autouse mock returns a generic error message
    assert status["error_message"] is not None


def test_check_model_status_modified_date(temp_models_dir):
    """Test status check returns modification date."""
    model_path = temp_models_dir / "recent_model.pkl"

    import time

    time.sleep(0.1)  # Small delay to ensure different modification time
    model_path.write_text("content")

    status = check_random_forest_model_status()

    assert status["exists"] is True
    assert status["modification_date"] is not None


def test_check_model_status_deprecated_features(temp_models_dir, mock_validate_model_custom):
    """Test status check detects deprecated features."""
    # Create the default model file path
    model_path = Path(tempfile.gettempdir()) / "test_models" / "random_forest_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text("model data")

    # Use custom mock for this test
    with mock_validate_model_custom((False, "Model uses deprecated raw OHLCV features")):
        status = check_random_forest_model_status()

        assert status["exists"] is True
        assert status["compatible"] is False
        assert status["uses_deprecated_features"] is True


def test_check_model_status_path_override(temp_models_dir):
    """Test that custom path overrides default path."""
    model_path = temp_models_dir / "custom_model.pkl"
    model_path.write_text("content")

    status = check_random_forest_model_status(model_path=str(model_path))

    assert status["exists"] is True
    assert status["model_path"] == str(model_path)


def test_check_model_status_none_path(temp_models_dir, mock_validate_model_default):
    """Test that None path uses default path."""
    # Ensure no model file exists in temp directory
    temp_model_path = Path(tempfile.gettempdir()) / "test_models" / "random_forest_model.pkl"
    if temp_model_path.exists():
        temp_model_path.unlink()

    status = check_random_forest_model_status(model_path=None)

    assert status["exists"] is False
    assert status["error_message"] is not None
    assert status["model_path"] != ""


def test_check_model_status_empty_file(temp_models_dir):
    """Test status check with empty model file."""
    # Create the default model file path
    model_path = Path(tempfile.gettempdir()) / "test_models" / "random_forest_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text("")

    status = check_random_forest_model_status()

    assert status["exists"] is True
    assert status["compatible"] is False
    assert status["error_message"] == "Model validation mocked"


def test_check_model_status_large_file(temp_models_dir):
    """Test status check with large model file."""
    model_path = temp_models_dir / "large_model.pkl"
    large_content = "x" * (10 * 1024 * 1024)  # 10MB
    model_path.write_text(large_content)

    status = check_random_forest_model_status()

    assert status["exists"] is True


def test_check_model_status_unicode_filename(temp_models_dir):
    """Test status check with unicode filename."""
    model_path = temp_models_dir / "模型_模型.pkl"
    model_path.write_text("content")

    status = check_random_forest_model_status(model_path=str(model_path))

    assert status["exists"] is True
    assert status["model_path"] == str(model_path)


def test_display_model_status_existing_model(status_dict):
    """Test display_model_status with existing model."""
    with patch("builtins.print") as mock_print:
        display_model_status(
            {
                "exists": True,
                "compatible": True,
                "model_path": "/path/to/model.pkl",
                "modification_date": "2023-01-15 14:30:00",
            }
        )

        # Get all print calls
        calls = [call[0][0] for call in mock_print.call_args_list]
        output = "\n".join(calls)
        assert "Model file" in output
        assert "Compatible" in output
        assert "Compatible" in output
        assert "2023-01-15" in output


def test_display_model_status_missing_model():
    """Test display_model_status with missing model."""
    with patch("builtins.print") as mock_print:
        display_model_status(
            {
                "exists": False,
                "model_path": "/path/to/model.pkl",
                "error_message": "Model file not found",
            }
        )

        calls = [call[0][0] for call in mock_print.call_args_list]
        output = "\n".join(calls)
        assert "not found" in output.lower()


def test_display_model_status_incompatible_model():
    """Test display_model_status with incompatible model."""
    with patch("builtins.print") as mock_print:
        display_model_status(
            {
                "exists": True,
                "compatible": False,
                "model_path": "/path/to/model.pkl",
                "error_message": "Model uses deprecated features",
                "uses_deprecated_features": True,
            }
        )

        calls = [call[0][0] for call in mock_print.call_args_list]
        output = "\n".join(calls)
        assert "Incompatible" in output
        assert "deprecated" in output.lower()
        assert "deprecated" in output.lower()


def test_display_model_status_with_error():
    """Test display_model_status with error message."""
    with patch("builtins.print") as mock_print:
        display_model_status(
            {
                "exists": True,
                "compatible": False,
                "model_path": "/path/to/model.pkl",
                "error_message": "Validation failed",
            }
        )

        calls = [call[0][0] for call in mock_print.call_args_list]
        output = "\n".join(calls)
        assert "Warning" in output
        assert "Validation failed" in output


def test_display_model_status_default_empty_dict():
    """Test display_model_status with empty dictionary."""
    with patch("builtins.print") as mock_print:
        display_model_status({})

        calls = [call[0][0] for call in mock_print.call_args_list]
        output = "\n".join(calls)
        assert "Model file not found" in output


@pytest.fixture
def status_dict():
    """Provide a sample status dictionary for testing."""
    return {
        "exists": True,
        "compatible": True,
        "model_path": "/path/to/model.pkl",
        "modification_date": "2023-01-15 14:30:00",
        "error_message": None,
        "uses_deprecated_features": False,
    }


def test_check_model_status_with_specific_path():
    """Test status check with specific model path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "specific_model.pkl"
        model_path.write_text("content")

        status = check_random_forest_model_status(model_path=str(model_path))

        assert status["exists"] is True
        assert status["model_path"] == str(model_path)


def test_check_model_status_multiple_calls():
    """Test that multiple calls to check_random_forest_model_status work correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test.pkl"
        model_path.write_text("content")

        status1 = check_random_forest_model_status()
        status2 = check_random_forest_model_status()

        assert status1["exists"] is True
        assert status2["exists"] is True
        assert status1["model_path"] == status2["model_path"]

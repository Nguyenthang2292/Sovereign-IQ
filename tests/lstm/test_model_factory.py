
import pytest
import torch

from config.lstm import WINDOW_SIZE_LSTM
from modules.lstm.models.model_factory import create_cnn_lstm_attention_model
from modules.lstm.models.model_factory import create_cnn_lstm_attention_model

"""
Tests for model factory module.
"""



# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def default_input_size():
    """Default input size for testing."""
    return 10


@pytest.fixture
def default_look_back():
    """Default look_back for testing."""
    return WINDOW_SIZE_LSTM


# ============================================================================
# Model Creation Tests
# ============================================================================


class TestModelCreation:
    """Test suite for model creation."""

    @pytest.mark.parametrize(
        "model_name,config",
        [
            ("lstm", {"use_attention": False, "use_cnn": False}),
            ("lstm_attention", {"use_attention": True, "use_cnn": False}),
            ("cnn_lstm", {"use_attention": False, "use_cnn": True, "look_back": 20}),
            ("cnn_lstm_attention", {"use_attention": True, "use_cnn": True, "look_back": 20}),
        ],
    )
    def test_create_model_variants(self, default_input_size, model_name, config):
        """Test creating different model variants."""
        model = create_cnn_lstm_attention_model(input_size=default_input_size, **config)
        assert model is not None
        assert hasattr(model, "forward")

    @pytest.mark.parametrize("output_mode", ["classification", "regression"])
    def test_create_model_output_modes(self, default_input_size, output_mode):
        """Test creating model in different output modes."""
        kwargs = {"input_size": default_input_size, "output_mode": output_mode}
        if output_mode == "regression":
            kwargs["num_classes"] = 1

        model = create_cnn_lstm_attention_model(**kwargs)
        assert model is not None
        assert hasattr(model, "output_mode")
        assert model.output_mode == output_mode

    def test_create_model_custom_look_back(self, default_input_size):
        """Test creating model with custom look_back."""
        custom_look_back = 30
        model = create_cnn_lstm_attention_model(input_size=default_input_size, look_back=custom_look_back)
        assert model is not None

    @pytest.mark.parametrize(
        "model_type,kwargs",
        [
            ("attention", {"use_attention": True, "num_heads": 4, "dropout": 0.2, "hidden_size": 64}),
            ("encoder", {"use_attention": False, "use_cnn": False, "dropout": 0.2, "hidden_size": 64}),
            ("cnn", {"use_cnn": True, "cnn_features": 32, "lstm_hidden": 16, "dropout": 0.2}),
        ],
    )
    def test_create_model_with_valid_kwargs(self, default_input_size, model_type, kwargs):
        """Test creating models with valid kwargs."""
        model = create_cnn_lstm_attention_model(input_size=default_input_size, **kwargs)
        assert model is not None


# ============================================================================
# Input Validation Tests
# ============================================================================


class TestInputValidation:
    """Test suite for input validation."""

    @pytest.mark.parametrize(
        "invalid_input_size,error_msg",
        [
            (0, "input_size must be positive integer"),
            (-1, "input_size must be positive integer"),
            (10.5, "input_size must be positive integer"),
        ],
    )
    def test_invalid_input_size(self, invalid_input_size, error_msg):
        """Test creating model with invalid input_size."""
        with pytest.raises(ValueError, match=error_msg):
            create_cnn_lstm_attention_model(input_size=invalid_input_size)

    @pytest.mark.parametrize(
        "invalid_look_back,error_msg",
        [
            (0, "look_back must be positive integer"),
            (-1, "look_back must be positive integer"),
            (10.5, "look_back must be positive integer"),
        ],
    )
    def test_invalid_look_back(self, default_input_size, invalid_look_back, error_msg):
        """Test creating model with invalid look_back."""
        with pytest.raises(ValueError, match=error_msg):
            create_cnn_lstm_attention_model(input_size=default_input_size, look_back=invalid_look_back)

    def test_invalid_output_mode(self, default_input_size):
        """Test creating model with invalid output_mode."""
        with pytest.raises(ValueError, match="output_mode must be 'classification' or 'regression'"):
            create_cnn_lstm_attention_model(input_size=default_input_size, output_mode="invalid")

    @pytest.mark.parametrize(
        "model_config,expected_error",
        [
            ({"use_attention": True, "invalid_param": 123}, "Unknown parameters for LSTM-Attention model"),
            (
                {"use_attention": False, "use_cnn": False, "invalid_param": 123},
                "Unknown parameters for LSTM encoder model",
            ),
            ({"use_cnn": True, "invalid_param": 123}, "Unknown parameters for CNN-LSTM-Attention model"),
        ],
    )
    def test_unknown_kwargs(self, default_input_size, model_config, expected_error):
        """Test creating model with unknown kwargs."""
        with pytest.raises(ValueError, match=expected_error):
            create_cnn_lstm_attention_model(input_size=default_input_size, **model_config)


# ============================================================================
# Forward Pass Tests
# ============================================================================


class TestForwardPass:
    """Test suite for model forward pass."""

    @pytest.mark.parametrize(
        "model_name,config,seq_len",
        [
            ("lstm", {"use_attention": False, "use_cnn": False}, WINDOW_SIZE_LSTM),
            ("lstm_attention", {"use_attention": True, "use_cnn": False}, WINDOW_SIZE_LSTM),
            ("cnn_lstm", {"use_attention": False, "use_cnn": True, "look_back": 20}, 20),
            ("cnn_lstm_attention", {"use_attention": True, "use_cnn": True, "look_back": 20}, 20),
        ],
    )
    def test_forward_pass_variants(self, default_input_size, model_name, config, seq_len):
        """Test forward pass of different model variants."""
        model = create_cnn_lstm_attention_model(input_size=default_input_size, **config)
        model.eval()

        batch_size = 2
        dummy_input = torch.randn(batch_size, seq_len, default_input_size)

        with torch.no_grad():
            output = model(dummy_input)

        assert output is not None
        assert output.shape[0] == batch_size
        assert len(output.shape) == 2  # (batch_size, num_classes or 1)

    def test_forward_pass_classification_output_shape(self, default_input_size, default_look_back):
        """Test classification model output shape."""
        model = create_cnn_lstm_attention_model(input_size=default_input_size, output_mode="classification")
        model.eval()

        batch_size = 2
        dummy_input = torch.randn(batch_size, default_look_back, default_input_size)

        with torch.no_grad():
            output = model(dummy_input)

        # Classification should output (batch_size, num_classes) where num_classes=3
        assert output.shape == (batch_size, 3)

    def test_forward_pass_regression_output_shape(self, default_input_size, default_look_back):
        """Test regression model output shape."""
        model = create_cnn_lstm_attention_model(
            input_size=default_input_size,
            output_mode="regression",
            num_classes=1,  # Explicitly set num_classes for regression
        )
        model.eval()

        batch_size = 2
        dummy_input = torch.randn(batch_size, default_look_back, default_input_size)

        with torch.no_grad():
            output = model(dummy_input)

        # Regression should output (batch_size, 1)
        assert output.shape == (batch_size, 1)

    def test_forward_pass_different_batch_sizes(self, default_input_size, default_look_back):
        """Test forward pass with different batch sizes."""
        model = create_cnn_lstm_attention_model(input_size=default_input_size)
        model.eval()

        for batch_size in [1, 4, 8]:
            dummy_input = torch.randn(batch_size, default_look_back, default_input_size)

            with torch.no_grad():
                output = model(dummy_input)

            assert output.shape[0] == batch_size

    def test_forward_pass_training_mode(self, default_input_size, default_look_back):
        """Test forward pass in training mode (model.train())."""
        model = create_cnn_lstm_attention_model(input_size=default_input_size)
        model.train()  # Set to training mode

        batch_size = 2
        dummy_input = torch.randn(batch_size, default_look_back, default_input_size)

        output = model(dummy_input)

        assert output is not None
        assert output.shape[0] == batch_size

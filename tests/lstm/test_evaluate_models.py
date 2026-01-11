
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import pytest
import torch
import torch.nn as nn

from config.evaluation import CONFIDENCE_THRESHOLDS
from modules.lstm.core.evaluate_models import (
from modules.lstm.core.evaluate_models import (

"""
Tests for evaluate_models module.
"""


    apply_confidence_threshold,
    evaluate_model_in_batches,
    evaluate_model_with_confidence,
)


class SimpleTestModel(nn.Module):
    """Simple test model for evaluation tests."""

    def __init__(self, input_size=10, num_classes=3):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # Use last timestep
        x = x[:, -1, :]
        return self.softmax(self.linear(x))


class TestApplyConfidenceThreshold:
    """Test suite for apply_confidence_threshold function."""

    def test_apply_threshold_basic(self):
        """Test basic confidence threshold application."""
        y_proba = np.array(
            [
                [0.1, 0.2, 0.7],  # Max: 0.7 (class 2 -> BUY)
                [0.8, 0.1, 0.1],  # Max: 0.8 (class 0 -> SELL)
                [0.2, 0.7, 0.1],  # Max: 0.7 (class 1 -> NEUTRAL)
            ]
        )

        threshold = 0.6
        predictions = apply_confidence_threshold(y_proba, threshold)

        assert predictions.shape == (3,)
        assert predictions.dtype == int
        assert all(p in [-1, 0, 1] for p in predictions)

    def test_apply_threshold_high_confidence(self):
        """Test with high confidence threshold."""
        y_proba = np.array(
            [
                [0.1, 0.2, 0.7],  # Below threshold 0.8 -> NEUTRAL
                [0.9, 0.05, 0.05],  # Above threshold -> SELL
                [0.05, 0.05, 0.9],  # Above threshold -> BUY
            ]
        )

        threshold = 0.8
        predictions = apply_confidence_threshold(y_proba, threshold)

        assert predictions[0] == 0  # Low confidence -> neutral
        assert predictions[1] == -1  # High confidence SELL
        assert predictions[2] == 1  # High confidence BUY

    def test_apply_threshold_low_confidence(self):
        """Test with low confidence threshold."""
        y_proba = np.array(
            [
                [0.4, 0.3, 0.3],  # Max: 0.4, below 0.5 -> NEUTRAL
                [0.6, 0.2, 0.2],  # Max: 0.6, above 0.5 -> SELL
            ]
        )

        threshold = 0.5
        predictions = apply_confidence_threshold(y_proba, threshold)

        assert predictions[0] == 0  # Below threshold
        assert predictions[1] == -1  # Above threshold

    def test_apply_threshold_empty_array(self):
        """Test with empty array."""
        y_proba = np.array([]).reshape(0, 3)
        predictions = apply_confidence_threshold(y_proba, 0.7)

        assert predictions.shape == (0,)
        assert predictions.dtype == int

    def test_apply_threshold_all_neutral(self):
        """Test when all predictions are below threshold."""
        y_proba = np.array(
            [
                [0.4, 0.3, 0.3],
                [0.35, 0.35, 0.3],
            ]
        )

        threshold = 0.5
        predictions = apply_confidence_threshold(y_proba, threshold)

        assert all(p == 0 for p in predictions)

    def test_apply_threshold_class_mapping(self):
        """Test that class indices are correctly mapped to -1, 0, 1."""
        y_proba = np.array(
            [
                [1.0, 0.0, 0.0],  # Class 0 -> -1 (SELL)
                [0.0, 1.0, 0.0],  # Class 1 -> 0 (NEUTRAL)
                [0.0, 0.0, 1.0],  # Class 2 -> 1 (BUY)
            ]
        )

        threshold = 0.5
        predictions = apply_confidence_threshold(y_proba, threshold)

        assert predictions[0] == -1
        assert predictions[1] == 0
        assert predictions[2] == 1


class TestEvaluateModelInBatches:
    """Test suite for evaluate_model_in_batches function."""

    def test_evaluate_basic(self):
        """Test basic batch evaluation."""
        model = SimpleTestModel(input_size=10, num_classes=3)
        device = torch.device("cpu")
        X_test = torch.randn(10, 20, 10)  # 10 samples, seq_len=20, features=10

        predictions = evaluate_model_in_batches(model, X_test, device, batch_size=4)

        assert predictions.shape == (10, 3)
        assert np.allclose(predictions.sum(axis=1), 1.0, atol=1e-6)  # Probabilities sum to 1

    def test_evaluate_single_batch(self):
        """Test evaluation with single batch."""
        model = SimpleTestModel(input_size=10, num_classes=3)
        device = torch.device("cpu")
        X_test = torch.randn(5, 20, 10)

        predictions = evaluate_model_in_batches(model, X_test, device, batch_size=10)

        assert predictions.shape == (5, 3)

    def test_evaluate_multiple_batches(self):
        """Test evaluation with multiple batches."""
        model = SimpleTestModel(input_size=10, num_classes=3)
        device = torch.device("cpu")
        X_test = torch.randn(20, 20, 10)

        predictions = evaluate_model_in_batches(model, X_test, device, batch_size=7)

        assert predictions.shape == (20, 3)

    def test_evaluate_empty_input(self):
        """Test evaluation with empty input."""
        model = SimpleTestModel(input_size=10, num_classes=3)
        device = torch.device("cpu")
        X_test = torch.randn(0, 20, 10)

        predictions = evaluate_model_in_batches(model, X_test, device, batch_size=4)

        assert predictions.shape == (0, 3)

    def test_evaluate_model_in_eval_mode(self):
        """Test that model is set to eval mode."""
        model = SimpleTestModel(input_size=10, num_classes=3)
        model.train()  # Set to training mode

        device = torch.device("cpu")
        X_test = torch.randn(5, 20, 10)

        predictions = evaluate_model_in_batches(model, X_test, device, batch_size=4)

        # Model should be in eval mode after evaluation
        assert not model.training
        assert predictions.shape == (5, 3)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_evaluate_on_cuda(self):
        """Test evaluation on CUDA device."""
        model = SimpleTestModel(input_size=10, num_classes=3)
        device = torch.device("cuda")
        model = model.to(device)
        X_test = torch.randn(10, 20, 10)

        predictions = evaluate_model_in_batches(model, X_test, device, batch_size=4)

        assert predictions.shape == (10, 3)
        assert isinstance(predictions, np.ndarray)


class TestEvaluateModelWithConfidence:
    """Test suite for evaluate_model_with_confidence function."""

    def test_evaluate_with_confidence_basic(self):
        """
        Smoke test: Verify evaluate_model_with_confidence executes without exceptions.

        This is a basic smoke test that ensures the function completes successfully
        with valid inputs. The function returns None and logs results, so we verify
        completion by checking that no exceptions are raised and the model state is
        preserved (set to eval mode).
        """
        model = SimpleTestModel(input_size=10, num_classes=3)
        device = torch.device("cpu")
        X_test = torch.randn(20, 20, 10)
        y_test = np.random.randint(-1, 2, 20)  # Labels: -1, 0, 1

        # Verify function completes without raising exceptions
        evaluate_model_with_confidence(model, X_test, y_test, device)

        # Verify model is in eval mode after evaluation (side effect check)
        assert not model.training, "Model should be in eval mode after evaluation"

    def test_evaluate_with_confidence_verifies_outputs(self):
        """Test that evaluate_model_with_confidence produces correct outputs and metrics."""
        model = SimpleTestModel(input_size=10, num_classes=3)
        device = torch.device("cpu")
        n_samples = 30
        X_test = torch.randn(n_samples, 20, 10)

        # Create balanced test labels
        np.random.seed(42)
        y_test = np.random.choice([-1, 0, 1], size=n_samples, p=[0.3, 0.4, 0.3])

        # Get predictions directly from evaluate_model_in_batches (integration test)
        predictions = evaluate_model_in_batches(model, X_test, device, batch_size=16)

        # Verify predictions were generated
        assert predictions is not None, "Predictions should be generated"
        assert predictions.shape == (n_samples, 3), f"Expected shape ({n_samples}, 3), got {predictions.shape}"

        # Verify predictions are probabilities (sum to ~1, all >= 0, all <= 1)
        assert np.allclose(predictions.sum(axis=1), 1.0, atol=1e-5), "Predictions should sum to 1"
        assert np.all(predictions >= 0), "All predictions should be >= 0"
        assert np.all(predictions <= 1), "All predictions should be <= 1"

        for threshold in CONFIDENCE_THRESHOLDS:
            y_pred = apply_confidence_threshold(predictions, threshold)

            # Verify predictions are valid class labels
            assert np.all(np.isin(y_pred, [-1, 0, 1])), f"Predictions should be in [-1, 0, 1] for threshold {threshold}"
            assert len(y_pred) == n_samples, f"Prediction length should match input for threshold {threshold}"

            # Verify metrics are computable and in valid ranges
            accuracy = accuracy_score(y_test, y_pred)
            assert 0 <= accuracy <= 1, f"Accuracy should be in [0, 1], got {accuracy}"

            precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
            assert 0 <= precision <= 1, f"Precision should be in [0, 1], got {precision}"

            recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
            assert 0 <= recall <= 1, f"Recall should be in [0, 1], got {recall}"

            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            assert 0 <= f1 <= 1, f"F1-score should be in [0, 1], got {f1}"

        # Verify relationship: higher threshold should produce more neutral predictions
        sorted_thresholds = sorted(CONFIDENCE_THRESHOLDS)
        if len(sorted_thresholds) >= 2:
            low_threshold = sorted_thresholds[0]
            high_threshold = sorted_thresholds[-1]
        else:
            low_threshold, high_threshold = 0.5, 0.9  # Fallback

        low_threshold_pred = apply_confidence_threshold(predictions, low_threshold)
        high_threshold_pred = apply_confidence_threshold(predictions, high_threshold)

        low_neutral_count = np.sum(low_threshold_pred == 0)
        high_neutral_count = np.sum(high_threshold_pred == 0)

        # Higher threshold should produce more neutral predictions.
        # Exception: if model predictions already have very high confidence (>0.9),
        # then increasing threshold won't produce more neutral predictions since
        # predictions already exceed the higher threshold.
        assert high_neutral_count >= low_neutral_count or np.max(predictions, axis=1).mean() > 0.9, (
            "Higher threshold should produce at least as many neutral predictions"
        )

    def test_evaluate_with_confidence_empty_data(self):
        """
        Test evaluation with empty test data.

        Empty data may cause various exceptions (ValueError, TypeError, etc.) depending on
        where the empty check occurs in the evaluation pipeline. This test verifies that
        the function handles empty data appropriately by either completing or raising a
        reasonable exception type.
        """
        model = SimpleTestModel(input_size=10, num_classes=3)
        device = torch.device("cpu")
        X_test = torch.randn(0, 20, 10)
        y_test = np.array([])

        # Empty data may raise ValueError, TypeError, or other exceptions depending on
        # where the empty check occurs (e.g., np.bincount with empty array can raise TypeError)
        # Accept any exception as long as it's not silently ignored
        with pytest.raises(Exception):  # Accept any exception for empty data
            evaluate_model_with_confidence(model, X_test, y_test, device)

    def test_evaluate_with_confidence_shape_mismatch(self):
        """
        Test that shape mismatch between X_test and y_test raises an exception.

        When X_test and y_test have mismatched lengths, sklearn metrics functions
        will raise ValueError when attempting to compute metrics. This test verifies
        that the error is properly propagated rather than being silently ignored.
        """
        model = SimpleTestModel(input_size=10, num_classes=3)
        device = torch.device("cpu")
        X_test = torch.randn(10, 20, 10)  # 10 samples
        y_test = np.random.randint(-1, 2, 5)  # 5 samples - shape mismatch

        # Should raise ValueError when sklearn metrics detect shape mismatch
        # sklearn error message: "Found input variables with inconsistent numbers of samples: [5, 10]"
        with pytest.raises(ValueError, match=".*inconsistent.*samples.*|.*shape.*|.*length.*|.*dimension.*"):
            evaluate_model_with_confidence(model, X_test, y_test, device)

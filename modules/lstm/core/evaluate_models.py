
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import numpy as np
import torch

from config.evaluation import CONFIDENCE_THRESHOLD, CONFIDENCE_THRESHOLDS
from modules.common.ui.logging import log_analysis, log_error, log_warn
from modules.common.ui.logging import log_analysis, log_error, log_warn

"""
Model evaluation utilities for LSTM models.
Provides functions for batch evaluation and confidence-based evaluation.
"""



# Default number of classes for trading signals: SELL (-1), NEUTRAL (0), BUY (1)
DEFAULT_NUM_CLASSES = 3


def apply_confidence_threshold(y_proba: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply confidence threshold to model predictions to improve reliability.

    This function converts probability predictions to class labels based on a confidence
    threshold. If the maximum probability is below the threshold, the prediction defaults
    to neutral (0).

    Args:
        y_proba: Array of prediction probabilities for each class (n_samples, n_classes)
        threshold: Confidence threshold to apply (0.0-1.0)

    Returns:
        Array of class predictions (-1, 0, 1) after applying threshold
    """
    if y_proba.size == 0:
        return np.array([], dtype=int)

    # Vectorized implementation for better performance
    max_confidence = np.max(y_proba, axis=1)
    high_confidence_mask = max_confidence >= threshold

    # Initialize all predictions as neutral (0)
    predictions = np.zeros(y_proba.shape[0], dtype=int)

    # For high confidence predictions, convert argmax indices to -1,0,1 (classes are 0,1,2 -> -1,0,1)
    if np.any(high_confidence_mask):
        pred_indices = np.argmax(y_proba[high_confidence_mask], axis=1)
        predictions[high_confidence_mask] = pred_indices - 1  # Convert 0,1,2 to -1,0,1

    return predictions


def _infer_output_classes(
    model: torch.nn.Module, X_test: torch.Tensor, device: torch.device, default_classes: int = DEFAULT_NUM_CLASSES
) -> int:
    """
    Infer number of output classes from model via dummy forward pass.

    Args:
        model: PyTorch model to infer classes from
        X_test: Test data tensor to infer shape from
        device: Device to run inference on
        default_classes: Default number of classes if inference fails

    Returns:
        Number of output classes (inferred or default)
    """
    if len(X_test.shape) >= 3:
        seq_len, n_features = X_test.shape[1], X_test.shape[2]

        # Validate edge case: zero-sized dimensions
        if seq_len == 0 or n_features == 0:
            log_warn(
                f"Cannot infer classes from zero-sized dimensions: "
                f"seq_len={seq_len}, n_features={n_features}. "
                f"Using default_classes={default_classes}"
            )
            return default_classes

        dummy_input = torch.zeros(1, seq_len, n_features).to(device)
        try:
            with torch.no_grad():
                dummy_output = model(dummy_input).cpu()
                return dummy_output.shape[1] if len(dummy_output.shape) > 1 else 1
        except RuntimeError as e:
            log_error(f"Runtime error during model inference (CUDA/CPU error): {e}")
        except (TypeError, ValueError) as e:
            log_error(f"Type/value error during model inference (shape mismatch): {e}")
        except Exception as e:
            log_error(f"Unexpected error during model inference: {type(e).__name__}: {e}")
    else:
        log_warn(
            f"X_test has invalid shape for inference: {X_test.shape} "
            f"(expected at least 3 dimensions). Using default_classes={default_classes}"
        )
    return default_classes


def evaluate_model_in_batches(
    model: torch.nn.Module, X_test: torch.Tensor, device: torch.device, batch_size: int = 32
) -> np.ndarray:
    """
    Evaluate model in batches to avoid CUDA out of memory errors.

    Args:
        model: PyTorch model to evaluate
        X_test: Test data tensor (n_samples, sequence_length, n_features)
        device: Device to run evaluation on (CPU/CUDA)
        batch_size: Batch size for evaluation

    Returns:
        numpy array: Prediction probabilities (n_samples, n_classes)
    """

    model.eval()
    all_predictions = []

    # Handle empty input
    if len(X_test) == 0:
        # Return empty array with correct shape (0, num_classes)
        num_classes = _infer_output_classes(model, X_test, device)
        return np.zeros((0, num_classes))

    # Process in smaller batches to avoid OOM
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_end = min(i + batch_size, len(X_test))
            batch_X = X_test[i:batch_end].to(device)

            try:
                batch_pred = model(batch_X).cpu()
                all_predictions.append(batch_pred)

                # Clear intermediate tensors
                del batch_X, batch_pred

                # Clear GPU cache periodically
                if device.type == "cuda" and i % (batch_size * 10) == 0:
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    log_warn(f"OOM during batch {i}-{batch_end}, reducing batch size")
                    # Try with smaller batch size
                    if batch_size > 1:
                        smaller_batch = max(1, batch_size // 2)
                        return evaluate_model_in_batches(model, X_test, device, smaller_batch)
                    else:
                        log_error("Cannot reduce batch size further, OOM persists at batch_size=1")
                        raise e
                else:
                    raise e

    # Concatenate all predictions and handle errors robustly
    try:
        if not all_predictions:
            # This shouldn't happen if len(X_test) > 0, but handle it anyway
            log_warn("No predictions produced despite non-empty input")
            num_classes = _infer_output_classes(model, X_test, device)
            return np.zeros((0, num_classes))
        concat_pred = torch.cat(all_predictions, dim=0)
        if concat_pred.shape[0] != len(X_test):
            log_error(
                f"Concatenated predictions have wrong number of samples: {concat_pred.shape[0]} vs X_test {len(X_test)}"
            )
            raise RuntimeError(
                f"Prediction shape mismatch: got {concat_pred.shape[0]} predictions "
                f"for {len(X_test)} input samples. This indicates a critical error in batch processing."
            )
        return concat_pred.numpy()

    except RuntimeError:
        # Re-raise critical errors (shape mismatches, OOM, CUDA errors)
        # RuntimeError indicates fundamental processing failure that cannot be recovered
        raise

    except Exception as err:
        log_error(f"Error during prediction concatenation: {err}")
        # Safe fallback: infer number of classes
        num_classes = _infer_output_classes(model, X_test, device)
        return np.zeros((len(X_test), num_classes))


def evaluate_model_with_confidence(
    model: torch.nn.Module, X_test: torch.Tensor, y_test: np.ndarray, device: torch.device
) -> None:
    """
    Evaluate LSTM model with multiple confidence thresholds for trading signals.

    Performs comprehensive evaluation using various confidence thresholds to assess
    trading signal reliability and determine optimal settings for live trading.

    Args:
        model: Trained PyTorch LSTM model
        X_test: Test features (n_samples, sequence_length, n_features)
        y_test: Test labels {-1: SELL, 0: NEUTRAL, 1: BUY} (n_samples,)
        device: Evaluation device (CPU/CUDA)

    Returns:
        None: Logs evaluation results and threshold recommendations

    Note:
        Uses batch evaluation to handle GPU memory constraints.
        Evaluates multiple thresholds from config.evaluation.CONFIDENCE_THRESHOLDS.
    """
    # Use batch evaluation to avoid memory issues
    evaluation_batch_size = 16 if device.type == "cuda" else 32  # Very small batch for GPU

    log_analysis(f"Evaluating model with {len(X_test)} test samples in batches of {evaluation_batch_size}...")

    try:
        y_pred_prob = evaluate_model_in_batches(model, X_test, device, evaluation_batch_size)
    except Exception as e:
        log_error(f"Failed to evaluate model in batches: {e}")
        return

    log_analysis(f"Test set class distribution: {np.bincount(y_test + 1, minlength=3)}")  # Shift for counting

    # Use config confidence thresholds
    for threshold in CONFIDENCE_THRESHOLDS:
        log_analysis("\n" + "=" * 50)
        log_analysis(f"CONFIDENCE THRESHOLD: {threshold:.1%}")
        log_analysis("=" * 50)

        y_pred = apply_confidence_threshold(y_pred_prob, threshold)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        log_analysis(f"Accuracy: {accuracy:.3f}")
        log_analysis(f"Precision: {precision:.3f}")
        log_analysis(f"Recall: {recall:.3f}")
        log_analysis(f"F1-Score: {f1:.3f}")

        # Signal distribution (using bincount with shift: -1->0, 0->1, 1->2)
        total = len(y_pred)
        if total == 0:
            log_warn("No predictions to analyze")
            continue

        signal_counts = np.bincount(y_pred + 1, minlength=3)  # Shift -1,0,1 to 0,1,2 for indexing
        log_analysis("Signal Distribution:")
        log_analysis(f"  SELL (-1): {signal_counts[0]:.0f} ({signal_counts[0] / total * 100:.1f}%)")
        log_analysis(f"  NEUTRAL (0): {signal_counts[1]:.0f} ({signal_counts[1] / total * 100:.1f}%)")
        log_analysis(f"  BUY (1): {signal_counts[2]:.0f} ({signal_counts[2] / total * 100:.1f}%)")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=[-1, 0, 1])
        log_analysis("Confusion Matrix:")
        log_analysis("     SELL  NEUTRAL  BUY")
        for i, label in enumerate(["SELL", "NEUTRAL", "BUY"]):
            log_analysis(f"{label:>8}: {cm[i]}")

        # Trading-specific metrics
        buy_precision = precision_score(y_test, y_pred, labels=[1], average="macro", zero_division=0)
        sell_precision = precision_score(y_test, y_pred, labels=[-1], average="macro", zero_division=0)

        log_analysis("Trading Metrics:")
        log_analysis(f"  BUY Signal Precision: {buy_precision:.3f}")
        log_analysis(f"  SELL Signal Precision: {sell_precision:.3f}")

        # Calculate trading score
        buy_freq = signal_counts[2] / total
        sell_freq = signal_counts[0] / total
        trading_score = (buy_precision * buy_freq + sell_precision * sell_freq) / (buy_freq + sell_freq + 1e-6)
        log_analysis(f"  Trading Score: {trading_score:.3f}")

    # Recommend optimal threshold
    log_analysis("RECOMMENDATION:")
    log_analysis("For conservative trading: Use threshold 0.75+ (higher precision, fewer signals)")
    log_analysis("For active trading: Use threshold 0.60-0.65 (balanced precision/frequency)")
    log_analysis(f"Current default: {CONFIDENCE_THRESHOLD} (used in get_latest_LSTM_signal)")

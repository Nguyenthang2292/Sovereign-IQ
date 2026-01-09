"""
Utility functions for LSTM models (loading and inference).
"""
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from modules.common.ui.logging import (
    log_info,
    log_error,
    log_warn,
    log_model,
    log_debug,
)
from config.lstm import MODELS_DIR, WINDOW_SIZE_LSTM, KALMAN_PROCESS_VARIANCE, KALMAN_OBSERVATION_VARIANCE
from config.model_features import MODEL_FEATURES
from config.evaluation import CONFIDENCE_THRESHOLD
from modules.lstm.utils.indicator_features import generate_indicator_features
from modules.lstm.utils.kalman_filter import apply_kalman_to_ohlc
from modules.lstm.models.lstm_models import LSTMAttentionModel, LSTMModel
from modules.lstm.models.model_factory import create_cnn_lstm_attention_model


def _load_checkpoint_safe(model_path: Path):
    """
    Safely load PyTorch checkpoint with support for sklearn scalers.
    
    Tries to load with weights_only=True and safe globals for sklearn scalers.
    Falls back to weights_only=False if needed (for trusted internal checkpoints).
    
    Args:
        model_path: Path to checkpoint file
        
    Returns:
        Loaded checkpoint dictionary
    """
    # Try loading with weights_only=True and safe globals for sklearn scalers
    try:
        # Add sklearn scalers to safe globals
        torch.serialization.add_safe_globals([MinMaxScaler, StandardScaler])
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        return checkpoint
    except Exception:
        # Fallback to weights_only=False for internal checkpoints
        # This is safe because these are checkpoints we created ourselves
        log_warn("Could not load checkpoint with weights_only=True, using weights_only=False (trusted source)")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        return checkpoint


def load_lstm_model(model_path: Optional[Path] = None) -> Optional[nn.Module]:
    """
    Load LSTM model from saved checkpoint.
    
    This function supports both old format (with 'input_size', 'use_attention') 
    and new format (with 'model_config' dictionary).
    
    Args:
        model_path: Path to the model file. If None, uses default path.
        
    Returns:
        Loaded model or None if failed
    """
    if model_path is None:
        # Try default model filename
        model_path = MODELS_DIR / "lstm_model.pth"
    elif isinstance(model_path, str):
        # Convert string to Path object
        model_path = Path(model_path)
    
    # If default model doesn't exist, try to find the latest model
    if not model_path.exists() and model_path == MODELS_DIR / "lstm_model.pth":
        if MODELS_DIR.exists():
            model_files = list(MODELS_DIR.glob("*.pth"))
            if model_files:
                # Sort by modification time, newest first
                model_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                model_path = model_files[0]
                log_info(f"Default model not found, using latest: {model_path.name}")
    
    if not model_path.exists():
        log_error(f"Model file not found: {model_path}")
        return None
    
    try:
        checkpoint = _load_checkpoint_safe(model_path)
        
        # Handle new format (with model_config)
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            input_size = config['input_size']
            use_attention = config.get('use_attention', False)
            use_cnn = config.get('use_cnn', False)
            attention_heads = config.get('attention_heads', 8)
            look_back = config.get('look_back', WINDOW_SIZE_LSTM)
            output_mode = config.get('output_mode', 'classification')
            num_classes = config.get('num_classes', 3 if output_mode == 'classification' else 1)
            
            # Create model using factory
            model = create_cnn_lstm_attention_model(
                input_size=input_size,
                use_attention=use_attention,
                use_cnn=use_cnn,
                look_back=look_back,
                output_mode=output_mode,
                num_heads=attention_heads,
                num_classes=num_classes,
            )
            model_name_parts = []
            if use_cnn:
                model_name_parts.append('CNN')
            model_name_parts.append('LSTM')
            if use_attention:
                model_name_parts.append('Attention')
            log_model(f"Loading {'-'.join(model_name_parts)} model")
            
            # Log original_look_back if available (for traceability when fallback occurred)
            original_look_back = config.get('original_look_back')
            if original_look_back is not None and original_look_back != look_back:
                log_model(
                    f"Model was trained with reduced look_back: {look_back} "
                    f"(original: {original_look_back}). This indicates insufficient data during training."
                )
        else:
            # Handle old format (backward compatibility)
            input_size = checkpoint.get('input_size')
            if input_size is None:
                raise ValueError("Cannot determine input_size from checkpoint")
            use_attention = checkpoint.get('use_attention', False)
            attention_heads = checkpoint.get('attention_heads', 8)
            
            if use_attention:
                model = LSTMAttentionModel(
                    input_size=input_size,
                    num_heads=attention_heads
                )
                log_model(f"Loading LSTM-Attention model with {attention_heads} heads (old format)")
            else:
                model = LSTMModel(input_size=input_size)
                log_model("Loading standard LSTM model (old format)")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        log_model(f"Successfully loaded model from {model_path}")
        return model
        
    except Exception as e:
        log_error(f"Error loading model: {e}")
        return None


def get_latest_signal(
    df_market_data: pd.DataFrame, 
    model: nn.Module, 
    scaler: Optional[MinMaxScaler] = None,
    look_back: int = WINDOW_SIZE_LSTM,
    use_kalman_filter: Optional[bool] = None,
    kalman_params: Optional[dict] = None
) -> str:
    """
    Generate trading signal from LSTM model using latest market data.
    
    Note: This function uses the model's current device. It does not move the model,
    but instead moves input tensors to match the model's device to avoid side effects.
    
    Args:
        df_market_data: DataFrame with OHLC market data
        model: Trained LSTM model (any variant). Model should already be on the desired device.
        scaler: Pre-fitted scaler for feature normalization (optional)
        look_back: Sequence length (default from config)
        use_kalman_filter: Whether to apply Kalman Filter. If None, checks model metadata.
        kalman_params: Kalman Filter parameters. If None, uses model metadata or config defaults.
        
    Returns:
        Trading signal: 'BUY', 'SELL', or 'NEUTRAL'
    """
    # Check the model_device, warn if None
    try:
        model_device = next(model.parameters()).device
    except Exception as e:
        log_error(f"Could not determine model device: {e}")
        return 'NEUTRAL'
    
    # Input dataframe check
    if df_market_data.empty:
        log_warn("Empty input DataFrame for signal generation")
        return 'NEUTRAL'
    
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df_market_data.columns]
    if missing_cols:
        log_warn(f"Missing OHLC columns: {missing_cols}. Available: {list(df_market_data.columns)}")
        return 'NEUTRAL'
    
    # Determine if Kalman Filter should be applied
    # Check model metadata if use_kalman_filter not explicitly provided
    if use_kalman_filter is None:
        use_kalman_filter = getattr(model, '_use_kalman_filter', False)
    
    # Get Kalman parameters from model metadata or use provided/defaults
    if use_kalman_filter and kalman_params is None:
        kalman_params = getattr(model, '_kalman_params', None)
        if kalman_params is None:
            kalman_params = {
                'process_variance': KALMAN_PROCESS_VARIANCE,
                'observation_variance': KALMAN_OBSERVATION_VARIANCE
            }
    
    # Apply Kalman Filter if needed
    df_for_features = df_market_data.copy()
    if use_kalman_filter:
        try:
            df_for_features = apply_kalman_to_ohlc(df_market_data.copy(), **kalman_params)
            log_model("Applied Kalman Filter to OHLC data for inference")
        except Exception as e:
            log_warn(f"Error applying Kalman Filter during inference: {e}, using original data")
            df_for_features = df_market_data.copy()
    
    # Generate/check features from (possibly smoothed) OHLC data
    df_features = generate_indicator_features(df_for_features)
    if df_features.empty:
        log_warn("Empty DataFrame after feature calculation")
        return 'NEUTRAL'
        
    if len(df_features) < look_back:
        log_warn(f"Insufficient data: {len(df_features)} < {look_back}")
        return 'NEUTRAL'
    
    available_features = [col for col in MODEL_FEATURES if col in df_features.columns]
    if not available_features:
        log_error(f"No valid features found from {MODEL_FEATURES}")
        return 'NEUTRAL'
    
    # Validate feature count matches model expectations using check
    # Try multiple ways to get input_size based on model type
    expected_input_size = None
    
    # For CNN-LSTM models: check self.input_size
    if hasattr(model, 'input_size'):
        expected_input_size = model.input_size
    # For standard LSTM models: check lstm_layers[0].input_size or self.lstm.input_size
    elif hasattr(model, 'lstm_layers') and len(model.lstm_layers) > 0:
        first_lstm = model.lstm_layers[0]
        if hasattr(first_lstm, 'input_size'):
            expected_input_size = first_lstm.input_size
    elif hasattr(model, 'lstm') and hasattr(model.lstm, 'input_size'):
        expected_input_size = model.lstm.input_size
    
    if expected_input_size and len(available_features) != expected_input_size:
        log_error(f"Feature count mismatch: model expects {expected_input_size}, got {len(available_features)}")
        log_error(f"Available features: {available_features[:10]}...")  # Show first 10 features
        return 'NEUTRAL'
    
    if len(available_features) < len(MODEL_FEATURES):
        log_warn(f"Using {len(available_features)}/{len(MODEL_FEATURES)} features")
        
    if scaler is None:
        log_error("No scaler provided - predictions will be unreliable without proper scaling")
        scaled_features = df_features[available_features].values
    else:
        try:
            # Convert DataFrame to array to avoid feature names warning
            feature_array = df_features[available_features].values
            scaled_features = scaler.transform(feature_array)
        except Exception as e:
            log_error(f"Scaler transform failed: {e}")
            return 'NEUTRAL'
    
    # Check for NaN/infinite values before feeding to model
    if not np.all(np.isfinite(scaled_features)):
        log_error("Found NaN or infinite values in the scaled features.")
        return 'NEUTRAL'
    
    # Prepare the model input

    # Convert numpy array to tensor efficiently
    window_data = scaled_features[-look_back:]
    input_window = torch.tensor(window_data, dtype=torch.float32, device=model_device).unsqueeze(0)

    try:
        # Debug: Log input window info
        log_debug(f"Input window shape: {input_window.shape}, dtype: {input_window.dtype}")
        log_debug(f"Input window stats: min={input_window.min().item():.4f}, max={input_window.max().item():.4f}, mean={input_window.mean().item():.4f}")
        
        model.eval()
        with torch.no_grad():
            model_output = model(input_window)
            
            # Debug: Log model output info
            log_debug(f"Model output shape: {model_output.shape}, dtype: {model_output.dtype}")
            log_debug(f"Model output stats: min={model_output.min().item():.4f}, max={model_output.max().item():.4f}, mean={model_output.mean().item():.4f}")
            
            # Check if output is 2D (batch, classes) or already 1D
            if len(model_output.shape) == 2:
                prediction_probs = model_output[0].cpu().numpy()
            else:
                prediction_probs = model_output.cpu().numpy()
            
            # Debug: Log prediction_probs info
            log_debug(f"Prediction probs shape: {prediction_probs.shape}")
            log_debug(f"Prediction probs values: {prediction_probs}")
            log_debug(f"Prediction probs sum: {np.sum(prediction_probs):.6f}")
            
            # Check for NaN/Inf
            if not np.all(np.isfinite(prediction_probs)):
                log_error(f"Found NaN or Inf in prediction_probs: {prediction_probs}")
                return 'NEUTRAL'
            
            # Verify probabilities sum to ~1 (should be after softmax)
            prob_sum = np.sum(prediction_probs)
            
            # If probabilities don't sum to 1, they might be logits - apply softmax
            if abs(prob_sum - 1.0) > 0.01:
                log_warn(f"Prediction probs don't sum to 1.0 (sum={prob_sum:.6f}), applying softmax normalization")
                # Apply softmax to normalize (handle both logits and unnormalized probs)
                exp_probs = np.exp(prediction_probs - np.max(prediction_probs))  # Numerical stability
                prediction_probs = exp_probs / np.sum(exp_probs)
                log_debug(f"After softmax - probs: {prediction_probs}, sum: {np.sum(prediction_probs):.6f}")
            
            # Ensure probabilities are valid (all >= 0, sum = 1)
            if np.any(prediction_probs < 0):
                log_warn(f"Found negative probabilities, clipping to 0: {prediction_probs}")
                prediction_probs = np.maximum(prediction_probs, 0.0)
                prediction_probs = prediction_probs / np.sum(prediction_probs)  # Renormalize
        
        predicted_class_idx = np.argmax(prediction_probs)
        predicted_class = predicted_class_idx - 1  # Convert 0,1,2 to -1,0,1
        confidence = float(np.max(prediction_probs))
        
        # Additional validation: confidence should be reasonable
        if confidence <= 0.0 or confidence > 1.0:
            log_error(f"Invalid confidence value: {confidence}, prediction_probs: {prediction_probs}")
            return 'NEUTRAL'
        
        # If confidence is very low (e.g., uniform distribution), log warning
        if confidence < (1.0 / len(prediction_probs)):
            log_warn(f"Very low confidence ({confidence:.6f}), model output may be uniform or unconfident")
        
        # Debug: Log final prediction info
        log_debug(f"Predicted class index: {predicted_class_idx}, predicted_class: {predicted_class}")
        log_debug(f"Confidence: {confidence:.6f}, all probs: {prediction_probs}")
        
        # Determine model type
        model_type = "LSTM"
        if hasattr(model, '__class__'):
            class_name = model.__class__.__name__
            if 'Attention' in class_name:
                model_type = "LSTM-Attention"
            if 'CNN' in class_name:
                model_type = f"CNN-{model_type}"
        
        device_info = "GPU" if model_device.type in ('cuda', 'mps') else "CPU"
        log_model(f"{model_type} ({device_info}) - Class: {predicted_class}, Confidence: {confidence:.3f}, Probs: [{prediction_probs[0]:.3f}, {prediction_probs[1]:.3f}, {prediction_probs[2]:.3f}]")
        
        if confidence >= CONFIDENCE_THRESHOLD:
            if predicted_class == 1:
                log_model(f"HIGH CONFIDENCE BUY signal ({confidence:.1%})")
                return 'BUY'
            elif predicted_class == -1:
                log_model(f"HIGH CONFIDENCE SELL signal ({confidence:.1%})")
                return 'SELL'
            else:
                log_model(f"HIGH CONFIDENCE NEUTRAL signal ({confidence:.1%})")
                return 'NEUTRAL'
        else:
            log_model(f"LOW CONFIDENCE - Returning NEUTRAL ({confidence:.1%})")
            return 'NEUTRAL'
            
    except Exception as e:
        log_error(f"Error generating LSTM signal: {e}")
        return 'NEUTRAL'


# Backward compatibility aliases
def load_lstm_attention_model(model_path: Optional[Path] = None, use_attention: bool = True) -> Optional[nn.Module]:
    """
    Load LSTM model (backward compatibility).
    
    Args:
        model_path: Path to the model file
        use_attention: Ignored - determined from checkpoint
        
    Returns:
        Loaded model or None if failed
    """
    return load_lstm_model(model_path)


def get_latest_lstm_attention_signal(
    df_market_data: pd.DataFrame, 
    model: nn.Module, 
    scaler: Optional[MinMaxScaler] = None
) -> str:
    """
    Generate trading signal (backward compatibility).
    
    Args:
        df_market_data: DataFrame with OHLC market data
        model: Trained LSTM model
        scaler: Pre-fitted scaler (optional)
        
    Returns:
        Trading signal: 'BUY', 'SELL', or 'NEUTRAL'
    """
    return get_latest_signal(df_market_data, model, scaler)


def load_model_and_scaler(model_path: Optional[Path] = None) -> tuple[Optional[nn.Module], Optional[MinMaxScaler], Optional[int]]:
    """
    Load LSTM model and scaler from checkpoint.
    
    This is a convenience function that loads both the model and its associated
    scaler from a checkpoint file. It wraps `load_lstm_model()` and extracts
    the scaler and look_back from the checkpoint.
    
    Args:
        model_path: Path to model checkpoint (can be Path object or string). If None, uses default path.
        
    Returns:
        Tuple of (model, scaler, look_back) or (None, None, None) if failed
    """
    if model_path is None:
        model_path = MODELS_DIR / "lstm_model.pth"
    elif isinstance(model_path, str):
        # Convert string to Path object
        model_path = Path(model_path)
    
    if not model_path.exists():
        # Try to find the latest model file if default doesn't exist
        if model_path == MODELS_DIR / "lstm_model.pth":
            log_warn(f"Default model file not found: {model_path}")
            log_info("Searching for latest model file...")
            
            if MODELS_DIR.exists():
                model_files = list(MODELS_DIR.glob("*.pth"))
                if model_files:
                    # Sort by modification time, newest first
                    model_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    model_path = model_files[0]
                    log_info(f"Using latest model: {model_path.name}")
                else:
                    log_error(f"No model files found in {MODELS_DIR}")
                    return None, None, None
            else:
                log_error(f"Models directory does not exist: {MODELS_DIR}")
                return None, None, None
        else:
            # User-specified model path doesn't exist
            log_error(f"Model file not found: {model_path}")
            log_info(f"Available models in {MODELS_DIR}:")
            if MODELS_DIR.exists():
                model_files = list(MODELS_DIR.glob("*.pth"))
                if model_files:
                    # Sort by modification time for display
                    model_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    for f in model_files:
                        log_info(f"  - {f.name}")
                else:
                    log_info("  (no models found)")
            return None, None, None
    
    try:
        checkpoint = _load_checkpoint_safe(model_path)
        
        # Load model
        model = load_lstm_model(model_path)
        if model is None:
            return None, None, None
        
        # Load scaler and look_back from checkpoint
        scaler = checkpoint.get('scaler')
        look_back = checkpoint.get('model_config', {}).get('look_back', WINDOW_SIZE_LSTM)
        
        # Load Kalman Filter metadata from checkpoint
        data_info = checkpoint.get('data_info', {})
        use_kalman_filter = data_info.get('use_kalman_filter', False)
        kalman_params = data_info.get('kalman_params', None)
        
        if scaler is None:
            log_warn("Scaler not found in checkpoint. Predictions may be unreliable.")
        else:
            log_info(f"Loaded scaler from checkpoint")
        
        if use_kalman_filter:
            log_info(f"Model was trained with Kalman Filter preprocessing")
        
        log_info(f"Loaded model from {model_path}")
        log_info(f"Model look_back: {look_back}")
        
        # Store Kalman Filter info in model for inference
        if hasattr(model, '__dict__'):
            model._use_kalman_filter = use_kalman_filter
            model._kalman_params = kalman_params
        
        return model, scaler, look_back
        
    except Exception as e:
        log_error(f"Error loading model: {e}")
        import traceback
        log_error(f"Traceback: {traceback.format_exc()}")
        return None, None, None

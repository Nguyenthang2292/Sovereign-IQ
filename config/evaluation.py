"""
Evaluation Configuration.

Shared configuration constants for model evaluation across different models.
These constants are used by LSTM, Random Forest, and other models for
confidence-based signal generation and evaluation.
"""

# Confidence Thresholds
CONFIDENCE_THRESHOLD = 0.6  # Default confidence threshold for signal generation
CONFIDENCE_THRESHOLDS = [0.5, CONFIDENCE_THRESHOLD, 0.7, 0.8, 0.9]  # Thresholds for model evaluation


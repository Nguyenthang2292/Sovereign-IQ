"""
Auto Trade Configuration Defaults
"""

SIGNAL_SELECTOR_DEFAULTS = {
    "weight_xgboost": 0.4,
    "weight_gemini": 0.6,
    "min_confidence_threshold": 0.7,
}

XGBOOST_FILTER_DEFAULTS = {
    "min_confidence": 0.6,
    "history_limit": 1500,
    "prediction_timeframe": "5m",
    "on_error": "drop",
    "min_required_candles": 250,
    "cache_ttl": 300,  # 5 minutes cache expiration
    "require_model": True,  # Fail-fast if model doesn't load
    "max_consecutive_failures": 3,  # Circuit breaker threshold
    "prob_sum_tolerance": 0.01,  # ±1% tolerance for probability sum
    "min_confidence_delta": 0.05,  # Minimum delta between predictions
}

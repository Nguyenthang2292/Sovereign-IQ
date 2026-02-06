"""
Auto Trade Configuration Defaults
"""

SIGNAL_SELECTOR_DEFAULTS = {
    "weight_xgboost": 0.4,
    "weight_gemini": 0.6,
    "min_confidence_threshold": 0.7,
}

# Pre-trained XGBoost filter defaults (legacy mode)
XGBOOST_FILTER_DEFAULTS = {
    "min_confidence": 0.3,
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

# Per-symbol XGBoost filter defaults (new mode - trains fresh model per symbol)
XGBOOST_PER_SYMBOL_DEFAULTS = {
    "min_confidence": 0.55,  # Slightly higher threshold since we're training fresh
    "training_timeframe": "1h",  # Timeframe for training data
    "training_limit": 1500,  # Number of historical candles to fetch
    "min_required_candles": 200,  # Minimum candles required for training
    "on_error": "drop",  # Error handling policy: "drop" or "pass"
    "max_workers": 4,  # Max parallel workers for training
    "use_cache": False,  # Fresh training by default (no caching)
    "handle_class_imbalance": True,  # Use class weights to handle imbalance
    "skip_if_imbalanced": True,  # Skip symbols with >80% in one class
}

ATC_SCANNER_DEFAULTS = {
    "weights": {"1h": 0.5, "15m": 0.3, "5m": 0.2},
    "threshold": 0.6,
    "timeframes": ["1h", "15m", "5m"],
    "min_signal": 0.0,
    "use_signal_strength": False,
    "enable_cache": True,
    "cache_ttl_seconds": 60,
    "batch_size": 50,
    "use_rust_cache": True,  # Use Rust ScanCache for 10-20x performance improvement
}

"""
Test script for config - Configuration constants and values.
"""

import pytest
import config


def test_default_symbol():
    """Test DEFAULT_SYMBOL constant."""
    assert config.DEFAULT_SYMBOL == "BTC/USDT"


def test_default_quote():
    """Test DEFAULT_QUOTE constant."""
    assert config.DEFAULT_QUOTE == "USDT"


def test_default_timeframe():
    """Test DEFAULT_TIMEFRAME constant."""
    assert config.DEFAULT_TIMEFRAME == "15m"


def test_default_limit():
    """Test DEFAULT_LIMIT constant."""
    assert config.DEFAULT_LIMIT == 1500


def test_default_exchanges():
    """Test DEFAULT_EXCHANGES list."""
    assert isinstance(config.DEFAULT_EXCHANGES, list)
    assert len(config.DEFAULT_EXCHANGES) > 0
    assert "binance" in config.DEFAULT_EXCHANGES


def test_default_exchange_string():
    """Test DEFAULT_EXCHANGE_STRING is properly formatted."""
    assert isinstance(config.DEFAULT_EXCHANGE_STRING, str)
    assert "," in config.DEFAULT_EXCHANGE_STRING or len(config.DEFAULT_EXCHANGES) == 1


def test_prediction_windows():
    """Test PREDICTION_WINDOWS dictionary."""
    assert isinstance(config.PREDICTION_WINDOWS, dict)
    assert "1h" in config.PREDICTION_WINDOWS
    assert config.PREDICTION_WINDOWS["1h"] == "24h"


def test_target_horizon():
    """Test TARGET_HORIZON constant."""
    assert isinstance(config.TARGET_HORIZON, int)
    assert config.TARGET_HORIZON > 0


def test_target_labels():
    """Test TARGET_LABELS list."""
    assert isinstance(config.TARGET_LABELS, list)
    assert len(config.TARGET_LABELS) == 3
    assert "DOWN" in config.TARGET_LABELS
    assert "NEUTRAL" in config.TARGET_LABELS
    assert "UP" in config.TARGET_LABELS


def test_label_to_id():
    """Test LABEL_TO_ID mapping."""
    assert isinstance(config.LABEL_TO_ID, dict)
    assert config.LABEL_TO_ID["DOWN"] == 0
    assert config.LABEL_TO_ID["NEUTRAL"] == 1
    assert config.LABEL_TO_ID["UP"] == 2


def test_id_to_label():
    """Test ID_TO_LABEL mapping."""
    assert isinstance(config.ID_TO_LABEL, dict)
    assert config.ID_TO_LABEL[0] == "DOWN"
    assert config.ID_TO_LABEL[1] == "NEUTRAL"
    assert config.ID_TO_LABEL[2] == "UP"


def test_label_id_consistency():
    """Test LABEL_TO_ID and ID_TO_LABEL are inverse mappings."""
    for label, idx in config.LABEL_TO_ID.items():
        assert config.ID_TO_LABEL[idx] == label
    for idx, label in config.ID_TO_LABEL.items():
        assert config.LABEL_TO_ID[label] == idx


def test_model_features():
    """Test MODEL_FEATURES list."""
    assert isinstance(config.MODEL_FEATURES, list)
    assert len(config.MODEL_FEATURES) > 0
    assert "close" in config.MODEL_FEATURES
    assert "volume" in config.MODEL_FEATURES


def test_xgboost_params():
    """Test XGBOOST_PARAMS dictionary."""
    assert isinstance(config.XGBOOST_PARAMS, dict)
    assert "n_estimators" in config.XGBOOST_PARAMS
    assert "learning_rate" in config.XGBOOST_PARAMS
    assert "max_depth" in config.XGBOOST_PARAMS
    assert config.XGBOOST_PARAMS["random_state"] == 42


def test_deep_learning_config():
    """Test deep learning configuration constants."""
    assert hasattr(config, "DEEP_MAX_ENCODER_LENGTH")
    assert hasattr(config, "DEEP_MAX_PREDICTION_LENGTH")
    assert hasattr(config, "DEEP_BATCH_SIZE")
    assert hasattr(config, "DEEP_MODEL_HIDDEN_SIZE")
    assert hasattr(config, "DEEP_MODEL_LEARNING_RATE")
    
    # Test values are reasonable
    assert config.DEEP_MAX_ENCODER_LENGTH > 0
    assert config.DEEP_MAX_PREDICTION_LENGTH > 0
    assert config.DEEP_BATCH_SIZE > 0
    assert 0 < config.DEEP_MODEL_DROPOUT < 1
    assert config.DEEP_MODEL_LEARNING_RATE > 0


def test_portfolio_config():
    """Test portfolio manager configuration constants."""
    assert hasattr(config, "BENCHMARK_SYMBOL")
    assert hasattr(config, "DEFAULT_REQUEST_PAUSE")
    assert hasattr(config, "DEFAULT_VAR_CONFIDENCE")
    
    assert config.DEFAULT_REQUEST_PAUSE > 0
    assert 0 < config.DEFAULT_VAR_CONFIDENCE < 1


def test_data_split_ratios():
    """Test data split ratios sum to 1.0."""
    train = config.DEEP_TRAIN_RATIO
    val = config.DEEP_VAL_RATIO
    test = config.DEEP_TEST_RATIO
    
    assert abs((train + val + test) - 1.0) < 0.001
    assert train > 0
    assert val > 0
    assert test > 0


def test_feature_selection_config():
    """Test feature selection configuration constants."""
    assert hasattr(config, "DEEP_FEATURE_SELECTION_METHOD")
    assert hasattr(config, "DEEP_FEATURE_SELECTION_TOP_K")
    assert hasattr(config, "DEEP_FEATURE_COLLINEARITY_THRESHOLD")
    
    assert config.DEEP_FEATURE_SELECTION_METHOD in ["mutual_info", "boruta", "f_test", "combined"]
    assert config.DEEP_FEATURE_SELECTION_TOP_K > 0
    assert 0 < config.DEEP_FEATURE_COLLINEARITY_THRESHOLD < 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


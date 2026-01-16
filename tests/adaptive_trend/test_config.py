"""
Tests for utils/config module.
"""

from modules.adaptive_trend.utils.config import ATCConfig, create_atc_config_from_dict


def test_atc_config_defaults():
    """Test that ATCConfig has correct default values."""
    config = ATCConfig()

    assert config.ema_len == 28
    assert config.hma_len == 28
    assert config.wma_len == 28
    assert config.dema_len == 28
    assert config.lsma_len == 28
    assert config.kama_len == 28
    assert config.ema_w == 1.0
    assert config.hma_w == 1.0
    assert config.wma_w == 1.0
    assert config.dema_w == 1.0
    assert config.lsma_w == 1.0
    assert config.kama_w == 1.0
    assert config.robustness == "Medium"
    assert config.lambda_param == 0.02
    assert config.decay == 0.03
    assert config.cutout == 0
    assert config.long_threshold == 0.1
    assert config.short_threshold == -0.1
    assert config.calculation_source == "close"
    assert config.limit == 1500
    assert config.timeframe == "15m"


def test_atc_config_custom_values():
    """Test that ATCConfig accepts custom values."""
    config = ATCConfig(
        ema_len=14,
        hma_len=21,
        wma_len=28,
        dema_len=35,
        lsma_len=42,
        kama_len=50,
        ema_w=2.0,
        hma_w=1.5,
        wma_w=1.0,
        dema_w=0.5,
        lsma_w=1.0,
        kama_w=1.0,
        robustness="Narrow",
        lambda_param=0.01,
        decay=0.05,
        cutout=10,
        long_threshold=0.2,
        short_threshold=-0.2,
        calculation_source="open",
        limit=2000,
        timeframe="1h",
    )

    assert config.ema_len == 14
    assert config.hma_len == 21
    assert config.wma_len == 28
    assert config.dema_len == 35
    assert config.lsma_len == 42
    assert config.kama_len == 50
    assert config.ema_w == 2.0
    assert config.hma_w == 1.5
    assert config.wma_w == 1.0
    assert config.dema_w == 0.5
    assert config.lsma_w == 1.0
    assert config.kama_w == 1.0
    assert config.robustness == "Narrow"
    assert config.lambda_param == 0.01
    assert config.decay == 0.05
    assert config.cutout == 10
    assert config.long_threshold == 0.2
    assert config.short_threshold == -0.2
    assert config.calculation_source == "open"
    assert config.limit == 2000
    assert config.timeframe == "1h"


def test_atc_config_robustness_values():
    """Test that ATCConfig accepts different robustness values."""
    config_narrow = ATCConfig(robustness="Narrow")
    config_medium = ATCConfig(robustness="Medium")
    config_wide = ATCConfig(robustness="Wide")

    assert config_narrow.robustness == "Narrow"
    assert config_medium.robustness == "Medium"
    assert config_wide.robustness == "Wide"


def test_create_atc_config_from_dict_full():
    """Test create_atc_config_from_dict with all parameters."""
    params = {
        "limit": 2000,
        "ema_len": 14,
        "hma_len": 21,
        "wma_len": 28,
        "dema_len": 35,
        "lsma_len": 42,
        "kama_len": 50,
        "ema_w": 2.0,
        "hma_w": 1.5,
        "wma_w": 1.0,
        "dema_w": 0.5,
        "lsma_w": 1.0,
        "kama_w": 1.0,
        "robustness": "Wide",
        "lambda_param": 0.01,
        "decay": 0.05,
        "cutout": 10,
        "long_threshold": 0.15,
        "short_threshold": -0.15,
        "calculation_source": "high",
    }
    timeframe = "4h"

    config = create_atc_config_from_dict(params, timeframe=timeframe)

    assert isinstance(config, ATCConfig)
    assert config.timeframe == "4h"
    assert config.limit == 2000
    assert config.ema_len == 14
    assert config.hma_len == 21
    assert config.wma_len == 28
    assert config.dema_len == 35
    assert config.lsma_len == 42
    assert config.kama_len == 50
    assert config.ema_w == 2.0
    assert config.hma_w == 1.5
    assert config.wma_w == 1.0
    assert config.dema_w == 0.5
    assert config.lsma_w == 1.0
    assert config.kama_w == 1.0
    assert config.robustness == "Wide"
    assert config.lambda_param == 0.01
    assert config.decay == 0.05
    assert config.cutout == 10
    assert config.long_threshold == 0.15
    assert config.short_threshold == -0.15
    assert config.calculation_source == "high"


def test_create_atc_config_from_dict_partial():
    """Test create_atc_config_from_dict with partial parameters."""
    params = {
        "ema_len": 14,
        "robustness": "Narrow",
    }

    config = create_atc_config_from_dict(params)

    assert isinstance(config, ATCConfig)
    assert config.ema_len == 14
    assert config.robustness == "Narrow"
    # Should use defaults for missing parameters
    assert config.hma_len == 28
    assert config.limit == 1500
    assert config.timeframe == "15m"  # Default timeframe
    assert config.lambda_param == 0.02
    assert config.decay == 0.03


def test_create_atc_config_from_dict_empty():
    """Test create_atc_config_from_dict with empty dict."""
    params = {}

    config = create_atc_config_from_dict(params)

    assert isinstance(config, ATCConfig)
    # Should use all defaults
    assert config.ema_len == 28
    assert config.robustness == "Medium"
    assert config.limit == 1500
    assert config.timeframe == "15m"


def test_create_atc_config_from_dict_custom_timeframe():
    """Test create_atc_config_from_dict with custom timeframe."""
    params = {"ema_len": 14}

    config = create_atc_config_from_dict(params, timeframe="1d")

    assert config.timeframe == "1d"
    assert config.ema_len == 14


def test_create_atc_config_from_dict_default_timeframe():
    """Test create_atc_config_from_dict uses default timeframe when not specified."""
    params = {"ema_len": 14}

    config = create_atc_config_from_dict(params)

    assert config.timeframe == "15m"  # Default


def test_atc_config_immutability():
    """Test that ATCConfig is a dataclass (can be modified but structure is fixed)."""
    config = ATCConfig()

    # Dataclass allows attribute modification
    config.ema_len = 14
    assert config.ema_len == 14

    # But structure is defined
    assert hasattr(config, "ema_len")
    assert hasattr(config, "robustness")
    assert hasattr(config, "lambda_param")


def test_atc_config_all_parameters():
    """Test that ATCConfig has all expected parameters."""
    config = ATCConfig()

    expected_params = [
        "ema_len",
        "hma_len",
        "wma_len",
        "dema_len",
        "lsma_len",
        "kama_len",
        "ema_w",
        "hma_w",
        "wma_w",
        "dema_w",
        "lsma_w",
        "kama_w",
        "robustness",
        "lambda_param",
        "decay",
        "cutout",
        "long_threshold",
        "short_threshold",
        "calculation_source",
        "limit",
        "timeframe",
    ]

    for param in expected_params:
        assert hasattr(config, param), f"Missing parameter: {param}"


def test_atc_config_weights_defaults():
    """Test that ATCConfig has correct default weights."""
    config = ATCConfig()

    assert config.ema_w == 1.0
    assert config.hma_w == 1.0
    assert config.wma_w == 1.0
    assert config.dema_w == 1.0
    assert config.lsma_w == 1.0
    assert config.kama_w == 1.0


def test_atc_config_thresholds_defaults():
    """Test that ATCConfig has correct default thresholds."""
    config = ATCConfig()

    assert config.long_threshold == 0.1
    assert config.short_threshold == -0.1


def test_atc_config_calculation_source_defaults():
    """Test that ATCConfig has correct default calculation_source."""
    config = ATCConfig()

    assert config.calculation_source == "close"


def test_atc_config_calculation_source_values():
    """Test that ATCConfig accepts different calculation_source values."""
    config_close = ATCConfig(calculation_source="close")
    config_open = ATCConfig(calculation_source="open")
    config_high = ATCConfig(calculation_source="high")
    config_low = ATCConfig(calculation_source="low")

    assert config_close.calculation_source == "close"
    assert config_open.calculation_source == "open"
    assert config_high.calculation_source == "high"
    assert config_low.calculation_source == "low"

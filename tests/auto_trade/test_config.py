"""
Unit Tests for Configuration Management
========================================

Tests configuration creation, validation, export/import.

Run: pytest tests/auto_trade/test_config.py -v
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.auto_trade.config import (
    AutoTradeConfig,
    MartingaleConfig,
    RiskConfig,
    ScanningConfig,
    get_aggressive_config,
    get_config,
    get_conservative_config,
    get_testnet_config,
    load_config,
    save_config,
)


class TestConfigCreation:
    """Test configuration creation."""

    def test_default_config(self):
        """Test creating default configuration."""
        config = AutoTradeConfig()

        assert config is not None
        assert config.risk.leverage == 2
        assert config.martingale.enabled is True
        assert config.scanning.scan_interval == 300

    def test_get_config_singleton(self):
        """Test singleton pattern."""
        config1 = get_config()
        config2 = get_config()

        # Should be same instance (singleton)
        assert config1 is config2


class TestConfigValidation:
    """Test configuration validation."""

    def test_valid_config(self):
        """Test that valid config passes validation."""
        config = AutoTradeConfig()
        # Should not raise
        config.validate()

    def test_invalid_leverage(self):
        """Test validation fails for invalid leverage."""
        config = AutoTradeConfig()
        config.risk.leverage = 200  # Too high

        with pytest.raises(ValueError, match="Leverage must be between"):
            config.validate()

    def test_invalid_scan_interval(self):
        """Test validation fails for invalid scan interval."""
        config = AutoTradeConfig()
        config.scanning.scan_interval = 30  # Too low

        with pytest.raises(ValueError, match="Scan interval must be at least"):
            config.validate()

    def test_invalid_martingale_steps(self):
        """Test validation fails for invalid Martingale steps."""
        config = AutoTradeConfig()
        config.martingale.max_steps = 15  # Too high

        with pytest.raises(ValueError, match="Martingale max steps"):
            config.validate()

    def test_invalid_position_size(self):
        """Test validation fails for invalid position size."""
        config = AutoTradeConfig()
        config.risk.position_size_percent = 1.5  # > 100%

        with pytest.raises(ValueError, match="Position size percent"):
            config.validate()


class TestConfigExportImport:
    """Test configuration export and import."""

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = AutoTradeConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "risk" in config_dict
        assert "martingale" in config_dict
        assert "scanning" in config_dict

    def test_to_json(self, tmp_path):
        """Test exporting config to JSON."""
        config = AutoTradeConfig()
        json_path = tmp_path / "test_config.json"

        json_str = config.to_json(str(json_path))

        assert Path(json_path).exists()
        assert isinstance(json_str, str)

        # Should be valid JSON
        data = json.loads(json_str)
        assert "risk" in data

    def test_from_json(self, tmp_path):
        """Test importing config from JSON."""
        # Create config and export
        config1 = AutoTradeConfig()
        config1.risk.max_open_positions = 5
        json_path = tmp_path / "test_config.json"
        config1.to_json(str(json_path))

        # Import
        config2 = AutoTradeConfig.from_json(str(json_path))

        assert config2.risk.max_open_positions == 5

    def test_load_config_from_file(self, tmp_path):
        """Test load_config with file path."""
        # Create config file
        config = AutoTradeConfig()
        config.risk.leverage = 3
        json_path = tmp_path / "config.json"
        config.to_json(str(json_path))

        # Load
        loaded = load_config(str(json_path))

        assert loaded.risk.leverage == 3


class TestPresetConfigs:
    """Test preset configuration templates."""

    def test_conservative_config(self):
        """Test conservative config preset."""
        config = get_conservative_config()

        assert config.risk.leverage == 1
        assert config.risk.max_open_positions == 1
        assert config.martingale.enabled is False

    def test_aggressive_config(self):
        """Test aggressive config preset."""
        config = get_aggressive_config()

        assert config.risk.leverage == 3
        assert config.risk.max_open_positions == 5
        assert config.martingale.max_steps == 4

    def test_testnet_config(self):
        """Test testnet config preset."""
        config = get_testnet_config()

        assert config.binance.testnet is True
        assert config.dry_run is True
        assert "test" in config.database.path


class TestConfigSummary:
    """Test configuration summary generation."""

    def test_get_summary(self):
        """Test getting configuration summary."""
        config = AutoTradeConfig()
        summary = config.get_summary()

        assert isinstance(summary, str)
        assert "RISK MANAGEMENT" in summary
        assert "MARTINGALE" in summary
        assert "SCANNING" in summary


class TestSubConfigs:
    """Test sub-configuration classes."""

    def test_scanning_config(self):
        """Test ScanningConfig."""
        scanning = ScanningConfig()

        assert scanning.scan_interval == 300
        assert scanning.symbol_sample_percentage == 1.0
        assert "ATC" in scanning.enabled_scanners

    def test_risk_config(self):
        """Test RiskConfig."""
        risk = RiskConfig()

        assert risk.leverage == 2
        assert risk.position_size_percent == 0.95
        assert risk.max_open_positions == 3

    def test_martingale_config(self):
        """Test MartingaleConfig."""
        martingale = MartingaleConfig()

        assert martingale.enabled is True
        assert martingale.max_steps == 3
        assert martingale.multiplier == 2.0


class TestConfigModification:
    """Test modifying configuration values."""

    def test_modify_risk_params(self):
        """Test modifying risk parameters."""
        config = AutoTradeConfig()
        config.risk.leverage = 5
        config.risk.max_open_positions = 10

        # Should pass validation
        config.validate()

        assert config.risk.leverage == 5
        assert config.risk.max_open_positions == 10

    def test_modify_scanning_params(self):
        """Test modifying scanning parameters."""
        config = AutoTradeConfig()
        config.scanning.scan_interval = 600
        config.scanning.symbol_sample_percentage = 0.5

        # Should pass validation
        config.validate()

        assert config.scanning.scan_interval == 600
        assert config.scanning.symbol_sample_percentage == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

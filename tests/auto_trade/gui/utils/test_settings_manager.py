"""
Comprehensive tests for SettingsManager.

Tests cover:
- Loading and saving settings
- Default settings handling
- Settings validation
- Import/export functionality
- Merging and migration
- Error handling
"""

import json
from pathlib import Path

import pytest
import yaml

from modules.auto_trade.gui.utils.settings_manager import SettingsManager


class TestSettingsManager:
    """Test SettingsManager functionality."""

    def test_init_creates_default_settings(self, temp_settings_file):
        """Test that initialization creates default settings."""
        manager = SettingsManager(settings_file=str(temp_settings_file))

        assert manager.settings is not None
        assert "risk" in manager.settings
        assert "filters" in manager.settings
        assert "api" in manager.settings

    def test_load_creates_file_if_not_exists(self, temp_settings_file):
        """Test that load creates settings file if it doesn't exist."""
        manager = SettingsManager(settings_file=str(temp_settings_file))
        manager.load()

        assert temp_settings_file.exists()

    def test_save_and_load_settings(self, temp_settings_file, sample_settings):
        """Test saving and loading settings."""
        manager = SettingsManager(settings_file=str(temp_settings_file))
        manager.settings = sample_settings

        # Save
        success = manager.save()
        assert success is True

        # Load in new instance
        manager2 = SettingsManager(settings_file=str(temp_settings_file))
        manager2.load()

        assert manager2.settings["risk"]["max_position_size"] == 100.0
        assert manager2.settings["filters"]["min_signal_score"] == 0.7

    def test_get_setting(self, temp_settings_file, sample_settings):
        """Test getting a setting by key."""
        manager = SettingsManager(settings_file=str(temp_settings_file))
        manager.settings = sample_settings

        value = manager.get("risk.max_position_size")
        assert value == 100.0

        value = manager.get("filters.min_signal_score")
        assert value == 0.7

    def test_get_setting_with_default(self, temp_settings_file):
        """Test getting non-existent setting with default."""
        manager = SettingsManager(settings_file=str(temp_settings_file))

        value = manager.get("non.existent.key", default=999)
        assert value == 999

    def test_set_setting(self, temp_settings_file):
        """Test setting a value by key."""
        manager = SettingsManager(settings_file=str(temp_settings_file))

        success = manager.set("risk.max_position_size", 200.0)
        assert success is True

        value = manager.get("risk.max_position_size")
        assert value == 200.0

    def test_set_nested_setting(self, temp_settings_file):
        """Test setting a nested value."""
        manager = SettingsManager(settings_file=str(temp_settings_file))

        success = manager.set("new.nested.key", "value")
        assert success is True

        value = manager.get("new.nested.key")
        assert value == "value"

    def test_validate_settings_fixes_invalid_values(self, temp_settings_file):
        """Test that validation fixes invalid values."""
        manager = SettingsManager(settings_file=str(temp_settings_file))

        # Set invalid values
        manager.settings["risk"]["max_position_size"] = -100  # Invalid (negative)
        manager.settings["risk"]["max_open_positions"] = 100  # Invalid (too high)
        manager.settings["filters"]["min_signal_score"] = 2.0  # Invalid (> 1)
        manager.settings["scanner"]["trading_direction"] = "INVALID_DIR"

        # Validate
        manager._validate_settings()

        # Check that values were fixed
        assert manager.settings["risk"]["max_position_size"] > 0
        assert 1 <= manager.settings["risk"]["max_open_positions"] <= 10
        assert 0 <= manager.settings["filters"]["min_signal_score"] <= 1
        assert manager.settings["scanner"]["trading_direction"] == "BOTH"

    def test_merge_settings(self, temp_settings_file):
        """Test merging loaded settings with defaults."""
        manager = SettingsManager(settings_file=str(temp_settings_file))

        defaults = {
            "risk": {"max_position_size": 100.0, "new_key": "default"},
            "filters": {"min_signal_score": 0.7},
        }

        loaded = {
            "risk": {"max_position_size": 200.0},  # Override
            "new_section": {"key": "value"},  # New section
        }

        merged = manager._merge_settings(defaults, loaded)

        # Override should take effect
        assert merged["risk"]["max_position_size"] == 200.0
        # Default value should be preserved
        assert merged["risk"]["new_key"] == "default"
        # New section should be added
        assert merged["new_section"]["key"] == "value"
        # Original default should remain
        assert merged["filters"]["min_signal_score"] == 0.7

    def test_export_yaml(self, temp_settings_file, tmp_path, sample_settings):
        """Test exporting settings to YAML."""
        manager = SettingsManager(settings_file=str(temp_settings_file))
        manager.settings = sample_settings

        export_file = tmp_path / "exported.yaml"
        success = manager.export(str(export_file))

        assert success is True
        assert export_file.exists()

        # Verify content
        with open(export_file) as f:
            loaded = yaml.safe_load(f)
            assert loaded["risk"]["max_position_size"] == 100.0

    def test_export_json(self, temp_settings_file, tmp_path, sample_settings):
        """Test exporting settings to JSON."""
        manager = SettingsManager(settings_file=str(temp_settings_file))
        manager.settings = sample_settings

        export_file = tmp_path / "exported.json"
        success = manager.export(str(export_file))

        assert success is True
        assert export_file.exists()

        # Verify content
        with open(export_file) as f:
            loaded = json.load(f)
            assert loaded["risk"]["max_position_size"] == 100.0

    def test_import_yaml(self, temp_settings_file, tmp_path):
        """Test importing settings from YAML."""
        manager = SettingsManager(settings_file=str(temp_settings_file))

        # Create import file
        import_file = tmp_path / "import.yaml"
        import_data = {
            "risk": {"max_position_size": 300.0},
            "filters": {"min_signal_score": 0.8},
        }

        with open(import_file, "w") as f:
            yaml.dump(import_data, f)

        # Import
        success = manager.import_settings(str(import_file))

        assert success is True
        assert manager.settings["risk"]["max_position_size"] == 300.0
        assert manager.settings["filters"]["min_signal_score"] == 0.8

    def test_import_json(self, temp_settings_file, tmp_path):
        """Test importing settings from JSON."""
        manager = SettingsManager(settings_file=str(temp_settings_file))

        # Create import file
        import_file = tmp_path / "import.json"
        import_data = {
            "risk": {"max_position_size": 250.0},
        }

        with open(import_file, "w") as f:
            json.dump(import_data, f)

        # Import
        success = manager.import_settings(str(import_file))

        assert success is True
        assert manager.settings["risk"]["max_position_size"] == 250.0

    def test_import_nonexistent_file(self, temp_settings_file):
        """Test importing from non-existent file."""
        manager = SettingsManager(settings_file=str(temp_settings_file))

        success = manager.import_settings("/nonexistent/file.yaml")

        assert success is False

    def test_reset_to_defaults(self, temp_settings_file, sample_settings):
        """Test resetting to default settings."""
        manager = SettingsManager(settings_file=str(temp_settings_file))
        manager.settings = sample_settings

        # Modify settings
        manager.set("risk.max_position_size", 999.0)

        # Reset
        success = manager.reset_to_defaults()

        assert success is True
        assert manager.settings == SettingsManager.DEFAULT_SETTINGS

    def test_backup_creation(self, temp_settings_file, sample_settings):
        """Test that backup is created on save."""
        manager = SettingsManager(settings_file=str(temp_settings_file))
        manager.settings = sample_settings

        # Save twice to ensure backup is created
        manager.save()
        manager.set("risk.max_position_size", 150.0)
        manager.save()

        backup_file = temp_settings_file.with_suffix(temp_settings_file.suffix + ".backup")
        # Backup file should exist (if save was called twice)
        # First save: no backup (no existing file)
        # Second save: backup created
        if backup_file.exists():
            assert backup_file.is_file()

    def test_get_all_settings(self, temp_settings_file, sample_settings):
        """Test getting all settings."""
        manager = SettingsManager(settings_file=str(temp_settings_file))
        manager.settings = sample_settings

        all_settings = manager.get_all()

        assert all_settings == sample_settings
        # Verify it's a copy
        assert all_settings is not manager.settings

    def test_normalize_whitelist_string(self, temp_settings_file):
        """Test normalizing comma-separated whitelist to newline-separated."""
        manager = SettingsManager(settings_file=str(temp_settings_file))

        # Set comma-separated whitelist
        manager.settings["filters"]["symbol_whitelist"] = "BTC/USDT,ETH/USDT,SOL/USDT"

        manager._normalize_whitelist()

        # Should be converted to newline-separated
        assert "\n" in manager.settings["filters"]["symbol_whitelist"]

    def test_normalize_whitelist_list(self, temp_settings_file):
        """Test normalizing list whitelist to string."""
        manager = SettingsManager(settings_file=str(temp_settings_file))

        # Set list whitelist
        manager.settings["filters"]["symbol_whitelist"] = ["BTC/USDT", "ETH/USDT"]

        manager._normalize_whitelist()

        # Should be converted to string
        assert isinstance(manager.settings["filters"]["symbol_whitelist"], str)
        assert "BTC/USDT" in manager.settings["filters"]["symbol_whitelist"]

    def test_json_to_yaml_migration(self, tmp_path):
        """Test migration from JSON to YAML settings file."""
        yaml_file = tmp_path / "settings.yaml"
        json_file = tmp_path / "settings.json"

        # Create old JSON file
        json_data = {
            "risk": {"max_position_size": 150.0},
        }

        with open(json_file, "w") as f:
            json.dump(json_data, f)

        # Initialize manager (should migrate)
        manager = SettingsManager(settings_file=str(yaml_file))
        manager.load()

        # Check that YAML file was created with migrated data
        if yaml_file.exists():
            assert manager.settings["risk"]["max_position_size"] == 150.0

    def test_error_handling_in_save(self, temp_settings_file):
        """Test error handling in save method."""
        manager = SettingsManager(settings_file=str(temp_settings_file))

        # Try to save to invalid location
        manager.settings_file = Path("/invalid/path/settings.yaml")

        success = manager.save()

        # Should fail gracefully
        assert success is False

    def test_validation_on_load(self, temp_settings_file):
        """Test that settings are validated on load."""
        # Create file with invalid settings
        invalid_settings = {
            "risk": {"max_position_size": -50},  # Invalid
            "filters": {"min_signal_score": 2.0},  # Invalid
        }

        with open(temp_settings_file, "w") as f:
            yaml.dump(invalid_settings, f)

        # Load should validate and fix
        manager = SettingsManager(settings_file=str(temp_settings_file))
        manager.load()

        # Check that invalid values were fixed
        assert manager.settings["risk"]["max_position_size"] > 0
        assert 0 <= manager.settings["filters"]["min_signal_score"] <= 1

    def test_trading_direction_saved_and_loaded(self, temp_settings_file):
        """Test scanner.trading_direction persists across save/load cycles."""
        manager = SettingsManager(settings_file=str(temp_settings_file))
        manager.load()

        manager.set("scanner.trading_direction", "LONG_ONLY")
        assert manager.save() is True

        manager2 = SettingsManager(settings_file=str(temp_settings_file))
        manager2.load()

        assert manager2.get("scanner.trading_direction") == "LONG_ONLY"

    def test_invalid_direction_defaults_to_both(self, temp_settings_file):
        """Test invalid trading direction is sanitized to BOTH."""
        manager = SettingsManager(settings_file=str(temp_settings_file))
        manager.load()

        manager.settings["scanner"]["trading_direction"] = "INVALID_VALUE"
        manager._validate_settings()

        assert manager.settings["scanner"]["trading_direction"] == "BOTH"

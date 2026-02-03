import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


class SettingsManager:
    """
    Manage application settings persistence
    Load/save settings from/to JSON file
    """

    DEFAULT_SETTINGS = {
        "risk": {
            "max_position_size": 100.0,
            "max_open_positions": 3,
            "max_daily_loss": 50.0,
            "default_leverage": "10x",
        },
        "filters": {
            "min_signal_score": 0.7,
            "enable_xgboost": True,
            "symbol_whitelist": "BTC/USDT\nETH/USDT\nSOL/USDT",
            "min_volume": 50.0,
            "timeframe": "1h",
        },
        "api": {"exchange": "Demo", "api_key": "", "api_secret": ""},
        "tp_sl": {"default_tp": 5.0, "default_sl": 2.5, "trailing_stop": False, "mode": "Percentage"},
        "scanner": {
            "scan_interval": 5,
            "timeframe": "1h",
            "symbol_list": "Top 20",
            "auto_start": True,
            "running": False,
        },
        "ui": {
            "theme": "dark",
            "font_size": 12,
            "window_size": {"width": 1200, "height": 800},
            "last_active_tab": "Dashboard",
            "column_visibility": {},
            "widget_order": {},
        },
    }

    def __init__(self, settings_file: str = None):
        """
        Initialize SettingsManager

        Args:
            settings_file: Path to settings JSON file. If None, uses default path
        """
        if settings_file is None:
            # Default settings file in project root
            self.settings_file = Path(__file__).parent.parent.parent / "settings.json"
        else:
            self.settings_file = Path(settings_file)

        self.settings: Dict = self.DEFAULT_SETTINGS.copy()
        self._ensure_settings_directory()

    def _ensure_settings_directory(self):
        """Ensure settings directory exists"""
        self.settings_file.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> Dict:
        """
        Load settings from file

        Returns:
            Dictionary containing all settings
        """
        try:
            if self.settings_file.exists():
                with open(self.settings_file, "r", encoding="utf-8") as f:
                    loaded_settings = json.load(f)

                # Merge with defaults to handle new/missing keys
                self.settings = self._merge_settings(self.DEFAULT_SETTINGS, loaded_settings)

                # Validate settings
                self._validate_settings()

                print(f"Settings loaded from {self.settings_file}")
            else:
                print(f"No settings file found, using defaults")
                self.save()  # Create default settings file
        except Exception as e:
            print(f"Error loading settings: {e}, using defaults")
            self.settings = self.DEFAULT_SETTINGS.copy()

        return self.settings

    def save(self) -> bool:
        """
        Save current settings to file

        Returns:
            True if save successful, False otherwise
        """
        try:
            # Validate before saving
            self._validate_settings()

            # Create backup
            self._create_backup()

            # Save to file
            with open(self.settings_file, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)

            print(f"Settings saved to {self.settings_file}")
            return True
        except Exception as e:
            print(f"Error saving settings: {e}")
            return False

    def _merge_settings(self, defaults: Dict, loaded: Dict) -> Dict:
        """
        Merge loaded settings with defaults

        Args:
            defaults: Default settings dictionary
            loaded: Loaded settings dictionary

        Returns:
            Merged settings dictionary
        """
        merged = defaults.copy()

        for key, value in loaded.items():
            if key in merged:
                if isinstance(value, dict) and isinstance(merged[key], dict):
                    # Recursively merge nested dictionaries
                    merged[key] = self._merge_settings(merged[key], value)
                else:
                    merged[key] = value
            else:
                # New key in loaded settings
                merged[key] = value

        return merged

    def _validate_settings(self):
        """Validate settings and fix invalid values"""
        try:
            # Validate risk settings
            if self.settings["risk"]["max_position_size"] <= 0:
                self.settings["risk"]["max_position_size"] = 100.0

            if self.settings["risk"]["max_open_positions"] < 1 or self.settings["risk"]["max_open_positions"] > 10:
                self.settings["risk"]["max_open_positions"] = 3

            if self.settings["risk"]["max_daily_loss"] < 0:
                self.settings["risk"]["max_daily_loss"] = 50.0

            # Validate filters
            if not 0 <= self.settings["filters"]["min_signal_score"] <= 1:
                self.settings["filters"]["min_signal_score"] = 0.7

            if self.settings["filters"]["min_volume"] < 0:
                self.settings["filters"]["min_volume"] = 50.0

            # Validate TP/SL
            if self.settings["tp_sl"]["default_tp"] <= 0:
                self.settings["tp_sl"]["default_tp"] = 5.0

            if self.settings["tp_sl"]["default_sl"] <= 0:
                self.settings["tp_sl"]["default_sl"] = 2.5

            # Validate scanner
            if self.settings["scanner"]["scan_interval"] < 1 or self.settings["scanner"]["scan_interval"] > 60:
                self.settings["scanner"]["scan_interval"] = 5

        except Exception as e:
            print(f"Settings validation error: {e}, using defaults")
            self.settings = self.DEFAULT_SETTINGS.copy()

    def _create_backup(self):
        """Create backup of current settings file"""
        try:
            if self.settings_file.exists():
                backup_file = self.settings_file.with_suffix(".json.backup")
                with open(self.settings_file, "r", encoding="utf-8") as src:
                    with open(backup_file, "w", encoding="utf-8") as dst:
                        dst.write(src.read())
                print(f"Backup created at {backup_file}")
        except Exception as e:
            print(f"Error creating backup: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a setting value by key

        Args:
            key: Setting key (supports dot notation e.g., 'risk.max_position_size')
            default: Default value if key not found

        Returns:
            Setting value or default
        """
        keys = key.split(".")
        value = self.settings

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value if default is None or value is not None else default

    def set(self, key: str, value: Any) -> bool:
        """
        Set a setting value by key

        Args:
            key: Setting key (supports dot notation e.g., 'risk.max_position_size')
            value: Value to set

        Returns:
            True if successful, False otherwise
        """
        try:
            keys = key.split(".")
            current = self.settings

            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]

            current[keys[-1]] = value
            return True
        except Exception as e:
            print(f"Error setting {key}: {e}")
            return False

    def export(self, export_path: str) -> bool:
        """
        Export settings to a file

        Args:
            export_path: Path to export file

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
            print(f"Settings exported to {export_path}")
            return True
        except Exception as e:
            print(f"Error exporting settings: {e}")
            return False

    def import_settings(self, import_path: str) -> bool:
        """
        Import settings from a file

        Args:
            import_path: Path to import file

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(import_path, "r", encoding="utf-8") as f:
                imported_settings = json.load(f)

            # Merge with defaults
            self.settings = self._merge_settings(self.DEFAULT_SETTINGS, imported_settings)

            # Validate imported settings
            self._validate_settings()

            # Save
            self.save()

            print(f"Settings imported from {import_path}")
            return True
        except Exception as e:
            print(f"Error importing settings: {e}")
            return False

    def reset_to_defaults(self) -> bool:
        """
        Reset all settings to default values

        Returns:
            True if successful, False otherwise
        """
        try:
            self.settings = self.DEFAULT_SETTINGS.copy()
            self.save()
            print("Settings reset to defaults")
            return True
        except Exception as e:
            print(f"Error resetting settings: {e}")
            return False

    def get_all(self) -> Dict:
        """Get all settings"""
        return self.settings.copy()

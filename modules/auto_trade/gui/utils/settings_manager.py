import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class SettingsManager:
    """
    Manage application settings persistence.
    Load/save settings from/to YAML file (settings.yaml).
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
        "api": {"exchange": "Demo", "mode": "DRY_RUN", "api_key": "", "api_secret": ""},
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

    def __init__(self, settings_file: Optional[str] = None):
        """
        Initialize SettingsManager.

        Args:
            settings_file: Path to settings YAML file. If None, uses settings.yaml in auto_trade dir.
        """
        if settings_file is None:
            self.settings_file = Path(__file__).parent.parent.parent / "settings.yaml"
        else:
            self.settings_file = Path(settings_file)

        self.settings: Dict = self.DEFAULT_SETTINGS.copy()
        self._ensure_settings_directory()

    def _ensure_settings_directory(self):
        """Ensure settings directory exists"""
        self.settings_file.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> Dict:
        """
        Load settings from file (YAML, or JSON once for migration).

        Returns:
            Dictionary containing all settings
        """
        try:
            if self.settings_file.exists():
                loaded_settings = self._load_file(self.settings_file)
                if loaded_settings is not None:
                    self.settings = self._merge_settings(self.DEFAULT_SETTINGS, loaded_settings)
                    self._validate_settings()
                    self._normalize_whitelist()
                    print(f"Settings loaded from {self.settings_file}")
                else:
                    self._use_defaults_and_save()
            else:
                # Migrate from settings.json if present
                json_path = self.settings_file.with_suffix(".json")
                if json_path.exists():
                    loaded_settings = self._load_json(json_path)
                    if loaded_settings:
                        self.settings = self._merge_settings(self.DEFAULT_SETTINGS, loaded_settings)
                        self._validate_settings()
                        self._normalize_whitelist()
                        self.save()
                        print(f"Settings migrated from {json_path} to {self.settings_file}")
                    else:
                        self._use_defaults_and_save()
                else:
                    print("No settings file found, using defaults")
                    self.save()
        except Exception as e:
            print(f"Error loading settings: {e}, using defaults")
            self.settings = self.DEFAULT_SETTINGS.copy()

        return self.settings

    def _load_file(self, path: Path) -> Optional[Dict]:
        """Load dict from YAML or JSON file by extension."""
        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        if suffix == ".json":
            return self._load_json(path)
        return None

    def _load_json(self, path: Path) -> Optional[Dict]:
        """Load dict from JSON file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _use_defaults_and_save(self):
        self.settings = self.DEFAULT_SETTINGS.copy()
        self.save()

    def _normalize_whitelist(self):
        """Ensure symbol_whitelist is a string with newlines for compatibility."""
        w = self.settings.get("filters", {}).get("symbol_whitelist")
        if isinstance(w, str) and "\n" not in w and w.strip():
            self.settings.setdefault("filters", {})["symbol_whitelist"] = w.replace(",", "\n").strip()
        elif isinstance(w, list):
            self.settings.setdefault("filters", {})["symbol_whitelist"] = "\n".join(str(s) for s in w)

    def save(self) -> bool:
        """
        Save current settings to YAML file.

        Returns:
            True if save successful, False otherwise
        """
        try:
            self._validate_settings()
            self._create_backup()

            with open(self.settings_file, "w", encoding="utf-8") as f:
                yaml.dump(
                    self.settings,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )

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

            # Validate API mode
            valid_modes = ["PRODUCTION", "DEMO", "DRY_RUN"]
            if self.settings["api"].get("mode") not in valid_modes:
                self.settings["api"]["mode"] = "DRY_RUN"

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
        """Create backup of current settings file."""
        try:
            if self.settings_file.exists():
                backup_file = self.settings_file.with_suffix(self.settings_file.suffix + ".backup")
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
        Export settings to a file (YAML or JSON by extension).

        Args:
            export_path: Path to export file (.yaml, .yml, or .json)

        Returns:
            True if successful, False otherwise
        """
        try:
            path = Path(export_path)
            with open(path, "w", encoding="utf-8") as f:
                if path.suffix.lower() in (".yaml", ".yml"):
                    yaml.dump(self.settings, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
                else:
                    json.dump(self.settings, f, indent=2, ensure_ascii=False)
            print(f"Settings exported to {export_path}")
            return True
        except Exception as e:
            print(f"Error exporting settings: {e}")
            return False

    def import_settings(self, import_path: str) -> bool:
        """
        Import settings from a file (YAML or JSON).

        Args:
            import_path: Path to import file

        Returns:
            True if successful, False otherwise
        """
        try:
            path = Path(import_path)
            if not path.exists():
                print(f"Import file not found: {import_path}")
                return False

            loaded = self._load_file(path)
            if loaded is None:
                print(f"Could not parse settings from {import_path}")
                return False

            self.settings = self._merge_settings(self.DEFAULT_SETTINGS, loaded)
            self._validate_settings()
            self._normalize_whitelist()
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

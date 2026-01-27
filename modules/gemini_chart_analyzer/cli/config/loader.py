"""Configuration loader module for batch scanner."""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional

from modules.common.utils import log_error, log_success, log_debug
from modules.common.system.managers.hardware_manager import get_hardware_manager


def list_configuration_files() -> List[Path]:
    """List all configuration JSON/YAML files in project root.

    Returns:
        List of Path objects for configuration files
    """
    project_root = Path(__file__).parent.parent.parent.parent
    patterns = [
        "batch_scanner_config_*.json",
        "batch_scanner_config_*.yaml",
        "batch_scanner_config_*.yml",
        "standard_*.json",
        "standard_*.yaml",
        "standard_*.yml",
        "config*.json",
        "config*.yaml",
        "config*.yml",
    ]

    config_files = []
    seen_paths = set()
    for pattern in patterns:
        for path in project_root.glob(pattern):
            if path not in seen_paths:
                config_files.append(path)
                seen_paths.add(path)

    # Sort by modification time, most recent first
    config_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    return config_files


def load_configuration_from_file(config_path: Path) -> Optional[Dict[str, Any]]:
    """Load configuration from JSON or YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Dictionary containing configuration parameters, or None if failed
    """
    try:
        if not config_path.exists():
            log_error(f"Configuration file not found: {config_path}")
            return None

        ext = config_path.suffix.lower()
        with open(config_path, "r", encoding="utf-8") as f:
            if ext == ".json":
                config_data = json.load(f)
            elif ext in [".yaml", ".yml"]:
                config_data = yaml.safe_load(f)
            else:
                log_error(f"Unsupported configuration format: {ext}")
                return None

        # Post-process configuration to auto-calculate num_threads if needed
        config_data = _process_config_auto_values(config_data)

        log_success(f"Configuration loaded from: {config_path}")
        return config_data

    except (json.JSONDecodeError, yaml.YAMLError) as e:
        log_error(f"Parse error in configuration file: {e}")
        return None
    except Exception as e:
        log_error(f"Failed to load configuration: {type(e).__name__}: {e}")
        return None


def _process_config_auto_values(config: Dict[str, Any]) -> Dict[str, Any]:
    """Process configuration to replace auto-calculated values.

    Auto-calculates values like num_threads based on system resources.

    Args:
        config: Raw configuration dictionary

    Returns:
        Processed configuration dictionary with auto-calculated values
    """
    if not config:
        return config

    # Auto-calculate num_threads for approximate_ma_scanner
    if "approximate_ma_scanner" in config:
        ama_config = config["approximate_ma_scanner"]
        if isinstance(ama_config, dict):
            # Only process if feature is enabled
            is_enabled = ama_config.get("enabled", False)

            if is_enabled:
                num_threads = ama_config.get("num_threads")

                # Auto-calculate if null, "auto", or not set
                if num_threads is None or (isinstance(num_threads, str) and num_threads.lower() == "auto"):
                    hw_manager = get_hardware_manager()
                    hw_manager.detect_resources()
                    resources = hw_manager.get_resources()

                    # Calculate optimal thread count:
                    # Reserve 1-2 cores for system, use logical cores for threading
                    optimal_threads = max(1, resources.cpu_cores - 2)
                    ama_config["num_threads"] = optimal_threads

                    log_debug(
                        f"Auto-calculated num_threads for approximate_ma_scanner: "
                        f"{optimal_threads} (CPU cores: {resources.cpu_cores})"
                    )

    return config

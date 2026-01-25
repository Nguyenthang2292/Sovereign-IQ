"""Configuration loader module for batch scanner."""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional

from modules.common.utils import log_error, log_success


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

        log_success(f"Configuration loaded from: {config_path}")
        return config_data

    except (json.JSONDecodeError, yaml.YAMLError) as e:
        log_error(f"Parse error in configuration file: {e}")
        return None
    except Exception as e:
        log_error(f"Failed to load configuration: {type(e).__name__}: {e}")
        return None

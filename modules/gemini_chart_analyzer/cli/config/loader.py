"""Configuration loader module for batch scanner."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from modules.common.utils import log_error, log_success


def list_configuration_files() -> List[Path]:
    """List all configuration JSON files in project root.

    Returns:
        List of Path objects for configuration files
    """
    project_root = Path(__file__).parent.parent.parent.parent
    config_files = sorted(
        project_root.glob("batch_scanner_config_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,  # Most recent first
    )
    return config_files


def load_configuration_from_json(config_path: Path) -> Optional[Dict[str, Any]]:
    """Load configuration from JSON file.

    Args:
        config_path: Path to configuration JSON file

    Returns:
        Dictionary containing configuration parameters, or None if failed
    """
    try:
        if not config_path.exists():
            log_error(f"Configuration file not found: {config_path}")
            return None

        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        log_success(f"Configuration loaded from: {config_path}")
        return config_data

    except json.JSONDecodeError as e:
        log_error(f"Invalid JSON in configuration file: {e}")
        return None
    except Exception as e:
        log_error(f"Failed to load configuration: {type(e).__name__}: {e}")
        return None

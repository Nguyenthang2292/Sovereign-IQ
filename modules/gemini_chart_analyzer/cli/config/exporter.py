"""Configuration exporter module for batch scanner."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from modules.common.utils import log_error, log_success


def export_configuration(
    config_data: Dict[str, Any],
    output_path: Optional[Path] = None,
    format: str = "yaml",
) -> Path:
    """Export configuration to JSON or YAML file.

    Args:
        config_data: Dictionary containing all configuration parameters
        output_path: Optional path to output file
        format: 'json' or 'yaml' (default: 'yaml')

    Returns:
        Path to saved configuration file
    """
    if output_path is None:
        # Save to project root
        project_root = Path(__file__).parent.parent.parent.parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = "yaml" if format == "yaml" else "json"
        output_path = project_root / f"batch_scanner_config_{timestamp}.{ext}"

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare configuration dictionary with only non-None values
    config_export = {}
    for key, value in config_data.items():
        # Skip None values and empty strings for cleaner file
        if value is not None and value != "":
            # Handle special cases
            if isinstance(value, Path):
                config_export[key] = str(value)
            elif isinstance(value, datetime):
                config_export[key] = value.isoformat()
            else:
                config_export[key] = value

    # Save to file
    try:
        ext = output_path.suffix.lower()
        with open(output_path, "w", encoding="utf-8") as f:
            if ext == ".json":
                json.dump(config_export, f, indent=2, ensure_ascii=False, default=str)
            elif ext in [".yaml", ".yml"]:
                yaml.dump(config_export, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
            else:
                log_error(f"Unsupported export format: {ext}")
                raise ValueError(f"Unsupported export format: {ext}")

        log_success(f"Configuration exported to: {output_path}")
        return output_path
    except Exception as e:
        log_error(f"Failed to export configuration: {type(e).__name__}: {e}")
        raise

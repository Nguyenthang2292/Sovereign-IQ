"""Configuration exporter module for batch scanner."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from modules.common.utils import log_error, log_success


def export_configuration_to_json(
    config_data: Dict[str, Any],
    output_path: Optional[Path] = None,
) -> Path:
    """Export configuration to JSON file.

    Args:
        config_data: Dictionary containing all configuration parameters
        output_path: Optional path to output file (default: batch_scanner_config_{timestamp}.json in root)

    Returns:
        Path to saved configuration file
    """
    if output_path is None:
        # Save to project root
        project_root = Path(__file__).parent.parent.parent.parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = project_root / f"batch_scanner_config_{timestamp}.json"

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare configuration dictionary with only non-None values
    config_export = {}
    for key, value in config_data.items():
        # Skip None values and empty strings for cleaner JSON
        if value is not None and value != "":
            # Handle special cases
            if isinstance(value, Path):
                config_export[key] = str(value)
            elif isinstance(value, datetime):
                config_export[key] = value.isoformat()
            else:
                config_export[key] = value

    # Save to JSON file
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(config_export, f, indent=2, ensure_ascii=False, default=str)
        log_success(f"Configuration exported to: {output_path}")
        return output_path
    except Exception as e:
        log_error(f"Failed to export configuration: {type(e).__name__}: {e}")
        raise

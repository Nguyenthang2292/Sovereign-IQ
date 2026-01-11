#!/usr/bin/env python3
"""
Export OpenAPI specification from FastAPI app to docs directory.
Run this after changes to API to keep docs up-to-date.

Usage:
    python scripts/export_openapi.py
"""

import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from web.app import app
except ImportError as e:
    print(f"Error importing FastAPI app: {e}")
    sys.exit(1)

def export_openapi():
    """Export OpenAPI spec to YAML and JSON files."""
    
    # Get OpenAPI spec from FastAPI app
    openapi_spec = app.openapi()
    
    # Paths for output files
    docs_dir = project_root / "docs"
    yaml_path = docs_dir / "openapi.yaml"
    json_path = docs_dir / "openapi.json"
    
    # Ensure docs directory exists
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # Export YAML
    try:
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(openapi_spec, f, default_flow_style=False, sort_keys=False)
        print(f"✓ Exported OpenAPI YAML to: {yaml_path}")
    except Exception as e:
        print(f"✗ Error exporting YAML: {e}")
        return False
    
    # Export JSON
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(openapi_spec, f, indent=2, ensure_ascii=False)
        print(f"✓ Exported OpenAPI JSON to: {json_path}")
    except Exception as e:
        print(f"✗ Error exporting JSON: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Exporting OpenAPI specification...")
    if export_openapi():
        print("\n✓ OpenAPI export completed successfully")
        print("\nNext steps:")
        print("  - Review: docs/openapi.yaml and docs/openapi.json")
        print("  - Update docs/API_DOCUMENTATION.md if needed")
        print("  - Commit the updated files")
        sys.exit(0)
    else:
        print("\n✗ OpenAPI export failed")
        sys.exit(1)

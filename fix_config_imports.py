#!/usr/bin/env python3
"""
Fix Config Import Script

Replaces problematic imports like:
  from config.MODULE import X
With backward-compatible imports:
  from config import X

This fixes circular import issues while maintaining backward compatibility.
"""

import re
import sys
from pathlib import Path

# Modules that should use backward-compatible imports
# All config modules (from config.modules/ and config.shared/)
MODULES_TO_FIX = [
    # config.modules.*
    'auto_trade',
    'decision_matrix',
    'deep_learning',
    'gemini_chart_analyzer',
    'hmm',
    'iching',
    'lstm',
    'pairs_trading',
    'portfolio',
    'position_sizing',
    'random_forest',
    'range_oscillator',
    'spc',
    'spc_enhancements',
    'xgboost',
    # config.shared.*
    'evaluation',
    'forex_pairs',
    'model_features',
]

def fix_file(file_path: Path) -> bool:
    """Fix imports in a single file. Returns True if file was modified."""
    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content

        # Pattern: from config.MODULE import ...
        for module in MODULES_TO_FIX:
            # Single-line import: from config.MODULE import X, Y, Z
            pattern = rf'from config\.{module} import\s+([^\n]+)'
            replacement = r'from config import \1'
            content = re.sub(pattern, replacement, content)

        if content != original_content:
            file_path.write_text(content, encoding='utf-8')
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return False

def main():
    project_root = Path(__file__).parent

    if not project_root.exists():
        print(f"Project directory not found: {project_root}", file=sys.stderr)
        return 1

    files_fixed = 0
    files_checked = 0

    # Find all Python files in modules/ and core/ directories
    search_dirs = [project_root / "modules", project_root / "core"]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for py_file in search_dir.rglob("*.py"):
            files_checked += 1
            if fix_file(py_file):
                files_fixed += 1
                print(f"Fixed: {py_file.relative_to(project_root)}")

    print(f"\nComplete: Fixed {files_fixed} files out of {files_checked} checked")
    return 0

if __name__ == "__main__":
    sys.exit(main())

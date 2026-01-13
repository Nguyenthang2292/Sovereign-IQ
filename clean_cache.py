#!/usr/bin/env python3
"""
Cache cleanup utility script for removing Python bytecode files and directories.
This script can be run periodically to clean up __pycache__ folders and .pyc files.
"""

import os
import shutil
import sys
from pathlib import Path


def clean_pycache(root_dir="."):
    """
    Remove all __pycache__ directories and .pyc files from the project.

    Args:
        root_dir: Root directory to start cleaning from (default: current directory)
    """
    root_path = Path(root_dir).resolve()
    removed_dirs = 0
    removed_files = 0

    print(f"Cleaning Python cache files from: {root_path}")
    print("-" * 60)

    # Remove __pycache__ directories
    for pycache_dir in root_path.rglob("__pycache__"):
        # Skip virtual environment directories
        if ".venv" in pycache_dir.parts or "venv" in pycache_dir.parts:
            continue

        try:
            shutil.rmtree(pycache_dir)
            print(f"Removed directory: {pycache_dir}")
            removed_dirs += 1
        except Exception as e:
            print(f"Error removing {pycache_dir}: {e}")

    # Remove .pyc, .pyo files
    for py_file in root_path.rglob("*.pyc"):
        # Skip virtual environment directories
        if ".venv" in py_file.parts or "venv" in py_file.parts:
            continue

        try:
            py_file.unlink()
            removed_files += 1
        except Exception as e:
            print(f"Error removing {py_file}: {e}")

    for py_file in root_path.rglob("*.pyo"):
        # Skip virtual environment directories
        if ".venv" in py_file.parts or "venv" in py_file.parts:
            continue

        try:
            py_file.unlink()
            removed_files += 1
        except Exception as e:
            print(f"Error removing {py_file}: {e}")

    print("-" * 60)
    print("Cleanup complete!")
    print(f"Removed directories: {removed_dirs}")
    print(f"Removed files: {removed_files}")

    return removed_dirs + removed_files


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        target_dir = os.path.dirname(os.path.abspath(__file__))

    total_removed = clean_pycache(target_dir)
    sys.exit(0 if total_removed >= 0 else 1)

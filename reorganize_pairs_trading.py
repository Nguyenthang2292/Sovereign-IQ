"""
Script to reorganize pairs_trading module into sub-packages.
"""
import os
import shutil
from pathlib import Path

BASE_DIR = Path("modules/pairs_trading")

# Mapping: source_file -> destination_subpackage
FILE_MAPPINGS = {
    # Core components
    "pairs_analyzer.py": "core",
    "pair_metrics_computer.py": "core",
    "opportunity_scorer.py": "core",
    
    # Metrics
    "statistical_tests.py": "metrics",
    "hedge_ratio.py": "metrics",
    "zscore_metrics.py": "metrics",
    "risk_metrics.py": "metrics",
    
    # Analysis
    "performance_analyzer.py": "analysis",
    
    # UI
    "cli.py": "ui",
    "display.py": "ui",
    "utils.py": "ui",
}

def reorganize():
    """Reorganize files into sub-packages."""
    # Create sub-package directories
    for subpackage in ["core", "metrics", "analysis", "ui"]:
        subdir = BASE_DIR / subpackage
        subdir.mkdir(exist_ok=True)
        # Create __init__.py if it doesn't exist
        init_file = subdir / "__init__.py"
        if not init_file.exists():
            init_file.touch()
        print(f"Created/verified directory: {subdir}")
    
    # Remove 'analysis' file if it exists (not a directory)
    analysis_file = BASE_DIR / "analysis"
    if analysis_file.exists() and not analysis_file.is_dir():
        analysis_file.unlink()
        print(f"Removed file: {analysis_file}")
        # Recreate as directory
        analysis_file.mkdir(exist_ok=True)
        (analysis_file / "__init__.py").touch()
    
    # Move files
    for source_file, subpackage in FILE_MAPPINGS.items():
        source_path = BASE_DIR / source_file
        dest_path = BASE_DIR / subpackage / source_file
        
        if source_path.exists():
            if dest_path.exists():
                print(f"Warning: {dest_path} already exists, skipping...")
            else:
                shutil.move(str(source_path), str(dest_path))
                print(f"Moved: {source_file} -> {subpackage}/")
        else:
            print(f"Warning: {source_file} not found, skipping...")
    
    print("\nReorganization complete!")

if __name__ == "__main__":
    reorganize()


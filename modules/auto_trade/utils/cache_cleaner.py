"""Python bytecode cache cleaner utility.

Clears __pycache__ directories to ensure fresh module imports after code changes.
Useful after:
- Updating module code
- Building/rebuilding Rust extensions
- Modifying dataclass definitions
- Fixing import errors
"""

import shutil
from pathlib import Path
from typing import List, Optional


def clear_pycache_recursive(root_path: Path, verbose: bool = False) -> tuple[int, int]:
    """
    Recursively remove all __pycache__ directories under root_path.

    Args:
        root_path: Root directory to start searching from
        verbose: Print detailed information about cleared directories

    Returns:
        Tuple of (directories_removed, errors_encountered)
    """
    removed_count = 0
    error_count = 0

    if not root_path.exists():
        if verbose:
            print(f"  Path does not exist: {root_path}")
        return 0, 1

    try:
        for pycache_dir in root_path.rglob("__pycache__"):
            try:
                if verbose:
                    print(f"  Removing: {pycache_dir.relative_to(root_path)}")
                shutil.rmtree(pycache_dir)
                removed_count += 1
            except PermissionError:
                if verbose:
                    print(f"  Permission denied: {pycache_dir}")
                error_count += 1
            except Exception as e:
                if verbose:
                    print(f"  Error removing {pycache_dir}: {e}")
                error_count += 1

    except Exception as e:
        if verbose:
            print(f"  Error scanning directory: {e}")
        error_count += 1

    return removed_count, error_count


def clear_module_cache(module_names: Optional[List[str]] = None, verbose: bool = False) -> bool:
    """
    Clear Python bytecode cache for specific modules.

    Args:
        module_names: List of module names to clear (e.g., ["adaptive_trend_LTS_mini", "xgboost_LTS"])
                     If None, clears cache for all auto_trade related modules
        verbose: Print detailed information

    Returns:
        True if successful, False if errors occurred
    """
    project_root = Path(__file__).resolve().parents[3]  # Go up to crypto-probability/
    modules_dir = project_root / "modules"

    # Default modules to clear
    if module_names is None:
        module_names = [
            "adaptive_trend_LTS_mini",
            "xgboost_LTS",
            "auto_trade",
            "common",
        ]

    print("\n" + "=" * 60)
    print("  Clearing Python Bytecode Cache")
    print("=" * 60 + "\n")

    total_removed = 0
    total_errors = 0

    for module_name in module_names:
        module_path = modules_dir / module_name

        if not module_path.exists():
            if verbose:
                print(f"Module not found: {module_name}")
            continue

        print(f"Clearing cache for: {module_name}")
        removed, errors = clear_pycache_recursive(module_path, verbose=verbose)

        total_removed += removed
        total_errors += errors

        if not verbose:
            if removed > 0:
                print(f"  Removed {removed} cache directories")
            elif errors == 0:
                print(f"  No cache found (already clean)")

    # Summary
    print("\n" + "=" * 60)
    print("  Cache Clear Summary")
    print("=" * 60)
    print(f"  Directories removed: {total_removed}")
    print(f"  Errors: {total_errors}")
    print("=" * 60 + "\n")

    return total_errors == 0


def clear_all_cache(verbose: bool = False) -> bool:
    """
    Clear Python bytecode cache for the entire project.

    Args:
        verbose: Print detailed information

    Returns:
        True if successful, False if errors occurred
    """
    project_root = Path(__file__).resolve().parents[3]  # Go up to crypto-probability/

    print("\n" + "=" * 60)
    print("  Clearing ALL Python Bytecode Cache")
    print("=" * 60 + "\n")

    removed, errors = clear_pycache_recursive(project_root, verbose=verbose)

    # Summary
    print("\n" + "=" * 60)
    print("  Cache Clear Summary")
    print("=" * 60)
    print(f"  Directories removed: {removed}")
    print(f"  Errors: {errors}")
    print("=" * 60 + "\n")

    return errors == 0


if __name__ == "__main__":
    # Allow running as standalone script
    import argparse

    parser = argparse.ArgumentParser(description="Clear Python bytecode cache")
    parser.add_argument(
        "-m",
        "--modules",
        nargs="+",
        help="Specific modules to clear (e.g., adaptive_trend_LTS_mini xgboost_LTS)",
    )
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Clear cache for entire project",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.all:
        success = clear_all_cache(verbose=args.verbose)
    else:
        success = clear_module_cache(module_names=args.modules, verbose=args.verbose)

    import sys

    sys.exit(0 if success else 1)

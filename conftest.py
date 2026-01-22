"""
Pytest configuration file - Root conftest.py
This file ensures pytest runs with the correct Python environment and paths.
"""

import sys
from pathlib import Path

# Ensure the project root is in the Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Verify we're using the venv Python
# Check for venv in common locations
venv_locations = [
    project_root / ".venv" / "Scripts" / "python.exe",  # Windows
    project_root / ".venv" / "bin" / "python",  # Linux/Mac
    project_root / "venv" / "Scripts" / "python.exe",  # Alternative Windows
    project_root / "venv" / "bin" / "python",  # Alternative Linux/Mac
]

venv_python = None
for location in venv_locations:
    if location.exists():
        venv_python = location
        break

# Check if we're in a virtual environment
is_venv = hasattr(sys, "real_prefix") or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)

current_python = Path(sys.executable)

# Enforce venv usage
if venv_python:
    # If venv exists, we should be using it
    if not is_venv and current_python != venv_python:
        print("\n" + "=" * 60)
        print("ERROR: Not using virtual environment!")
        print("=" * 60)
        print(f"   Current Python: {current_python}")
        print(f"   Expected venv:  {venv_python}")
        print("\n   Please activate venv before running tests:")
        if sys.platform == "win32":
            print("   PowerShell: .\\.venv\\Scripts\\Activate.ps1")
            print("   CMD:        .venv\\Scripts\\activate.bat")
        else:
            print("   source .venv/bin/activate")
        print("\n   Or use the test scripts:")
        print("   PowerShell: .\\run_tests.ps1")
        print("   CMD:        run_tests.bat")
        print("=" * 60 + "\n")
        sys.exit(1)
    elif is_venv:
        # Verify we're using the correct venv
        if current_python != venv_python and not str(current_python).startswith(str(project_root / ".venv")):
            print("\n" + "=" * 60)
            print("WARNING: Using different virtual environment!")
            print("=" * 60)
            print(f"   Current Python: {current_python}")
            print(f"   Expected venv:  {venv_python}")
            print("=" * 60 + "\n")
elif not is_venv:
    # No venv found and not in any venv - warn but don't fail
    print("\n" + "=" * 60)
    print("WARNING: No virtual environment detected!")
    print("=" * 60)
    print("   It's recommended to use a virtual environment.")
    print("   Create one with: python -m venv .venv")
    print("=" * 60 + "\n")


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Add custom markers
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "memory_intensive: marks tests that use significant RAM")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "performance: marks tests that measure performance")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU/CUDA")

    # Print environment info
    print(f"\n{'=' * 60}")
    print(f"Python: {sys.version}")
    print(f"Project Root: {project_root}")
    print(f"Python Executable: {sys.executable}")
    print(f"{'=' * 60}\n")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add automatic markers."""
    for item in items:
        # Auto-mark GPU tests
        if "gpu" in item.nodeid.lower():
            item.add_marker("gpu")

        # Auto-mark memory intensive tests
        if "memory" in item.nodeid.lower():
            item.add_marker("memory_intensive")

        # Auto-mark integration tests
        if "integration" in item.nodeid.lower():
            item.add_marker("integration")

        # Auto-mark performance tests
        if "performance" in item.nodeid.lower():
            item.add_marker("performance")

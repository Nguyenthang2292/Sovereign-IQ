"""
Simple script to start all web applications.
"""

import subprocess
import sys
import time
from pathlib import Path

# Get script directory
SCRIPT_DIR = Path(__file__).parent
WEB_ROOT = SCRIPT_DIR.parent
APPS_DIR = WEB_ROOT / "apps"

# App configurations
APPS = {
    "gemini_analyzer": {
        "name": "Gemini Chart Analyzer",
        "backend_dir": APPS_DIR / "gemini_analyzer" / "backend",
        "backend_port": 8001,
        "frontend_dir": APPS_DIR / "gemini_analyzer" / "frontend",
        "frontend_port": 5173,
    },
    "atc_visualizer": {
        "name": "ATC Visualizer",
        "backend_dir": APPS_DIR / "atc_visualizer" / "backend",
        "backend_port": 8002,
        "frontend_dir": APPS_DIR / "atc_visualizer" / "frontend",
        "frontend_port": 5174,
    },
}


def start_app_backend(app_name, config):
    """Start app backend server."""
    print(f"\n🚀 Starting {config['name']} Backend...")
    backend_dir = config["backend_dir"]

    if not backend_dir.exists():
        print(f"❌ Backend directory not found: {backend_dir}")
        return None

    try:
        process = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=str(backend_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f"✅ {config['name']} Backend started on port {config['backend_port']}")
        return process
    except Exception as e:
        print(f"❌ Failed to start {config['name']} Backend: {e}")
        return None


def start_app_frontend(app_name, config):
    """Start app frontend dev server."""
    print(f"\n🎨 Starting {config['name']} Frontend...")
    frontend_dir = config["frontend_dir"]

    if not frontend_dir.exists():
        print(f"❌ Frontend directory not found: {frontend_dir}")
        return None

    try:
        process = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=str(frontend_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
        )
        print(f"✅ {config['name']} Frontend started on port {config['frontend_port']}")
        return process
    except Exception as e:
        print(f"❌ Failed to start {config['name']} Frontend: {e}")
        return None


def main():
    """Start all applications."""
    print("=" * 60)
    print("🚀 Starting All Web Applications")
    print("=" * 60)

    processes = []

    # Start all backends
    for app_name, config in APPS.items():
        process = start_app_backend(app_name, config)
        if process:
            processes.append((f"{config['name']} Backend", process))
        time.sleep(1)  # Wait between starts

    # Start all frontends
    for app_name, config in APPS.items():
        process = start_app_frontend(app_name, config)
        if process:
            processes.append((f"{config['name']} Frontend", process))
        time.sleep(1)  # Wait between starts

    print("\n" + "=" * 60)
    print("✅ All applications started!")
    print("=" * 60)
    print("\n📊 Access Points:")
    for app_name, config in APPS.items():
        print(f"\n{config['name']}:")
        print(f"  Backend:  http://localhost:{config['backend_port']}")
        print(f"  Frontend: http://localhost:{config['frontend_port']}")
        print(f"  API Docs: http://localhost:{config['backend_port']}/docs")

    print("\n" + "=" * 60)
    print("Press Ctrl+C to stop all applications")
    print("=" * 60)

    try:
        # Keep script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping all applications...")
        for name, process in processes:
            try:
                process.terminate()
                print(f"✅ Stopped {name}")
            except Exception as e:
                print(f"❌ Error stopping {name}: {e}")
        print("\n✅ All applications stopped!")


if __name__ == "__main__":
    main()

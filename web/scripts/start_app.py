"""
Start a specific web application.

Usage:
    python start_app.py gemini_analyzer
    python start_app.py atc_visualizer
"""

import argparse
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


def start_backend(config):
    """Start backend server."""
    print(f"\n🚀 Starting {config['name']} Backend...")
    backend_dir = config["backend_dir"]

    if not backend_dir.exists():
        print(f"❌ Backend directory not found: {backend_dir}")
        return None

    try:
        process = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=str(backend_dir),
        )
        print(f"✅ {config['name']} Backend started on port {config['backend_port']}")
        print(f"   API Docs: http://localhost:{config['backend_port']}/docs")
        return process
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return None


def start_frontend(config):
    """Start frontend dev server."""
    print(f"\n🎨 Starting {config['name']} Frontend...")
    frontend_dir = config["frontend_dir"]

    if not frontend_dir.exists():
        print(f"❌ Frontend directory not found: {frontend_dir}")
        return None

    try:
        process = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=str(frontend_dir),
            shell=True,
        )
        print(f"✅ {config['name']} Frontend started on port {config['frontend_port']}")
        print(f"   URL: http://localhost:{config['frontend_port']}")
        return process
    except Exception as e:
        print(f"❌ Failed to start frontend: {e}")
        return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Start a specific web application")
    parser.add_argument("app", choices=list(APPS.keys()), help="Application to start")
    parser.add_argument("--backend-only", action="store_true", help="Start backend only")
    parser.add_argument("--frontend-only", action="store_true", help="Start frontend only")

    args = parser.parse_args()

    config = APPS[args.app]

    print("=" * 60)
    print(f"🚀 Starting {config['name']}")
    print("=" * 60)

    processes = []

    # Start backend
    if not args.frontend_only:
        backend_process = start_backend(config)
        if backend_process:
            processes.append(("Backend", backend_process))
            time.sleep(2)  # Wait for backend to start

    # Start frontend
    if not args.backend_only:
        frontend_process = start_frontend(config)
        if frontend_process:
            processes.append(("Frontend", frontend_process))

    if not processes:
        print("\n❌ Failed to start any services")
        return 1

    print("\n" + "=" * 60)
    print("✅ Application started!")
    print("=" * 60)
    print("\nPress Ctrl+C to stop")
    print("=" * 60)

    try:
        # Keep script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping application...")
        for name, process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ Stopped {name}")
            except Exception as e:
                print(f"❌ Error stopping {name}: {e}")
                try:
                    process.kill()
                except:
                    pass
        print("\n✅ Application stopped!")

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
ATC Visualizer - Main Entry Point

Starts both backend and frontend servers automatically.
"""

import subprocess
import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
BACKEND_DIR = BASE_DIR / "web" / "atc_visualizer" / "backend"
FRONTEND_DIR = BASE_DIR / "web" / "atc_visualizer" / "frontend"


def check_requirements():
    """Check if required tools are installed."""
    missing = []

    try:
        result = subprocess.run(["python", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            missing.append("Python")
    except FileNotFoundError:
        missing.append("Python")

    npm_found = False
    npm_version = None

    npm_commands = ["npm", "npm.cmd", "node --version && npm --version"]

    for cmd in npm_commands:
        try:
            if "&&" in cmd:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            else:
                result = subprocess.run(cmd, capture_output=True, text=True, shell=sys.platform == "win32")

            if result.returncode == 0:
                npm_found = True
                npm_version = result.stdout.strip() or result.stderr.strip()
                break
        except (FileNotFoundError, subprocess.SubprocessError):
            continue

    if not npm_found:
        print("⚠️  npm/Node.js not found in PATH")

        try:
            result = subprocess.run("where npm", shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                npm_path = result.stdout.strip()
                print(f"   Found npm at: {npm_path}")

                result = subprocess.run(f'"{npm_path}" --version', shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    npm_version = result.stdout.strip()
                    npm_found = True
                    print(f"   Version: {npm_version}")
        except Exception as e:
            print(f"   Error searching for npm: {e}")

    if npm_found:
        print(f"✅ npm found (Version: {npm_version})")
    else:
        missing.append("npm/Node.js")

    if missing:
        print(f"\n❌ Missing requirements: {', '.join(missing)}")
        print("\nPlease install:")
        if "Python" in missing:
            print("  - Python 3.9+")
        if "npm/Node.js" in missing:
            print("  - Node.js 18+ and npm")
            print("\n💡 Troubleshooting tips:")
            print("  1. Restart your terminal/command prompt")
            print("  2. Verify Node.js is in your PATH:")
            print("     - Windows: echo %PATH%")
            print("     - Run: where npm")
            print("  3. Reinstall Node.js and check 'Add to PATH' option")
        sys.exit(1)

    print("✅ All requirements met")


def install_dependencies():
    """Install backend and frontend dependencies if needed."""
    print("\n📦 Checking dependencies...")

    backend_req = BACKEND_DIR / "requirements.txt"
    if backend_req.exists():
        print(f"  Installing backend dependencies from {backend_req}")
        subprocess.run(["pip", "install", "-r", str(backend_req)], check=True)
    else:
        print(f"  ⚠️  Backend requirements.txt not found at {backend_req}")

    frontend_json = FRONTEND_DIR / "package.json"
    if frontend_json.exists():
        print(f"  Installing frontend dependencies from {frontend_json}")

        use_shell = sys.platform == "win32"

        try:
            subprocess.run(["npm", "install"], cwd=str(FRONTEND_DIR), check=True, shell=use_shell)
        except subprocess.CalledProcessError as e:
            print(f"  ⚠️  npm install failed (exit code {e.returncode})")
            print("  💡 Trying with --legacy-peer-deps flag...")

            subprocess.run(["npm", "install", "--legacy-peer-deps"], cwd=str(FRONTEND_DIR), check=True, shell=use_shell)
            print("  ✅ Dependencies installed successfully with --legacy-peer-deps")
    else:
        print(f"  ⚠️  Frontend package.json not found at {frontend_json}")


def start_backend():
    """Start FastAPI backend server."""
    print("\n🚀 Starting Backend Server (FastAPI)...")
    print(f"   API Docs: http://localhost:5000/docs")

    backend_process = subprocess.Popen(["python", "api.py"], cwd=str(BACKEND_DIR))

    return backend_process


def start_frontend():
    """Start Vue.js frontend dev server."""
    print("\n🎨 Starting Frontend Server (Vue.js + Vite)...")
    print(f"   App URL: http://localhost:5173")

    use_shell = sys.platform == "win32"

    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=str(FRONTEND_DIR),
        shell=use_shell,
    )

    return frontend_process


def print_startup_info():
    """Print startup information."""
    print("\n" + "=" * 60)
    print("📊 ATC VISUALIZER STARTED")
    print("=" * 60)
    print("\n📍 Access Points:")
    print("   Frontend:   http://localhost:5173")
    print("   Backend API: http://localhost:5000")
    print("   API Docs:   http://localhost:5000/docs")
    print("\n💡 Press Ctrl+C to stop both servers")
    print("=" * 60)


def monitor_servers(backend_proc, frontend_proc):
    """Monitor both servers and handle shutdown."""
    try:
        backend_proc.wait()
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\n\n⏸️  Stopping servers...")

        if backend_proc.poll() is None:
            print("   Stopping backend...")
            backend_proc.terminate()
            backend_proc.wait(timeout=5)

        if frontend_proc.poll() is None:
            print("   Stopping frontend...")
            frontend_proc.terminate()
            frontend_proc.wait(timeout=5)

        print("\n✅ Servers stopped successfully")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        for proc in [backend_proc, frontend_proc]:
            if proc.poll() is None:
                proc.kill()


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("🚀 ATC VISUALIZER - STARTUP")
    print("=" * 60)
    print(f"\nPlatform: {sys.platform}")
    print(f"Python: {sys.version.split()[0]}")

    skip_npm_check = "--skip-npm-check" in sys.argv

    if not skip_npm_check:
        check_requirements()
    else:
        print("⚠️  Skipping npm/Node.js requirement check (--skip-npm-check)")
        print("✅ Requirements check skipped")

    if "--install" in sys.argv or "-i" in sys.argv:
        install_dependencies()
        print("\n✅ Dependencies installed!")
        return

    if "--no-install" not in sys.argv:
        install = input("\nInstall/Update dependencies? (y/n): ").strip().lower()
        if install == "y":
            install_dependencies()

    try:
        backend_proc = start_backend()
        frontend_proc = start_frontend()

        print_startup_info()
        monitor_servers(backend_proc, frontend_proc)

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to start: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

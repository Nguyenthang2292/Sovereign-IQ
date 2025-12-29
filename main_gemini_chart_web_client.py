"""
Main entry point for Vue.js App (Gemini Chart Analyzer Web Client).

This script provides convenient commands to:
- Run Vue development server
- Build Vue app for production

Usage:
    # Run Vue dev server (with hot reload)
    python main_gemini_chart_web_client.py dev

    # Build Vue app for production
    python main_gemini_chart_web_client.py build
"""

import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path

project_root = Path(__file__).parent

VUE_DIR = project_root / "web" / "static" / "vue"
# Detect Windows platform
IS_WINDOWS = sys.platform == "win32"


def run_npm_command(cmd_args, **kwargs):
    """Run npm command with proper Windows compatibility."""
    if IS_WINDOWS:
        kwargs.setdefault("shell", True)
    return subprocess.run(cmd_args, **kwargs)


def check_node_installed():
    """Check if Node.js and npm are installed."""
    # Use shutil.which to find executables in PATH (cross-platform)
    node_path = shutil.which("node")
    npm_path = shutil.which("npm")
    
    if not node_path or not npm_path:
        print("❌ Error: Node.js and npm are required but not found.")
        print("   Please install Node.js from https://nodejs.org/")
        print(f"   Current PATH: {os.environ.get('PATH', 'Not set')[:200]}...")
        return False
    
    # Verify they work by checking versions
    try:
        # Use global IS_WINDOWS and run_npm_command helper for consistency
        result_node = run_npm_command(
            ["node", "--version"],
            capture_output=True,
            check=True,
            text=True
        )
        result_npm = run_npm_command(
            ["npm", "--version"],
            capture_output=True,
            check=True,
            text=True
        )
        print(f"✅ Node.js {result_node.stdout.strip()}, npm {result_npm.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"❌ Error: Node.js/npm found but cannot execute: {e}")
        return False


def check_dependencies():
    """Check if node_modules exists, install if missing."""
    node_modules = VUE_DIR / "node_modules"
    if not node_modules.exists():
        print("📦 Installing dependencies...")
        try:
            os.chdir(VUE_DIR)
            run_npm_command(["npm", "install"], check=True)
        finally:
            os.chdir(project_root)
        print("✅ Dependencies installed!")
    else:
        print("✅ Dependencies already installed")

def run_dev_server():
    """Run Vue development server with hot reload."""
    if not check_node_installed():
        return 1
    
    check_dependencies()
    
    print("🚀 Starting Vue development server...")
    print("   Frontend: http://localhost:5173")
    print("   Press Ctrl+C to stop")
    
    os.chdir(VUE_DIR)
    try:
        run_npm_command(["npm", "run", "dev"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Development server stopped")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running dev server: {e}")
        return 1
    finally:
        os.chdir(project_root)


def build_vue_app():
    """Build Vue app for production."""
    if not check_node_installed():
        return 1
    
    check_dependencies()
    
    print("🔨 Building Vue app for production...")
    os.chdir(VUE_DIR)
    try:
        run_npm_command(["npm", "run", "build"], check=True)
        print("✅ Build completed! Output: web/static/vue/dist/")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        return 1
    finally:
        os.chdir(project_root)


def main():
    parser = argparse.ArgumentParser(
        description="Vue.js App Entry Point - Gemini Chart Analyzer (Frontend Only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_gemini_chart_web_client.py dev          # Run Vue dev server
  python main_gemini_chart_web_client.py build         # Build for production
        """
    )
    
    parser.add_argument(
        "command",
        choices=["dev", "build"],
        help="Command to execute"
    )
    
    args = parser.parse_args()
    
    if args.command == "dev":
        return run_dev_server()
    elif args.command == "build":
        return build_vue_app()


if __name__ == "__main__":
    sys.exit(main())


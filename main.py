import subprocess
import sys
import time
import signal
from pathlib import Path
import os


class ProcessManager:
    def __init__(self):
        self.processes = []

    def add_process(self, cmd, cwd, name, env=None):
        print(f"Starting {name}...")
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            shell=True if sys.platform == "win32" else False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.processes.append({"process": process, "name": name})
        return process

    def stop_all(self):
        print("\nStopping all servers...")
        for proc_info in self.processes:
            try:
                proc_info["process"].terminate()
                proc_info["process"].wait(timeout=5)
                print(f"Stopped {proc_info['name']}")
            except subprocess.TimeoutExpired:
                proc_info["process"].kill()
            except Exception as e:
                print(f"Error stopping {proc_info['name']}: {e}")


def start_backend_servers(manager):
    import os

    project_root = Path(__file__).resolve().parent

    backend_configs = [
        {
            "name": "ATC Visualizer Backend",
            "module": "web.apps.atc_visualizer.backend.main",
            "port": 8002,
        },
        {
            "name": "Gemini Analyzer Backend",
            "module": "web.apps.gemini_analyzer.backend.main",
            "port": 8001,
        },
    ]

    for config in backend_configs:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root)
        manager.add_process(
            cmd=[
                sys.executable,
                "-m",
                "uvicorn",
                f"{config['module']}:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(config["port"]),
            ],
            cwd=str(project_root),
            name=config["name"],
        )


def install_frontend_dependencies():
    """Install and update dependencies for all frontend apps"""
    project_root = Path(__file__).resolve().parent
    if sys.platform == "win32":
        # Use cmd.exe to avoid PowerShell execution policy issues
        npm_cmd = ["cmd.exe", "/c", "npm.cmd"]
    else:
        npm_cmd = ["npm"]

    frontend_configs = [
        {
            "name": "ATC Visualizer Frontend",
            "cwd": project_root / "web" / "apps" / "atc_visualizer" / "frontend",
        },
        {
            "name": "Gemini Analyzer Frontend",
            "cwd": project_root / "web" / "apps" / "gemini_analyzer" / "frontend",
        },
    ]

    print("\n" + "=" * 60)
    print("INSTALLING FRONTEND DEPENDENCIES")
    print("=" * 60 + "\n")

    for config in frontend_configs:
        cwd_path = config["cwd"]
        package_json = cwd_path / "package.json"

        if not package_json.exists():
            print(f"⚠️  Warning: {config['name']} - package.json not found, skipping...")
            continue

        print(f"📦 Installing dependencies for {config['name']}...")
        print(f"   Path: {cwd_path}\n")

        # Verify package.json exists
        if not package_json.exists():
            print(f"⚠️  Warning: {config['name']} - package.json not found at {package_json}, skipping...")
            continue

        try:
            # Run npm install
            result = subprocess.run(
                npm_cmd + ["install"],
                cwd=str(cwd_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=300,  # 5 minutes timeout
            )

            if result.returncode == 0:
                print(f"✅ {config['name']} - Dependencies installed successfully\n")
            else:
                print(f"❌ {config['name']} - Failed to install dependencies")
                print(f"   Error output: {result.stdout[-500:]}\n")  # Last 500 chars
                print("⚠️  Continuing anyway, but the app may not work correctly...\n")
        except subprocess.TimeoutExpired:
            print(f"⏱️  {config['name']} - Installation timeout (exceeded 5 minutes)")
            print("⚠️  Continuing anyway...\n")
        except Exception as e:
            print(f"❌ {config['name']} - Error during installation: {e}")
            print("⚠️  Continuing anyway...\n")

    print("=" * 60 + "\n")


def build_frontend_apps():
    """Build all frontend apps before starting dev servers"""
    project_root = Path(__file__).resolve().parent
    if sys.platform == "win32":
        # Use cmd.exe to avoid PowerShell execution policy issues
        npm_cmd = ["cmd.exe", "/c", "npm.cmd"]
    else:
        npm_cmd = ["npm"]

    frontend_configs = [
        {
            "name": "ATC Visualizer Frontend",
            "cwd": project_root / "web" / "apps" / "atc_visualizer" / "frontend",
        },
        {
            "name": "Gemini Analyzer Frontend",
            "cwd": project_root / "web" / "apps" / "gemini_analyzer" / "frontend",
        },
    ]

    print("\n" + "=" * 60)
    print("BUILDING FRONTEND APPS")
    print("=" * 60 + "\n")

    for config in frontend_configs:
        cwd_path = config["cwd"]
        package_json = cwd_path / "package.json"

        if not package_json.exists():
            print(f"⚠️  Warning: {config['name']} - package.json not found, skipping...")
            continue

        print(f"🔨 Building {config['name']}...")
        print(f"   Path: {cwd_path}\n")

        # Verify package.json exists
        if not package_json.exists():
            print(f"⚠️  Warning: {config['name']} - package.json not found at {package_json}, skipping...")
            continue

        try:
            # Run npm run build
            result = subprocess.run(
                npm_cmd + ["run", "build"],
                cwd=str(cwd_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=600,  # 10 minutes timeout for build
            )

            if result.returncode == 0:
                print(f"✅ {config['name']} - Build completed successfully\n")
            else:
                print(f"❌ {config['name']} - Build failed")
                print(f"   Error output: {result.stdout[-500:]}\n")  # Last 500 chars
                print("⚠️  Continuing anyway, but the app may not work correctly...\n")
        except subprocess.TimeoutExpired:
            print(f"⏱️  {config['name']} - Build timeout (exceeded 10 minutes)")
            print("⚠️  Continuing anyway...\n")
        except Exception as e:
            print(f"❌ {config['name']} - Error during build: {e}")
            print("⚠️  Continuing anyway...\n")

    print("=" * 60 + "\n")


def start_frontend_servers(manager):
    project_root = Path(__file__).resolve().parent

    frontend_configs = [
        {
            "name": "ATC Visualizer Frontend",
            "cwd": project_root / "web" / "apps" / "atc_visualizer" / "frontend",
        },
        {
            "name": "Gemini Analyzer Frontend",
            "cwd": project_root / "web" / "apps" / "gemini_analyzer" / "frontend",
        },
    ]

    if sys.platform == "win32":
        # Use cmd.exe to avoid PowerShell execution policy issues
        npm_cmd = ["cmd.exe", "/c", "npm.cmd"]
    else:
        npm_cmd = ["npm"]

    for config in frontend_configs:
        manager.add_process(
            cmd=npm_cmd + ["run", "dev"],
            cwd=config["cwd"],
            name=config["name"],
        )


def print_startup_info():
    print("=" * 60)
    print("CRYPTO PROBABILITY - WEB SERVERS")
    print("=" * 60)
    print("\nServers will be available at:")
    print("  ATC Visualizer Frontend:  http://localhost:5174")
    print("  ATC Visualizer Backend:   http://localhost:8002/docs")
    print("  Gemini Analyzer Frontend: http://localhost:5173")
    print("  Gemini Analyzer Backend:  http://localhost:8001/docs")
    print("\nPress Ctrl+C to stop all servers")
    print("=" * 60 + "\n")


def main():
    manager = ProcessManager()

    def signal_handler(sig, frame):
        manager.stop_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print_startup_info()

    try:
        # Install frontend dependencies first
        install_frontend_dependencies()
        
        # Build frontend apps before starting dev servers
        build_frontend_apps()
        
        # Start backend servers
        start_backend_servers(manager)
        time.sleep(2)
        
        # Start frontend servers
        start_frontend_servers(manager)

        print("All servers started successfully!\n")

        while True:
            time.sleep(1)
            for proc_info in manager.processes:
                return_code = proc_info["process"].poll()
                if return_code is not None:
                    print(f"\n{proc_info['name']} exited with code {return_code}")
                    print("Stopping all servers...")
                    manager.stop_all()
                    sys.exit(1)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\nError: {e}")
        manager.stop_all()
        sys.exit(1)


if __name__ == "__main__":
    main()

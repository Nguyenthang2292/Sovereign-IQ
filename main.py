import subprocess
import sys
import time
import signal
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


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

    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"

    for config in frontend_configs:
        manager.add_process(
            cmd=[npm_cmd, "run", "dev"],
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
        start_backend_servers(manager)
        time.sleep(2)
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

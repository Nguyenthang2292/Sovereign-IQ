"""
Kill processes on specified ports.
Usage:
    python kill_ports.py 8001 8002 5173 5174
    python kill_ports.py --all
"""

import argparse
import subprocess
import sys
import platform


def kill_port(port):
    """Kill process using the specified port."""
    system = platform.system()

    try:
        if system == "Windows":
            # Windows: netstat + taskkill
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True,
                text=True,
                check=True,
            )

            for line in result.stdout.split("\n"):
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        try:
                            subprocess.run(
                                ["taskkill", "/F", "/PID", pid],
                                capture_output=True,
                            )
                            print(f"✅ Killed process {pid} on port {port}")
                        except subprocess.CalledProcessError:
                            print(f"❌ Failed to kill process {pid} on port {port}")
        else:
            # Linux/Mac: lsof
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True,
                text=True,
            )

            if result.stdout.strip():
                pids = result.stdout.strip().split("\n")
                for pid in pids:
                    try:
                        subprocess.run(
                            ["kill", "-9", pid],
                            capture_output=True,
                        )
                        print(f"✅ Killed process {pid} on port {port}")
                    except subprocess.CalledProcessError:
                        print(f"❌ Failed to kill process {pid} on port {port}")
            else:
                print(f"ℹ️  No process found on port {port}")

    except FileNotFoundError:
        print(f"❌ Command not found on {system}")
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Kill processes on specified ports")
    parser.add_argument("ports", nargs="*", type=int, help="Ports to kill (e.g., 8001 8002 5173 5174)")
    parser.add_argument("--all", action="store_true", help="Kill all web app ports (8001, 8002, 5173, 5174)")

    args = parser.parse_args()

    ports_to_kill = args.ports

    if args.all:
        ports_to_kill = [8001, 8002, 5173, 5174]

    if not ports_to_kill:
        parser.print_help()
        return 1

    print("=" * 60)
    print(f"🛑 Killing processes on ports: {', '.join(map(str, ports_to_kill))}")
    print("=" * 60)

    for port in ports_to_kill:
        kill_port(port)

    print("=" * 60)
    print("✅ Done!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())

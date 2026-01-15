"""
Kill processes on ports 5000 and 5173 automatically.
"""

import subprocess
import sys
import time


def kill_port(port):
    """Kill process using specific port."""
    try:
        result = subprocess.run(f"netstat -ano | findstr :{port}", shell=True, capture_output=True, text=True)

        if result.returncode == 0 and result.stdout:
            lines = result.stdout.strip().split("\n")
            pids = []

            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    pid = parts[-1]
                    if pid not in pids:
                        pids.append(pid)

            if pids:
                print(f"  Found {len(pids)} process(es) on port {port}")

                for pid in pids:
                    try:
                        subprocess.run(f"taskkill /F /PID {pid}", shell=True, capture_output=True)
                        print(f"  ✅ Killed PID {pid}")
                    except Exception as e:
                        print(f"  ❌ Failed to kill PID {pid}: {e}")

                time.sleep(0.5)
            else:
                print(f"  🟢 Port {port} is available")
        else:
            print(f"  🟢 Port {port} is available")

    except Exception as e:
        print(f"  ⚠️  Error checking port {port}: {e}")


def main():
    """Main entry point."""
    print("=" * 60)
    print("🧹 Cleaning Up Ports")
    print("=" * 60)
    print()

    ports = [5000, 5173, 5174, 5175, 5176]

    for port in ports:
        print(f"🔍 Checking port {port}...")
        kill_port(port)
        print()

    print("=" * 60)
    print("✅ Cleanup Complete!")
    print("=" * 60)
    print()
    print("💡 Now you can run: python run_atc_visualizer.py")
    print("   Or: start_visualizer_clean.bat")


if __name__ == "__main__":
    main()

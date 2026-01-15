"""
Debug script to check if ports are in use.
"""

import socket
import sys


def check_port(port):
    """Check if a port is in use."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", port))
            return False
    except (socket.error, OSError):
        return True


def main():
    """Check ports 5000 and 5173."""
    print("=" * 60)
    print("🔍 Port Check")
    print("=" * 60)
    print()

    ports = [5000, 5173]

    for port in ports:
        in_use = check_port(port)
        status = "🔴 IN USE" if in_use else "🟢 AVAILABLE"
        print(f"Port {port}: {status}")

        if in_use:
            print(f"  ⚠️  Port {port} is already in use!")
            print(f"  💡 Kill the process using this port first:")

            if port == 5000:
                print(f"     Windows: netstat -ano | findstr :5000")
                print(f"              taskkill /F /PID <PID>")
            elif port == 5173:
                print(f"     Windows: netstat -ano | findstr :5173")
                print(f"              taskkill /F /PID <PID>")
        print()

    print("=" * 60)


if __name__ == "__main__":
    main()

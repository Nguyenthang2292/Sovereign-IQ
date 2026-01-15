"""
Check health of all web applications.
Usage:
    python health_check.py
"""

import argparse
import requests
import sys
from pathlib import Path
from typing import Dict, List, Tuple


# Configuration
APPS = {
    "gemini_analyzer": {
        "name": "Gemini Chart Analyzer",
        "backend_url": "http://localhost:8001/health",
        "frontend_url": "http://localhost:5173",
        "api_docs_url": "http://localhost:8001/docs",
    },
    "atc_visualizer": {
        "name": "ATC Visualizer",
        "backend_url": "http://localhost:8002/api/health",
        "frontend_url": "http://localhost:5174",
        "api_docs_url": "http://localhost:8002/docs",
    },
}


def check_url(url: str, timeout: int = 2) -> Tuple[bool, str]:
    """
    Check if a URL is accessible.

    Returns:
        (success, message)
    """
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            return True, "OK"
        else:
            return False, f"Status {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Connection refused"
    except requests.exceptions.Timeout:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def check_app(app_key: str, config: Dict) -> Dict:
    """
    Check health of an application.

    Returns:
        Health status dict
    """
    health = {
        "name": config["name"],
        "backend": False,
        "backend_msg": "",
        "frontend": False,
        "frontend_msg": "",
        "api_docs": False,
        "api_docs_msg": "",
    }

    # Check backend
    backend_ok, backend_msg = check_url(config["backend_url"])
    health["backend"] = backend_ok
    health["backend_msg"] = backend_msg

    # Check frontend
    frontend_ok, frontend_msg = check_url(config["frontend_url"])
    health["frontend"] = frontend_ok
    health["frontend_msg"] = frontend_msg

    # Check API docs
    docs_ok, docs_msg = check_url(config["api_docs_url"])
    health["api_docs"] = docs_ok
    health["api_docs_msg"] = docs_msg

    return health


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Check health of all web applications")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    args = parser.parse_args()

    print("=" * 80)
    print("🏥 Web Apps Health Check")
    print("=" * 80)

    results = {}

    for app_key, config in APPS.items():
        health = check_app(app_key, config)
        results[app_key] = health

        # Print results
        print(f"\n{health['name']}:")
        print(f"  Backend:  {'✅' if health['backend'] else '❌'} {health['backend_msg']}")
        print(f"  Frontend: {'✅' if health['frontend'] else '❌'} {health['frontend_msg']}")
        print(f"  API Docs: {'✅' if health['api_docs'] else '❌'} {health['api_docs_msg']}")

    # Summary
    print("\n" + "=" * 80)
    print("📊 Summary:")
    print("=" * 80)

    total_apps = len(APPS)
    backend_up = sum(1 for r in results.values() if r["backend"])
    frontend_up = sum(1 for r in results.values() if r["frontend"])

    print(f"\nBackend Services:  {backend_up}/{total_apps} up")
    print(f"Frontend Services: {frontend_up}/{total_apps} up")

    if args.json:
        import json

        print("\n" + "=" * 80)
        print("📄 JSON Output:")
        print("=" * 80)
        print(json.dumps(results, indent=2))

    print("\n" + "=" * 80)

    return 0 if (backend_up == total_apps and frontend_up == total_apps) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
        sys.exit(1)

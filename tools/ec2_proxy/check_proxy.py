"""
EC2 Proxy Health Check
======================
Kiểm tra SSH tunnel + xác nhận outbound IP = EC2 Elastic IP

Usage:
    python tools/ec2_proxy/check_proxy.py
    python tools/ec2_proxy/check_proxy.py --real   # Xem IP thật của máy
"""

import os
import socket
import sys
from pathlib import Path

# Ensure project root on path
_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root))

try:
    from dotenv import load_dotenv

    load_dotenv(_root / "modules" / "auto_trade" / ".env", override=False)
    load_dotenv(_root / ".env", override=False)
except ImportError:
    pass

try:
    import httpx
except ImportError:
    print("❌ httpx not installed. Run:")
    print("   pip install httpx[socks]")
    sys.exit(1)


PROXY_URL = f"socks5h://127.0.0.1:{os.getenv('EC2_PROXY_PORT', '1080')}"
CHECK_URL = "https://api.ipify.org?format=json"
BINANCE_PING = "https://api.binance.com/api/v3/ping"


def get_ip(use_proxy: bool = True) -> str | None:
    proxy = PROXY_URL if use_proxy else None
    try:
        with httpx.Client(proxy=proxy, timeout=10.0) as client:
            r = client.get(CHECK_URL)
            return r.json().get("ip")
    except Exception as e:
        return None


def is_port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            return True
    except Exception:
        return False


def test_binance(use_proxy: bool = True) -> bool:
    proxy = PROXY_URL if use_proxy else None
    try:
        with httpx.Client(proxy=proxy, timeout=10.0) as client:
            r = client.get(BINANCE_PING)
            return r.status_code == 200
    except Exception:
        return False


def main():
    real_mode = "--real" in sys.argv
    proxy_port = int(os.getenv("EC2_PROXY_PORT", "1080"))
    ec2_host = os.getenv("EC2_PROXY_HOST", "NOT_SET")
    enabled = os.getenv("EC2_PROXY_ENABLED", "false").lower() == "true"

    print("=" * 58)
    print("  Binance EC2 Proxy Health Check")
    print("=" * 58)
    print(f"  EC2_PROXY_ENABLED : {enabled}")
    print(f"  EC2_PROXY_HOST    : {ec2_host}")
    print(f"  SOCKS5 Port       : {proxy_port}")
    print()

    if real_mode:
        print("[MODE: Direct — showing real local IP]")
        ip = get_ip(use_proxy=False)
        print(f"  Real local IP: {ip or 'unknown'}")
        print("  (This IP changes when router restarts)")
        return

    # 1. Check port
    port_open = is_port_open(proxy_port)
    print(f"[1] SOCKS5 port {proxy_port} listening: {'✅ YES' if port_open else '❌ NO'}")
    if not port_open:
        print("    → SSH tunnel not running.")
        print("    → App will start it automatically on next launch.")
        print("    → Or run manually:")
        print(f"      ssh -D {proxy_port} -N -i <key.pem> ec2-user@{ec2_host}")
        print()

    # 2. Check outbound IP via proxy
    print(f"[2] Outbound IP via proxy...")
    proxy_ip = get_ip(use_proxy=True)
    if proxy_ip:
        print(f"    ✅ {proxy_ip}")
        expected = ec2_host
        if proxy_ip == expected:
            print(f"    ✅ Matches EC2 Elastic IP: {expected}")
        else:
            print(f"    ⚠️  EC2_PROXY_HOST={expected} but outbound={proxy_ip}")
    else:
        print("    ❌ Cannot get IP via proxy")

    # 3. Real IP for comparison
    real_ip = get_ip(use_proxy=False)
    print(f"[3] Real local IP (without proxy): {real_ip or 'unknown'}")
    if proxy_ip and real_ip and proxy_ip != real_ip:
        print("    ✅ Proxy is hiding your real IP correctly")
    elif proxy_ip and real_ip and proxy_ip == real_ip:
        print("    ⚠️  Same IP — proxy may not be routing correctly")

    # 4. Binance connectivity
    print(f"[4] Binance API reachable via proxy...")
    binance_ok = test_binance(use_proxy=True)
    print(f"    {'✅ YES' if binance_ok else '❌ NO'}")

    # Summary
    print()
    print("=" * 58)
    if proxy_ip and binance_ok:
        print("  ✅ EVERYTHING OK")
        print(f"  📌 Whitelist this IP on Binance: {proxy_ip}")
        if proxy_ip == ec2_host:
            print("  ✅ IP matches EC2_PROXY_HOST in .env")
    else:
        print("  ❌ Issues detected — see above")
        if not enabled:
            print("  ℹ️  EC2_PROXY_ENABLED=false in .env")
            print("     Set to true after deploying EC2")
    print("=" * 58)


if __name__ == "__main__":
    main()

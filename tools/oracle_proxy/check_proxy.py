"""
Binance Proxy Health Check
==========================
Verify rằng SSH SOCKS5 tunnel đang hoạt động và
outbound IP = Oracle Reserved IP đã whitelist trên Binance.

Usage:
    python tools/oracle_proxy/check_proxy.py
    python tools/oracle_proxy/check_proxy.py --no-proxy  # xem IP thật của máy
"""

import os
import sys
from pathlib import Path

# Ensure project root on path
_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root))

try:
    import httpx
except ImportError:
    print("❌ httpx not installed. Run: pip install httpx[socks]")
    sys.exit(1)

PROXY_URL = "socks5h://127.0.0.1:1080"
CHECK_URLS = [
    "https://api.ipify.org?format=json",
    "https://api4.my-ip.io/ip.json",
]
BINANCE_TEST = "https://api.binance.com/api/v3/ping"


def get_ip(use_proxy: bool = True) -> str | None:
    """Lấy public IP, có hoặc không qua proxy."""
    proxy = PROXY_URL if use_proxy else None
    label = f"via proxy {PROXY_URL}" if use_proxy else "direct (no proxy)"

    for url in CHECK_URLS:
        try:
            with httpx.Client(proxy=proxy, timeout=10.0) as client:
                resp = client.get(url)
                resp.raise_for_status()
                data = resp.json()
                ip = data.get("ip") or data.get("IP")
                if ip:
                    return ip
        except Exception:
            continue

    return None


def test_binance_connectivity(use_proxy: bool = True) -> bool:
    """Kiểm tra kết nối tới Binance API."""
    proxy = PROXY_URL if use_proxy else None
    try:
        with httpx.Client(proxy=proxy, timeout=10.0) as client:
            resp = client.get(BINANCE_TEST)
            return resp.status_code == 200
    except Exception as e:
        print(f"   Binance connectivity error: {e}")
        return False


def main():
    no_proxy = "--no-proxy" in sys.argv

    print("=" * 55)
    print("  Binance Proxy Health Check")
    print("=" * 55)

    if no_proxy:
        print("\n[MODE: Direct — no proxy]")
        ip = get_ip(use_proxy=False)
        if ip:
            print(f"  Your real IP: {ip}")
            print("  (This IP changes when router restarts)")
        else:
            print("  ❌ Could not detect IP")
        return

    # === Check proxy IP ===
    print(f"\n[1] Checking outbound IP via proxy ({PROXY_URL})...")
    proxy_ip = get_ip(use_proxy=True)

    if proxy_ip:
        print(f"  ✅ Outbound IP via proxy: {proxy_ip}")
        print(f"     → This is the IP to whitelist on Binance")
        print(f"     → This IP is FIXED (Oracle Reserved IP, never changes)")
    else:
        print("  ❌ Cannot reach internet via proxy")
        print("     Possible causes:")
        print("     • SSH tunnel is not running")
        print("     • Oracle VM is down")
        print("     • Port 1080 not listening")
        print()
        print("  Debug: Check if tunnel is running:")
        print("    netstat -an | Select-String 1080")
        print("    Get-ScheduledTask -TaskName BinanceProxyTunnel")
        return

    # === Check direct IP (for comparison) ===
    print(f"\n[2] Your real local IP (without proxy)...")
    real_ip = get_ip(use_proxy=False)
    if real_ip:
        print(f"  Local IP: {real_ip}")
        if real_ip == proxy_ip:
            print("  ⚠️  WARNING: Proxy IP = real IP — tunnel may not be working!")
        else:
            print(f"  ✅ OK — proxy hides your real IP correctly")

    # === Test Binance connectivity ===
    print(f"\n[3] Testing Binance API connectivity via proxy...")
    ok = test_binance_connectivity(use_proxy=True)
    if ok:
        print("  ✅ Binance API reachable via proxy")
    else:
        print("  ❌ Cannot reach Binance API via proxy")
        print("     If IP is correctly whitelisted, this should work.")

    print()
    print("=" * 55)
    if proxy_ip and ok:
        print(f"  ✅ ALL OK — whitelist this IP on Binance:")
        print(f"     {proxy_ip}")
    else:
        print("  ❌ Issues detected — see messages above")
    print("=" * 55)


if __name__ == "__main__":
    main()

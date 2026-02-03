"""
Debug script to compare ccxt initialization methods
"""

import os
import sys

import ccxt
from dotenv import load_dotenv

# Load .env
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path, override=True)

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

print("=" * 60)
print("DEBUGGING BINANCE API ENDPOINT CONFIGURATION")
print("=" * 60)

# Method 1: From test_demo_simple.py (WORKS)
print("\n1️⃣  Testing Method 1 (test_demo_simple.py style):")
print("-" * 60)
try:
    exchange1 = ccxt.binance(
        {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",
            },
        }
    )

    # Skip load_markets to avoid spot API calls
    print(f"✅ Exchange initialized")
    print(f"   Default type: {exchange1.options.get('defaultType')}")

    # Check URLs
    print(f"\n   Configured URLs:")
    if "urls" in exchange1.urls:
        for key, val in exchange1.urls.items():
            if isinstance(val, dict):
                print(f"     {key}:")
                for k, v in val.items():
                    print(f"       {k}: {v}")
            else:
                print(f"     {key}: {val}")

    # Try to fetch futures balance without load_markets
    print(f"\n   Testing futures balance...")
    try:
        # Use fapiPrivate endpoints directly
        balance = exchange1.fapiPrivateV2GetBalance()
        print(f"   ✅ Futures balance endpoint works!")
        print(f"      Found {len(balance)} balance entries")
    except Exception as e:
        print(f"   ❌ Error: {e}")

except Exception as e:
    print(f"❌ Method 1 failed: {e}")

# Method 2: From BinanceClient (FAILS)
print("\n\n2️⃣  Testing Method 2 (BinanceClient style):")
print("-" * 60)
try:
    exchange2 = ccxt.binance(
        {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",
                "adjustForTimeDifference": True,
            },
        }
    )

    print(f"✅ Exchange initialized")
    print(f"   Default type: {exchange2.options.get('defaultType')}")

    # Try fetch_balance (triggers load_markets which calls spot API)
    print(f"\n   Testing fetch_balance() method...")
    try:
        balance = exchange2.fetch_balance()
        print(f"   ✅ fetch_balance() works!")
        usdt = balance.get("USDT", {})
        print(f"      USDT: {usdt}")
    except Exception as e:
        error_msg = str(e)
        print(f"   ❌ fetch_balance() failed: {error_msg}")

        if "Invalid Api-Key" in error_msg or "-2008" in error_msg:
            print(f"\n   🔍 ROOT CAUSE FOUND:")
            print(f"      fetch_balance() triggers load_markets()")
            print(f"      load_markets() calls fetch_currencies()")
            print(f"      fetch_currencies() uses SPOT API endpoint")
            print(f"      Demo keys DON'T WORK with Spot API!")

except Exception as e:
    print(f"❌ Method 2 failed: {e}")

# Solution
print("\n\n" + "=" * 60)
print("💡 SOLUTION")
print("=" * 60)
print("\nDemo API keys ONLY work with Futures API endpoints.")
print("\nOptions:")
print("1. Skip fetch_balance() - use fetch_positions() instead")
print("2. Use direct fapi endpoints: exchange.fapiPrivateV2GetBalance()")
print("3. Set warnOnFetchOpenOrdersWithoutSymbol to False")
print("4. Don't call load_markets() for demo keys")
print("\nRecommended: Use direct futures endpoints instead of generic methods")

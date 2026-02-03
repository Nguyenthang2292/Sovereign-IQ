"""
API Key Permissions Diagnostic Tool
Checks what's wrong with your Binance demo API keys.
"""
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Fix import path
current_file = os.path.abspath(__file__)
auto_trade_dir = os.path.dirname(current_file)
modules_dir = os.path.dirname(auto_trade_dir)
project_root = os.path.dirname(modules_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
import ccxt
import requests

# Load .env
env_file_path = os.path.join(auto_trade_dir, ".env")
load_dotenv(env_file_path, override=True)

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

print("=" * 70)
print("BINANCE API KEY PERMISSIONS DIAGNOSTIC")
print("=" * 70)

print(f"\n🔑 API Key: {api_key[:15]}...{api_key[-5:]}")
print(f"🔐 Secret: {api_secret[:10]}...{api_secret[-5:]}")

# Test 1: Check API Key Format
print("\n" + "=" * 70)
print("TEST 1: API Key Format Validation")
print("=" * 70)

if len(api_key) >= 64:
    print("✅ API key length looks correct (64+ chars)")
else:
    print(f"⚠️  API key seems short ({len(api_key)} chars) - should be 64+")

if len(api_secret) >= 64:
    print("✅ Secret length looks correct (64+ chars)")
else:
    print(f"⚠️  Secret seems short ({len(api_secret)} chars) - should be 64+")

# Test 2: Try Public Endpoint (No Auth)
print("\n" + "=" * 70)
print("TEST 2: Public Endpoint (No Authentication)")
print("=" * 70)

try:
    response = requests.get("https://demo-fapi.binance.com/fapi/v1/ping", timeout=10)
    if response.status_code == 200:
        print("✅ Demo endpoint is reachable")
        print(f"   Response: {response.json()}")
    else:
        print(f"⚠️  Unexpected response: {response.status_code}")
except Exception as e:
    print(f"❌ Cannot reach demo endpoint: {e}")

# Test 3: Try Different Authentication Methods
print("\n" + "=" * 70)
print("TEST 3: Authentication Tests")
print("=" * 70)

configs_to_test = [
    {
        "name": "Demo Endpoints (demo-fapi.binance.com)",
        "config": {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
            "urls": {
                "api": {
                    "fapiPublic": "https://demo-fapi.binance.com/fapi/v1",
                    "fapiPrivate": "https://demo-fapi.binance.com/fapi/v1",
                    "fapiPrivateV2": "https://demo-fapi.binance.com/fapi/v2",
                }
            },
        },
    },
    {
        "name": "Production Endpoints (fapi.binance.com)",
        "config": {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        },
    },
]

for test_config in configs_to_test:
    print(f"\n🔹 Testing: {test_config['name']}")
    print("-" * 70)

    try:
        exchange = ccxt.binance(test_config["config"])

        # Try server time first (no auth needed)
        try:
            server_time = exchange.fetch_time()
            print(f"  ✅ Server time fetched: {server_time}")
        except Exception as e:
            print(f"  ⚠️  Server time failed: {e}")

        # Try account balance (requires auth)
        try:
            balance = exchange.fapiPrivateV2GetBalance()
            print(f"  ✅ Balance API successful!")
            usdt_data = next((item for item in balance if item["asset"] == "USDT"), None)
            if usdt_data:
                print(f"     USDT Balance: {float(usdt_data.get('balance', 0)):,.2f}")
        except Exception as e:
            error_str = str(e)
            print(f"  ❌ Balance API failed: {error_str}")

            if "-2015" in error_str:
                print("     → Error -2015: Invalid API-key, IP, or permissions")
                print("     → Possible causes:")
                print("        1. API key doesn't have 'Enable Futures' permission")
                print("        2. IP address is restricted")
                print("        3. API key is for Spot, not Futures")
                print("        4. Using demo keys on production endpoints (or vice versa)")
            elif "-2008" in error_str:
                print("     → Error -2008: Invalid API Key ID")
                print("     → API key is expired or doesn't exist")

    except Exception as e:
        print(f"  ❌ Exchange initialization failed: {e}")

# Test 4: IP Check
print("\n" + "=" * 70)
print("TEST 4: IP Address Check")
print("=" * 70)

try:
    ip_response = requests.get("https://api.ipify.org?format=json", timeout=10)
    if ip_response.status_code == 200:
        your_ip = ip_response.json().get("ip")
        print(f"🌐 Your current IP address: {your_ip}")
        print("\nIf your API key has IP restrictions, make sure this IP is whitelisted.")
        print("For testing, set API key to 'Unrestricted' in Binance API Management.")
    else:
        print("⚠️  Could not determine your IP address")
except Exception as e:
    print(f"❌ IP check failed: {e}")

# Summary and Recommendations
print("\n" + "=" * 70)
print("DIAGNOSIS SUMMARY & RECOMMENDATIONS")
print("=" * 70)

print("""
🔍 If you're seeing -2015 errors, check these settings in Binance:

1. **API Key Type**:
   ✅ Must create key while in "Demo Trading" mode
   ✅ Must have "Enable Futures" permission checked
   ❌ Spot-only keys will NOT work

2. **IP Restrictions**:
   ✅ Set to "Unrestricted" for testing
   ❌ If restricted, add your current IP to whitelist

3. **API Key Permissions** (check in Binance API Management):
   ✅ Enable Reading
   ✅ Enable Futures
   ❌ DO NOT enable Withdrawal (security risk)

4. **Correct Endpoint**:
   ✅ Demo keys → demo-fapi.binance.com
   ✅ Real keys → fapi.binance.com
   ❌ Don't mix demo keys with production endpoints

📋 Steps to Fix:
   1. Login to Binance
   2. Switch to "Demo Trading" mode (top right)
   3. Go to API Management
   4. Create NEW API key with:
      - Enable Reading: ✅
      - Enable Futures: ✅
      - IP Restriction: Unrestricted
   5. Copy keys immediately (secret shown only once)
   6. Update .env file
   7. Re-run this diagnostic

🔗 Binance Demo Trading:
   https://www.binance.com/en/futures/BTCUSDT (click Demo Trading button)
""")

print("=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)

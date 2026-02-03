"""
Diagnostic script to verify Binance endpoint configuration.
"""
import os
import sys
import io

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

# Load .env
env_file_path = os.path.join(auto_trade_dir, ".env")
load_dotenv(env_file_path, override=True)

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

print("=" * 70)
print("BINANCE ENDPOINT DIAGNOSTICS")
print("=" * 70)

print(f"\n📍 API Key (first 15 chars): {api_key[:15]}...")
print(f"📍 API Key (last 5 chars): ...{api_key[-5:]}")

# Test 1: Check CCXT's default URLs
print("\n" + "=" * 70)
print("TEST 1: CCXT Default Binance Configuration")
print("=" * 70)

exchange_default = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',
        'adjustForTimeDifference': True,
    }
})

print("\nDefault URLs:")
for key, value in exchange_default.urls.items():
    if isinstance(value, dict):
        print(f"\n  {key}:")
        for k, v in value.items():
            print(f"    {k}: {v}")
    else:
        print(f"  {key}: {value}")

# Test 2: Check with testnet=True config (demo-fapi)
print("\n" + "=" * 70)
print("TEST 2: Demo Configuration (demo-fapi.binance.com)")
print("=" * 70)

exchange_demo = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',
        'adjustForTimeDifference': True,
    },
    'urls': {
        'api': {
            'public': 'https://demo-fapi.binance.com/fapi/v1',
            'private': 'https://demo-fapi.binance.com/fapi/v1',
        }
    }
})

print("\nDemo URLs:")
for key, value in exchange_demo.urls.items():
    if isinstance(value, dict):
        print(f"\n  {key}:")
        for k, v in value.items():
            print(f"    {k}: {v}")
    else:
        print(f"  {key}: {value}")

# Test 3: Check CCXT's testnet option
print("\n" + "=" * 70)
print("TEST 3: CCXT's setSandboxMode(True)")
print("=" * 70)

exchange_sandbox = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',
        'adjustForTimeDifference': True,
    }
})

try:
    exchange_sandbox.set_sandbox_mode(True)
    print("\n✅ Sandbox mode enabled")
    print("\nSandbox URLs:")
    for key, value in exchange_sandbox.urls.items():
        if isinstance(value, dict):
            print(f"\n  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
except Exception as e:
    print(f"\n⚠️  Sandbox mode error: {e}")

# Test 4: Try actual API calls
print("\n" + "=" * 70)
print("TEST 4: API Call Tests")
print("=" * 70)

test_configs = [
    ("Production endpoints (default)", exchange_default),
    ("Demo endpoints (demo-fapi)", exchange_demo),
]

for name, exchange in test_configs:
    print(f"\n🔹 {name}")
    print("-" * 70)

    # Try balance check
    try:
        balance = exchange.fapiPrivateV2GetBalance()
        print(f"  ✅ Balance API call successful")
        usdt_data = next((item for item in balance if item["asset"] == "USDT"), None)
        if usdt_data:
            print(f"     Available USDT: {float(usdt_data.get('availableBalance', 0)):,.2f}")
    except Exception as e:
        error_msg = str(e)
        print(f"  ❌ Balance API call failed: {error_msg}")
        if "-2008" in error_msg:
            print(f"     → Invalid API Key ID")
        elif "-2015" in error_msg:
            print(f"     → Invalid API key, IP, or permissions")

print("\n" + "=" * 70)
print("DIAGNOSIS COMPLETE")
print("=" * 70)

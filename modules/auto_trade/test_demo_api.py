# Fix Windows console encoding
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Fix import path: thêm project root vào sys.path TRƯỚC khi import bất cứ thứ gì
current_file = os.path.abspath(__file__)
auto_trade_dir = os.path.dirname(current_file)  # modules/auto_trade
modules_dir = os.path.dirname(auto_trade_dir)  # modules
project_root = os.path.dirname(modules_dir)  # project root

# Thêm project root vào sys.path để import config package
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# DEBUG: In ra sys.path để kiểm tra
print("=" * 60)
print("DEBUG: sys.path after fix:")
for i, p in enumerate(sys.path[:5]):  # Chỉ in 5 path đầu
    print(f"  [{i}] {p}")
print("=" * 60)

# BÂY GIỜ mới import các modules từ project (sau khi fix path)
from dotenv import load_dotenv

print("✅ dotenv imported successfully")

try:
    from modules.auto_trade.execution.binance_client import BinanceClient
    from modules.auto_trade.number_utils import coerce_float

    print("✅ BinanceClient imported successfully")
except Exception as e:
    print(f"❌ Error importing BinanceClient: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Load environment variables từ file .env
# FORCE OVERRIDE: Ưu tiên .env file thay vì Windows environment variables
env_file_path = os.path.join(auto_trade_dir, ".env")

print(f"\n📂 Loading .env from: {env_file_path}")
print(f"   File exists: {os.path.exists(env_file_path)}")

# QUAN TRỌNG: override=True để đè lên Windows env vars
load_dotenv(env_file_path, override=True)

# Verify API key sau khi load
loaded_key = os.getenv("BINANCE_API_KEY", "")
print(f"\n🔑 Loaded API Key: {loaded_key[:15]}..." if loaded_key else "❌ No API key loaded")

# Check if keys exist
if not loaded_key or loaded_key == "YOUR_API_KEY_HERE":
    print("\n" + "=" * 60)
    print("⚠️  NO VALID API KEYS CONFIGURED")
    print("=" * 60)
    print("\n📝 To test with Binance API, you need to:")
    print("\n1. Get API Keys from Binance:")
    print("   a) Demo/Paper Trading (Recommended for testing):")
    print("      - Login to your Binance account")
    print("      - Go to Futures > Demo Trading")
    print("      - Generate Demo API keys (free virtual funds)")
    print("      - Demo REST API: https://demo-fapi.binance.com")
    print("")
    print("   b) Real Trading (Use with caution!):")
    print("      - Visit: https://www.binance.com/")
    print("      - Go to API Management")
    print("      - Create API keys with 'Enable Futures' permission")
    print("")
    print("2. Update .env file:")
    print(f"   Edit: {env_file_path}")
    print("   Set: BINANCE_API_KEY=your_key_here")
    print("        BINANCE_API_SECRET=your_secret_here")
    print("        BINANCE_TESTNET=true  (for demo) or false (for real)")
    print("")
    print("3. Re-run this test script")
    print("=" * 60)
    sys.exit(1)

print("=" * 60)


def test_demo_connection():
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    print("-" * 50)
    print("Testing Demo Connection using keys:")
    api_key_preview = f"{api_key[:10]}...{api_key[-5:]}" if api_key else "None"
    print(f"API Key: {api_key_preview}")
    print("-" * 50)

    if not api_key or not api_secret:
        print("❌ Error: API Key or Secret not found in .env")
        return False

    try:
        # ⚠️⚠️⚠️ WARNING: REAL ACCOUNT MODE - REAL MONEY AT RISK! ⚠️⚠️⚠️
        # testnet=False means using PRODUCTION endpoints (fapi.binance.com)
        # All operations will use REAL funds from your Binance account
        print("\n⚠️⚠️⚠️ WARNING: REAL ACCOUNT MODE ⚠️⚠️⚠️")
        print("You are testing with REAL money on production Binance Futures!")
        print("Current balance: $0.89 USDT")
        print("=" * 60 + "\n")

        # Khởi tạo client
        # testnet=False: Uses production endpoints (fapi.binance.com)
        # testnet=True: Uses demo endpoints (demo-fapi.binance.com)
        client = BinanceClient(
            api_key=api_key,
            api_secret=api_secret,
            testnet=False,  # ⚠️ REAL ACCOUNT - Production endpoints
        )

        print("✅ Client initialized (PRODUCTION MODE)")

        # Verify the endpoint being used
        print("\n📍 Endpoint Verification:")
        print(f"   fapiPrivateV2 endpoint: {client.exchange.urls['api'].get('fapiPrivateV2', 'NOT SET')}")
        print(f"   fapiPrivate endpoint: {client.exchange.urls['api'].get('fapiPrivate', 'NOT SET')}")

        # 1. Kiểm tra Balance (dùng trực tiếp Futures API - QUAN TRỌNG cho demo keys!)
        print("\nChecking Balance (using Futures API)...")

        # Demo keys KHÔNG hỗ trợ fetch_balance() vì nó gọi Spot API
        # Phải dùng trực tiếp futures endpoints
        try:
            # Call futures balance endpoint directly
            # Note: Don't pass _self parameter - CCXT handles it automatically
            futures_balance = client.exchange.fapiPrivateV2GetBalance()  # pyright: ignore[reportCallIssue]

            print("✅ Balance Check Successful")

            # Find USDT balance
            usdt_data = next((item for item in futures_balance if item["asset"] == "USDT"), None)

            if usdt_data:
                available = float(usdt_data.get("availableBalance", 0))
                wallet_balance = float(usdt_data.get("balance", 0))
                print(f"   Available USDT: {available:,.2f}")
                print(f"   Wallet Balance: {wallet_balance:,.2f}")
            else:
                print("   No USDT balance found")

        except Exception as balance_error:
            print(f"⚠️  Balance check failed: {balance_error}")
            print("   (This is OK - continuing with other checks...)")

        # 2. Kiểm tra Positions
        print("\nChecking Positions...")
        positions = client.exchange.fetch_positions()
        open_positions = [p for p in positions if coerce_float(p.get("contracts")) > 0]

        print("✅ Positions Check Successful")
        print(f"   Open Positions: {len(open_positions)}")

        for pos in open_positions:
            print(f"   - {pos['symbol']}: {pos['side']} {pos['contracts']} contracts @ {pos['entryPrice']}")

        # 3. Test Market Data (để đảm bảo kết nối ổn định)
        print("\nChecking Market Data (BTC/USDT)...")
        ticker = client.exchange.fetch_ticker("BTC/USDT")
        print("✅ Market Data Check Successful")
        print(f"   BTC/USDT Price: {ticker['last']:,.2f}")

        print("\n" + "=" * 50)
        print("✅✅✅ CONNECTION TO REAL ACCOUNT SUCCESSFUL! ✅✅✅")
        print("=" * 50)
        print("\n⚠️⚠️⚠️ IMPORTANT REMINDERS ⚠️⚠️⚠️")
        print("1. You are now connected to REAL Binance Futures")
        print("2. Any orders placed will use REAL money")
        print("3. Current balance: Very low ($0.89 USDT)")
        print("4. Consider:")
        print("   - Enable dry_run mode for testing order logic")
        print("   - Start with MINIMUM position sizes")
        print("   - Deposit more funds if needed for meaningful testing")
        print("   - Use stop losses on every position")
        print("=" * 50)
        return True

    except Exception as e:
        error_msg = str(e)
        print("\n" + "=" * 50)
        print(f"❌ KẾT NỐI THẤT BẠI: {error_msg}")
        print("=" * 50)

        # Provide helpful error messages
        if "Invalid Api-Key ID" in error_msg or "-2008" in error_msg:
            print("\n🔧 API Key is INVALID or EXPIRED")
            print("\n   Possible reasons:")
            print("   1. Demo API keys have expired (they typically expire after a few months)")
            print("   2. Using wrong API key type (spot keys won't work for futures)")
            print("   3. API key was deleted or revoked")
            print("\n   Solutions:")
            print("   1. Get NEW demo keys from: Binance Futures > Demo Trading")
            print("      Demo API endpoint: https://demo-fapi.binance.com")
            print("   2. Or use your real Binance account keys (be careful!)")
            print("   3. Update .env file with new keys")
            print("   4. Set BINANCE_TESTNET=true for demo, false for real")

        elif "Signature" in error_msg or "-1022" in error_msg:
            print("\n🔧 API Signature Error")
            print("   - Check if API SECRET is correct")
            print("   - Verify system time is synchronized")

        elif "IP" in error_msg or "banned" in error_msg.lower():
            print("\n🔧 IP Restriction Error")
            print("   - Check if IP whitelist is configured")
            print("   - VPN may cause issues")

        else:
            print("\n📋 Full traceback:")
            import traceback

            traceback.print_exc()

        return False


if __name__ == "__main__":
    success = test_demo_connection()
    sys.exit(0 if success else 1)

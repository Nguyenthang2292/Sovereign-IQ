"""
Simple Binance Demo API Test
Kiểm tra kết nối với Binance Demo Account mà không cần import toàn bộ project
"""
import io
import os
import sys

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import ccxt
from dotenv import load_dotenv

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

def test_demo_connection():
    """Test kết nối với Binance Demo Account"""

    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    print("-" * 60)
    print("🔍 Testing Binance Demo Connection")
    print("-" * 60)

    if not api_key or not api_secret:
        print("❌ Error: API Key or Secret not found in .env")
        print(f"   Looking for .env at: {env_path}")
        print("\n📝 Please create .env file with:")
        print("   BINANCE_API_KEY=your_api_key")
        print("   BINANCE_API_SECRET=your_api_secret")
        return False

    print(f"✅ API Key found: {api_key[:10]}...{api_key[-5:]}")
    print(f"✅ API Secret found: {'*' * 40}")

    try:
        # Khởi tạo ccxt exchange
        print("\n📡 Initializing Binance Futures client...")
        exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',  # Sử dụng futures
                'adjustForTimeDifference': True,  # Auto-sync time with server
                'recvWindow': 60000,  # 60 seconds tolerance for timestamp
            }
        })

        # Set sandbox mode nếu là testnet
        testnet = os.getenv("BINANCE_TESTNET", "false").lower() == "true"
        if testnet:
            exchange.set_sandbox_mode(True)
            print("⚠️  Sandbox mode enabled (testnet)")
        else:
            print("✅ Production mode (Demo Account)")

        # 1. Test Balance
        print("\n" + "=" * 60)
        print("📊 Test 1: Checking Balance...")
        print("=" * 60)

        balance = exchange.fetch_balance()

        print("✅ Balance fetch successful!")

        # Show USDT balance
        if 'USDT' in balance:
            usdt_free = balance['USDT'].get('free', 0)
            usdt_used = balance['USDT'].get('used', 0)
            usdt_total = balance['USDT'].get('total', 0)

            print("\n💰 USDT Balance:")
            print(f"   Free:  {usdt_free:>15,.2f} USDT")
            print(f"   Used:  {usdt_used:>15,.2f} USDT")
            print(f"   Total: {usdt_total:>15,.2f} USDT")

        # Show all non-zero balances
        non_zero_balances = {
            currency: info
            for currency, info in balance.items()
            if isinstance(info, dict) and info.get('total', 0) > 0
        }

        if len(non_zero_balances) > 1:  # More than just USDT
            print("\n📈 Other balances:")
            for currency, info in non_zero_balances.items():
                if currency != 'USDT':
                    print(f"   {currency}: {info.get('total', 0):.8f}")

        # 2. Test Positions
        print("\n" + "=" * 60)
        print("📊 Test 2: Checking Open Positions...")
        print("=" * 60)

        positions = exchange.fetch_positions()

        # Filter for open positions (contracts > 0)
        open_positions = [
            p for p in positions
            if float(p.get('contracts', 0)) > 0
        ]

        print("✅ Positions fetch successful!")
        print(f"\n📍 Open Positions: {len(open_positions)}")

        if open_positions:
            print("\nPosition Details:")
            for pos in open_positions:
                symbol = pos.get('symbol', 'N/A')
                side = pos.get('side', 'N/A')
                contracts = float(pos.get('contracts', 0))
                entry_price = float(pos.get('entryPrice', 0))
                unrealized_pnl = float(pos.get('unrealizedPnl', 0))

                pnl_emoji = "🟢" if unrealized_pnl > 0 else "🔴" if unrealized_pnl < 0 else "⚪"

                print(f"\n   {pnl_emoji} {symbol}")
                print(f"      Side: {side}")
                print(f"      Contracts: {contracts:.8f}")
                print(f"      Entry Price: ${entry_price:,.2f}")
                print(f"      Unrealized P&L: ${unrealized_pnl:,.2f}")
        else:
            print("   No open positions")

        # 3. Test Market Data
        print("\n" + "=" * 60)
        print("📊 Test 3: Checking Market Data (BTC/USDT)...")
        print("=" * 60)

        ticker = exchange.fetch_ticker('BTC/USDT')

        print("✅ Market data fetch successful!")
        print("\n₿  BTC/USDT:")
        print(f"   Last Price: ${ticker.get('last', 0) or 0:>12,.2f}")
        print(f"   Bid:        ${ticker.get('bid', 0) or 0:>12,.2f}")
        print(f"   Ask:        ${ticker.get('ask', 0) or 0:>12,.2f}")
        print(f"   24h High:   ${ticker.get('high', 0) or 0:>12,.2f}")
        print(f"   24h Low:    ${ticker.get('low', 0) or 0:>12,.2f}")
        print(f"   24h Volume: {ticker.get('baseVolume', 0) or 0:>12,.2f} BTC")

        # 4. Test Account Info (Optional - may not work with all demo keys)
        print("\n" + "=" * 60)
        print("📊 Test 4: Checking Account Info (Optional)...")
        print("=" * 60)

        try:
            account = exchange.fapiPrivateGetAccount()
            print("✅ Account info fetch successful!")

            # Total wallet balance
            total_wallet = float(account.get('totalWalletBalance', 0))
            total_unrealized_profit = float(account.get('totalUnrealizedProfit', 0))
            total_margin_balance = float(account.get('totalMarginBalance', 0))
            available_balance = float(account.get('availableBalance', 0))

            print("\n💼 Account Summary:")
            print(f"   Wallet Balance:     ${total_wallet:>12,.2f}")
            print(f"   Unrealized P&L:     ${total_unrealized_profit:>12,.2f}")
            print(f"   Margin Balance:     ${total_margin_balance:>12,.2f}")
            print(f"   Available Balance:  ${available_balance:>12,.2f}")
        except Exception:
            print("⚠️  Account info endpoint not available with these demo keys")
            print("   (This is expected - basic balance and positions work fine)")

        # Final Success Message
        print("\n" + "=" * 60)
        print("✅ ✅ ✅  CORE TESTS PASSED!  ✅ ✅ ✅")
        print("=" * 60)
        print("\n🎉 Binance Demo Connection is working!")
        print("✅ Balance check: PASSED")
        print("✅ Positions check: PASSED")
        print("✅ Market data check: PASSED")
        print("\n🚀 You can now use this API for paper trading")
        print("\n⚠️  Remember: This is DEMO money, not real funds")

        return True

    except ccxt.AuthenticationError as e:
        print("\n" + "=" * 60)
        print("❌ AUTHENTICATION ERROR")
        print("=" * 60)
        print(f"Error: {str(e)}")
        print("\n🔧 Possible fixes:")
        print("   1. Check your API Key and Secret in .env file")
        print("   2. Make sure you're using Demo Account keys")
        print("   3. Verify keys haven't expired")
        return False

    except ccxt.NetworkError as e:
        print("\n" + "=" * 60)
        print("❌ NETWORK ERROR")
        print("=" * 60)
        print(f"Error: {str(e)}")
        print("\n🔧 Possible fixes:")
        print("   1. Check your internet connection")
        print("   2. Check if Binance API is accessible")
        print("   3. Try again in a few moments")
        return False

    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ UNEXPECTED ERROR")
        print("=" * 60)
        print(f"Error: {str(e)}")
        print(f"Type: {type(e).__name__}")

        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "🔥" * 30)
    print("   BINANCE DEMO ACCOUNT TEST")
    print("🔥" * 30 + "\n")

    success = test_demo_connection()

    if success:
        print("\n✨ Test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Test failed. Please fix the issues above.")
        sys.exit(1)

"""
Simple Binance Demo API Test
Kiểm tra kết nối với Binance Demo Account mà không cần import toàn bộ project
"""
import os
from pathlib import Path

import ccxt
import pytest
from dotenv import load_dotenv


def _setup_env():
    """Load environment variables from .env file."""
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    env_path = project_root / "modules" / "auto_trade" / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    load_dotenv(project_root / ".env")
    return str(env_path)


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("CI") == "true" or not os.getenv("BINANCE_API_KEY"),
    reason="Integration test requires valid API credentials and should not run in CI"
)
def test_demo_connection():
    """Test kết nối với Binance Demo Account"""
    env_path = _setup_env()

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
        pytest.skip("API credentials not found")

    print(f"✅ API Key found: {api_key[:10]}...{api_key[-5:]}")
    print(f"✅ API Secret found: {'*' * 40}")

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

    # 2. Test Positions
    print("\n" + "=" * 60)
    print("📊 Test 2: Checking Open Positions...")
    print("=" * 60)
    positions = exchange.fetch_positions()
    open_positions = [p for p in positions if float(p.get('contracts', 0)) > 0]
    print("✅ Positions fetch successful!")
    print(f"\n📍 Open Positions: {len(open_positions)}")

    # 3. Test Market Data
    print("\n" + "=" * 60)
    print("📊 Test 3: Checking Market Data (BTC/USDT)...")
    print("=" * 60)
    ticker = exchange.fetch_ticker('BTC/USDT')
    print("✅ Market data fetch successful!")
    print(f"\n₿  BTC/USDT Last: ${ticker.get('last', 0) or 0:,.2f}")

    print("\n" + "=" * 60)
    print("✅ ✅ ✅  CORE TESTS PASSED!  ✅ ✅ ✅")
    print("=" * 60)

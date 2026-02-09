import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from modules.auto_trade.execution.binance_client import BinanceClient
from modules.auto_trade.number_utils import coerce_float


def _setup_env():
    """Setup environment variables. Called from main or tests."""
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    env_path = project_root / "modules" / "auto_trade" / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)
    load_dotenv(project_root / ".env", override=True)
    return str(env_path)


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("CI") == "true" or not os.getenv("BINANCE_API_KEY"),
    reason="Integration test requires valid API credentials and should not run in CI"
)
def test_demo_connection():
    _setup_env()
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    print("-" * 50)
    print("Testing Demo Connection using keys:")
    api_key_preview = f"{api_key[:10]}...{api_key[-5:]}" if api_key else "None"
    print(f"API Key: {api_key_preview}")
    print("-" * 50)

    if not api_key or not api_secret:
        print("❌ Error: API Key or Secret not found in .env")
        pytest.skip("API credentials not found")

    # Khởi tạo client (testnet from env for safety in CI)
    testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
    client = BinanceClient(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
    )

    print("✅ Client initialized")

    # 1. Balance (futures)
    print("\nChecking Balance (using Futures API)...")
    try:
        futures_balance = client.exchange.fapiPrivateV2GetBalance()  # pyright: ignore[reportCallIssue]
        print("✅ Balance Check Successful")
        usdt_data = next((item for item in futures_balance if item["asset"] == "USDT"), None)
        if usdt_data:
            available = float(usdt_data.get("availableBalance", 0))
            wallet_balance = float(usdt_data.get("balance", 0))
            print(f"   Available USDT: {available:,.2f}")
            print(f"   Wallet Balance: {wallet_balance:,.2f}")
    except Exception as balance_error:
        print(f"⚠️  Balance check failed: {balance_error}")

    # 2. Positions
    print("\nChecking Positions...")
    positions = client.exchange.fetch_positions()
    open_positions = [p for p in positions if coerce_float(p.get("contracts")) > 0]
    print("✅ Positions Check Successful")
    print(f"   Open Positions: {len(open_positions)}")

    # 3. Market Data
    print("\nChecking Market Data (BTC/USDT)...")
    ticker = client.exchange.fetch_ticker("BTC/USDT")
    print("✅ Market Data Check Successful")
    print(f"   BTC/USDT Price: {ticker['last']:,.2f}")

    print("\n" + "=" * 50)
    print("✅✅✅ CONNECTION SUCCESSFUL! ✅✅✅")
    print("=" * 50)

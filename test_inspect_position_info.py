#!/usr/bin/env python3
"""
Inspect full position structure including 'info' field.
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.auto_trade.execution.binance_client import BinanceClient
from modules.auto_trade.gui.utils.credential_manager import CredentialManager
from modules.auto_trade.gui.utils.settings_manager import SettingsManager


def main():
    """Inspect position full structure."""

    print("=" * 80)
    print("🔬 Inspect Position Full Structure")
    print("=" * 80)

    # Load settings
    settings = SettingsManager()
    settings.load()

    # Get credentials
    credential_manager = CredentialManager()
    api_config = credential_manager.load_credentials("binance")

    # Get testnet setting
    api_settings = settings.get("api", {})
    testnet = api_settings.get("mode", "").upper() == "TESTNET"

    print(f"\n🔑 Using API mode: {'TESTNET' if testnet else 'PRODUCTION'}")

    # Create Binance client
    client = BinanceClient(
        api_key=api_config.get("api_key") or "",
        api_secret=api_config.get("api_secret") or "",
        testnet=testnet,
        dry_run=False,
    )

    print("\n📊 Fetching positions...")
    print("-" * 80)

    try:
        positions = client.exchange.fetch_positions()
        open_positions = [p for p in positions if float(p.get("contracts", 0)) != 0]
        
        print(f"\n✅ Found {len(open_positions)} open position(s)\n")
        
        for i, pos in enumerate(open_positions, 1):
            print(f"\n{'=' * 80}")
            print(f"POSITION #{i}: {pos.get('symbol')}")
            print(f"{'=' * 80}")
            
            # Basic fields
            print(f"\n📌 Basic Fields:")
            print(f"  Symbol:       {pos.get('symbol')}")
            print(f"  Side:         {pos.get('side')}")
            print(f"  Contracts:    {pos.get('contracts')}")
            print(f"  Entry Price:  {pos.get('entryPrice')}")
            print(f"  Mark Price:   {pos.get('markPrice')}")
            print(f"  Leverage:     {pos.get('leverage')}")
            print(f"  Notional:     {pos.get('notional')}")
            
            # Check for TP/SL in main fields
            print(f"\n🎯 TP/SL Fields (if available):")
            print(f"  takeProfit:   {pos.get('takeProfit')}")
            print(f"  stopLoss:     {pos.get('stopLoss')}")
            print(f"  takeProfitPrice: {pos.get('takeProfitPrice')}")
            print(f"  stopLossPrice: {pos.get('stopLossPrice')}")
            
            # Inspect 'info' field (raw from Binance)
            if 'info' in pos:
                info = pos['info']
                print(f"\n📋 Info Field (Raw Binance Data):")
                print(f"  symbol:           {info.get('symbol')}")
                print(f"  positionSide:     {info.get('positionSide')}")
                print(f"  entryPrice:       {info.get('entryPrice')}")
                print(f"  markPrice:        {info.get('markPrice')}")
                print(f"  unRealizedProfit: {info.get('unRealizedProfit')}")
                print(f"  liquidationPrice: {info.get('liquidationPrice')}")
                
                # Check for TP/SL related fields
                print(f"\n  🎯 TP/SL Related Fields:")
                print(f"  stopPrice:        {info.get('stopPrice')}")
                print(f"  takeProfitPrice:  {info.get('takeProfitPrice')}")
                print(f"  stopLossPrice:    {info.get('stopLossPrice')}")
                print(f"  takeProfit:       {info.get('takeProfit')}")
                print(f"  stopLoss:         {info.get('stopLoss')}")
                
                # Print all keys in info to see what's available
                print(f"\n  📋 All available keys in 'info':")
                for key in sorted(info.keys()):
                    print(f"    - {key}: {info[key]}")
            
            # Full JSON structure
            print(f"\n🔬 Full Position Structure (JSON):")
            print(json.dumps(pos, indent=2, default=str))
            print()

        # Additional check: Try to get account information
        print(f"\n{'=' * 80}")
        print("💼 Checking Account Info for Position Risk")
        print(f"{'=' * 80}\n")
        
        try:
            # Binance Futures has positionRisk endpoint
            if hasattr(client.exchange, 'fapiPrivateV2GetPositionrisk'):
                risk_info = client.exchange.fapiPrivateV2GetPositionrisk()
                print("✅ Position Risk Data:")
                print(json.dumps(risk_info, indent=2, default=str))
            else:
                print("⚠️  positionRisk endpoint not available")
        except Exception as e:
            print(f"❌ Error fetching position risk: {e}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("💡 What to Look For:")
    print("=" * 80)
    print("""
1. Check 'info' field for:
   - stopPrice / stopLossPrice
   - takeProfitPrice / takeProfit
   
2. If these fields are '0' or empty:
   → TP/SL are separate orders (need different API call)
   
3. If these fields have values:
   → TP/SL are embedded in position (update code to read from here)
""")
    print()


if __name__ == "__main__":
    main()

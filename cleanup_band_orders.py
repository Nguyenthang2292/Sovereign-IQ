"""
One-shot script to cancel all duplicate TP/SL orders for BAND/USDT.
Run once to clean up the 14 accumulated conditional orders.
"""

import os
import sys

sys.path.insert(0, r"c:\Users\Admin\Desktop\i-ching\crypto-probability")
from dotenv import load_dotenv

load_dotenv(r"c:\Users\Admin\Desktop\i-ching\crypto-probability\modules\auto_trade\.env")

from modules.auto_trade.execution.binance.order_management import (
    _classify_order_kind,
    _ccxt_futures_symbol,
    _fetch_all_open_orders,
)
from modules.auto_trade.execution.binance_client import BinanceClient

api_key = os.getenv("BINANCE_API_KEY", "")
api_secret = os.getenv("BINANCE_API_SECRET", "")
testnet = os.getenv("BINANCE_TESTNET", "false").lower() == "true"

print(f"Connecting (testnet={testnet})...")
client = BinanceClient(api_key=api_key, api_secret=api_secret, testnet=testnet, dry_run=False)
exchange = client.exchange

symbol = "BAND/USDT:USDT"

# Get entry price from position for accurate classification
entry_price = 0.0
side = "long"
try:
    positions = exchange.fetch_positions([symbol])
    for p in positions:
        if float(p.get("contracts", 0)) != 0:
            ep = p.get("entryPrice") or (p.get("info") or {}).get("entryPrice", 0)
            entry_price = float(ep or 0)
            amt = float((p.get("info") or {}).get("positionAmt", 0))
            side = "long" if amt > 0 else "short"
            print(f"Position found: side={side}, entry={entry_price}")
            break
    else:
        print("No open position found (may have already been closed)")
except Exception as e:
    print(f"Could not fetch position: {e}")

orders = _fetch_all_open_orders(exchange, symbol)
print(f"\nFound {len(orders)} open orders for {symbol}:\n")

for o in orders:
    kind = _classify_order_kind(o, entry_price, side)
    sp = o.get("stopPrice") or (o.get("info") or {}).get("stopPrice", "N/A")
    print(f"  id={o['id']}  kind={kind}  stopPrice={sp}  type={o.get('type')}")

print()
confirm = input(f"\nCancel ALL {len(orders)} orders? [y/N] ").strip().lower()
if confirm != "y":
    print("Aborted.")
    sys.exit(0)

cancelled = 0
skipped = 0
for o in orders:
    kind = _classify_order_kind(o, entry_price, side)
    if kind in ("tp", "sl"):
        try:
            exchange.cancel_order(o["id"], symbol)
            print(f"  ✅ Cancelled {kind.upper()} order {o['id']}")
            cancelled += 1
        except Exception as e:
            print(f"  ❌ Failed to cancel {o['id']}: {e}")
    else:
        print(f"  ⏭  Skipped unknown order {o['id']}")
        skipped += 1

print(f"\nDone. Cancelled={cancelled}, Skipped={skipped}")
print("The next EnsureTPSL run will place exactly 1 TP + 1 SL cleanly.")

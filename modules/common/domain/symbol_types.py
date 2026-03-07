from typing import NewType

DbSymbol = NewType("DbSymbol", str)  # "BTCUSDT"       ← DynamoDB
CcxtSymbol = NewType("CcxtSymbol", str)  # "BTC/USDT"      ← CCXT spot/scanner
FuturesSymbol = NewType("FuturesSymbol", str)  # "BTC/USDT:USDT" ← CCXT futures API

import json
import random
import math
import time


def generate_ohlcv(num_bars=200):
    """Generate sample OHLCV data"""
    base_price = 100.0
    timestamp = []
    open_prices = []
    high_prices = []
    low_prices = []
    close_prices = []
    volumes = []

    start_time = int(time.time()) - (num_bars * 3600)

    for i in range(num_bars):
        timestamp.append(start_time + (i * 3600))

        # Generate realistic price movements
        change = random.uniform(-2, 2)
        open_price = base_price + change
        close_price = open_price + random.uniform(-1, 1)
        high_price = max(open_price, close_price) + random.uniform(0, 0.5)
        low_price = min(open_price, close_price) - random.uniform(0, 0.5)

        open_prices.append(round(open_price, 2))
        high_prices.append(round(high_price, 2))
        low_prices.append(round(low_price, 2))
        close_prices.append(round(close_price, 2))
        volumes.append(random.randint(1000, 100000))

        base_price = close_price

    return {
        "timestamp": timestamp,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volumes,
    }


def generate_symbol_data():
    """Generate data for one symbol"""
    crypto_assets = [
        "BTC",
        "ETH",
        "BNB",
        "SOL",
        "XRP",
        "ADA",
        "DOGE",
        "AVAX",
        "DOT",
        "MATIC",
        "LINK",
        "UNI",
        "ATOM",
        "LTC",
        "ETC",
        "XLM",
        "ALGO",
        "VET",
        "FIL",
        "THETA",
        "AAVE",
        "XMR",
        "EOS",
        "MKR",
        "XTZ",
        "NEO",
        "CAKE",
        "KSM",
        "WAVES",
        "DASH",
        "ZEC",
        "ENJ",
        "COMP",
        "SNX",
        "YFI",
        "SAND",
        "MANA",
        "AXS",
        "FTM",
        "RUNE",
        "ONE",
        "ZIL",
        "CEL",
        "CRV",
        "SUSHI",
        "BEL",
        "POND",
        "STX",
        "KAVA",
        "ANKR",
        "RVN",
        "IMX",
        "LDO",
        "APT",
        "ARB",
        "OP",
        "SUI",
        "SEI",
        "INJ",
        "TIA",
        "PEPE",
        "SHIB",
        "WLD",
        "BLUR",
        "FET",
        "AGIX",
        "RNDR",
        "CFX",
        "GMX",
        "LRC",
        "ENJ",
        "CHZ",
        "BAT",
        "GALA",
        "MINA",
        "QNT",
        "GRT",
        "FTNT",
        "SXP",
        "KNC",
        "REEF",
        "NULS",
        "STMX",
        "ICX",
        "ZRX",
        "SKL",
        "ANKR",
        "GTO",
        "TWT",
        "SPELL",
        "JASMY",
        "MASK",
        "DYDX",
        "GNS",
        "MAGIC",
        "GMT",
        "HOOK",
        "SSV",
        "LDO",
        "RPL",
    ]

    # Choose a random symbol
    symbol = random.choice(crypto_assets) + "USDT"

    # Generate timeframes
    timeframes = {}
    for tf in ["1h", "4h", "1d"]:
        timeframes[tf] = generate_ohlcv(200)

    return {"symbol": symbol, "timeframes": timeframes}


def generate_test_data(num_symbols=120):
    """Generate test data for profiling"""
    symbols = []
    for _ in range(num_symbols):
        symbols.append(generate_symbol_data())

    config = {
        "robustness": "Medium",
        "weights": {"1h": 0.5, "4h": 0.3, "1d": 0.2},
        "threshold": 0.3,
        "min_signal": 0.0,
        "use_signal_strength": True,
        "lambda_param": 0.02,
        "decay": 0.03,
        "cutout": 0,
        "equity_floor": 0.25,
        "ma_configs": [
            {"ma_type": "EMA", "length": 28, "weight": 1.0},
            {"ma_type": "HMA", "length": 28, "weight": 1.0},
            {"ma_type": "WMA", "length": 28, "weight": 1.0},
            {"ma_type": "DEMA", "length": 28, "weight": 1.0},
            {"ma_type": "LSMA", "length": 28, "weight": 1.0},
            {"ma_type": "KAMA", "length": 28, "weight": 1.0},
        ],
    }

    return {"symbols": symbols, "config": config}


if __name__ == "__main__":
    # Generate test data
    data = generate_test_data(120)

    # Write to file
    with open("test_data_120.json", "w") as f:
        json.dump(data, f)

    print(f"Generated test data with {len(data['symbols'])} symbols")
    print(f"File size: {len(json.dumps(data)) / 1024:.1f} KB")

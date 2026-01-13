def parse_trading_signal_simple(response_text):
    signal = {"entry_price": None, "stop_loss": None}

    # Extract entry price
    import re

    entry_match = re.search(r"entry:\s*([0-9]+\.?[0-9]*)", response_text, re.IGNORECASE)
    if entry_match:
        print(f"Entry match: {entry_match.group(1)}")
        signal["entry_price"] = float(entry_match.group(1))

    # Extract stop loss
    sl_match = re.search(r"sl:\s*([0-9]+\.?[0-9]*)", response_text, re.IGNORECASE)
    if sl_match:
        print(f"SL match: {sl_match.group(1)}")
        signal["stop_loss"] = float(sl_match.group(1))

    return signal


response = "SHORT position. Entry: 1.2000, SL: 1.2500, TP2: 1.3000"
result = parse_trading_signal_simple(response)
print(f"Result: {result}")

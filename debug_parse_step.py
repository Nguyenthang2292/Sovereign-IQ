import re

response = "SHORT position. Entry: 1.2000, SL: 1.2500, TP2: 1.3000"

price_patterns = {
    "entry": [
        r"(?:entry|giá\s*vào):\s*([0-9]+\.?[0-9]*)",
        r"(?:into|at):\s*([0-9]+\.?[0-9]*)",
    ],
    "stop_loss": [
        r"(?:stop\s*loss|sl|cắt\s*lỗ):\s*([0-9]+\.?[0-9]*)",
        r"sl:\s*([0-9]+\.?[0-9]*)",
    ],
    "take_profit_1": [
        r"(?:take\s*profit|tp|chốt\s*lời)\s*1?:\s*([0-9]+\.?[0-9]*)",
        r"tp1?\s*([0-9]+\.?[0-9]*)",
    ],
    "take_profit_2": [
        r"(?:take\s*profit|tp|chốt\s*lời)\s*2?:\s*([0-9]+\.?[0-9]*)",
        r"tp2?\s*([0-9]+\.?[0-9]*)",
    ],
}

for field, patterns in price_patterns.items():
    print(f"\nProcessing field: {field}")
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        print(f"  Pattern: {pattern}")
        print(f"  Match: {match}")
        if match:
            try:
                value = float(match.group(1))
                print(f"  Value: {value}")
                break
            except (ValueError, IndexError) as e:
                print(f"  Error: {e}")
                pass

import re

response = "SHORT position. Entry: 1.2000, SL: 1.2500, TP2: 1.3000"

entry_patterns = [
    r"(?:entry|giá\s*vào):\s*([0-9]+\.?[0-9]*)",
    r"(?:into|at):\s*([0-9]+\.?[0-9]*)",
]

for pattern in entry_patterns:
    match = re.search(pattern, response, re.IGNORECASE)
    print(f"Pattern: {pattern}")
    print(f"Match: {match}")
    if match:
        print(f"Group 1: {match.group(1)}")
    print("---")

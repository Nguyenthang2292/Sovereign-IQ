import re

response = "Tăng giá. Giá vào: 1.2345, cắt lỗ: 1.2200, chốt lời 1: 2800"

# Test Vietnamese pattern
tang_match = re.search(r"tang", response, re.IGNORECASE)
print(f"Tang match: {tang_match}")

# Check original text
text_lower = response.lower()
print(f"Text lower: {text_lower}")
print(f"Contains 'tang': {'tang' in text_lower}")

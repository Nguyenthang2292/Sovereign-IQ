"""
API Configuration for external services.
File này chứa các API keys và secrets - KHÔNG commit lên git!

Để sử dụng:
1. Đặt API keys của bạn vào biến môi trường (khuyến nghị) hoặc
2. Tạo file config/config_api.py từ config/config_api.py.example và điền API keys
3. File này đã được thêm vào .gitignore

LƯU Ý BẢO MẬT:
- KHÔNG hardcode API keys trực tiếp trong file này
- Luôn sử dụng biến môi trường hoặc file local không được commit
- Nếu file này đã bị commit với keys, hãy rotate (thay đổi) keys ngay lập tức
"""

import os

# Binance API Configuration
# Ưu tiên đọc từ biến môi trường, nếu không có thì để None
# Để bảo mật, luôn sử dụng biến môi trường:
#   export BINANCE_API_KEY='your-key-here'
#   export BINANCE_API_SECRET='your-secret-here'
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY") or None
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET") or None

# Google Gemini API Configuration
# Lấy API key từ: https://makersuite.google.com/app/apikey
# Hoặc: https://aistudio.google.com/app/apikey
# Ưu tiên đọc từ biến môi trường, nếu không có thì để None
# Để bảo mật, luôn sử dụng biến môi trường:
#   export GEMINI_API_KEY='your-api-key-here'
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or None


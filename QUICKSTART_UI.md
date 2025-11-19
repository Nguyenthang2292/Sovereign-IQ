# 🚀 Quick Start - Crypto Prediction UI

## Cài đặt nhanh

```bash
# 1. Cài đặt thư viện
pip install -r requirements.txt

# 2. Chạy UI
streamlit run crypto_ui.py
```

## Sử dụng

1. Mở trình duyệt tại `http://localhost:8501`
2. Điền thông tin ở sidebar:
   - Trading Pair: `BTC` hoặc `ETH`
   - Timeframe: `1h` (khuyến nghị)
   - Number of Candles: `1500` (khuyến nghị)
3. Click **"🚀 Predict"**
4. Đợi kết quả!

## Lưu ý

- Lần đầu chạy có thể mất vài phút để train model
- Cần kết nối internet để fetch data từ exchanges
- Model dự đoán cho 24 candles tiếp theo

## Troubleshooting

**Lỗi import streamlit:**
```bash
pip install streamlit plotly
```

**Lỗi kết nối exchange:**
- Thử chọn exchange khác trong sidebar
- Kiểm tra kết nối internet

**UI không hiển thị:**
- Kiểm tra terminal có lỗi không
- Refresh trình duyệt (Ctrl+Shift+R)


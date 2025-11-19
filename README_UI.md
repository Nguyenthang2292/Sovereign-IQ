# Crypto Prediction UI - Hướng dẫn sử dụng

## 📋 Yêu cầu

### Cài đặt thư viện

```bash
pip install -r requirements.txt
```

Hoặc cài đặt thủ công:
```bash
pip install streamlit plotly
```

## 🚀 Chạy ứng dụng

### Khởi động UI

```bash
streamlit run crypto_ui.py
```

Ứng dụng sẽ tự động mở trong trình duyệt tại địa chỉ: `http://localhost:8501`

## 📖 Hướng dẫn sử dụng

### 1. Cấu hình tham số

**Sidebar (Thanh bên trái):**

- **Trading Pair**: Nhập symbol (ví dụ: `BTC`, `ETH`) hoặc cặp đầy đủ (`BTC/USDT`)
- **Quote Currency**: Chọn đồng quote (USDT, USD, BTC, ETH)
- **Timeframe**: Chọn khung thời gian (30m, 1h, 4h, 1d...)
- **Number of Candles**: Số lượng nến để lấy dữ liệu (500-3000)
  - Nhiều hơn = nhiều dữ liệu huấn luyện hơn nhưng chậm hơn
  - Khuyến nghị: 1500-2000
- **Exchanges**: Chọn các sàn giao dịch để lấy dữ liệu
  - Nên chọn nhiều sàn để đảm bảo độ tin cậy

### 2. Thực hiện dự đoán

1. Điền thông tin vào sidebar
2. Click nút **"🚀 Predict"**
3. Đợi quá trình:
   - Fetching data (Lấy dữ liệu)
   - Calculating indicators (Tính toán chỉ báo)
   - Training model (Huấn luyện mô hình)
   - Making prediction (Đưa ra dự đoán)

### 3. Đọc kết quả

**Kết quả chính:**
- **Prediction**: Hướng dự đoán (UP/DOWN/NEUTRAL)
- **Confidence**: Độ tin cậy (%)

**Thông tin bổ sung:**
- Current Price: Giá hiện tại
- Market Volatility (ATR): Độ biến động thị trường
- Probability Breakdown: Phân tích xác suất cho từng hướng
- Price Targets: Mục tiêu giá (nếu không phải NEUTRAL)
- Price Chart: Biểu đồ giá tương tác
- Technical Indicators: Tóm tắt các chỉ báo kỹ thuật

## 🎨 Tính năng UI

### Biểu đồ tương tác
- **Price Chart**: Biểu đồ nến với volume
- **Probability Chart**: Biểu đồ cột hiển thị xác suất

### Màu sắc dự đoán
- 🟢 **UP**: Màu xanh lá (tăng giá)
- 🔴 **DOWN**: Màu đỏ (giảm giá)
- 🟡 **NEUTRAL**: Màu vàng (đi ngang)

### Thông tin chi tiết
- Expandable sections cho Technical Indicators và Data Information
- Responsive design, tự động điều chỉnh theo màn hình

## 💡 Mẹo sử dụng

1. **Để có kết quả tốt nhất:**
   - Sử dụng ít nhất 1500 candles
   - Chọn nhiều exchanges
   - Timeframe 1h hoặc 4h thường cho kết quả tốt

2. **Hiểu rõ dự đoán:**
   - Model dự đoán cho **24 candles** tiếp theo
   - Threshold động dựa trên biến động lịch sử
   - Precision của UP/DOWN quan trọng hơn accuracy tổng thể

3. **Xử lý lỗi:**
   - Nếu không lấy được dữ liệu: Thử lại hoặc chọn exchange khác
   - Nếu training chậm: Giảm số lượng candles
   - Nếu không đủ dữ liệu: Tăng limit

## 🔧 Troubleshooting

### Lỗi import
```bash
# Đảm bảo đã cài đặt đầy đủ
pip install -r requirements.txt
```

### Lỗi kết nối exchange
- Kiểm tra kết nối internet
- Thử chọn exchange khác
- Một số exchange có thể bị chặn ở một số quốc gia

### UI không hiển thị
- Kiểm tra terminal có lỗi không
- Thử refresh trình duyệt
- Xóa cache: `Ctrl + Shift + R` (Windows) hoặc `Cmd + Shift + R` (Mac)

## 📝 Lưu ý

- **Không phải lời khuyên đầu tư**: Đây là công cụ phân tích, không phải lời khuyên tài chính
- **Rủi ro**: Trading cryptocurrency có rủi ro cao, chỉ đầu tư số tiền bạn có thể mất
- **Backtesting**: Luôn backtest trước khi sử dụng trong thực tế
- **Model accuracy**: Model có thể sai, luôn kết hợp với phân tích kỹ thuật khác

## 🆚 So sánh với CLI

| Tính năng | CLI (`crypto_simple_enhance.py`) | UI (`crypto_ui.py`) |
|-----------|----------------------------------|---------------------|
| Dễ sử dụng | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Tốc độ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Visualization | ❌ | ✅ |
| Interactive | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Automation | ✅ | ⭐⭐ |
| Metrics detail | ✅ | ⭐⭐⭐ |

## 🔗 Liên kết

- File chính: `crypto_simple_enhance.py`
- File UI: `crypto_ui.py`
- Requirements: `requirements.txt`


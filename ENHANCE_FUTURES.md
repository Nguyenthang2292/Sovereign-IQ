## 1. Nâng Cấp Thuật Toán & Chiến Lược (Quantitative & Strategy)

Hiện tại hệ thống dùng tương quan và z-score rolling window nên phản ứng chậm. Các cải tiến đề xuất:

### 1.A fnding-Rate Arbitrage (Delta Neutral)
- Đã có `HedgeFinder`, tận dụng để kiếm funding chênh lệch.  
- **Chiến lược:** Long spot (hoặc futures sàn A) và short futures Binance để triệt tiêu delta. Ăn funding khi thị trường uptrend.  
- **Nâng cấp:** Viết thuật toán quét funding giữa các sàn, tính phí giao dịch, trượt giá để biết lúc nào thực sự có lợi nhuận.

### 1.B Tối Ưu Danh Mục Markowitz
- **Vấn đề:** Hedge hiện tại dạng 1-1.  
- **Giải pháp:** Áp dụng MPT (Mean-Variance) để tìm tỷ trọng nhiều coin tối ưu Sharpe.  
- **Thư viện:** `PyPortfolioOpt`.

## 2. Cải Tiến Machine Learning (AI/Prediction)

Pipeline XGBoost hiện xử lý dữ liệu tabular. Cần nâng cấp feature và kiến trúc:

### 2.A Feature Engineering Nâng Cao
- **Order Book Imbalance:** Thu thập bid/ask volume để đo áp lực mua/bán ngắn hạn (<5 phút).  
- **On-chain Data:** Với timeframe lớn (H4, D1), đưa thêm exchange inflow/outflow, whale alert.  
- **Sentiment:** Crawl tiêu đề/news/Twitter, dùng VADER hoặc BERT tính điểm cảm xúc đưa vào feature.

### 2.B Deep Learning cho Time-Series
- **LSTM/GRU:** Giữ bộ nhớ dài hạn cho chuỗi giá.  
- **Temporal Fusion Transformer (TFT):** Kiến trúc SOTA, giải thích được tầm quan trọng feature theo thời gian.  
- **Thư viện:** PyTorch Forecasting, Darts.

### 2.C Meta-Labeling (Marcos Lopez de Prado)
- **Model 1:** Dùng indicator (RSI, Bollinger, ...) xác định tín hiệu nền.  
- **Model 2:** XGBoost dự đoán xác suất tín hiệu Model 1 sẽ thành công → loại bỏ false positive, tăng win-rate.

## 3. Nâng Cấp Hệ Thống & Kiến Trúc (System Engineering)

### 3.A Backtesting Engine
- **Lý do:** Trước khi chạy bot thật cần mô phỏng quá khứ.  
- **Yêu cầu:** Bao gồm phí giao dịch, trượt giá, funding.  
- **Thực hiện:** Xây class `Backtester` nhận strategy + dữ liệu lịch sử, chạy mô phỏng và vẽ equity curve.  
- **Tham khảo:** `backtrader`, `vectorbt`.

### 3.B Event-Driven Architecture
- **Hiện tại:** Fetch → Analyze → Print theo vòng lặp.  
- **Đề xuất:**  
  - Luồng WebSocket nhận giá real-time.  
  - Khi giá đổi → phát event.  
  - Strategy lắng nghe event → tính toán → phát signal.  
  - Execution nhận signal → đặt lệnh.  
- **Lợi ích:** Phản ứng mili-giây, không bị delay do sleep loop.

### 3.C Web Dashboard
- **Lý do:** Thay CLI bằng UI theo dõi trực quan.  
- **Công cụ:** Streamlit (dựng nhanh bằng Python).  
- **Tính năng:** Biểu đồ PnL real-time, danh sách vị thế, nút “Emergency Close All”.

## Lộ Trình Đề Xuất
- **Ngắn hạn:** Refactor `crypto_pairs_trading.py` vào hệ thống module + thêm backtest đơn giản với vectorbt.  
- **Trung hạn:** Nghiên cứu Kalman Filter cho pairs trading và đưa dữ liệu order book vào XGBoost.  
- **Dài hạn:** Xây web dashboard (Streamlit) và chuyển sang kiến trúc event-driven để trade real-time.
# Hệ thống Trading dựa trên Orderbook (Sổ Lệnh)

> **Nguồn**: Tài liệu nghiên cứu nội bộ — Binance Futures Orderbook  
> **Cập nhật**: 2026-03-01

---

## Giới thiệu

Thiết kế một hệ thống trading dựa trên dữ liệu **Orderbook (Sổ lệnh)** của Binance đòi hỏi sự kết hợp giữa hạ tầng cực nhanh và các mô hình toán học nhạy bén.

Tài liệu này trình bày các cách tiếp cận chính, chia theo phương pháp luận và kỹ thuật, từ đơn giản đến phức tạp.

---

## Mục lục

1. [Các cách tiếp cận chính](#1-các-cách-tiếp-cận-chính)
   - [1.1 Orderbook Microstructure](#11-orderbook-microstructure)
   - [1.2 Market Making](#12-market-making)
   - [1.3 Machine Learning & Deep Learning](#13-machine-learning--deep-learning)
   - [1.4 Infrastructure & Độ trễ](#14-infrastructure--độ-trễ)
2. [So sánh các phương pháp](#2-so-sánh-các-phương-pháp)
3. [Kết hợp Market Making + OBI (Tiêu chuẩn vàng)](#3-kết-hợp-market-making--obi-tiêu-chuẩn-vàng)
   - [3.1 Chiến thuật Skewing](#31-chiến-thuật-skewing)
   - [3.2 Quản lý Inventory thông minh](#32-quản-lý-inventory-thông-minh)
   - [3.3 Dynamic Spread](#33-dynamic-spread)
   - [3.4 Mô hình toán học tổng hợp](#34-mô-hình-toán-học-tổng-hợp)
4. [Universe Selection — Chọn lọc danh mục](#4-universe-selection--chọn-lọc-danh-mục)
   - [4.1 Lọc theo Volatility & Liquidity](#41-lọc-theo-volatility--liquidity)
   - [4.2 Tiền xử lý Micro-structure](#42-tiền-xử-lý-micro-structure)
   - [4.3 Phân loại Market Regime](#43-phân-loại-market-regime)
   - [4.4 Data Pipeline lý tưởng](#44-data-pipeline-lý-tưởng)
   - [4.5 Bảng tiêu chí lọc](#45-bảng-tiêu-chí-lọc)
5. [Kiến trúc hệ thống HFT hoàn chỉnh](#5-kiến-trúc-hệ-thống-hft-hoàn-chỉnh)
6. [MFT/Scalping — Hướng tiếp cận cho cá nhân](#6-mftscalping--hướng-tiếp-cận-cho-cá-nhân)
   - [6.1 Aggregation Orderbook](#61-aggregation-orderbook)
   - [6.2 Time-Weighted OBI](#62-time-weighted-obi)
   - [6.3 Order Flow Cumulative Delta](#63-order-flow-cumulative-delta)
7. [So sánh HFT vs MFT/Scalping](#7-so-sánh-hft-vs-mftscalping)

---

## 1. Các cách tiếp cận chính

### 1.1 Orderbook Microstructure

Đây là phương pháp phổ biến nhất, tập trung vào việc khai thác các **tín hiệu ngắn hạn (alpha)** từ sự thay đổi của các tầng giá.

#### Order Book Imbalance (OBI)

Tính toán sự chênh lệch giữa khối lượng bên mua (bid) và bên bán (ask):

$$I = \frac{Q_{bid} - Q_{ask}}{Q_{bid} + Q_{ask}}$$

Nếu khối lượng mua ở các tầng đầu tiên vượt trội, có xác suất cao giá sẽ **tăng trong vài mili giây tới**.

#### Weighted Mid-Price

Thay vì dùng giá Mid-price thông thường, sử dụng giá có **trọng số theo khối lượng** để phản ánh sát hơn áp lực thị trường:

$$P_{mid}^{W} = \frac{Q_{ask} \cdot P_{bid}^{best} + Q_{bid} \cdot P_{ask}^{best}}{Q_{bid} + Q_{ask}}$$

#### Order Flow Toxicity (VPIN)

Đo lường mức độ **"độc hại"** của dòng lệnh để nhận biết khi nào các nhà giao dịch có thông tin nội bộ (informed traders) đang đẩy mạnh giao dịch — giúp hệ thống tránh bị "úp bô" khi làm Market Maker.

---

### 1.2 Market Making

Thay vì dự đoán hướng đi của giá, bạn đóng vai trò là **người cung cấp thanh khoản** — đặt cả lệnh mua và lệnh bán để ăn chênh lệch (spread).

#### Chiến thuật Avellaneda-Stoikov

Một mô hình cổ điển trong HFT giúp tối ưu hóa việc đặt lệnh dựa trên:

- **Biến động thị trường (volatility)**
- **Vị thế kho hàng (inventory)** hiện tại

Giúp tránh rủi ro tích lũy một chiều quá nhiều.

#### Dự đoán Spread

Sử dụng dữ liệu lịch sử để dự đoán khi nào Spread sẽ **giãn rộng** hoặc **thu hẹp** để điều chỉnh vị trí đặt lệnh tối ưu.

---

### 1.3 Machine Learning & Deep Learning

Sử dụng các mô hình phức tạp để nhận diện các **"pattern"** mà mắt người hoặc các công thức tuyến tính không thấy được.

| Mô hình | Ứng dụng |
|---------|----------|
| **CNN** | Coi dữ liệu Orderbook (giá và khối lượng ở 20-50 tầng) như một bức ảnh 2D để nhận diện hình thái tích lũy/phân phối lệnh |
| **LSTM/Transformers** | Xử lý chuỗi thời gian của các biến động sổ lệnh để dự đoán biến động giá trong $n$ tích tắc tiếp theo |
| **Reinforcement Learning** | Huấn luyện Agent tự đưa ra quyết định mua/bán/hủy lệnh; phần thưởng = lợi nhuận, hình phạt = drawdown |

---

### 1.4 Infrastructure & Độ trễ

Trong HFT, đôi khi **"thuật toán hay không bằng tốc độ nhanh"**.

| Yếu tố | Chi tiết |
|--------|---------|
| **Colocation** | Binance dùng AWS → đặt server tại cùng Region (Tokyo - `ap-northeast-1`) để latency thấp nhất |
| **Ngôn ngữ** | Ưu tiên **C++ hoặc Rust** cho vòng lặp thực thi chính; tránh Python |
| **FIX Protocol** | Nếu có volume lớn, đăng ký cổng FIX của Binance thay vì REST/WebSocket thông thường |

---

## 2. So sánh các phương pháp

| Phương pháp | Độ phức tạp | Ưu điểm | Nhược điểm |
|------------|------------|---------|-----------|
| **Imbalance (OBI)** | Thấp | Dễ triển khai, phản ứng nhanh | Dễ bị nhiễu bởi lệnh giả (spoofing) |
| **Market Making** | Trung bình | Lợi nhuận ổn định trong thị trường sideway | Rủi ro cao khi thị trường có xu hướng mạnh |
| **Deep Learning** | Cao | Khả năng thích nghi tốt với dữ liệu lớn | Đòi hỏi GPU mạnh, dễ bị overfitting |

> **Lời khuyên**: Binance có cơ chế chống Spoofing khá phức tạp. Nếu chỉ nhìn vào Depth (độ sâu), bạn rất dễ bị lừa. **Cách tốt nhất hiện nay** là kết hợp **Order Flow Imbalance với Trade Data** (dữ liệu khớp lệnh thực tế) để xác nhận tín hiệu.

---

## 3. Kết hợp Market Making + OBI (Tiêu chuẩn vàng)

Việc kết hợp Market Making (MM) và Order Book Imbalance (OBI) không chỉ khả thi mà còn là **"tiêu chuẩn vàng"** trong các hệ thống HFT hiện đại.

Trong giới trading chuyên nghiệp, kỹ thuật này được gọi là **Skewing (Lệch báo giá)**. Thay vì đặt các lệnh chờ (limit orders) cân xứng hai đầu giá Mid-price, OBI được dùng như một "la bàn" để điều chỉnh vị trí đặt lệnh nhằm giảm thiểu rủi ro **Adverse Selection**.

---

### 3.1 Chiến thuật Skewing

**Thông thường**, một Market Maker đặt lệnh cân xứng:

```
Bid = Mid − Spread/2
Ask = Mid + Spread/2
```

**Khi kết hợp OBI**, thêm giá trị biến thiên $\delta$ dựa trên độ lệch sổ lệnh:

| Trạng thái OBI | Hành động | Mục đích |
|---------------|-----------|----------|
| **OBI > 0** (áp lực mua mạnh) | Đẩy cả Bid & Ask lên cao | Dễ khớp lệnh mua (đi cùng xu hướng), khó bị khớp lệnh bán |
| **OBI < 0** (áp lực bán mạnh) | Hạ cả Bid & Ask xuống | Ưu tiên khớp lệnh bán, tránh mua vào khi giá đang rơi |

---

### 3.2 Quản lý Inventory thông minh

Một trong những rủi ro lớn nhất của MM là **cầm quá nhiều hàng khi giá giảm** (hoặc short quá nhiều khi giá tăng).

**Ví dụ xử lý:**  
Tình huống: Đang cầm quá nhiều BTC (Long Inventory) và muốn giảm vị thế.  
→ Nếu OBI đang âm (phe bán chiếm ưu thế): hệ thống **chủ động hạ giá Ask xuống cực thấp** (thậm chí sát Mid) để "thoát hàng" nhanh nhất trước khi thị trường sập sâu hơn.

---

### 3.3 Dynamic Spread

OBI không chỉ cho biết hướng đi mà còn cho biết **độ biến động sắp tới**:

| Điều kiện | Hành động | Lý do |
|-----------|-----------|-------|
| $OBI \approx 0$ (cân bằng) | **Thu hẹp Spread** | Thị trường ổn định → tăng volume, kiếm lời từ spread nhiều hơn |
| $\|OBI\|$ cực lớn | **Giãn Spread hoặc rút lệnh** | Dấu hiệu "pump/dump" → tránh làm "bia đỡ đạn" cho Informed Traders |

---

### 3.4 Mô hình toán học tổng hợp

Công thức cập nhật giá đặt lệnh:

$$P_{quote} = P_{mid} \pm \left(\frac{Spread_{min}}{2} + \alpha \cdot \text{Inventory} - \beta \cdot \text{OBI}\right)$$

Trong đó:

- $\alpha$: Hệ số nhạy cảm với **lượng hàng đang cầm** (inventory risk)
- $\beta$: Hệ số nhạy cảm với **áp lực sổ lệnh** (OBI signal)

**Tại sao kết hợp này hiệu quả:**

| Vấn đề của MM thuần túy | Cách OBI giải quyết |
|------------------------|---------------------|
| Adverse Selection (bị khớp ngay trước khi giá chạy ngược) | Dự báo trước hướng đi ngắn hạn để né hoặc đi cùng |
| Inventory Risk (kẹt hàng) | Dùng áp lực thị trường để điều chỉnh giá thoát tối ưu |
| Spread cố định | Tự động giãn khi thấy "bom" sắp nổ (imbalance cao) |

---

## 4. Universe Selection — Chọn lọc danh mục

Trong HFT, bước này không chỉ là "nên" mà là **bắt buộc**. Nếu cố gắng chạy HFT trên tất cả các cặp tiền của Binance (hàng trăm cặp), hệ thống sẽ bị **"nghẽn cổ chai"** dữ liệu và lãng phí tài nguyên vào các coin rác.

---

### 4.1 Lọc theo Volatility & Liquidity

#### Loại bỏ "Coin Chết"

Nếu một symbol có Volume 24h quá thấp (ví dụ < $1M$), Orderbook sẽ rất mỏng. Chỉ một lệnh nhỏ cũng làm giá nhảy vọt → chiến thuật MM bị quét Stop-loss liên tục.

#### Tính toán ADR (Average Daily Range)

$$Volatility = \frac{High - Low}{Open} \times 100\%$$

Lọc các coin có biên độ dao động đủ lớn để bù đắp chi phí giao dịch (Trading Fee) và độ trượt giá (Slippage).

#### Lọc theo Gap (Lỗ hổng)

Nếu sổ lệnh có quá nhiều khoảng trống giữa các bước giá (tick) → dấu hiệu thanh khoản kém → tránh.

---

### 4.2 Tiền xử lý Micro-structure

#### Tick Size Analysis

Một số coin có bước giá (Tick size) quá lớn so với giá trị (ví dụ: bước giá 0.1 trên giá 10.0 = 1%). Điều này khiến Spread luôn bị giãn rộng cưỡng ép. Cần chọn symbol có **Tick size đủ mịn**.

#### Outlier Removal (Lọc nhiễu)

Loại bỏ các spikes (râu nến) do lỗi API hoặc lệnh "Fat finger" (đặt nhầm số lượng cực lớn nhưng bị hủy ngay). Những dữ liệu này nếu đưa vào OBI sẽ làm lệch hoàn toàn dự báo.

#### Log-Normalization

Thay vì dùng giá tuyệt đối, chuyển sang tính toán theo **log-return** để đồng bộ hóa dữ liệu giữa các cặp tiền khác nhau (BTC giá $60k vs PEPE giá $0.00001):

$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

---

### 4.3 Phân loại Market Regime

Trước khi chạy MM + OBI, phân loại symbol đang ở trạng thái nào trong **1h qua**:

| Regime | Chiến thuật tối ưu |
|--------|-------------------|
| **Trending** (Có xu hướng) | OBI cực kỳ hiệu quả để "bơi theo cá mập" |
| **Ranging** (Đi ngang) | MM thuần túy (Avellaneda-Stoikov) ăn đậm nhờ thu phí Spread |
| **High Uncertainty** (Biến động mạnh/tin tức) | Tự động **Kill-switch** để bảo vệ vốn |

---

### 4.4 Data Pipeline lý tưởng

```
Scanner (mỗi 15-30 phút)
    └── Quét toàn bộ Binance qua REST API
    └── Lọc ra Top 20-30 symbol: Volume > X, Volatility > Y
            ↓
Streamer (thời gian thực)
    └── Chỉ kết nối WebSocket vào 20-30 symbol đã chọn
            ↓
Processor
    └── Làm sạch dữ liệu
    └── Tính toán OBI
    └── Đẩy vào Execution Engine
```

---

### 4.5 Bảng tiêu chí lọc

| Tiêu chí | Ngưỡng gợi ý | Lý do |
|---------|--------------|-------|
| Volume 24h | > $10,000,000 USD | Đảm bảo đủ thanh khoản để vào/ra lệnh |
| Spread % | < 0.05% | Spread quá rộng → khó khớp lệnh đối ứng |
| Tick Size Ratio | < 0.01% của giá | Giá di chuyển mượt, không bị "nhảy bậc thang" |
| 1h Volatility | > 0.5% | Có biến động mới có cơ hội ăn chênh lệch |

> **Lời khuyên**: Đừng lọc quá chặt ở khung 4h vì HFT sống nhờ những biến động nhỏ trong tích tắc. **1h là khung thời gian "vàng"** để đánh giá tính chất của một symbol.

---

## 5. Kiến trúc hệ thống HFT hoàn chỉnh

Để xây dựng hệ thống HFT có thể "sống sót" trên thị trường crypto, cần nhiều hơn là chỉ một thuật toán hay. Hệ thống chuyên nghiệp được chia thành các **lớp (layers) tách biệt**.

### Component 1: Local Order Book Manager

Binance gửi dữ liệu qua WebSocket dưới dạng **"diff updates"** (chỉ gửi những thay đổi). Cần duy trì một bản sao cập nhật của Orderbook trong RAM.

- **Yêu cầu**: Cấu trúc dữ liệu cực nhanh (B-Tree hoặc Skip List)
- **Mục tiêu**: Tìm kiếm, chèn, xóa giá trong **dưới 1 micro-giây**

### Component 2: Pre-Trade Risk Engine

Một lỗi nhỏ trong code HFT có thể khiến thuật toán đặt hàng nghìn lệnh sai trong 1 giây → cháy tài khoản (Flash Crash).

| Cơ chế | Chức năng |
|--------|-----------|
| **Fat-finger Protection** | Chặn lệnh có giá quá xa Mid-price hoặc số lượng bất thường |
| **Max Position/Exposure** | Giới hạn tổng lượng hàng tối đa được cầm |
| **Rate Limiter** | Không vi phạm giới hạn số lệnh/giây của Binance (tránh bị ban IP) |

### Component 3: Order Management System (OMS)

| Tính năng | Mô tả |
|-----------|-------|
| **Lifecycle Tracking** | Theo dõi trạng thái lệnh: New → Partially Filled → Filled/Canceled |
| **Smart Order Routing** | Nếu đánh đa sàn, chọn sàn có giá tốt nhất |
| **Post-trade Reconciliation** | Đối soát số dư giữa Database và thực tế trên sàn |

### Component 4: Event-Driven Backtester

Không thể dùng thư viện như Backtrader hay Pandas cho HFT vì chúng **"nhìn trước tương lai"**.

Cần mô phỏng:

- Từng tick một (tick-by-tick)
- Độ trễ thực tế (Latency)
- **Vị trí hàng đợi (Queue Position)**: Đặt lệnh ở tầng $60,001 — phải có bao nhiêu lệnh khớp trước thì mới đến lượt?

### Component 5: Historical Data Store

Dữ liệu Orderbook L2 rất khổng lồ (vài GB/ngày cho một cặp tiền).

- ✅ **Dùng**: ClickHouse, QuestDB, InfluxDB (Time-series DB)
- ❌ **Tránh**: MySQL/PostgreSQL (sẽ "nghẹt thở" với hàng triệu dòng tick data)

### Component 6: Monitoring & Alerting

| Công cụ | Mục đích |
|---------|----------|
| **Grafana/Prometheus** | Theo dõi PnL, Fill rate, End-to-end Latency theo thời gian thực |
| **Kill-switch** | Dừng toàn bộ hệ thống ngay lập tức nếu Drawdown vượt ngưỡng |

### Component 7: Post-Trade Analysis

| Phân tích | Mục đích |
|-----------|----------|
| **Alpha Decay** | OBI signal còn hiệu quả bao lâu sau khi xuất hiện? |
| **Slippage Analysis** | Đo độ trượt giá thực tế vs kỳ vọng để tối ưu thuật toán đặt lệnh |

### Tổng kết Pipeline vận hành

| Bước | Thành phần | Công nghệ gợi ý |
|------|-----------|----------------|
| Nhận tin | WebSocket Client | Rust (Tokio) / C++ (Boost.Asio) |
| Xử lý | Local Order Book + OBI Logic | In-memory Data Structures |
| Kiểm tra | Pre-Trade Risk | Hard-coded rules (Low latency) |
| Thực thi | OMS / API Connector | Binance API (Signed Requests) |
| Lưu trữ | Data Logger | ClickHouse |
| Giám sát | Health Check | Prometheus + Grafana |

> **Lưu ý quan trọng**: Trong HFT, thời gian của máy chủ bạn và máy chủ Binance lệch nhau **100ms** cũng là thảm họa. Cần sử dụng **PTP (Precision Time Protocol)** để đồng bộ timestamp tuyệt đối.

---

## 6. MFT/Scalping — Hướng tiếp cận cho cá nhân

Khi chuyển từ xử lý mili-giây (HFT) sang khung thời gian **dưới 60 giây (Sub-minute)**, bạn đang chuyển từ HFT sang **MFT (Medium-Frequency Trading)** hoặc **Scalping**.

Thay vì bắt những biến động "nhiễu" siêu nhỏ, bạn bắt những **"con sóng nhỏ"** của dòng tiền — phù hợp với trader cá nhân hoặc đội ngũ nhỏ.

---

### 6.1 Aggregation Orderbook

Thay vì nhìn vào từng bước giá sát nút (ví dụ 0.1 USD), gộp chúng thành các **"thùng" (bins)** rộng hơn.

**Cách làm**: BTC đang ở 60,000 → thay vì xem mức 60,001, 60,002... gộp tất cả lệnh trong khoảng 60,000–60,010 thành một tầng duy nhất.

$$\text{Aggregated Volume} = \sum_{i=1}^{n} Q_i \quad \text{với } P_i \in \left[P_{mid},\ P_{mid} \times (1 + X\%)\right]$$

**Lợi ích:**

- Giảm đáng kể khối lượng dữ liệu cần xử lý
- Loại bỏ nhiễu từ các lệnh "spoofing" nhỏ lẻ sát Mid-price
- Phản ánh rõ hơn các **"vùng cản"** thực sự của các tay chơi lớn

---

### 6.2 Time-Weighted OBI

Với khung 60s, một tín hiệu OBI tại một thời điểm (snapshot) không còn đủ tin cậy. Thay vào đó, dùng **Trung bình trượt OBI**:

1. Tính OBI mỗi giây một lần
2. Lấy trung bình của 60 giây gần nhất

**Tín hiệu vào lệnh**: Nếu OBI trung bình liên tục duy trì ở mức dương (> 0.6) trong suốt 1 phút → dấu hiệu của một **lực mua chủ động bền bỉ**, an toàn hơn nhiều so với một cú nhảy OBI trong 1ms.

---

### 6.3 Order Flow Cumulative Delta

Ở tần suất thấp hơn, kết hợp Orderbook với **dữ liệu khớp lệnh thực tế (Trades)**:

$$\Delta = \text{Lượng mua chủ động} - \text{Lượng bán chủ động} \quad \text{(trong 60 giây)}$$

**Tín hiệu xác nhận mạnh**:

- Orderbook lệch về bên mua: **OBI dương**
- **VÀ** Delta cũng dương

→ Xác suất giá tăng trong vài phút tới **rất cao**.

**Chiến thuật Sniper**: Đợi OBI và Delta cùng đồng thuận trong 30–60s rồi mới vào lệnh bằng lệnh Market hoặc Limit sát giá.

---

## 7. So sánh HFT vs MFT/Scalping

| Đặc điểm | HFT (< 10ms) | MFT/Scalping (< 60s) |
|----------|-------------|----------------------|
| **Hạ tầng** | C++, Colocation, AWS Direct Connect | Python (Pandas/NumPy), Cloud thông thường |
| **Dữ liệu** | Từng thay đổi nhỏ (Diff depth) | Sổ lệnh đã gộp tầng (Aggregated depth) |
| **Tín hiệu** | Phản ứng tức thì với OBI snapshot | Phân tích xu hướng dòng tiền (Flow analysis) |
| **Chi phí** | Rất cao (hàng chục nghìn USD/tháng) | Thấp (vài chục đến vài trăm USD/tháng) |
| **Đối thủ** | Các quỹ lớn, ngân hàng đầu tư | Các bot cá nhân, trader nhỏ lẻ |
| **Database** | ClickHouse, QuestDB (Time-series) | SQLite, Parquet files |
| **OBI signal** | OBI tức thời (milliseconds) | Time-Weighted OBI trung bình 60s |

> **Khuyến nghị cho cá nhân/team nhỏ**: Chọn **MFT/Scalping** — không cần colocation hay C++, Python hoàn toàn đủ sức xử lý ở khung < 60s. Cạnh tranh với bot cá nhân khác thay vì các quỹ lớn với hạ tầng hàng triệu USD.

---

## Tài liệu liên quan

- [`2026-03-01-order-book-imbalance-gate-design.md`](./2026-03-01-order-book-imbalance-gate-design.md) — Thiết kế tích hợp Order Book Imbalance Gate vào hệ thống `auto_trade` hiện tại

---

*Tài liệu được tổ chức và viết lại từ nghiên cứu nội bộ — 2026-03-01*

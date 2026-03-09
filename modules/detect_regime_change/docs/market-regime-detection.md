# Phân Tích Chuyên Sâu: Thuật Toán & Khung Công Việc Phát Hiện Thay Đổi Trạng Thái Thị Trường Tài Chính

> **Phạm vi:** Nghiên cứu và tổng hợp các phương pháp từ thống kê cổ điển đến học sâu, giai đoạn 2024–2025.

---

## Mục Lục

1. [Bản Chất Phi Tĩnh và Sự Cần Thiết của Nhận Diện Trạng Thái](#1-bản-chất-phi-tĩnh-và-sự-cần-thiết-của-nhận-diện-trạng-thái)
2. [Mô Hình Markov Ẩn (HMM)](#2-mô-hình-markov-ẩn-hidden-markov-models---hmm)
3. [Phân Cụm Dựa Trên Khoảng Cách Wasserstein](#3-phân-cụm-dựa-trên-khoảng-cách-wasserstein-và-hình-học-phân-phối)
4. [Thuật Toán Phát Hiện Điểm Thay Đổi (CPD)](#4-thuật-toán-phát-hiện-điểm-thay-đổi-change-point-detection---cpd)
5. [Học Sâu và Mạng Thần Kinh](#5-học-sâu-và-các-kiến-trúc-mạng-thần-kinh-trong-nhận-diện-trạng-thái)
6. [Khung Làm Việc Hybrid và Nghiên Cứu Mới Nhất 2024–2025](#6-khung-làm-việc-hybrid-và-nghiên-cứu-mới-nhất-20242025)
7. [Đặc Trưng Đầu Vào và Tiền Xử Lý](#7-đặc-trưng-đầu-vào-và-các-yếu-tố-kỹ-thuật-trong-mô-hình-hóa)
8. [Phân Tích So Sánh: Độ Trễ và Độ Chính Xác](#8-phân-tích-so-sánh-về-độ-trễ-và-độ-chính-xác)
9. [Kết Luận và Định Hướng Chiến Lược](#9-kết-luận-và-định-hướng-chiến-lược)
10. [Tham Khảo Thư Viện và Công Cụ](#10-tham-khảo-thư-viện-và-công-cụ)

---

## 1. Bản Chất Phi Tĩnh và Sự Cần Thiết của Nhận Diện Trạng Thái

Việc xác định các **trạng thái thị trường** (market regimes) đóng vai trò là nền tảng trong quản lý rủi ro và tối ưu hóa chiến lược đầu tư hiện đại. Thị trường tài chính vốn dĩ không vận hành theo một quy luật tĩnh lặng mà luôn dịch chuyển qua các giai đoạn có đặc tính thống kê khác biệt — từ những chu kỳ tăng trưởng ổn định với biến động thấp đến những giai đoạn khủng hoảng cực đoan với sự hội tụ tương quan tài sản mạnh mẽ.

### 1.1 Vì Sao Nhận Diện Trạng Thái Quan Trọng?

Khả năng phát hiện sớm sự thay đổi trạng thái — được định nghĩa là những **cụm điều kiện thị trường dai dẳng** ảnh hưởng đến hiệu quả của các yếu tố đầu tư — giúp các nhà quản trị danh mục:

- Thích ứng với sự **phi tĩnh** (non-stationarity) của dữ liệu
- Tối đa hóa lợi nhuận theo từng giai đoạn
- Giảm thiểu **rủi ro đuôi** (tail risk) trong các kịch bản khủng hoảng

Nhiều thất bại của mô hình học máy xuất phát từ việc dữ liệu huấn luyện không còn đại diện cho phân phối hiện tại — khi thị trường chuyển dịch, các mối quan hệ tuyến tính và phi tuyến tính giữa các biến số thường bị phá vỡ. Do đó, phát hiện thay đổi cấu trúc là công cụ trung tâm để kích hoạt quá trình **tái huấn luyện mô hình** hoặc **điều chỉnh tỷ trọng danh mục**.

### 1.2 Bốn Trạng Thái Thị Trường Chính

Dựa trên dữ liệu thị trường Mỹ từ năm 1995 đến 2024 (State Street Global Advisors):

| Trạng Thái | Tần Suất | Lợi Nhuận TB (Cổ Phiếu) | Độ Tin Cậy | Đặc Điểm Chính |
|---|---|---|---|---|
| **Emerging Expansion** | 42.34% | Dương (trung bình) | Trung bình | Giai đoạn chuyển tiếp, biến động cao hơn Robust Expansion |
| **Robust Expansion** | 25.35% | +2.9% / tháng | 88% | Hiệu suất tốt nhất, rủi ro thấp |
| **Cautious Decline** | 19.16% | -0.7% / tháng | Thấp | Lợi nhuận giảm, bất định gia tăng |
| **Market Turmoil** | 13.16% | -3.4% / tháng | Rất thấp | Khủng hoảng, trái phiếu vượt trội cổ phiếu |

---

## 2. Mô Hình Markov Ẩn (Hidden Markov Models - HMM)

Mô hình Markov Ẩn (HMM) là phương pháp tiếp cận phổ biến nhất để mô hình hóa các trạng thái thị trường **không quan sát trực tiếp được**. HMM giả định rằng trạng thái của thị trường là một biến ẩn $S_t$ tuân theo một chuỗi Markov bậc nhất, nơi xác suất chuyển sang trạng thái tiếp theo chỉ phụ thuộc vào trạng thái hiện tại.

### 2.1 Cơ Chế Thống Kê và Toán Học

Một mô hình HMM điển hình bao gồm $K$ trạng thái ẩn (thường là 2–3 trạng thái: Bull, Bear, Side-way). Mỗi trạng thái gắn liền với một phân phối quan sát Gaussian:

$$r_t \mid (S_t = i) \sim \mathcal{N}(\mu_i,\, \sigma_i^2)$$

Trong đó:
- $\mu_i$ — lợi nhuận trung bình đặc trưng cho trạng thái $i$
- $\sigma_i$ — biến động đặc trưng cho trạng thái $i$

Quá trình chuyển đổi được điều khiển bởi **ma trận chuyển trạng thái** $P$, với $p_{ij}$ là xác suất chuyển từ trạng thái $i$ sang $j$.

HMM cung cấp **xác suất hậu nghiệm** (smoothed probabilities) cho mỗi trạng thái tại mọi thời điểm, cho phép nhà đầu tư không chỉ xác định trạng thái hiện tại mà còn đánh giá độ tin cậy của phân loại. Ví dụ: trong các giai đoạn ổn định trước 2020, xác suất trạng thái biến động thấp duy trì gần 1, nhưng sẽ giảm đột ngột khi xảy ra cú sốc COVID-19.

### 2.2 Markov Switching Autoregressive (MSAR)

Khi chuỗi dữ liệu có sự phụ thuộc vào các giá trị trễ, mô hình **Markov Switching Autoregressive (MSAR)** mở rộng khả năng của HMM:

$$y_t = \alpha(S_t) + \sum_{j=1}^k \phi_j(S_t)\,y_{t-j} + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}\!\left(0,\, \sigma(S_t)^2\right)$$

Khung này cho phép nắm bắt các động lực phức tạp hơn của chu kỳ kinh tế và sự bền bỉ của xu hướng giá. Thư viện Python `statsmodels` cung cấp `MarkovAutoregression` và `MarkovRegression` cho việc triển khai.

### 2.3 Ưu Điểm và Hạn Chế

| Khía Cạnh | Nhận Xét |
|---|---|
| **Minh bạch** | Cao — các trạng thái có thể dán nhãn rõ ràng qua $\mu_i$, $\sigma_i$ |
| **Giả định Gaussian** | Dễ sai số trong khủng hoảng (đuôi béo, nhảy vọt) |
| **Tính Markov** | Không có bộ nhớ dài hạn — có thể không phản ánh đúng chu kỳ siêu dài |
| **Khởi tạo tham số** | Nhạy cảm với initialization — cần khởi tạo cẩn thận |

---

## 3. Phân Cụm Dựa Trên Khoảng Cách Wasserstein và Hình Học Phân Phối

### 3.1 Sự Thất Bại của Khoảng Cách Euclidean

Thuật toán **K-means** truyền thống sử dụng khoảng cách Euclidean thường thất bại trong phát hiện trạng thái thị trường vì:

- Xử lý các điểm dữ liệu như thực thể **độc lập** — không xét cấu trúc phân phối
- Trong tài chính, **hình dạng phân phối** (độ lệch, kurtosis) mang thông tin quan trọng hơn vị trí tuyệt đối
- Một giai đoạn lợi nhuận = 0 với biến động cực cao phải được phân loại khác biệt hoàn toàn với lợi nhuận = 0 biến động thấp

### 3.2 Thuật Toán Wasserstein K-means (WK-means)

**Khoảng cách Wasserstein** (Earth Mover's Distance) đo lường chi phí tối thiểu để biến đổi một phân phối xác suất thành một phân phối khác. WK-means:

1. Chia dữ liệu thành các **cửa sổ lăn** (rolling windows)
2. Coi mỗi cửa sổ như một phân phối thực nghiệm
3. Áp dụng phân cụm dựa trên khoảng cách Wasserstein thay vì Euclidean

Đối với phân phối một chiều, $W_1$ tính hiệu quả qua hàm phân phối tích lũy nghịch đảo:

$$W_1(\mu, \nu) = \int_0^1 \left|F_\mu^{-1}(t) - F_\nu^{-1}(t)\right| dt$$

### 3.3 Ưu Điểm Vượt Trội

| Ưu Điểm | Mô Tả |
|---|---|
| **Nhận thức phân phối** | Nhạy với mọi thay đổi hình dạng, kể cả rủi ro đuôi và bất đối xứng |
| **Phi tham số** | Không yêu cầu giả định Gaussian |
| **Ranh giới mượt mà** | Tạo ranh giới trạng thái tự nhiên hơn, phản ánh sự chuyển dịch dần dần |

### 3.4 So Sánh HMM vs. Wasserstein Clustering

| Tiêu Chí | HMM | WK-means |
|---|---|---|
| **Cơ chế cốt lõi** | Xác suất chuyển trạng thái (Markov) | Khoảng cách giữa các phân phối |
| **Giả định phân phối** | Thường là Gaussian | Phi tham số (tự do) |
| **Xác suất chuyển đổi** | Có (rõ ràng) | Không trực tiếp |
| **Độ nhạy rủi ro đuôi** | Kém (nếu dùng Gaussian) | Rất tốt |
| **Tính diễn giải** | Cao | Trung bình |
| **Yêu cầu dữ liệu** | Phù hợp dữ liệu nhỏ | Cần đủ dữ liệu cho cửa sổ phân phối |

---

## 4. Thuật Toán Phát Hiện Điểm Thay Đổi (Change Point Detection - CPD)

Phát hiện điểm thay đổi tập trung vào việc xác định các **thời điểm mà tại đó** các đặc tính thống kê (trung bình, phương sai, xu hướng) của chuỗi thời gian thay đổi đột ngột.

### 4.1 PELT — Pruned Exact Linear Time

PELT là thuật toán tối ưu toàn cục dựa trên **quy hoạch động**. Bài toán CPD cốt lõi:

$$Q_{\text{PELT}}(x_{1:n}) = \min_{m,\, \tau_{1:m}} \sum_{i=1}^{m+1} \left[C\!\left(x_{\tau_{i-1}+1:\tau_i}\right) + \beta\right]$$

Trong đó:
- $C$ — hàm chi phí (ví dụ: log-likelihood âm)
- $m$ — số lượng điểm thay đổi
- $\beta$ — tham số phạt (penalty) để tránh overfitting

**Quy tắc cắt tỉa (pruning):** Nếu điểm $u$ không thể là điểm thay đổi tối ưu tại $v$, nó bị loại khỏi tính toán tương lai → đạt độ phức tạp **$O(n)$** trung bình trong khi vẫn đảm bảo nghiệm **tối ưu toàn cục**.

### 4.2 Các Phương Pháp CPD Khác

| Thuật Toán | Loại Tìm Kiếm | Độ Phức Tạp | Đặc Điểm |
|---|---|---|---|
| **PELT** | Exact (tối ưu toàn cục) | $O(n)$ trung bình | Cần chọn $\beta$ phù hợp (BIC) |
| **Binary Segmentation** | Greedy (tham lam) | $O(n \log n)$ | Nhanh, dễ triển khai, có thể bỏ sót cấu trúc phức tạp |
| **WBS** | Greedy + ngẫu nhiên | $O(n \log n)$ | Cải thiện BinSeg với khoảng ngẫu nhiên |
| **CUSUM** | Online | $O(n)$ | Phát hiện trực tuyến, nhạy cảm với nhiễu |
| **Optimal Partitioning** | Exact | $O(n^2)$ | Không có pruning |
| **ED-PELT** | Non-parametric | $O(n)$ | Dựa trên hàm phân phối thực nghiệm |

### 4.3 Lựa Chọn Tham Số Phạt

- **BIC (Bayesian Information Criterion):** $\beta = k \ln(n)$, cân bằng giữa số điểm thay đổi và độ khớp dữ liệu
- **AIC:** $\beta = 2k$, ít phạt hơn BIC, phù hợp khi cần phát hiện nhiều điểm thay đổi nhỏ hơn

---

## 5. Học Sâu và Các Kiến Trúc Mạng Thần Kinh trong Nhận Diện Trạng Thái

Các mô hình học sâu mang lại khả năng nắm bắt **quan hệ phi tuyến tính** và **phụ thuộc dài hạn** mà thống kê truyền thống thường bỏ qua.

### 5.1 Mạng LSTM và RNN

**Long Short-Term Memory (LSTM)** được thiết kế để giải quyết vấn đề triệt tiêu đạo hàm, cho phép ghi nhớ các mẫu thị trường qua khoảng thời gian dài.

Trong nhận diện trạng thái, LSTM thường được dùng để:
- Phân loại cửa sổ thời gian vào các nhãn trạng thái đã định
- Dự báo xác suất chuyển đổi trạng thái

Việc tích hợp **cơ chế Attention** vào LSTM giúp mô hình tập trung vào các thời điểm nhạy cảm, cải thiện độ chính xác dự báo so với ARIMA truyền thống.

### 5.2 Autoencoders cho Phát Hiện Stress Thị Trường

**LSTM-Autoencoder** là phương pháp **không giám sát** mạnh mẽ gồm:
- **Encoder:** Nén chuỗi thời gian vào không gian ẩn (latent space)
- **Decoder:** Tái cấu trúc lại chuỗi từ biểu diễn ẩn

**Cơ chế phát hiện:**

```
Thị trường bình thường  →  Reconstruction Error thấp
Thị trường khủng hoảng  →  Reconstruction Error tăng vọt → Signal cảnh báo
```

**Variational Autoencoder (VAE):** Cung cấp cấu trúc latent space theo phân phối xác suất ưu tiên, cho phép:
- Mô hình hóa sự không chắc chắn
- Trích xuất nhân tố rủi ro hệ thống phi tuyến tính

Nghiên cứu 2025 chỉ ra kết hợp VAE + Boosting (LightGBM) cải thiện đáng kể độ chính xác dự báo biến động so với PCA.

### 5.3 Mạng Thần Kinh Đồ Thị (Graph Neural Networks - GNN)

Thị trường tài chính là một **mạng lưới tài sản liên kết chặt chẽ**. Thay đổi trạng thái thường biểu hiện qua thay đổi **cấu trúc tương quan** của mạng lưới.

#### Hệ Thống CRISP (2025)
*(Crisis-Resilient Investment through Spatio-temporal Patterns)*

| Thành Phần | Vai Trò |
|---|---|
| **GNN** | Học quan hệ không gian giữa các tài sản |
| **BiLSTM** | Học động lực thời gian |
| **Graph Attention** | Tự động lọc ~92.5% kết nối "nhiễu" |

CRISP:
- Dự báo **cấu trúc đồ thị cho 5 ngày tiếp theo**
- Thích ứng nhanh hơn với các biến động ngắn hạn so với phương pháp ngưỡng tĩnh
- Nhận diện các cơ chế khủng hoảng khác nhau (lây lan tín dụng 2008, lạm phát 2022) mà **không cần nhãn giám sát**

---

## 6. Khung Làm Việc Hybrid và Nghiên Cứu Mới Nhất 2024–2025

### 6.1 RegimeFolio

Đề xuất cuối 2024, RegimeFolio là hệ thống **tích hợp đa tầng**:

```
Tầng 1: Phát hiện trạng thái
         └─ Phân loại dựa trên VIX → các vùng biến động

Tầng 2: Dự báo chuyên biệt  
         └─ Ensemble (Random Forest + Gradient Boosting)
            được huấn luyện riêng cho từng trạng thái

Tầng 3: Tối ưu hóa danh mục
         └─ Mean-Variance thích ứng hằng ngày
            dựa trên tín hiệu dự báo trạng thái
```

**Kết quả thực nghiệm** (34 cổ phiếu vốn hóa lớn, US, 2020–2024):
- **Sharpe Ratio: 1.17** — vượt hẳn các mô hình không nhận thức trạng thái

### 6.2 Mô Hình Lai ARIMA-HMM

Kết hợp:
- **ARIMA** — xử lý thành phần tuyến tính ngắn hạn
- **HMM** — xử lý các trạng thái ẩn dài hạn

Hiệu quả vượt trội tại **thị trường mới nổi** (Ấn Độ): **Sharpe Ratio đạt 4.63**.

### 6.3 Diffusion-Augmented Reinforcement Learning (DARL, 2025)

- Sử dụng **diffusion models** để tạo kịch bản khủng hoảng giả định (ví dụ: "2025 Tariff Crisis")
- Huấn luyện RL agents có khả năng **chống chịu tốt hơn** trước các sự cố không lường trước
- Định hướng tiên tiến nhất trong giai đoạn 2025–2026

---

## 7. Đặc Trưng Đầu Vào và Các Yếu Tố Kỹ Thuật trong Mô Hình Hóa

### 7.1 Các Nhóm Đặc Trưng Quan Trọng

#### Nhóm 1 — Lợi Nhuận và Phong Cách (17 chuỗi)
- Lợi nhuận các chỉ số vốn hóa lớn / nhỏ
- Phong cách: **Value** (giá trị), **Momentum** (đà tăng), **Quality** (chất lượng), **Low Risk** (rủi ro thấp)

#### Nhóm 2 — Biến Động và Bất Định (6 chuỗi)
- Chênh lệch **CDS** (Credit Default Swap)
- Thanh khoản trái phiếu doanh nghiệp
- **Option skews** (độ lệch quyền chọn)
- Chỉ số bất định hàng hóa / kinh tế

#### Nhóm 3 — Cấu Trúc Thị Trường
| Khái Niệm | Ý Nghĩa |
|---|---|
| **CHoCH** (Change of Character) | Nhận diện sự thay đổi tâm lý thị trường |
| **BoS** (Break of Structure) | Xác nhận sự tiếp diễn của xu hướng |

### 7.2 Tiền Xử Lý và Robustness

**Residualization (loại bỏ tác động biến động theo thời gian):**
- Đảm bảo trạng thái được xác định dựa trên **cấu trúc cốt lõi** thay vì chỉ phản chiếu cú sốc ngắn hạn

**Robust Scaling:**
- Ngăn các biến số đơn lẻ có biên độ lớn gây ảnh hưởng quá mức

**Kiểm chứng robustness:**
- So sánh giai đoạn "Market Turmoil" với suy thoái NBER và sụt giảm S&P 500
- Mô hình hỗn hợp **t-distributed** đạt **F1-score 73%–78%** trong nhận diện khủng hoảng nghiêm trọng (30 năm)

---

## 8. Phân Tích So Sánh về Độ Trễ và Độ Chính Xác

### 8.1 Thách Thức về Độ Trễ (Detection Lag)

Có một **sự đánh đổi cố hữu** giữa:

```
Tốc độ phát hiện cao  ←→  Tính chính xác / độ tin cậy cao
     ↑                           ↓
Nhiều tín hiệu giả           Bỏ lỡ điểm quay đầu
  (false positives)            (late confirmation)
```

**Nguyên nhân gây trễ:**
- Cửa sổ lăn quá dài trong WK-means
- Tham số phạt $\beta$ quá cao trong PELT
- Thị trường thực hiện phần lớn chuyển động **trước khi HMM xác nhận** thay đổi trạng thái

### 8.2 Ưu Thế của Mô Hình Nhận Thức Tương Quan

Các nghiên cứu mới nhất (CRISP, GAE) chuyển hướng từ phát hiện thay đổi **mức độ giá** sang phát hiện thay đổi **cấu trúc mạng lưới**.

> **Sự hội tụ đột ngột của tương quan** giữa các cổ phiếu không liên quan là **leading indicator** cho khủng hoảng hệ thống — xuất hiện **trước** khi giá thực sự sụp đổ.

### 8.3 Bảng So Sánh Tổng Hợp

| Loại Mô Hình | Cơ Chế | Tốc Độ Phát Hiện | Độ Tin Cậy |
|---|---|---|---|
| **HMM (Thống kê)** | Xác suất hậu nghiệm | Trung bình – Chậm | Cao (minh bạch) |
| **CPD Offline (PELT)** | Tối ưu hóa toàn cục | Chậm (cần dữ liệu sau sự kiện) | Rất cao |
| **Deep Learning (LSTM)** | Nhận diện mẫu hình | Nhanh | Trung bình (rủi ro nhiễu) |
| **Mạng Lưới (GNN/CRISP)** | Hội tụ tương quan | Rất nhanh (cận thời gian thực) | Cao (nhận diện rủi ro hệ thống) |

---

## 9. Kết Luận và Định Hướng Chiến Lược

Việc phát hiện trạng thái thị trường không còn là bài toán thống kê đơn lẻ mà đã trở thành **khung công việc tích hợp đa kỷ luật**:

- **Nền tảng thống kê** (HMM) → sự minh bạch và kiểm soát
- **Tính linh hoạt phi tham số** (Wasserstein) → độ nhạy với rủi ro đuôi
- **Quan hệ không gian-thời gian** (GNN) → phát hiện khủng hoảng hệ thống sớm hơn

### 9.1 Nguyên Tắc Triển Khai Thực Tiễn

| Tình Huống | Phương Pháp Khuyên Dùng |
|---|---|
| Phân tích lịch sử / kiểm định | PELT + HMM |
| Giám sát thời gian thực | GNN / CRISP |
| Bảo vệ danh mục tự động | LSTM-Autoencoder (reconstruction error) |
| Thị trường mới nổi | ARIMA-HMM hybrid |
| Mô phỏng kịch bản cực đoan | DARL (Diffusion + RL) |

### 9.2 Định Hướng 2025–2026

Xu hướng tiếp theo sẽ tập trung vào:

1. **Generative AI** để mô phỏng các trạng thái chưa từng có trong lịch sử
2. **Cơ chế Attention** để dự báo chuyển dịch cấu trúc **trước** khi biểu hiện qua lợi nhuận
3. **Multi-asset regime detection** đồng thời trên nhiều lớp tài sản (equity, fixed income, crypto)

---

## 10. Tham Khảo Thư Viện và Công Cụ

| Thư Viện / Nền Tảng | Ngôn Ngữ | Thuật Toán Hỗ Trợ | Ứng Dụng Chính |
|---|---|---|---|
| **statsmodels** | Python | HMM, Markov Regression, MSAR | Phân tích kinh tế lượng, nghiên cứu học thuật |
| **ruptures** | Python | PELT, BinSeg, Bottom-up, Kernel CPD | Phân đoạn chuỗi thời gian offline |
| **hmmlearn** | Python | Gaussian HMM, GMM-HMM | Triển khai HMM nhanh chóng |
| **PyTorch / TensorFlow** | Python | LSTM, Autoencoders, GNN, Transformer | Xây dựng mô hình deep learning tùy chỉnh |
| **QuantConnect** | Python / C# | HMM tích hợp dữ liệu thị trường trực tiếp | Backtesting và giao dịch live |

### 10.1 Ví Dụ Code Nhanh

```python
# PELT với thư viện ruptures
import ruptures as rpt

signal = ...  # chuỗi thời gian lợi nhuận
algo = rpt.Pelt(model="rbf").fit(signal)
breakpoints = algo.predict(pen=10)  # pen = beta

# HMM với hmmlearn
from hmmlearn.hmm import GaussianHMM
import numpy as np

model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000)
model.fit(returns.reshape(-1, 1))
hidden_states = model.predict(returns.reshape(-1, 1))
```

---

*Tài liệu được tổng hợp và cấu trúc lại từ các nghiên cứu học thuật và báo cáo ngành tài chính, giai đoạn 2024–2025.*

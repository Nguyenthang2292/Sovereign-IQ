# Enhancement Roadmap - Crypto Probability Trading System

Tài liệu này mô tả các đề xuất nâng cấp và cải tiến cho hệ thống Crypto Probability Trading System, bao gồm các hướng phát triển ngắn hạn, trung hạn và dài hạn.

## 📋 Mục lục

- [1. Nâng Cấp Thuật Toán & Chiến Lược](#1-nâng-cấp-thuật-toán--chiến-lược)
- [2. Cải Tiến Machine Learning](#2-cải-tiến-machine-learning)
- [3. Nâng Cấp HMM Module](#3-nâng-cấp-hmm-module)
- [4. Nâng Cấp Hệ Thống & Kiến Trúc](#4-nâng-cấp-hệ-thống--kiến-trúc)
- [5. Tích Hợp Dữ Liệu Nâng Cao](#5-tích-hợp-dữ-liệu-nâng-cao)
- [6. Lộ Trình Triển Khai](#6-lộ-trình-triển-khai)

---

## 1. Nâng Cấp Thuật Toán & Chiến Lược

### 1.1 Funding Rate Arbitrage (Delta Neutral)

**Trạng thái hiện tại:**
- ✅ Đã có `HedgeFinder` trong `modules/portfolio/hedge_finder.py`
- ✅ Hỗ trợ multi-exchange qua `ExchangeManager`
- ⚠️ Chưa có module chuyên dụng cho funding rate arbitrage

**Đề xuất:**
- **Chiến lược:** Long spot (hoặc futures sàn A) và short futures Binance để triệt tiêu delta, ăn funding khi thị trường uptrend
- **Module mới:** `modules/funding_arbitrage/`
  - `funding_scanner.py`: Quét funding rates giữa các sàn
  - `arbitrage_calculator.py`: Tính toán lợi nhuận sau phí giao dịch và trượt giá
  - `execution_manager.py`: Quản lý execution với delta neutral constraints
- **Tích hợp:** Sử dụng `HedgeFinder` để tìm hedge pairs, mở rộng để hỗ trợ funding arbitrage

**Yêu cầu kỹ thuật:**
- Real-time funding rate monitoring
- Tính toán phí giao dịch chính xác (maker/taker)
- Mô phỏng trượt giá (slippage)
- Delta neutral position management

**Thư viện đề xuất:**
- `ccxt` (đã có) - Fetch funding rates
- `numpy` (đã có) - Tính toán delta neutral

---

### 1.2 Tối Ưu Danh Mục Markowitz (Modern Portfolio Theory)

**Trạng thái hiện tại:**
- ✅ Đã có `PortfolioRiskCalculator` với VaR, Beta calculation
- ✅ Đã có `PortfolioCorrelationAnalyzer` cho correlation analysis
- ⚠️ Hedge hiện tại dạng 1-1, chưa có portfolio optimization

**Đề xuất:**
- **Module mới:** `modules/portfolio/optimization.py`
  - `markowitz_optimizer.py`: Mean-Variance Optimization (MVO)
  - `efficient_frontier.py`: Tính toán efficient frontier
  - `risk_parity.py`: Risk Parity portfolio allocation
- **Tích hợp:** Mở rộng `PortfolioRiskCalculator` để hỗ trợ portfolio optimization

**Tính năng:**
- Tối ưu Sharpe ratio
- Tối ưu với constraints (min/max weights, sector limits)
- Risk parity allocation
- Black-Litterman model (nếu có views)

**Thư viện đề xuất:**
- `PyPortfolioOpt`: Portfolio optimization library
- `scipy.optimize` (đã có): Optimization algorithms

**Ví dụ sử dụng:**
```python
from modules.portfolio.optimization import MarkowitzOptimizer

optimizer = MarkowitzOptimizer(
    returns=returns_df,
    risk_free_rate=0.02
)
optimal_weights = optimizer.optimize_sharpe()
```

---

### 1.3 Nâng Cấp Pairs Trading

**Trạng thái hiện tại:**
- ✅ Đã có `modules/pairs_trading/` với comprehensive quantitative metrics
- ✅ Hỗ trợ Kalman Filter cho dynamic hedge ratio
- ✅ Cointegration tests (ADF, Johansen)
- ⚠️ Chưa có backtesting engine cho pairs trading

**Đề xuất cải tiến:**
- **Backtesting Engine:** `modules/pairs_trading/backtesting.py`
  - Walk-forward backtesting
  - In-sample/out-of-sample validation
  - Performance metrics (Sharpe, Calmar, Max Drawdown)
- **Advanced Strategies:**
  - Momentum pairs (đã có preset nhưng cần mở rộng)
  - Pairs rotation strategy
  - Multi-pairs portfolio
- **Risk Management:**
  - Dynamic position sizing dựa trên volatility
  - Stop-loss và take-profit tự động
  - Correlation breakdown detection

---

### 1.4 Advanced Execution Algorithms

**Trạng thái hiện tại:**
- ⚠️ Hầu hết các lệnh được gửi là Market hoặc Limit đơn giản.
- ⚠️ Chưa có phân chia lệnh lớn (large orders) để tránh trượt giá (slippage).

**Đề xuất:**
- **Module mới:** `modules/execution/`
  - `twap.py`: Time-Weighted Average Price execution.
  - `vwap.py`: Volume-Weighted Average Price execution.
  - `iceberg.py`: Ẩn khối lượng thực tế của lệnh.
  - `chase_limit.py`: Tự động điều chỉnh giá limit để khớp lệnh mà không dùng market order.
- **Lợi ích:**
  - Giảm impact cost và slippage cho các lệnh lớn.
  - Tối ưu hóa điểm vào lệnh (entry) và ra lệnh (exit).

---

### 1.5 Risk Management Core (Quản trị rủi ro chuyên sâu)

**Trạng thái hiện tại:**
- ⚠️ Quản trị rủi ro phân tán trong từng strategy.
- ⚠️ Chưa có hệ thống "Circuit Breaker" toàn cục.

**Đề xuất:**
- **Module mới:** `modules/risk_management/`
  - `kelly_criterion.py`: Tính toán size lệnh tối ưu dựa trên win-rate và payoff ratio.
  - `vol_target.py`: Điều chỉnh size để duy trì độ biến động danh mục mục tiêu (ví dụ: 15% annualized vol).
  - `circuit_breaker.py`: Tự động ngắt trading nếu drawdown trong ngày vượt quá giới hạn (ví dụ: -5%).
- **Tích hợp:**
  - Hoạt động như một "gatekeeper" chặn lệnh trước khi gửi ra sàn.

---

## 2. Cải Tiến Machine Learning

### 2.1 Feature Engineering Nâng Cao

**Trạng thái hiện tại:**
- ✅ XGBoost module với feature engineering cơ bản
- ✅ TFT (Temporal Fusion Transformer) đã được implement
- ⚠️ Chưa có order book data, on-chain data, sentiment data

**Đề xuất:**

#### 2.1.1 Order Book Imbalance
- **Module mới:** `modules/common/orderbook/`
  - `orderbook_fetcher.py`: Fetch order book data từ exchanges
  - `imbalance_calculator.py`: Tính toán bid/ask imbalance
  - `orderflow_analyzer.py`: Phân tích order flow (delta, volume profile)
- **Tích hợp:** Thêm order book features vào XGBoost và TFT pipelines
- **Timeframe:** Ngắn hạn (<5 phút) cho scalping strategies

#### 2.1.2 On-Chain Data
- **Module mới:** `modules/common/onchain/`
  - `exchange_flow.py`: Exchange inflow/outflow
  - `whale_tracker.py`: Whale wallet monitoring
  - `network_metrics.py`: Network metrics (hash rate, active addresses)
- **Tích hợp:** Với timeframe lớn (H4, D1) cho swing trading
- **Data sources:**
  - Glassnode API
  - CryptoQuant API
  - Blockchain.com API

#### 2.1.3 Sentiment Analysis
- **Module mới:** `modules/common/sentiment/`
  - `news_crawler.py`: Crawl news từ crypto news sites
  - `twitter_scraper.py`: Twitter sentiment analysis
  - `sentiment_analyzer.py`: VADER hoặc BERT-based sentiment scoring
- **Tích hợp:** Thêm sentiment features vào ML models
- **Thư viện đề xuất:**
  - `vaderSentiment`: VADER sentiment analysis
  - `transformers`: BERT-based models
  - `tweepy`: Twitter API

---

### 2.2 Deep Learning cho Time-Series

**Trạng thái hiện tại:**
- ✅ Đã có TFT (Temporal Fusion Transformer) implementation
- ✅ Data pipeline và feature selection đã được implement
- ⚠️ Có thể mở rộng với các kiến trúc khác

**Đề xuất bổ sung:**

#### 2.2.1 LSTM/GRU Models
- **Module:** `modules/deeplearning/models/lstm.py`
- **Use case:** Giữ bộ nhớ dài hạn cho chuỗi giá
- **Tích hợp:** Thêm vào model registry, so sánh performance với TFT

#### 2.2.2 Transformer Variants
- **N-BEATS:** Neural Basis Expansion Analysis
- **Informer:** Long sequence time-series forecasting
- **Autoformer:** Decomposition architecture

**Thư viện đề xuất:**
- `PyTorch Forecasting` (đã có trong roadmap)
- `Darts`: Time series forecasting library

---

### 2.3 Meta-Labeling (Marcos Lopez de Prado)

**Đề xuất:**
- **Module mới:** `modules/metalabeling/`
  - `base_signal_generator.py`: Model 1 - Indicator-based signals
  - `meta_classifier.py`: Model 2 - XGBoost predicts success probability
  - `signal_filter.py`: Filter signals based on meta-classifier confidence
- **Workflow:**
  1. Model 1 (Indicators) tạo tín hiệu nền
  2. Model 2 (XGBoost) dự đoán xác suất tín hiệu Model 1 sẽ thành công
  3. Chỉ execute signals với confidence > threshold
- **Lợi ích:** Loại bỏ false positives, tăng win-rate

**Tích hợp:**
- Sử dụng existing indicators từ `modules/common/IndicatorEngine.py`
- Sử dụng XGBoost từ `modules/xgboost/`

---

### 2.4 Hyperparameter Optimization (AutoML)

**Trạng thái hiện tại:**
- ⚠️ Sử dụng hyperparameters cố định trong `config.py`.
- ⚠️ Chưa có cơ chế tự động tìm params tối ưu cho từng symbol/timeframe cụ thể.

**Đề xuất:**
- **Module mới:** `modules/xgboost/optimization.py`
  - `hyperparameter_tuner.py`: Tích hợp Optuna để tìm kiếm bộ tham số tốt nhất.
  - `study_manager.py`: Quản lý và lưu trữ kết quả optimization studies.
- **Workflow:**
  1. Định nghĩa search space (learning_rate, max_depth, subsample, colsample_bytree, v.v.).
  2. Chạy optimization loop sử dụng TimeSeriesSplit cross-validation (tận dụng logic trong `model.py`).
  3. Tự động cập nhật config hoặc lưu best params cho từng symbol.
- **Thư viện đề xuất:** `optuna`.

---

### 2.5 Model Persistence & MLOps

**Trạng thái hiện tại:**
- ⚠️ Retrain model mỗi khi chạy script -> Tốn tài nguyên và thời gian, không hiệu quả cho high-frequency hoặc testing liên tục.
- ⚠️ Không lưu lại lịch sử model để so sánh hiệu suất theo thời gian.

**Đề xuất:**
- **Module mới:** `modules/xgboost/persistence.py`
  - `model_registry.py`: Chức năng Lưu/Load model (sử dụng joblib hoặc pickle).
  - `version_control.py`: Quản lý metadata của model (accuracy, timestamp, params, training data range).
- **Workflow:**
  - Khi khởi động, kiểm tra xem có model đã train (còn hạn, ví dụ < 1h) cho symbol hiện tại không.
  - Nếu có -> Load model và predict ngay lập tức.
  - Nếu không hoặc model quá cũ -> Retrain -> Save model mới vào registry.

---

### 2.6 Interpretability (Explainable AI - XAI)

**Đề xuất:**
- **Module mới:** `modules/xgboost/explanation.py`
  - `shap_analyzer.py`: Tính toán SHAP (SHapley Additive exPlanations) values.
  - `feature_importance.py`: Visualization mức độ ảnh hưởng của từng feature đối với quyết định UP/DOWN.
- **Lợi ích:**
  - "White-box" mô hình: Hiểu tại sao model đưa ra dự đoán đó.
  - Ví dụ: Model có thể chỉ ra rằng "RSI > 80" đang đóng góp 60% vào quyết định "DOWN".
  - Giúp trader tự tin hơn khi vào lệnh hoặc lọc bỏ các tín hiệu vô lý.

---

## 3. Nâng Cấp HMM Module

**Trạng thái hiện tại:**
- ✅ Đã có 3 HMM strategies: Swings, KAMA, True High-Order HMM
- ✅ Strategy registry và signal combiner
- ✅ Multiple voting mechanisms
- ⚠️ Có thể mở rộng với các biến thể HMM nâng cao

### 3.1 Multivariate HMM (HMM Đa Biến)

**Đề xuất:**
- **Mở rộng:** `modules/hmm/core/multivariate.py`
- **Ý tưởng:** Thay vì chỉ quan sát một chuỗi dữ liệu, mô hình quan sát vector đa chiều
- **Dữ liệu đề xuất:**
  - Price Returns + Volume
  - Order Flow (Delta)
  - On-chain metrics (Exchange Inflow/Outflow)
- **Lợi ích:** Trạng thái thị trường được xác định chính xác hơn
  - Ví dụ: Giá tăng nhưng Vol giảm → "Bull Trap" thay vì "Uptrend"

**Kế thừa:** Module `kama.py` đã bắt đầu hướng này với feature matrix

---

### 3.2 Autoregressive HMM (AR-HMM)

**Đề xuất:**
- **Module mới:** `modules/hmm/core/ar_hmm.py`
- **Vấn đề:** HMM tiêu chuẩn giả định observations độc lập, không phù hợp với momentum
- **Giải pháp:** AR-HMM - observation tiếp theo phụ thuộc vào cả hidden state VÀ observations trước đó
- **Ứng dụng:** Hiệu quả cho trending markets
- **Kế thừa:** Có thể build trên cấu trúc `high_order.py`

---

### 3.3 Hierarchical HMM (HHMM)

**Đề xuất:**
- **Module mới:** `modules/hmm/core/hierarchical.py`
- **Cấu trúc:**
  - **Lớp trên:** Chế độ thị trường dài hạn (Bull/Bear/Sideways)
  - **Lớp dưới:** Biến động ngắn hạn (Pullback, Rally, Noise)
- **Lợi ích:** Lọc nhiễu, tránh over-trading
  - Ví dụ: Nếu lớp trên là Bull, lớp dưới chỉ kích hoạt LONG khi có Pullback

---

### 3.4 HMM-GARCH

**Đề xuất:**
- **Module mới:** `modules/hmm/core/hmm_garch.py`
- **Ý tưởng:** Mỗi hidden state gắn với một GARCH model riêng để dự báo volatility
- **Ứng dụng:** Quản lý rủi ro
  - Khi HMM chuyển sang "High Volatility" → tự động giảm position size hoặc nới rộng stop-loss
- **Thư viện:** `arch` (ARCH/GARCH models)

---

### 3.5 Factorial HMM

**Đề xuất:**
- **Module mới:** `modules/hmm/core/factorial.py`
- **Ý tưởng:** Tách biến động giá thành nhiều nguồn độc lập (Factors)
- **Cơ chế:**
  - Chuỗi Markov 1: Trạng thái thị trường chung (Bitcoin/Total Market)
  - Chuỗi Markov 2: Trạng thái nội tại của Altcoin
- **Lợi ích:** Phân loại tín hiệu
  - Coin tăng do "nước lên thuyền lên" (Beta) hay do nội tại mạnh (Alpha)

---

### 3.6 Input-Output HMM (IO-HMM)

**Đề xuất:**
- **Module mới:** `modules/hmm/core/io_hmm.py`
- **Ý tưởng:** Transition matrix không cố định mà thay đổi động dựa trên biến số vĩ mô
- **Input variables:**
  - BTC Dominance
  - Fear & Greed Index
  - Macro indicators (nếu có)
- **Triển khai:** Transition probabilities được tính lại dựa trên inputs
  - Ví dụ: P(Sideways → Dump) tăng khi Fear & Greed Index < 20

---

## 4. Nâng Cấp Hệ Thống & Kiến Trúc

### 4.1 Backtesting Engine

**Trạng thái hiện tại:**
- ⚠️ Chưa có backtesting engine chuyên dụng
- ✅ Có Historical Simulation VaR trong `PortfolioRiskCalculator`
- ✅ Có performance analysis trong `pairs_trading`

**Đề xuất:**
- **Module mới:** `modules/backtesting/`
  - `backtester.py`: Core backtesting engine
  - `strategy_interface.py`: Standardized strategy interface
  - `performance_analyzer.py`: Performance metrics calculation
  - `visualization.py`: Equity curve, drawdown charts
- **Tính năng:**
  - Walk-forward backtesting
  - In-sample/out-of-sample validation
  - Transaction cost modeling (fees, slippage)
  - Funding cost calculation (cho futures)
  - Multi-strategy backtesting
- **Tích hợp:**
  - Tất cả strategies hiện có (ATC, Range Oscillator, SPC, HMM, XGBoost)
  - Pairs trading strategies
  - Portfolio optimization strategies

**Thư viện đề xuất:**
- `vectorbt`: Vectorized backtesting
- `backtrader`: Event-driven backtesting framework

**Ví dụ sử dụng:**
```python
from modules.backtesting import Backtester
from modules.adaptive_trend import ATCStrategy

strategy = ATCStrategy()
backtester = Backtester(
    strategy=strategy,
    data=historical_data,
    initial_capital=10000,
    commission=0.001
)
results = backtester.run()
backtester.plot_equity_curve()
```

---

### 4.2 Event-Driven Architecture

**Trạng thái hiện tại:**
- ✅ Fetch → Analyze → Print workflow (polling-based)
- ✅ Multi-exchange support với fallback
- ⚠️ Chưa có real-time streaming

**Đề xuất:**
- **Module mới:** `modules/realtime/`
  - `websocket_manager.py`: WebSocket connection management
  - `event_bus.py`: Event bus cho pub/sub pattern
  - `stream_processor.py`: Real-time data processing
  - `signal_emitter.py`: Emit signals khi có events
- **Kiến trúc:**
  ```
  WebSocket → Event Bus → Strategy Listeners → Signal Emitter → Execution Engine
  ```
- **Tính năng:**
  - Real-time price updates
  - Order book streaming
  - Trade execution events
  - Strategy signal events
- **Lợi ích:**
  - Phản ứng mili-giây
  - Không bị delay do sleep loop
  - Scalable với nhiều strategies

**Thư viện đề xuất:**
- `ccxt` (đã có) - WebSocket support
- `asyncio` (built-in) - Async event handling
- `websockets`: WebSocket client/server

---

### 4.3 Web Dashboard

**Trạng thái hiện tại:**
- ✅ CLI interfaces cho tất cả modules
- ✅ Colorama cho colored output
- ⚠️ Chưa có web UI

**Đề xuất:**
- **Module mới:** `modules/dashboard/`
  - `app.py`: Streamlit main application
  - `pages/`: Multi-page dashboard
    - `overview.py`: Tổng quan portfolio
    - `strategies.py`: Strategy performance
    - `signals.py`: Real-time signals
    - `backtesting.py`: Backtesting results
    - `settings.py`: Configuration
- **Tính năng:**
  - Real-time PnL charts
  - Position tracking
  - Signal monitoring
  - Strategy performance comparison
  - Emergency "Close All" button
  - Configuration management
- **Tích hợp:**
  - Tất cả modules hiện có
  - Real-time data từ WebSocket (nếu có)
  - Backtesting results visualization

**Thư viện đề xuất:**
- `streamlit` (đã có trong requirements-ocr.txt)
- `plotly`: Interactive charts
- `pandas` (đã có): Data manipulation

---

### 4.4 Database & Data Persistence

**Đề xuất:**
- **Module mới:** `modules/database/`
  - `models.py`: SQLAlchemy models
  - `repository.py`: Data access layer
  - `migrations/`: Database migrations
- **Dữ liệu lưu trữ:**
  - Historical OHLCV data
  - Strategy signals và results
  - Backtesting results
  - Portfolio positions
  - Performance metrics
- **Database options:**
  - SQLite (development)
  - PostgreSQL (production)
  - TimescaleDB (time-series optimization)

**Thư viện đề xuất:**
- `SQLAlchemy`: ORM
- `Alembic`: Database migrations
- `TimescaleDB`: Time-series database extension

---

### 4.5 System Ops & Monitoring

**Trạng thái hiện tại:**
- ⚠️ Log file Text đơn giản.
- ⚠️ Chưa có cảnh báo hệ thống (system health alerts) qua Telegram/Discord (chỉ có tín hiệu trade).

**Đề xuất:**
- **Module mới:** `modules/ops/`
  - `heartbeat.py`: Gửi tín hiệu "I'm alive" định kỳ.
  - `rate_limit_guard.py`: Quản lý tập trung API rate limits để tránh bị ban IP.
  - `alert_bot.py`: Bot chuyên dụng báo lỗi (Exceptions, Disconnects) và trạng thái tài nguyên (RAM/CPU).
- **Lợi ích:**
  - Tăng độ tin cậy của trading bot chạy 24/7.
  - Phát hiện sớm sự cố hạ tầng.

---

## 5. Tích Hợp Dữ Liệu Nâng Cao

### 5.1 Multi-Timeframe Analysis

**Đề xuất:**
- **Module mới:** `modules/common/multitimeframe/`
  - `analyzer.py`: Phân tích signals trên nhiều timeframes
  - `consensus.py`: Tạo consensus từ multiple timeframes
- **Use case:**
  - Higher timeframe xác định trend
  - Lower timeframe xác định entry point
  - Conflict resolution giữa timeframes

---

### 5.2 Cross-Asset Analysis

**Đề xuất:**
- **Mở rộng:** `modules/common/crossasset/`
  - `correlation_matrix.py`: Correlation giữa các assets
  - `spillover_analyzer.py`: Phân tích spillover effects
  - `market_regime.py`: Xác định market regime từ multiple assets
- **Tích hợp:**
  - BTC dominance analysis
  - Stock market correlation (nếu có data)
  - Commodity correlation

---

### 5.3 Alternative Data Sources

**Đề xuất:**
- **Social Media:**
  - Reddit sentiment
  - Telegram channel analysis
  - Discord activity
- **Options Data:**
  - Put/Call ratios
  - Options flow
  - Implied volatility
- **Derivatives:**
  - Futures basis
  - Perpetual funding rates
  - Options skew

---

## 6. Lộ Trình Triển Khai

### 6.1 Ngắn Hạn (1-3 tháng)

**Priority: High**

1. **Backtesting Engine** (`modules/backtesting/`)
   - Core backtesting framework
   - Tích hợp với existing strategies
   - Basic performance metrics

2. **Funding Rate Arbitrage** (`modules/funding_arbitrage/`)
   - Funding rate scanner
   - Arbitrage calculator
   - Delta neutral execution

3. **Order Book Features** (`modules/common/orderbook/`)
   - Order book data fetching
   - Imbalance calculation
   - Tích hợp vào XGBoost pipeline

4. **Web Dashboard** (`modules/dashboard/`)
   - Basic Streamlit app
   - Portfolio overview
   - Signal monitoring

5. **Risk Management Core** (`modules/risk_management/`)
   - Circuit Breakers
   - Kelly Criterion

---

### 6.2 Trung Hạn (3-6 tháng)

**Priority: Medium**

1. **Markowitz Optimization** (`modules/portfolio/optimization.py`)
   - Mean-Variance Optimization
   - Efficient frontier
   - Risk parity

2. **Meta-Labeling** (`modules/metalabeling/`)
   - Base signal generator
   - Meta-classifier
   - Signal filtering

3. **Hyperparameter Optimization & MLOps** (`modules/xgboost/`)
   - Tích hợp Optuna cho XGBoost
   - Cơ chế Save/Load model (Persistence)

4. **Advanced HMM Variants**
   - Multivariate HMM
   - AR-HMM
   - HMM-GARCH

5. **Event-Driven Architecture** (`modules/realtime/`)
   - WebSocket manager
   - Event bus
   - Real-time processing

6. **On-Chain Data** (`modules/common/onchain/`)
   - Exchange flow
   - Whale tracking
   - Network metrics

7. **Advanced Execution** (`modules/execution/`)
   - TWAP/VWAP algo
   - Iceberg orders
   
8. **System Ops** (`modules/ops/`)
   - Heartbeat & Health check
   - Rate limit manager

---

### 6.3 Dài Hạn (6-12 tháng)

**Priority: Low (Research & Development)**

1. **Hierarchical HMM** (`modules/hmm/core/hierarchical.py`)
2. **Factorial HMM** (`modules/hmm/core/factorial.py`)
3. **Input-Output HMM** (`modules/hmm/core/io_hmm.py`)
4. **Sentiment Analysis** (`modules/common/sentiment/`)
5. **Database & Persistence** (`modules/database/`)
6. **Explainable AI (XAI)**
   - SHAP integration cho XGBoost
7. **Advanced Deep Learning Models**
   - N-BEATS
   - Informer
   - Autoformer

---

## 7. Ghi Chú Kỹ Thuật

### 7.1 Module Structure Standard

Tất cả modules mới nên tuân theo cấu trúc chuẩn:

```
modules/new_module/
├── __init__.py
├── README.md
├── core/              # Core logic
├── config/            # Configuration (nếu cần)
├── cli/               # CLI interface (nếu cần)
├── utils/             # Utilities
└── tests/             # Tests (trong tests/new_module/)
```

### 7.2 Testing Requirements

- Unit tests cho tất cả core functions
- Integration tests cho workflows
- Performance tests cho real-time components
- Backtesting validation cho strategies

### 7.3 Documentation Requirements

- README.md cho mỗi module
- Docstrings cho tất cả public functions
- Examples và usage guides
- Architecture diagrams (nếu phức tạp)

---

## 8. Tài Nguyên & Tham Khảo

### 8.1 Thư Viện Đề Xuất

- **Portfolio Optimization:** `PyPortfolioOpt`
- **Backtesting:** `vectorbt`, `backtrader`
- **Time-Series:** `Darts`, `PyTorch Forecasting`
- **Sentiment:** `vaderSentiment`, `transformers`
- **Database:** `SQLAlchemy`, `TimescaleDB`
- **Web:** `streamlit`, `plotly`

### 8.2 Papers & Research

- **Meta-Labeling:** "Advances in Financial Machine Learning" - Marcos Lopez de Prado
- **HMM Variants:** "Hidden Markov Models for Time Series" - Walter Zucchini
- **Portfolio Optimization:** "Modern Portfolio Theory" - Harry Markowitz
- **Event-Driven Architecture:** "Designing Data-Intensive Applications" - Martin Kleppmann

---

**Last Updated:** 2025
**Version:** 2.0
**Maintainer:** Crypto Probability Team

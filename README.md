# Crypto Probability / Dự Đoán Giá Cryptocurrency bằng AI

A comprehensive cryptocurrency trading analysis system using Machine Learning, Deep Learning, and Quantitative Strategies.

Hệ thống phân tích giao dịch tiền điện tử toàn diện sử dụng Machine Learning, Deep Learning và các Chiến Lược Định Lượng.

## 🚀 Features / Tính Năng

### Core Capabilities / Khả Năng Cốt Lõi

-   **Multi-Exchange Support / Hỗ Trợ Đa Sàn**: Automatically fetches data from Binance, Kraken, KuCoin, Gate.io, OKX, Bybit, MEXC, Huobi with smart fallback / Tự động lấy dữ liệu từ nhiều sàn với cơ chế chuyển đổi thông minh
-   **Advanced Indicators / Chỉ Báo Nâng Cao**: SMA, RSI, ATR, MACD, Bollinger Bands, Stochastic RSI, OBV, Candlestick Patterns, KAMA / Sử dụng đầy đủ các chỉ báo kỹ thuật
-   **Multiple ML Models / Nhiều Mô Hình ML**:
    - XGBoost for directional prediction / XGBoost cho dự đoán hướng
    - Temporal Fusion Transformer (TFT) for deep learning predictions / TFT cho dự đoán deep learning
    - HMM-KAMA for state-based analysis / HMM-KAMA cho phân tích trạng thái
-   **Pairs Trading / Giao Dịch Cặp**: Identify and analyze pairs trading opportunities with quantitative metrics / Xác định và phân tích cơ hội giao dịch cặp với các metrics định lượng
-   **Portfolio Management / Quản Lý Danh Mục**: Risk calculation, correlation analysis, hedge finding / Tính toán rủi ro, phân tích tương quan, tìm hedge

## 📁 Project Structure / Cấu Trúc Dự Án

```
crypto-probability/
├── main_xgboost_prediction.py      # XGBoost prediction CLI
├── main_deeplearning_prediction.py # Deep Learning (TFT) training
├── main_pairs_trading.py           # Pairs trading analysis
├── main_portfolio_manager.py       # Portfolio risk management
├── main_hmm.py                     # HMM Signal Combiner (High-Order HMM + HMM-KAMA)
├── modules/                        # Core modules
│   ├── common/                     # Shared utilities
│   │   ├── DataFetcher.py         # Multi-exchange data fetching
│   │   ├── ExchangeManager.py     # Exchange connection management
│   │   ├── IndicatorEngine.py     # Technical indicators
│   │   └── indicators/            # Indicator implementations
│   ├── xgboost/                    # XGBoost prediction module
│   ├── deeplearning/              # Deep learning module (TFT)
│   ├── pairs_trading/             # Pairs trading strategies
│   ├── portfolio/                 # Portfolio management
│   └── hmm/                       # HMM-KAMA analysis
├── tests/                          # Comprehensive test suite
├── docs/                           # Documentation
└── artifacts/                      # Model checkpoints and outputs
```

## 🔧 Installation / Cài Đặt

### Prerequisites / Yêu Cầu

- Python 3.8+ (Python 3.10+ recommended / Khuyến nghị Python 3.10+)
- pip

### Setup / Thiết Lập

1. **Clone the repository / Clone repository:**
   ```bash
   git clone <repository-url>
   cd crypto-probability
   ```

2. **Install dependencies / Cài đặt các thư viện:**

   **Basic requirements / Yêu cầu cơ bản:**
   ```bash
   pip install -r requirements.txt
   ```

   **For Deep Learning / Cho Deep Learning:**
   ```bash
   pip install -r requirements-ml.txt
   ```

   **For development / Cho phát triển:**
   ```bash
   pip install -r requirements-dev.txt
   ```

3. **Configure API keys (optional) / Cấu hình API keys (tùy chọn):**
   
   Create `modules/config_api.py` with your exchange API keys:
   ```python
   BINANCE_API_KEY = "your_key"
   BINANCE_API_SECRET = "your_secret"
   ```

## 📖 Usage / Cách Sử Dụng

### 1. XGBoost Prediction / Dự Đoán XGBoost

Predict next price movement using XGBoost classifier:

```bash
python main_xgboost_prediction.py
```

**Options:**
- `--symbol SYMBOL`: Trading pair (e.g., BTC/USDT)
- `--timeframe TIMEFRAME`: Timeframe (1h, 4h, 1d)
- `--limit LIMIT`: Number of candles to fetch
- `--no-prompt`: Skip interactive prompts

**Example:**
```bash
python main_xgboost_prediction.py --symbol BTC/USDT --timeframe 1h --limit 500
```

### 2. Deep Learning (TFT) / Deep Learning (TFT)

Train Temporal Fusion Transformer model for price prediction:

```bash
python main_deeplearning_prediction.py
```

**Options:**
- `--symbol SYMBOL`: Trading pair to train on
- `--timeframe TIMEFRAME`: Data timeframe
- `--epochs EPOCHS`: Number of training epochs
- `--batch-size SIZE`: Batch size
- `--gpu`: Use GPU if available
- `--phase PHASE`: Training phase (1=regression, 2=classification)

**Example:**
```bash
python main_deeplearning_prediction.py --symbol BTC/USDT --timeframe 1h --epochs 10 --gpu
```

### 3. Pairs Trading / Giao Dịch Cặp

Identify pairs trading opportunities:

```bash
python main_pairs_trading.py
```

**Options:**
- `--pairs-count N`: Number of pairs to analyze
- `--sort-by SCORE`: Sort by `opportunity_score` or `quantitative_score`
- `--require-cointegration`: Only show cointegrated pairs
- `--max-half-life N`: Maximum half-life threshold
- `--min-quantitative-score N`: Minimum quantitative score (0-100)
- `--max-pairs N`: Maximum pairs to display

**Example:**
```bash
python main_pairs_trading.py --sort-by quantitative_score --require-cointegration --min-quantitative-score 70
```

### 4. Portfolio Manager / Quản Lý Danh Mục

Manage portfolio risk and find hedges:

```bash
python main_portfolio_manager.py
```

**Features:**
- Portfolio risk calculation (VaR, Beta)
- Correlation analysis
- Automatic hedge finding
- Real-time position tracking

### 5. HMM-KAMA Analysis / Phân Tích HMM-KAMA

State-based analysis using Hidden Markov Model with KAMA:

```bash
python main_hmm.py
```

**Options:**
- `--symbol SYMBOL`: Trading pair to analyze
- `--timeframe TIMEFRAME`: Data timeframe
- `--window-size N`: HMM window size
- `--window-kama N`: KAMA window size
- `--fast-kama N`: Fast KAMA parameter
- `--slow-kama N`: Slow KAMA parameter
- `--orders-argrelextrema N`: Order for swing detection
- `--strict-mode`: Use strict mode for swing-to-state conversion

## 🧪 Testing / Kiểm Thử

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run specific test module
pytest tests/xgboost/
pytest tests/deeplearning/
pytest tests/pairs_trading/
pytest tests/portfolio/
pytest tests/common/

# Run with coverage
pytest --cov=modules --cov-report=html
```

## 📚 Documentation / Tài Liệu

Detailed documentation is available in the `docs/` directory:

- **Common**: Exchange management, data fetching, indicators
- **XGBoost**: Prediction model documentation
- **Deep Learning**: TFT model architecture and training
- **Pairs Trading**: Strategy documentation and quantitative metrics
- **Portfolio**: Risk management and hedge finding

See `docs/README.md` for the full documentation index.

## 🏗️ Architecture / Kiến Trúc

### Module Organization / Tổ Chức Module

- **`modules/common/`**: Shared utilities used across all modules
  - `DataFetcher`: Multi-exchange data fetching with fallback
  - `ExchangeManager`: Exchange connection and API management
  - `IndicatorEngine`: Technical indicator computation
  - `Position`: Position data structure

- **`modules/xgboost/`**: XGBoost prediction pipeline
  - Feature engineering and labeling
  - Model training and prediction
  - Classification report generation

- **`modules/deeplearning/`**: Deep learning models
  - Temporal Fusion Transformer (TFT) implementation
  - Data pipeline and preprocessing
  - Feature selection
  - Model training with PyTorch Lightning

- **`modules/pairs_trading/`**: Pairs trading strategies
  - Performance analysis
  - Pair metrics computation (cointegration, half-life, etc.)
  - Opportunity scoring
  - Risk metrics

- **`modules/portfolio/`**: Portfolio management
  - Risk calculation (VaR, Beta)
  - Correlation analysis
  - Hedge finding

- **`modules/hmm/`**: Hidden Markov Model analysis
  - HMM-KAMA state detection
  - High-Order HMM signal generation
  - Signal combiner with conflict resolution
  - Mean reversion analysis

## 🔍 Key Features / Tính Năng Chính

### Quantitative Metrics / Metrics Định Lượng

The pairs trading module includes comprehensive quantitative metrics:

- **Cointegration Tests**: ADF and Johansen tests
- **Mean Reversion**: Half-life, Hurst exponent
- **Risk Metrics**: Sharpe ratio, max drawdown, Calmar ratio
- **Statistical Tests**: Z-score analysis, correlation metrics
- **Classification Metrics**: F1-score, precision, recall

See `QUANT_METRICS_USAGE_REPORT.md` for detailed usage.

### Smart Data Fetching / Lấy Dữ Liệu Thông Minh

- Automatic exchange fallback when data is stale
- Multi-exchange support for redundancy
- Caching to reduce API calls
- Freshness checking

### Advanced Indicators / Chỉ Báo Nâng Cao

- **Trend**: SMA, EMA, MACD
- **Momentum**: RSI, Stochastic RSI
- **Volatility**: ATR, Bollinger Bands
- **Volume**: OBV, Volume indicators
- **Candlestick Patterns**: Doji, Engulfing, Three White Soldiers, etc.
- **Custom**: KAMA (Kaufman Adaptive Moving Average)

## ⚙️ Configuration / Cấu Hình

Configuration is managed in `modules/config.py`. Key settings:

- Exchange selection and priorities
- Timeframe defaults
- Model hyperparameters
- Risk thresholds
- Trading parameters

For API keys, create `modules/config_api.py` (not tracked in git).

## 🛠️ Development / Phát Triển

### Code Structure / Cấu Trúc Code

- Modular design with clear separation of concerns
- Comprehensive test coverage
- Type hints where applicable
- Documentation strings

### Running Tests / Chạy Tests

```bash
# All tests
pytest

# Specific module
pytest tests/xgboost/ -v

# With coverage
pytest --cov=modules --cov-report=term-missing
```

### Code Quality / Chất Lượng Code

- Linting: `pylint` or `flake8`
- Formatting: `black` (recommended)
- Type checking: `mypy` (optional)

## 📄 License / Giấy Phép

See `LICENSE` file for details.

## ⚠️ Disclaimer / Tuyên Bố Miễn Trừ

**Not Financial Advice / Không Phải Lời Khuyên Tài Chính**: 

This tool is for educational and research purposes only. Trading cryptocurrency involves high risk and can result in significant financial losses. Always do your own research and never invest more than you can afford to lose.

Công cụ này chỉ dành cho mục đích giáo dục và nghiên cứu. Giao dịch tiền điện tử có rủi ro cao và có thể dẫn đến tổn thất tài chính đáng kể. Luôn tự nghiên cứu và không bao giờ đầu tư nhiều hơn số tiền bạn có thể mất.

## 🤝 Contributing / Đóng Góp

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## 📞 Support / Hỗ Trợ

For issues, questions, or contributions, please open an issue on the repository.

---

**Last Updated**: 2024
**Version**: 2.0

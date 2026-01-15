# Sovereign-IQ

A comprehensive cryptocurrency trading analysis system using Machine Learning, Deep Learning, and Quantitative Strategies.

## 🚀 Features

### Core Capabilities

- **Multi-Exchange Support**: Automatically fetches data from Binance, Kraken, KuCoin, Gate.io, OKX, Bybit, MEXC, Huobi with smart fallback.
- **Advanced Indicators**: SMA, RSI, ATR, MACD, Bollinger Bands, Stochastic RSI, OBV, Candlestick Patterns, KAMA, plus custom indicators.
- **Multiple ML Models**:
    - XGBoost for directional prediction.
    - Temporal Fusion Transformer (TFT) for deep learning forecasts.
    - **HMM (Hidden Markov Model)**: Three HMM strategies:
        - HMM-Swings: Swing detection-based state classification
        - HMM-KAMA: KAMA-based HMM with ARM and K-Means clustering
        - True High-Order HMM: State space expansion with automatic order optimization
- **Adaptive Trend Classification (ATC)**: Multi-layer trend analysis with robustness filtering.
- **Range Oscillator**: Advanced oscillator-based signal generation with multiple strategies.
- **Simplified Percentile Clustering (SPC)**: Cluster-based market regime detection with multiple strategies.
- **Decision Matrix**: Pseudo Random Forest-like voting system for combining multiple indicators.
- **Pairs Trading**: Identify and analyze mean-reversion or momentum pairs with extensive quantitative metrics.
- **Portfolio Management**: Risk calculation, correlation analysis, hedge finding.

## 📁 Project Structure

```
crypto-probability/
├── main_hybrid.py                           # ATC + Range Oscillator + SPC (Hybrid Analyzer)
├── main_voting.py                            # ATC + Range Oscillator + SPC (Voting Analyzer)
├── core/                                     # Core analyzers
│   ├── hybrid_analyzer.py                   # Sequential filtering + voting
│   ├── voting_analyzer.py                   # Pure voting system
│   ├── signal_calculators.py                # Signal calculation helpers
│   └── README.md                            # Analyzer workflow comparison
├── main/                                     # Individual module entry points
│   ├── main_atc.py                          # Adaptive Trend Classification
│   ├── main_atc_oscillator.py               # ATC + Range Oscillator
│   ├── main_xgboost.py                      # XGBoost Prediction
│   ├── main_simplified_percentile_clustering.py  # SPC Analysis
│   ├── main_deeplearning_prediction.py      # Deep Learning TFT
│   ├── main_hmm.py                          # HMM Signal Combiner
│   ├── main_pairs_trading.py                # Pairs Trading
│   └── main_portfolio_manager.py            # Portfolio Manager
├── modules/                                  # Core modules
│   ├── common/                              # Shared utilities
│   │   ├── DataFetcher.py                  # Multi-exchange data fetching
│   │   ├── ExchangeManager.py              # Exchange connection management
│   │   ├── IndicatorEngine.py              # Technical indicators
│   │   ├── indicators/                     # Indicator implementations
│   │   └── quantitative_metrics/            # Quantitative analysis tools
│   ├── adaptive_trend/                      # Adaptive Trend Classification
│   ├── range_oscillator/                    # Range Oscillator strategies
│   ├── simplified_percentile_clustering/   # SPC clustering and strategies
│   ├── decision_matrix/                     # Decision Matrix voting system
│   ├── xgboost/                             # XGBoost prediction module
│   ├── deeplearning/                        # Deep learning module (TFT)
│   ├── pairs_trading/                       # Pairs trading strategies
│   ├── portfolio/                           # Portfolio management
│   └── hmm/                                 # HMM module (3 strategies)
│       ├── core/                            # HMM implementations
│       │   ├── swings.py                   # HMM-Swings strategy
│       │   ├── kama.py                     # HMM-KAMA strategy
│       │   └── high_order.py               # True High-Order HMM
│       ├── signals/                         # Signal processing
│       │   ├── strategy.py                 # Strategy interface
│       │   ├── registry.py                 # Strategy registry
│       │   ├── combiner.py                 # Signal combiner
│       │   ├── voting.py                   # Voting mechanisms
│       │   └── ...
│       └── README.md                        # HMM module documentation
├── config/                                   # Configuration files
│   ├── common.py                            # Common settings
│   ├── hmm.py                               # HMM configuration
│   ├── decision_matrix.py                   # Decision Matrix config
│   └── ...
├── cli/                                      # CLI utilities
│   ├── argument_parser.py                  # Argument parsing
│   └── display.py                           # Display utilities
├── tests/                                    # Comprehensive test suite
├── artifacts/                                # Model checkpoints and outputs
├── setup/                                    # Setup scripts and documentation
│   ├── QUICK_START_API_KEYS.md             # Quick start guide for API keys
│   ├── SECURITY.md                          # Security best practices
│   └── setup_api_keys.*                     # Setup scripts (Windows/Linux/Mac)
└── README*.md                               # Documentation files
```

## 🔧 Installation

### Prerequisites

- Python 3.9+ (Python 3.10+ recommended)
- pip

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd crypto-probability
   ```

2. **Configure API Keys:** (Chạy sau bước 1, trước hoặc cùng lúc với cài đặt dependencies)
   ```bash
   # Windows (PowerShell)
   .\setup\setup_api_keys.ps1
   
   # Windows (Command Prompt)
   setup\setup_api_keys.bat
   
   # Linux/Mac
   chmod +x setup/setup_api_keys.sh
   ./setup/setup_api_keys.sh
   ```
   **Lưu ý:** Các script này an toàn để chạy lại (idempotent) và việc cấu hình API keys độc lập với cài đặt dependencies, nên bạn có thể chạy trước hoặc sau bước 3. Xem [setup/QUICK_START_API_KEYS.md](./setup/QUICK_START_API_KEYS.md) để biết thêm chi tiết.

3. **Install dependencies:**

   **Basic requirements:**
   ```bash
   pip install -r requirements.txt
   ```

   **For Deep Learning:**
   ```bash
   pip install -r requirements-ml.txt
   ```

   **For development:**
   ```bash
   pip install -r requirements-dev.txt
   ```

3. **Configure API keys (optional):**
   
   **Recommended: Use environment variables (see [setup/QUICK_START_API_KEYS.md](./setup/QUICK_START_API_KEYS.md)):**
   ```bash
   # Windows (PowerShell)
   .\setup\setup_api_keys.ps1
   
   # Windows (Command Prompt)
   setup\setup_api_keys.bat
   
   # Linux/Mac
   chmod +x setup/setup_api_keys.sh
   ./setup/setup_api_keys.sh
   ```
   
   ⚠️ **NOT RECOMMENDED / DEPRECATED: Hardcoded API keys in `config/config_api.py`:**
   ```python
   BINANCE_API_KEY = "your_key"
   BINANCE_API_SECRET = "your_secret"
   ```
   **⚠️ Lưu ý:** Phương pháp hardcoded API keys KHÔNG được khuyến nghị vì rủi ro bảo mật. Vui lòng sử dụng biến môi trường như hướng dẫn ở trên. Xem [setup/SECURITY.md](./setup/SECURITY.md) để biết thêm chi tiết về bảo mật.

## 📖 Usage

### 1. Adaptive Trend Classification (ATC)

Analyze market trends using multi-layer adaptive classification:

```bash
python main/main_atc.py
```

**Features:**
- Multi-layer trend analysis with robustness filtering
- Auto scan across multiple symbols
- Interactive timeframe selection
- Configurable moving averages and parameters

### 2. ATC + Range Oscillator

Combine ATC signals with Range Oscillator confirmation:

```bash
python main_atc_oscillator.py
```

**Features:**
- Sequential filtering: ATC → Range Oscillator
- Parallel processing for performance
- Fallback to ATC-only if no oscillator confirmation

### 3. ATC + Range Oscillator + SPC (Hybrid Analyzer)

Hybrid approach combining sequential filtering and voting:

```bash
python main_hybrid.py
```

**Workflow:**
1. ATC auto scan to find initial LONG/SHORT signals
2. Filter by Range Oscillator confirmation (sequential filtering)
3. Calculate SPC signals for remaining symbols (if enabled)
4. Apply Decision Matrix voting system (if enabled)
5. Display final filtered results

**Features:**
- Early filtering reduces computational load
- Range Oscillator acts as early filter to remove false positives
- Fallback mechanism when no symbols pass Range Oscillator filter
- Optional SPC and Decision Matrix for additional filtering
- Sequential approach is easier to debug and monitor

**Options:**
- `--enable-spc`: Enable SPC signal calculation
- `--use-decision-matrix`: Enable Decision Matrix voting
- `--timeframe TIMEFRAME`: Data timeframe (1h, 4h, 1d, etc.)
- `--no-menu`: Skip interactive prompts

### 4. ATC + Range Oscillator + SPC (Voting Analyzer)

Pure voting system without sequential filtering:

```bash
python main_voting.py
```

**Workflow:**
1. ATC auto scan to find initial LONG/SHORT signals
2. Calculate signals from all indicators in parallel for all symbols
3. Apply voting system with weighted impact and cumulative vote
4. Filter symbols based on voting results
5. Display final results with voting metadata

**Features:**
- Parallel signal calculation for all indicators
- All symbols evaluated with full indicator information
- Voting system considers all indicators simultaneously
- More flexible, not dependent on filtering order
- Higher resource usage but potentially better accuracy

**Options:**
- `--enable-spc`: Enable SPC signal calculation
- `--timeframe TIMEFRAME`: Data timeframe
- `--no-menu`: Skip interactive prompts

**See Also:**
- `core/README.md`: Detailed workflow comparison between Hybrid and Voting analyzers

### 5. Simplified Percentile Clustering (SPC)

Standalone SPC analysis:

```bash
python main/main_simplified_percentile_clustering.py
```

**Features:**
- Cluster-based market regime detection
- Multiple strategies: Cluster Transition, Regime Following, Mean Reversion
- Configurable clustering parameters

### 6. XGBoost Prediction

Predict next price movement using XGBoost classifier:

```bash
python main/main_xgboost.py
```

**Options:**
- `--symbol SYMBOL`: Trading pair (e.g., BTC/USDT)
- `--timeframe TIMEFRAME`: Timeframe (1h, 4h, 1d)
- `--limit LIMIT`: Number of candles to fetch
- `--no-prompt`: Skip interactive prompts

**Example:**
```bash
python main/main_xgboost.py --symbol BTC/USDT --timeframe 1h --limit 500
```

### 7. Deep Learning (TFT)

Train Temporal Fusion Transformer model for price prediction:

```bash
python main/main_deeplearning_prediction.py
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
python main/main_deeplearning_prediction.py --symbol BTC/USDT --timeframe 1h --epochs 10 --gpu
```

### 8. Pairs Trading

Identify pairs trading opportunities:

```bash
python main/main_pairs_trading.py
```

**Options:**
- `--pairs-count N`: Number of pairs to analyze
- `--sort-by SCORE`: Sort by `opportunity_score` or `quantitative_score`
- `--require-cointegration`: Only show cointegrated pairs
- `--max-half-life N`: Maximum half-life threshold
- `--min-quantitative-score N`: Minimum quantitative score (0-100)
- `--max-pairs N`: Maximum pairs to display
- `--strategy {reversion,momentum}`: Switch between mean-reversion (default) and momentum mode. Without `--no-menu`, the interactive prompt also lets you choose and preview each strategy.

**Example:**
```bash
python main/main_pairs_trading.py --sort-by quantitative_score --require-cointegration --min-quantitative-score 70
```

### 9. Portfolio Manager

Manage portfolio risk and find hedges:

```bash
python main/main_portfolio_manager.py
```

**Features:**
- Portfolio risk calculation (VaR, Beta)
- Correlation analysis
- Automatic hedge finding
- Real-time position tracking

### 10. HMM Signal Combiner

State-based analysis using multiple Hidden Markov Model strategies:

```bash
python main/main_hmm.py
```

**HMM Strategies:**
1. **HMM-Swings**: Uses swing detection (high/low points) to identify market states
2. **HMM-KAMA**: Uses KAMA (Kaufman Adaptive Moving Average) as features with ARM and K-Means
3. **True High-Order HMM**: High-order HMM with state space expansion and automatic order optimization

**Features:**
- Combines signals from all 3 HMM strategies
- Multiple voting mechanisms (simple majority, weighted, confidence-weighted, threshold-based)
- Strategy registry for dynamic strategy management
- Conflict resolution and confidence calculation
- Automatic order selection for High-Order HMM using BIC

**Options:**
- `--symbol SYMBOL`: Trading pair to analyze (e.g., BTC/USDT)
- `--timeframe TIMEFRAME`: Data timeframe (1h, 4h, 1d, etc.)
- `--limit N`: Number of candles to fetch
- `--window-size N`: HMM window size
- `--window-kama N`: KAMA window size
- `--fast-kama N`: Fast KAMA parameter
- `--slow-kama N`: Slow KAMA parameter
- `--orders-argrelextrema N`: Order for swing detection
- `--strict-mode`: Use strict mode for swing-to-state conversion

**Example:**
```bash
python main/main_hmm.py --symbol BTC/USDT --timeframe 1h --limit 500
```

**See Also:**
- `modules/hmm/README.md`: Detailed HMM module documentation

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run specific test module
pytest tests/adaptive_trend/
pytest tests/range_oscillator/
pytest tests/simplified_percentile_clustering/
pytest tests/hmm/
pytest tests/xgboost/
pytest tests/deeplearning/
pytest tests/pairs_trading/
pytest tests/portfolio/
pytest tests/common/
pytest tests/decision_matrix/
pytest tests/test_main_hybrid.py
pytest tests/test_main_voting.py

# Run with coverage
pytest --cov=modules --cov-report=html
```

## 📚 Documentation

Detailed documentation is available in various markdown files:

- **README.md**: Main project documentation (this file)
- **README_DECISION_MATRIX.md**: Decision Matrix voting system documentation
- **OSCILLATOR_SIGNAL_EXPLANATION.md**: Range Oscillator signal explanation
- **COMBINED_STRATEGY_IMPROVEMENTS.md**: Range Oscillator combined strategy improvements
- **CODE_REVIEW.md**: Code review and best practices
- **TFT_IMPLEMENTATION_ROADMAP.md**: Deep learning TFT implementation roadmap
- **ENHANCE_FUTURES.md**: Futures trading enhancements

Module-specific documentation:
- **modules/adaptive_trend/README.md**: Adaptive Trend Classification
- **modules/range_oscillator/**: Range Oscillator strategies
- **modules/simplified_percentile_clustering/README.md**: SPC clustering
- **modules/xgboost/README.md**: XGBoost prediction
- **modules/deeplearning/README.md**: Deep learning TFT
- **modules/hmm/README.md**: HMM module (3 strategies, signal combination)
- **modules/pairs_trading/README.md**: Pairs trading strategies
- **modules/portfolio/README.md**: Portfolio management
- **core/README.md**: Hybrid vs Voting analyzer workflow comparison

## 🏗️ Architecture

### Module Organization

- **`modules/common/`**: Shared utilities used across all modules
  - `DataFetcher`: Multi-exchange data fetching with fallback
  - `ExchangeManager`: Exchange connection and API management
  - `IndicatorEngine`: Technical indicator computation
  - `Position`: Position data structure

- **`modules/adaptive_trend/`**: Adaptive Trend Classification (ATC)
  - Multi-layer trend analysis
  - Robustness filtering
  - Signal detection and scanning
  - Interactive CLI

- **`modules/range_oscillator/`**: Range Oscillator strategies
  - Multiple signal generation strategies (2-9)
  - Combined strategy with weighted voting
  - Dynamic strategy selection
  - Adaptive weights based on performance

- **`modules/simplified_percentile_clustering/`**: SPC clustering
  - Percentile-based clustering
  - Multiple strategies: Cluster Transition, Regime Following, Mean Reversion
  - Vote aggregation with weighted voting
  - Configurable clustering parameters

- **`modules/decision_matrix/`**: Decision Matrix voting system
  - Pseudo Random Forest-like classifier
  - Weighted voting based on accuracy
  - Feature importance calculation
  - Cumulative vote with threshold

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
  - **HMM-Swings**: Swing detection-based state classification
  - **HMM-KAMA**: KAMA-based HMM with ARM and K-Means clustering
  - **True High-Order HMM**: State space expansion with automatic order optimization
  - **Signal Combiner**: Registry-based strategy combination with multiple voting mechanisms
  - **Strategy Interface**: Standardized interface for all HMM strategies
  - Conflict resolution and confidence calculation

## 🔍 Key Features

### Quantitative Metrics

The pairs trading module includes comprehensive quantitative metrics:

- **Cointegration Tests**: ADF and Johansen tests
- **Mean Reversion**: Half-life, Hurst exponent
- **Risk Metrics**: Sharpe ratio, max drawdown, Calmar ratio
- **Statistical Tests**: Z-score analysis, correlation metrics
- **Classification Metrics**: F1-score, precision, recall

See `QUANT_METRICS_USAGE_REPORT.md` for detailed usage.

### Smart Data Fetching

- Automatic exchange fallback when data is stale
- Multi-exchange support for redundancy
- Caching to reduce API calls
- Freshness checking

### Advanced Indicators

- **Trend**: SMA, EMA, MACD, ATC (Adaptive Trend Classification)
- **Momentum**: RSI, Stochastic RSI
- **Volatility**: ATR, Bollinger Bands, Range Oscillator
- **Volume**: OBV, Volume indicators
- **Candlestick Patterns**: Doji, Engulfing, Three White Soldiers, etc.
- **Custom**: KAMA (Kaufman Adaptive Moving Average), SPC (Simplified Percentile Clustering)

### Signal Combination Strategies

- **Hybrid Analyzer**: Sequential filtering (ATC → Range Oscillator → SPC) with optional Decision Matrix
  - Early filtering reduces computational load
  - Range Oscillator acts as early filter
  - Fallback mechanism when no symbols pass filter
- **Voting Analyzer**: All indicators calculate signals in parallel, then vote through Decision Matrix
  - All symbols evaluated with full indicator information
  - More flexible, not dependent on filtering order
- **SPC Aggregation**: Three SPC strategies (Cluster Transition, Regime Following, Mean Reversion) aggregated into single vote
- **HMM Signal Combiner**: Three HMM strategies (Swings, KAMA, High-Order) combined with multiple voting mechanisms
- **Decision Matrix**: Weighted voting system based on historical accuracy and signal strength

## ⚙️ Configuration

Configuration is managed in `config/` directory. Key configuration files:

- **`config/common.py`**: Common settings (exchanges, timeframes, defaults)
- **`config/hmm.py`**: HMM configuration (strategies, voting mechanisms, parameters)
- **`config/decision_matrix.py`**: Decision Matrix voting configuration
- **`config/range_oscillator.py`**: Range Oscillator strategy configuration
- **`config/spc.py`**: SPC clustering configuration
- **`config/pairs_trading.py`**: Pairs trading parameters
- **`config/portfolio.py`**: Portfolio management settings
- **`config/xgboost.py`**: XGBoost model configuration
- **`config/deep_learning.py`**: Deep learning TFT configuration

⚠️ **NOT RECOMMENDED / DEPRECATED:** For API keys, you can create `config/config_api.py` (not tracked in git), but **environment variables are the recommended approach**. See [setup/SECURITY.md](./setup/SECURITY.md) for details:
```python
BINANCE_API_KEY = "your_key"
BINANCE_API_SECRET = "your_secret"
# ... other exchange keys
```

## 🛠️ Development

### Code Structure

- Modular design with clear separation of concerns
- Comprehensive test coverage
- Type hints where applicable
- Documentation strings

### Running Tests

```bash
# All tests
pytest

# Specific module
pytest tests/xgboost/ -v

# With coverage
pytest --cov=modules --cov-report=term-missing
```

### Code Quality

- Linting: `pylint` or `flake8`
- Formatting: `black` (recommended)
- Type checking: `mypy` (optional)

## 📄 License

See `LICENSE` file for details.

## ⚠️ Disclaimer

**Not Financial Advice**: This tool is for educational and research purposes only. Trading cryptocurrency involves high risk and can result in significant financial losses. Always do your own research and never invest more than you can afford to lose.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## 📞 Support

For issues, questions, or contributions, please open an issue on the repository.

---

**Last Updated**: 2025
**Version**: 3.0

---

## 📖 Quick Start Guide

### For New Users

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Try Hybrid Analyzer (recommended for beginners):**
   ```bash
   python main_hybrid.py --timeframe 1h --enable-spc
   ```

3. **Try individual modules:**
   ```bash
   # HMM Signal Combiner
   python main/main_hmm.py --symbol BTC/USDT --timeframe 1h
   
   # XGBoost Prediction
   python main/main_xgboost.py --symbol BTC/USDT --timeframe 1h
   
   # Pairs Trading
   python main/main_pairs_trading.py
   ```

### For Advanced Users

- **Customize configuration**: Edit files in `config/` directory
- **Add new strategies**: Follow module structure in `modules/`
- **Extend analyzers**: Modify `core/hybrid_analyzer.py` or `core/voting_analyzer.py`
- **Add new indicators**: Extend `modules/common/IndicatorEngine.py`

### Workflow Selection Guide

**Use Hybrid Analyzer (`main_hybrid.py`) when:**
- You want to reduce computational load with early filtering
- Range Oscillator is important for your strategy
- You need a sequential, easy-to-debug workflow
- You want fallback mechanisms

**Use Voting Analyzer (`main_voting.py`) when:**
- You want to consider all indicators simultaneously
- You have sufficient computational resources
- You prefer a pure voting approach
- You want maximum flexibility

See `core/README.md` for detailed comparison.

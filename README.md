# Crypto Probability

A comprehensive cryptocurrency trading analysis system using Machine Learning, Deep Learning, and Quantitative Strategies.

## 🚀 Features

### Core Capabilities

- **Multi-Exchange Support**: Automatically fetches data from Binance, Kraken, KuCoin, Gate.io, OKX, Bybit, MEXC, Huobi with smart fallback.
- **Advanced Indicators**: SMA, RSI, ATR, MACD, Bollinger Bands, Stochastic RSI, OBV, Candlestick Patterns, KAMA, plus custom indicators.
- **Multiple ML Models**:
    - XGBoost for directional prediction.
    - Temporal Fusion Transformer (TFT) for deep learning forecasts.
    - HMM-KAMA for state-based signal analysis.
- **Adaptive Trend Classification (ATC)**: Multi-layer trend analysis with robustness filtering.
- **Range Oscillator**: Advanced oscillator-based signal generation with multiple strategies.
- **Simplified Percentile Clustering (SPC)**: Cluster-based market regime detection with multiple strategies.
- **Decision Matrix**: Pseudo Random Forest-like voting system for combining multiple indicators.
- **Pairs Trading**: Identify and analyze mean-reversion or momentum pairs with extensive quantitative metrics.
- **Portfolio Management**: Risk calculation, correlation analysis, hedge finding.

## 📁 Project Structure

```
crypto-probability/
├── main_atc.py                              # Adaptive Trend Classification (ATC)
├── main_atc_oscillator.py                   # ATC + Range Oscillator combined
├── main_hybrid.py                           # ATC + Range Oscillator + SPC (Hybrid)
├── main_voting.py                            # ATC + Range Oscillator + SPC (Pure Voting)
├── main/
│   ├── main_atc.py                          # Adaptive Trend Classification
│   ├── main_atc_oscillator.py               # ATC + Range Oscillator
│   ├── main_xgboost.py                      # XGBoost Prediction
│   ├── main_simplified_percentile_clustering.py  # SPC Analysis
│   ├── main_deeplearning_prediction.py      # Deep Learning TFT
│   ├── main_hmm.py                          # HMM Signal Combiner
│   ├── main_pairs_trading.py                # Pairs Trading
│   └── main_portfolio_manager.py            # Portfolio Manager
├── main_hmm.py                              # HMM Signal Combiner (High-Order HMM + HMM-KAMA)
├── modules/                                 # Core modules
│   ├── common/                              # Shared utilities
│   │   ├── DataFetcher.py                  # Multi-exchange data fetching
│   │   ├── ExchangeManager.py              # Exchange connection management
│   │   ├── IndicatorEngine.py              # Technical indicators
│   │   └── indicators/                     # Indicator implementations
│   ├── adaptive_trend/                      # Adaptive Trend Classification
│   ├── range_oscillator/                    # Range Oscillator strategies
│   ├── simplified_percentile_clustering/   # SPC clustering and strategies
│   ├── decision_matrix/                     # Decision Matrix voting system
│   ├── xgboost/                             # XGBoost prediction module
│   ├── deeplearning/                        # Deep learning module (TFT)
│   ├── pairs_trading/                       # Pairs trading strategies
│   ├── portfolio/                          # Portfolio management
│   └── hmm/                                 # HMM-KAMA analysis
├── tests/                                   # Comprehensive test suite
├── artifacts/                               # Model checkpoints and outputs
└── README*.md                               # Documentation files
```

## 🔧 Installation

### Prerequisites

- Python 3.8+ (Python 3.10+ recommended)
- pip

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd crypto-probability
   ```

2. **Install dependencies:**

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
   
   Create `modules/config_api.py` with your exchange API keys:
   ```python
   BINANCE_API_KEY = "your_key"
   BINANCE_API_SECRET = "your_secret"
   ```

## 📖 Usage

### 1. Adaptive Trend Classification (ATC)

Analyze market trends using multi-layer adaptive classification:

```bash
python main_atc.py
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

### 3. ATC + Range Oscillator + SPC (Hybrid)

Hybrid approach combining sequential filtering and voting:

```bash
python main_hybrid.py
```

**Workflow:**
1. ATC auto scan
2. Range Oscillator filtering
3. SPC signal calculation (all 3 strategies)
4. Decision Matrix voting system

**Features:**
- All 3 SPC strategies (Cluster Transition, Regime Following, Mean Reversion)
- SPC votes aggregated into single vote
- Decision Matrix for final confirmation

### 4. ATC + Range Oscillator + SPC (Pure Voting)

Pure voting system without sequential filtering:

```bash
python main_voting.py
```

**Workflow:**
1. ATC auto scan
2. Calculate all indicator signals in parallel
3. Decision Matrix voting system

**Features:**
- Parallel signal calculation
- All indicators vote simultaneously
- Weighted voting based on accuracy

### 5. Simplified Percentile Clustering (SPC)

Standalone SPC analysis:

```bash
python main_simplified_percentile_clustering.py
```

**Features:**
- Cluster-based market regime detection
- Multiple strategies: Cluster Transition, Regime Following, Mean Reversion
- Configurable clustering parameters

### 6. XGBoost Prediction

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

### 7. Deep Learning (TFT)

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

### 8. Pairs Trading

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
- `--strategy {reversion,momentum}`: Switch between mean-reversion (default) and momentum mode. Without `--no-menu`, the interactive prompt also lets you choose and preview each strategy.

**Example:**
```bash
python main_pairs_trading.py --sort-by quantitative_score --require-cointegration --min-quantitative-score 70
```

### 9. Portfolio Manager

Manage portfolio risk and find hedges:

```bash
python main_portfolio_manager.py
```

**Features:**
- Portfolio risk calculation (VaR, Beta)
- Correlation analysis
- Automatic hedge finding
- Real-time position tracking

### 10. HMM-KAMA Analysis

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

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run specific test module
pytest tests/adaptive_trend/
pytest tests/range_oscillator/
pytest tests/simplified_percentile_clustering/
pytest tests/xgboost/
pytest tests/deeplearning/
pytest tests/pairs_trading/
pytest tests/portfolio/
pytest tests/common/

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
- **modules/pairs_trading/README.md**: Pairs trading strategies
- **modules/portfolio/README.md**: Portfolio management

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
  - HMM-KAMA state detection
  - High-Order HMM signal generation
  - Signal combiner with conflict resolution
  - Mean reversion analysis

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

- **Hybrid Approach**: Sequential filtering (ATC → Range Oscillator → SPC) with optional Decision Matrix
- **Pure Voting**: All indicators calculate signals in parallel, then vote through Decision Matrix
- **SPC Aggregation**: Three SPC strategies (Cluster Transition, Regime Following, Mean Reversion) aggregated into single vote
- **Decision Matrix**: Weighted voting system based on historical accuracy and signal strength

## ⚙️ Configuration

Configuration is managed in `modules/config.py`. Key settings:

- Exchange selection and priorities
- Timeframe defaults
- Model hyperparameters
- Risk thresholds
- Trading parameters

For API keys, create `modules/config_api.py` (not tracked).

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

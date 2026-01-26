# Sovereign-IQ

A comprehensive cryptocurrency trading analysis system combining Machine Learning, Deep Learning, and Quantitative Strategies for professional trading analysis and signal generation.

## 🚀 Features

### Core Capabilities

- **Multi-Exchange Support**:
  - *Crypto:* Automated data fetching from 8+ exchanges (Binance, Kraken, KuCoin, Gate.io, OKX, Bybit, MEXC, Huobi) with intelligent fallback mechanisms.

  - *Forex*: Fetches and analyzes forex market data (EUR/USD, GBP/USD, USD/JPY, etc.) alongside crypto, with seamless integration into all analyzers and strategies. Forex data is fetched via dedicated modules and can be used for cross-market analysis, hedging, and portfolio diversification.

- **Advanced Technical Indicators**: SMA, EMA, RSI, ATR, MACD, Bollinger Bands, Stochastic RSI, OBV, Candlestick Patterns, KAMA, and custom indicators.
- **Machine Learning Models**:
  - **XGBoost**: Multi-class classification with hyperparameter optimization and volatility-based labeling
  - **LSTM**: CNN-LSTM-Attention architecture for temporal sequence modeling
  - **Temporal Fusion Transformer (TFT)**: Advanced deep learning for time series forecasting
  - **Random Forest**: Ensemble learning with feature importance analysis
  - **Hidden Markov Model (HMM)**: Three distinct strategies:
    - HMM-Swings: Swing detection-based state classification
    - HMM-KAMA: KAMA-based HMM with ARM and K-Means clustering
    - True High-Order HMM: State space expansion with automatic order optimization using BIC
 - **Trading Strategies**:
   - **Adaptive Trend Classification (ATC)**: Multi-layer trend analysis with 6 MA types and equity-based weighting
   - **Adaptive Trend LTS (ATC-LTS)**: High-performance ATC with Rust/CUDA extensions and Dask out-of-core processing
   - **Range Oscillator**: Advanced oscillator-based signals with 8 strategies (mean reversion, momentum, divergence, etc.)
   - **Simplified Percentile Clustering (SPC)**: Cluster-based market regime detection with 3 strategies
   - **Decision Matrix**: Weighted voting system combining multiple indicators with accuracy-based weights
   - **Pairs Trading**: Mean-reversion and momentum strategies with comprehensive quantitative metrics
   - **Portfolio Management**: Risk calculation (VaR, Beta), correlation analysis, and hedge finding
 - **AI-Powered Analysis**: Google Gemini integration for intelligent chart interpretation and batch market scanning
 - **Web Interface**: Modern Vue.js + FastAPI applications for real-time visualization and analysis
 - **Dask Integration**: Out-of-core processing for unlimited datasets with Rust+Dask hybrid execution (80-90% memory reduction)

## 📁 Project Structure

```
crypto-probability/
├── main_complex_hybrid.py                   # Hybrid Analyzer (Sequential Filtering)
├── main_complex_voting.py                   # Voting Analyzer (Pure Voting)
├── main_complex_atc_oscillator.py           # ATC + Range Oscillator
├── main_lstm.py                             # LSTM Model Training/Prediction
├── main_position_sizing.py                  # Position Sizing Calculator
├── main_gemini_chart_analyzer.py            # AI-Powered Chart Analysis
├── main_gemini_chart_batch_scanner.py       # Batch Market Scanning
├── main_gemini_chart_web_server.py          # Web Server for Chart Analysis
├── main_cal_position_totals.py              # Position Totals Calculator
│
├── core/                                     # Core Analyzers
│   ├── hybrid_analyzer.py                   # Sequential filtering + voting
│   ├── voting_analyzer.py                   # Pure voting system
│   ├── signal_calculators.py                # Signal calculation helpers
│   └── README.md                            # Analyzer workflow comparison
│
├── modules/                                  # Core Modules
│   ├── common/                              # Shared Utilities
│   │   ├── core/                           # Business components
│   │   │   ├── data_fetcher.py            # Multi-exchange data fetching
│   │   │   ├── exchange_manager.py        # Exchange connection management
│   │   │   ├── indicator_engine.py        # Technical indicator orchestration
│   │   │   └── data_fetcher_forex.py      # Forex data fetching
│   │   ├── models/                         # Data models (Position)
│   │   ├── ui/                             # UI utilities (progress bars, logging)
│   │   ├── utils/                          # Domain utilities
│   │   ├── indicators/                     # Technical indicator implementations
│   │   └── quantitative_metrics/           # Quantitative analysis
│   │       ├── statistical_tests/         # ADF, Johansen tests
│   │       ├── mean_reversion/            # Half-life, Hurst, Z-score
│   │       ├── hedge_ratios/              # OLS, Kalman filter
│   │       ├── risk/                      # Sharpe, drawdown, Calmar
│   │       └── classification/            # Prediction metrics
│   │
 │   ├── adaptive_trend/                      # Adaptive Trend Classification (ATC)
 │   │   ├── analyzer.py                     # Main ATC analyzer
 │   │   ├── scanner.py                      # Multi-symbol scanner
 │   │   ├── compute_atc_signals.py         # Signal calculation
 │   │   ├── compute_equity.py              # Equity curve calculation
 │   │   └── signal_detection.py            # Signal detection logic
 │   │
 │   ├── adaptive_trend_LTS/                 # Adaptive Trend LTS (ATC-LTS)
 │   │   ├── core/                          # Core components
 │   │   │   ├── compute_atc_signals/       # Signal computation with Rust/CUDA
 │   │   │   ├── scanner/                   # Dask-enabled scanner
 │   │   │   ├── backtesting/               # Dask backtesting
 │   │   │   ├── gpu_backend/              # CUDA kernels
 │   │   │   └── utils/                    # Utilities & config
 │   │   ├── rust_extensions/               # Rust extensions (PyO3)
 │   │   ├── docs/                          # Documentation
 │   │   │   └── dask_usage_guide.md        # Dask integration guide
 │   │   └── benchmarks/                   # Performance benchmarks
 │   │
│   ├── range_oscillator/                    # Range Oscillator Strategies
│   │   ├── oscillator_analyzer.py         # Main oscillator analyzer
│   │   └── strategies/                    # 8 strategy implementations
│   │
│   ├── simplified_percentile_clustering/   # SPC Clustering
│   │   ├── clustering.py                  # Clustering logic
│   │   ├── features.py                    # Feature engineering
│   │   ├── centers.py                     # Cluster center calculation
│   │   ├── strategies/                    # 3 strategy implementations
│   │   └── aggregation.py                 # Vote aggregation
│   │
│   ├── decision_matrix/                     # Decision Matrix Voting
│   │   ├── analyzer.py                    # Voting analyzer
│   │   └── voting.py                      # Voting mechanisms
│   │
│   ├── xgboost/                             # XGBoost ML Module
│   │   ├── model.py                       # Model training/prediction
│   │   ├── labeling.py                    # Dynamic labeling system
│   │   ├── optimization.py                # Hyperparameter optimization
│   │   ├── cli/                           # CLI interface
│   │   └── utils.py                       # Utilities
│   │
│   ├── lstm/                                # LSTM Deep Learning
│   │   ├── unified_trainer.py             # Training pipeline
│   │   ├── preprocessing.py               # Data preprocessing
│   │   ├── model_factory.py               # Model construction
│   │   ├── attention.py                   # Attention mechanisms
│   │   └── cnn_lstm_attention.py          # CNN-LSTM-Attention model
│   │
│   ├── deeplearning/                        # Temporal Fusion Transformer (TFT)
│   │   ├── model.py                       # TFT implementation
│   │   ├── data_pipeline.py               # Data preprocessing
│   │   ├── dataset.py                     # PyTorch datasets
│   │   ├── training.py                    # Training loop
│   │   └── feature_selection/             # Feature selection methods
│   │
│   ├── random_forest/                       # Random Forest Module
│   │   └── model.py                       # RF classifier
│   │
│   ├── hmm/                                 # Hidden Markov Model
│   │   ├── core/                          # 3 HMM implementations
│   │   │   ├── swings.py                 # Swing-based HMM
│   │   │   ├── kama.py                   # KAMA-based HMM
│   │   │   └── high_order.py             # High-Order HMM
│   │   ├── signals/                       # Signal processing
│   │   │   ├── strategy.py               # Strategy interface
│   │   │   ├── registry.py               # Strategy registry
│   │   │   ├── combiner.py               # Signal combiner
│   │   │   └── voting.py                 # Voting mechanisms
│   │   └── cli/                           # CLI interface
│   │
│   ├── pairs_trading/                       # Pairs Trading Strategies
│   │   ├── core/                          # Core analyzers
│   │   ├── analysis/                      # Performance analysis
│   │   ├── utils/                         # Selection utilities
│   │   └── cli/                           # CLI interface
│   │
│   ├── portfolio/                           # Portfolio Management
│   │   ├── risk_calculator.py             # Risk metrics
│   │   ├── correlation_analyzer.py        # Correlation analysis
│   │   └── hedge_finder.py                # Hedge identification
│   │
│   ├── position_sizing/                     # Position Sizing
│   │   ├── kelly_calculator.py            # Kelly Criterion
│   │   └── hybrid_signal_calculator.py    # Signal combination
│   │
│   ├── gemini_chart_analyzer/               # AI Chart Analysis
│   │   ├── core/                          # Analysis engine
│   │   ├── scanners/                      # Market scanners
│   │   └── cli/                           # CLI interface
│   │
│   └── backtester/                          # Backtesting Framework
│
├── web/                                      # Web Applications
│   ├── shared/                             # Shared utilities
│   ├── apps/
│   │   ├── gemini_analyzer/               # Port 8001 (Gemini Chart Analysis)
│   │   │   ├── backend/                  # FastAPI backend
│   │   │   └── frontend/                 # Vue.js frontend
│   │   └── atc_visualizer/                # Port 8002 (ATC Visualization)
│   │       ├── backend/                  # FastAPI backend
│   │       └── frontend/                 # Vue.js frontend
│   ├── scripts/                            # Management scripts
│   └── docs/                               # Web documentation
│
├── config/                                   # Configuration Files
│   ├── common.py                           # Common settings
│   ├── config_api.py                       # API keys (supports env vars)
│   ├── decision_matrix.py                  # Decision Matrix config
│   ├── range_oscillator.py                 # Oscillator config
│   ├── spc.py                              # SPC config
│   ├── pairs_trading.py                    # Pairs trading config
│   ├── portfolio.py                        # Portfolio config
│   ├── position_sizing.py                  # Position sizing config
│   ├── xgboost.py                          # XGBoost config
│   ├── lstm.py                             # LSTM config
│   ├── deep_learning.py                    # TFT config
│   ├── hmm.py                              # HMM config
│   ├── random_forest.py                    # Random Forest config
│   ├── gemini_chart_analyzer.py            # Gemini AI config
│   ├── forex_pairs.py                      # Forex configurations
│   └── model_features.py                   # Feature definitions
│
├── cli/                                      # CLI Utilities
│   ├── argument_parser.py                  # Argument parsing
│   └── display.py                          # Display utilities
│
├── tests/                                    # Comprehensive Test Suite
│   ├── adaptive_trend/                     # ATC tests
│   ├── xgboost/                            # XGBoost tests
│   ├── deeplearning/                       # TFT tests
│   ├── lstm/                               # LSTM tests
│   ├── pairs_trading/                      # Pairs trading tests
│   ├── portfolio/                          # Portfolio tests
│   ├── simplified_percentile_clustering/  # SPC tests
│   ├── hmm/                                # HMM tests
│   ├── position_sizing/                    # Position sizing tests
│   ├── random_forest/                      # Random Forest tests
│   ├── web/                                # Web API tests
│   ├── e2e/                                # End-to-end tests
│   └── performance/                        # Performance tests
│
├── artifacts/                                # Model Checkpoints & Outputs
│   ├── models/                             # Trained models
│   └── deep/                               # Deep learning artifacts
│
├── setup/                                    # Setup Scripts & Documentation
│   ├── QUICK_START_API_KEYS.md            # API key setup guide
│   ├── SECURITY.md                         # Security best practices
│   ├── setup_api_keys.ps1                 # Windows PowerShell setup
│   ├── setup_api_keys.bat                 # Windows Command Prompt setup
│   └── setup_api_keys.sh                  # Linux/Mac setup
│
├── scripts/                                  # Utility Scripts
│   └── export_openapi.py                   # OpenAPI schema export
│
├── docs/                                     # Documentation
│   ├── api_documentation.md                # API documentation
│   ├── openapi.yaml                        # OpenAPI specification
│   └── setup_tradingview_credentials.md    # TradingView setup
│
└── README*.md                               # Project Documentation
```

## 🔧 Installation

### Prerequisites

- Python 3.9+ (Python 3.10+ recommended)
- pip (package installer)
- Git

### Setup

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd crypto-probability
   ```

2. **Install dependencies:**

   **Basic requirements (required):**

   ```bash
   pip install -r requirements.txt
   ```

   **Machine Learning dependencies (optional, for XGBoost, LSTM, TFT):**

   ```bash
   pip install -r requirements-ml.txt
   ```

   **Development dependencies (optional, for testing and development):**

   ```bash
   pip install -r requirements-dev.txt
   ```

3. **Configure API keys:**

   Use environment variables for secure API key management. Run the appropriate setup script for your platform:

   ```bash
   # Windows (PowerShell)
   .\setup\setup_api_keys.ps1
   
   # Windows (Command Prompt)
   setup\setup_api_keys.bat
   
   # Linux/Mac
   chmod +x setup/setup_api_keys.sh
   ./setup/setup_api_keys.sh
   ```

   The setup scripts are idempotent and safe to run multiple times. See [setup/QUICK_START_API_KEYS.md](./setup/QUICK_START_API_KEYS.md) for detailed instructions and [setup/SECURITY.md](./setup/SECURITY.md) for security best practices.

## 📖 Usage

### 1. Hybrid Analyzer (Sequential Filtering)

The Hybrid Analyzer uses sequential filtering to efficiently identify trading opportunities:

```bash
python main_complex_hybrid.py --timeframe 1h --enable-spc --use-decision-matrix
```

**Workflow:**

1. ATC auto scan to find initial LONG/SHORT signals
2. Filter by Range Oscillator confirmation (sequential filtering)
3. Calculate SPC signals for remaining symbols (if enabled)
4. Apply Decision Matrix voting system (if enabled)
5. Display final filtered results

**Features:**

- Early filtering reduces computational load by eliminating symbols progressively
- Range Oscillator acts as confirmation filter to remove false positives
- Fallback mechanism when no symbols pass filters
- Sequential approach is easier to debug and monitor
- Resource-efficient for large symbol pools

**Key Options:**

- `--timeframe TIMEFRAME`: Data timeframe (1h, 4h, 1d, etc.)
- `--enable-spc`: Enable SPC signal calculation
- `--use-decision-matrix`: Enable Decision Matrix voting
- `--enable-xgboost`: Enable XGBoost predictions
- `--enable-hmm`: Enable HMM signal combination
- `--enable-random-forest`: Enable Random Forest classifier
- `--no-menu`: Skip interactive prompts

**When to Use:**

- Limited computational resources
- Need faster results with early filtering
- Range Oscillator is important for your strategy
- Prefer sequential, easy-to-debug workflow

### 2. Voting Analyzer (Pure Voting System)

The Voting Analyzer evaluates all indicators simultaneously using a weighted voting system:

```bash
python main_complex_voting.py --timeframe 1h --enable-spc
```

**Workflow:**

1. ATC auto scan to find initial LONG/SHORT signals
2. Calculate signals from all indicators in parallel for all symbols
3. Apply voting system with weighted impact and cumulative vote
4. Filter symbols based on voting results
5. Display final results with voting metadata

**Features:**

- Parallel signal calculation for all enabled indicators
- All symbols evaluated with complete indicator information
- Voting system considers all indicators simultaneously
- More flexible, not dependent on filtering order
- Potentially better accuracy with higher resource usage

**Key Options:**

- `--timeframe TIMEFRAME`: Data timeframe
- `--enable-spc`: Enable SPC signal calculation
- `--enable-xgboost`: Enable XGBoost predictions
- `--enable-hmm`: Enable HMM signal combination
- `--enable-random-forest`: Enable Random Forest classifier
- `--no-menu`: Skip interactive prompts

**When to Use:**

- Sufficient computational resources available
- Want to consider all indicators simultaneously
- Prefer pure voting approach without sequential filtering
- Need maximum flexibility in signal combination

**See Also:**

- `core/README.md`: Detailed workflow comparison

### 3. XGBoost Prediction

Multi-class price movement prediction using XGBoost with hyperparameter optimization:

```bash
python modules/xgboost/cli/main.py --symbol BTC/USDT --timeframe 1h --limit 500
```

**Features:**

- Multi-class classification (UP, NEUTRAL, DOWN)
- Dynamic volatility-based labeling with triple-barrier method
- Hyperparameter optimization using Optuna
- Time-series cross-validation
- Feature importance analysis

**Key Options:**

- `--symbol SYMBOL`: Trading pair (e.g., BTC/USDT)
- `--timeframe TIMEFRAME`: Timeframe (1h, 4h, 1d)
- `--limit LIMIT`: Number of candles to fetch
- `--no-prompt`: Skip interactive prompts

### 4. LSTM Deep Learning

CNN-LSTM-Attention model for temporal sequence prediction:

```bash
python main_lstm.py --symbol BTC/USDT --timeframe 1h --epochs 50
```

**Features:**

- CNN-LSTM-Attention architecture
- Multi-head attention mechanisms
- Positional encoding for temporal awareness
- Focal loss for imbalanced classes
- Kalman filter for noise reduction

**Key Options:**

- `--symbol SYMBOL`: Trading pair
- `--timeframe TIMEFRAME`: Data timeframe
- `--epochs EPOCHS`: Training epochs
- `--batch-size SIZE`: Batch size

### 5. Temporal Fusion Transformer (TFT)

Advanced deep learning model for time series forecasting:

```bash
python modules/deeplearning/cli/main.py --symbol BTC/USDT --timeframe 1h --epochs 10 --gpu
```

**Features:**

- Temporal Fusion Transformer architecture
- Quantile regression for uncertainty estimation
- Advanced feature selection (Mutual Information, Boruta-like, F-test)
- Fractional differentiation
- Triple barrier labeling

**Key Options:**

- `--symbol SYMBOL`: Trading pair
- `--timeframe TIMEFRAME`: Data timeframe
- `--epochs EPOCHS`: Training epochs
- `--gpu`: Enable GPU acceleration
- `--phase PHASE`: Training phase (1=regression, 2=classification)

### 6. Hidden Markov Model (HMM)

State-based market analysis combining three HMM strategies:

```bash
python modules/hmm/cli/main.py --symbol BTC/USDT --timeframe 1h --limit 500
```

**HMM Strategies:**

1. **HMM-Swings**: Swing detection-based state classification using highs/lows
2. **HMM-KAMA**: KAMA-based features with ARM and K-Means clustering
3. **True High-Order HMM**: State space expansion with automatic order optimization using BIC

**Features:**

- Combines signals from all 3 strategies
- Multiple voting mechanisms (simple majority, weighted, confidence-weighted, threshold-based)
- Strategy registry for dynamic management
- Automatic order selection for High-Order HMM
- Conflict resolution and confidence calculation

**Key Options:**

- `--symbol SYMBOL`: Trading pair
- `--timeframe TIMEFRAME`: Data timeframe
- `--limit N`: Number of candles
- `--window-size N`: HMM window size
- `--window-kama N`: KAMA window size
- `--strict-mode`: Use strict mode for swing-to-state conversion

**See Also:**

- `modules/hmm/README.md`: Detailed HMM documentation

### 7. Pairs Trading

Identify and analyze mean-reversion or momentum pairs:

```bash
python modules/pairs_trading/cli/main.py --sort-by quantitative_score --require-cointegration --min-quantitative-score 70
```

**Features:**

- Mean-reversion and momentum strategies
- Comprehensive quantitative metrics (cointegration, half-life, Hurst exponent)
- Risk metrics (Sharpe ratio, max drawdown, Calmar ratio)
- Multi-timeframe performance analysis
- Opportunity scoring system

**Key Options:**

- `--pairs-count N`: Number of pairs to analyze
- `--sort-by SCORE`: Sort by `opportunity_score` or `quantitative_score`
- `--require-cointegration`: Only show cointegrated pairs
- `--max-half-life N`: Maximum half-life threshold
- `--min-quantitative-score N`: Minimum quantitative score (0-100)
- `--strategy {reversion,momentum}`: Strategy type

### 8. Portfolio Management

Risk analysis and portfolio optimization:

```bash
python modules/portfolio/cli/main.py
```

**Features:**

- Portfolio risk calculation (VaR, Beta)
- Correlation analysis with caching
- Automatic hedge finding
- Real-time position tracking
- Multi-asset risk assessment

### 9. Gemini AI Chart Analysis

AI-powered chart interpretation using Google Gemini:

```bash
python main_gemini_chart_analyzer.py --symbol BTC/USDT --timeframes 15m,1h,4h,1d
```

**Features:**

- Single and multi-timeframe chart analysis
- Deep analysis mode (analyze each timeframe separately)
- Batch analysis mode (multi-timeframe batch charts)
- Weighted signal aggregation across timeframes
- Real-time log streaming

**Batch Market Scanning:**

```bash
python main_gemini_chart_batch_scanner.py --timeframes 1h,4h --limit 100
```

Scan entire markets in batches (100 symbols per batch) with AI-powered analysis.

### 10. Position Sizing

Kelly Criterion-based position sizing calculator:

```bash
python main_position_sizing.py --symbol BTC/USDT --timeframe 1h
```

**Features:**

- Kelly Criterion calculations
- Hybrid signal combination
- Risk management tools
- Indicator-based position sizing

### 11. Web Interface

Modern web applications for visualization and analysis:

```bash
# Start all web apps
python web/scripts/start_all.py

# Start specific app
python web/scripts/start_app.py gemini_analyzer
python web/scripts/start_app.py atc_visualizer
```

**Gemini Analyzer (Port 8001):**

- Web interface for AI chart analysis
- Single and multi-timeframe analysis
- Batch market scanning
- Real-time log streaming

**ATC Visualizer (Port 8002):**

- Real-time OHLCV chart visualization
- 6 Moving Average types overlay
- Signal markers (buy/sell arrows)
- Interactive parameter controls

## 🧪 Testing

The project includes a comprehensive test suite optimized for memory efficiency (80-90% RAM reduction). The test suite covers module-level, integration, and end-to-end tests with advanced memory management through 3 optimization phases.

### 🚀 Recommended Test Commands

```bash
# Run all tests with optimizations (recommended)
pytest

# Use helper script for best results
python run_tests.py

# Windows batch script
run_tests.bat

# Run with memory profiling (detailed)
pytest tests/ -c pytest_memory.ini

# Skip memory-intensive tests
pytest tests/ --skip-memory-intensive

# Single-threaded execution (no parallel processing)
pytest tests/ -n 0
```

### 📊 Memory Optimization Results

| Phase | Optimization | RAM Reduction | Total Reduction |
|-------|-------------|----------------|----------------|
| Phase 1 | Garbage collection + data reduction | 50-60% | 50-60% |
| Phase 2 | Session fixtures + parallel processing | 30-40% | **70-80%** |
| Phase 3 | Lazy loading + monitoring | 10-20% | **80-90%** |

### 🧪 Test Categories

```bash
# Core analyzer tests
pytest tests/test_main_hybrid.py       # Hybrid analyzer
pytest tests/test_main_voting.py       # Voting analyzer

# Module-specific tests
pytest tests/adaptive_trend/           # ATC tests
pytest tests/range_oscillator/         # Range Oscillator tests
pytest tests/simplified_percentile_clustering/  # SPC tests
pytest tests/hmm/                      # HMM tests
pytest tests/xgboost/                  # XGBoost tests
pytest tests/deeplearning/             # TFT tests
pytest tests/lstm/                     # LSTM tests
pytest tests/pairs_trading/            # Pairs trading tests
pytest tests/portfolio/                # Portfolio tests
pytest tests/position_sizing/          # Position sizing tests
pytest tests/random_forest/            # Random Forest tests
pytest tests/common/                   # Common utilities tests
pytest tests/decision_matrix/          # Decision Matrix tests
pytest tests/web/                      # Web API tests

# Special test categories
pytest tests/e2e/                      # End-to-end tests
pytest tests/performance/              # Performance tests
pytest tests/backtester/test_session_fixtures.py -v  # Session fixtures demo
```

### 📈 Coverage and Reporting

```bash
# Run with coverage report
pytest --cov=modules --cov-report=html
pytest --cov=modules --cov-report=term-missing

# Verbose output
pytest -v

# Run tests matching pattern
pytest -k "test_signal"

# Test specific functionality
pytest -k "signal and (hybrid or voting)"
```

### 🛠️ Test Infrastructure

**Configuration Files:**

- `pytest.ini`: Default configuration with parallel processing (xdist)
- `pytest_memory.ini`: Memory-optimized configuration with profiling
- `conftest.py`: Shared fixtures and test configuration
- `conftest_optimized.py`: Memory-optimized test fixtures

**Helper Scripts:**

- `run_tests.py`: Python helper script for optimized test execution
- `run_tests.bat`: Windows batch script
- `.vscode/settings.json`: VSCode integration with memory profiling

### 🔧 Troubleshooting

**VSCode Issues:**

```bash
# If VSCode doesn't use default settings
# 1. Restart VSCode
# 2. Ensure Python extension testing is enabled
# 3. Verify .vscode/settings.json syntax
```

**Memory Profiling Too Verbose:**

```bash
# Disable memory profiling
pytest tests/ --memory-profile=no
```

**Parallel Processing Issues:**

```bash
# Check xdist installation
pip install pytest-xdist

# Force single-threaded execution
pytest tests/ -n 0
```

See `tests/docs/test_memory_usage_guide.md` for detailed memory optimization documentation.

## 📚 Documentation

### Root Documentation

- **README.md**: Main project documentation (this file)
- **CODE_REVIEW.md**: Code review guidelines and best practices
- **ENHANCE_FUTURES.md**: Futures trading enhancements and roadmap
- **BATCH_SCANNER_FLOW.md**: Batch scanner workflow documentation

### Core Documentation

- **core/README.md**: Detailed comparison between Hybrid and Voting analyzers

### Module Documentation

- **modules/adaptive_trend/README.md**: Adaptive Trend Classification system
- **modules/range_oscillator/**: Range Oscillator strategies and implementations
  - `OSCILLATOR_SIGNAL_EXPLANATION.md`: Signal generation explanation
  - `COMBINED_STRATEGY_IMPROVEMENTS.md`: Combined strategy improvements
- **modules/simplified_percentile_clustering/README.md**: SPC clustering system
- **modules/decision_matrix/README_DECISION_MATRIX.md**: Decision Matrix voting system
- **modules/xgboost/README.md**: XGBoost prediction pipeline
  - `TARGET_HORIZON_EXPLANATION.md`: Target horizon explanation
- **modules/deeplearning/README.md**: Temporal Fusion Transformer (TFT)
  - `TFT_IMPLEMENTATION_ROADMAP.md`: Implementation roadmap
- **modules/lstm/README.md**: LSTM model architecture and training
- **modules/hmm/README.md**: HMM strategies and signal combination
- **modules/pairs_trading/README.md**: Pairs trading strategies
  - `QUANT_METRICS_USAGE_REPORT.md`: Quantitative metrics usage
- **modules/portfolio/README.md**: Portfolio management system
- **modules/gemini_chart_analyzer/README.md**: AI-powered chart analysis

### Setup Documentation

- **setup/README.md**: Setup instructions overview
- **setup/QUICK_START_API_KEYS.md**: Quick start guide for API keys
- **setup/SECURITY.md**: Security best practices for API keys

### Web Documentation

- **web/README.md**: Web applications overview
- **web/docs/ARCHITECTURE.md**: Web architecture documentation
- **web/docs/ADDING_NEW_APP.md**: Guide for adding new web applications
- **web/MIGRATION_SUMMARY.md**: Migration guide for web components
- **web/apps/gemini_analyzer/README.md**: Gemini Analyzer app documentation
- **web/apps/atc_visualizer/README.md**: ATC Visualizer app documentation

### API Documentation

- **docs/api_documentation.md**: REST API documentation
- **docs/openapi_extended.md**: Extended OpenAPI documentation
- **docs/openapi.yaml**: OpenAPI specification (YAML)
- **docs/openapi.json**: OpenAPI specification (JSON)
- **docs/setup_tradingview_credentials.md**: TradingView integration setup

### Test Documentation

- **tests/docs/test_memory_usage_guide.md**: Memory optimization for tests

## 🏗️ Architecture

### System Architecture

The system is built on a modular, layered architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   CLI Apps   │  │  Web Apps    │  │  API Endpoints   │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                    Core Analyzer Layer                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Hybrid     │  │   Voting     │  │  Signal Calc.    │  │
│  │   Analyzer   │  │   Analyzer   │  │  Helpers         │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                    Strategy/Module Layer                     │
│  ┌─────────┐ ┌─────────┐ ┌────────┐ ┌────────┐ ┌────────┐ │
│  │   ATC   │ │  Range  │ │  SPC   │ │  Pairs │ │  HMM   │ │
│  │         │ │  Osc.   │ │        │ │Trading │ │        │ │
│  └─────────┘ └─────────┘ └────────┘ └────────┘ └────────┘ │
│  ┌─────────┐ ┌─────────┐ ┌────────┐ ┌────────┐ ┌────────┐ │
│  │ XGBoost │ │  LSTM   │ │  TFT   │ │  R.F.  │ │Gemini  │ │
│  │         │ │         │ │        │ │        │ │  AI    │ │
│  └─────────┘ └─────────┘ └────────┘ └────────┘ └────────┘ │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                   Common Utilities Layer                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ DataFetcher  │  │  Exchange    │  │  Indicator       │  │
│  │              │  │  Manager     │  │  Engine          │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Quantitative │  │  Technical   │  │  Risk & Risk     │  │
│  │  Metrics     │  │  Indicators  │  │  Management      │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                    Data Sources Layer                        │
│  ┌─────────┐ ┌─────────┐ ┌────────┐ ┌────────┐ ┌────────┐ │
│  │ Binance │ │ Kraken  │ │KuCoin  │ │Gate.io │ │  OKX   │ │
│  └─────────┘ └─────────┘ └────────┘ └────────┘ └────────┘ │
│  ┌─────────┐ ┌─────────┐ ┌────────┐                        │
│  │  Bybit  │ │  MEXC   │ │ Huobi  │     (Smart Fallback)   │
│  └─────────┘ └─────────┘ └────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

### Module Organization

#### **Common Utilities (`modules/common/`)**

Shared infrastructure used across all modules:

- **Data Layer**:
  - `DataFetcher`: Multi-exchange data fetching with caching and smart fallback
  - `ExchangeManager`: Exchange connection management (authenticated & public)
  - `ForexDataFetcher`: Forex market data fetching

- **Indicator Layer**:
  - `IndicatorEngine`: Technical indicator orchestration (CORE, XGBOOST, DEEP_LEARNING profiles)
  - `indicators/`: 50+ technical indicator implementations
  - `quantitative_metrics/`: Statistical and quantitative analysis tools

- **Infrastructure**:
  - `models/`: Data structures (Position)
  - `ui/`: Progress bars, logging, formatting
  - `utils/`: Domain-specific utilities

#### **Trading Strategy Modules**

- **`adaptive_trend/`**: Multi-layer trend classification with 6 MA types
- **`range_oscillator/`**: 8 oscillator-based strategies with weighted voting
- **`simplified_percentile_clustering/`**: Cluster-based regime detection with 3 strategies
- **`decision_matrix/`**: Weighted voting system for signal combination
- **`pairs_trading/`**: Mean-reversion and momentum pairs with quantitative metrics
- **`portfolio/`**: Risk management, correlation analysis, hedge finding
- **`position_sizing/`**: Kelly Criterion and risk-based sizing

#### **Machine Learning Modules**

- **`xgboost/`**: Multi-class classifier with hyperparameter optimization
  - Dynamic volatility-based labeling
  - Triple-barrier method
  - Time-series cross-validation
  
- **`lstm/`**: CNN-LSTM-Attention architecture
  - Multi-head attention
  - Positional encoding
  - Focal loss for imbalanced data
  
- **`deeplearning/`**: Temporal Fusion Transformer
  - Quantile regression
  - Advanced feature selection
  - Fractional differentiation
  
- **`random_forest/`**: Ensemble learning with feature importance

- **`hmm/`**: Hidden Markov Models
  - 3 strategies (Swings, KAMA, High-Order)
  - Strategy registry and combiner
  - Multiple voting mechanisms

#### **Specialized Modules**

- **`gemini_chart_analyzer/`**: AI-powered chart analysis using Google Gemini
  - Single and multi-timeframe analysis
  - Batch market scanning
  - Weighted signal aggregation

- **`backtester/`**: Strategy backtesting framework
- **`smart_money_concept/`**: Institutional trading concepts
- **`iching/`**: I Ching divination integration
- **`targets/`**: ATR-based target calculation

### Design Principles

1. **Modularity**: Each module is self-contained with clear interfaces
2. **Extensibility**: Easy to add new strategies and indicators
3. **Testability**: Comprehensive test coverage for all components
4. **Configurability**: Centralized configuration system
5. **Performance**: Caching, parallel processing, and optimization
6. **Reliability**: Smart fallback mechanisms and error handling

## 🔍 Key Features

### 1. Multi-Exchange Data Infrastructure

**Smart Data Fetching:**

- Automatic fallback across 8+ exchanges when data is unavailable or stale
- Intelligent caching system to minimize API calls
- Data freshness validation
- Support for both spot and futures markets
- Forex market data integration

**Supported Exchanges:**

- Binance (primary)
- Kraken, KuCoin, Gate.io, OKX, Bybit, MEXC, Huobi (fallback)

### 2. Technical Indicators (50+)

**Trend Indicators:**

- Moving Averages: SMA, EMA, WMA, HMA, DEMA, LSMA
- MACD (Moving Average Convergence Divergence)
- ATC (Adaptive Trend Classification)
- KAMA (Kaufman Adaptive Moving Average)

**Momentum Indicators:**

- RSI (Relative Strength Index)
- Stochastic RSI
- Momentum oscillators

**Volatility Indicators:**

- ATR (Average True Range)
- Bollinger Bands
- Range Oscillator (8 strategies)
- Keltner Channels

**Volume Indicators:**

- OBV (On-Balance Volume)
- Volume-weighted indicators
- Accumulation/Distribution

**Pattern Recognition:**

- Candlestick patterns (Doji, Engulfing, Hammer, Shooting Star, etc.)
- Three White Soldiers, Three Black Crows
- Morning Star, Evening Star

### 3. Signal Combination Strategies

**Hybrid Analyzer (Sequential Filtering):**

- Sequential filtering: ATC → Range Oscillator → SPC → Decision Matrix
- Early filtering reduces computational load by 70-90%
- Range Oscillator acts as confirmation filter
- Fallback mechanisms when filters don't match
- Optimal for resource-constrained environments

**Voting Analyzer (Pure Voting):**

- All indicators calculate signals in parallel
- Weighted voting based on historical accuracy
- All symbols evaluated with complete information
- More flexible, not dependent on filtering order
- Higher accuracy potential with more resources

**Decision Matrix:**

- Pseudo Random Forest-like voting system
- Weighted votes based on indicator accuracy
- Cumulative vote threshold filtering
- Feature importance calculation

**SPC Aggregation:**

- Three strategies combined (Cluster Transition, Regime Following, Mean Reversion)
- Weighted voting with configurable weights
- Cluster-based market regime detection

**HMM Signal Combiner:**

- Three HMM strategies (Swings, KAMA, High-Order)
- Multiple voting mechanisms (simple majority, weighted, confidence-weighted, threshold-based)
- Automatic strategy registration and management
- Conflict resolution and confidence scoring

### 4. Quantitative Metrics

**Statistical Tests:**

- Augmented Dickey-Fuller (ADF) test for stationarity
- Johansen cointegration test
- Correlation analysis with caching

**Mean Reversion Metrics:**

- Half-life calculation
- Hurst exponent
- Z-score analysis
- Ornstein-Uhlenbeck process parameters

**Risk Metrics:**

- Sharpe ratio
- Maximum drawdown
- Calmar ratio
- Sortino ratio
- Value at Risk (VaR)
- Beta calculation

**Hedge Ratios:**

- Ordinary Least Squares (OLS) regression
- Kalman filter adaptive estimation

**Classification Metrics:**

- F1-score, precision, recall
- Confusion matrix
- Direction prediction accuracy

### 5. Machine Learning Capabilities

**XGBoost:**

- Multi-class classification (UP, NEUTRAL, DOWN)
- Dynamic volatility-based labeling
- Triple-barrier method with adaptive thresholds
- Hyperparameter optimization with Optuna
- Time-series cross-validation
- Feature importance analysis

**LSTM:**

- CNN-LSTM-Attention architecture
- Multi-head attention mechanisms
- Positional encoding
- Focal loss for imbalanced classes
- Kalman filter noise reduction

**Temporal Fusion Transformer:**

- State-of-the-art time series forecasting
- Quantile regression for uncertainty
- Advanced feature selection (MI, Boruta, F-test)
- Fractional differentiation
- Interpretable attention mechanisms

**Random Forest:**

- Ensemble learning
- Feature importance ranking
- Out-of-bag error estimation

**Hidden Markov Models:**

- 3 distinct strategies
- State space modeling
- Automatic order optimization using BIC
- Regime detection and classification

### 6. AI-Powered Analysis

**Google Gemini Integration:**

- Intelligent chart interpretation
- Single and multi-timeframe analysis
- Batch market scanning (100 symbols per batch)
- Deep analysis mode
- Weighted signal aggregation
- Natural language insights

### 7. Web Interface

**Modern Tech Stack:**

- FastAPI backend (high-performance async)
- Vue.js 3 frontend (reactive UI)
- Real-time log streaming
- RESTful API with OpenAPI documentation

**Applications:**

- Gemini Analyzer (Port 8001)
- ATC Visualizer (Port 8002)
- Extensible architecture for new apps

## ⚙️ Configuration

All configuration is centralized in the `config/` directory:

### Core Configuration

- **`common.py`**: Common settings (exchanges, timeframes, symbols, defaults)
- **`config_api.py`**: API keys (supports environment variables - recommended)
- **`model_features.py`**: Feature definitions for ML models
- **`evaluation.py`**: Evaluation metrics and thresholds

### Strategy Configuration

- **`decision_matrix.py`**: Decision Matrix voting system
- **`range_oscillator.py`**: Range Oscillator strategies
- **`spc.py`**: Simplified Percentile Clustering
- **`pairs_trading.py`**: Pairs trading parameters
- **`portfolio.py`**: Portfolio management settings
- **`position_sizing.py`**: Position sizing parameters

### Machine Learning Configuration

- **`xgboost.py`**: XGBoost model parameters
- **`lstm.py`**: LSTM architecture and training
- **`deep_learning.py`**: Temporal Fusion Transformer (TFT)
- **`random_forest.py`**: Random Forest parameters
- **`hmm.py`**: HMM strategies and voting mechanisms

### Specialized Configuration

- **`gemini_chart_analyzer.py`**: Google Gemini AI settings
- **`forex_pairs.py`**: Forex pair configurations
- **`iching.py`**: I Ching integration settings

### API Key Management

**Recommended: Environment Variables**

Use the setup scripts to configure API keys securely:

```bash
# Windows (PowerShell)
.\setup\setup_api_keys.ps1

# Windows (Command Prompt)
setup\setup_api_keys.bat

# Linux/Mac
./setup/setup_api_keys.sh
```

The scripts create environment variables that are automatically loaded by the application. See [setup/SECURITY.md](./setup/SECURITY.md) for security best practices.

## 🛠️ Development

### Code Structure

The codebase follows best practices for maintainability and extensibility:

- **Modular Design**: Clear separation of concerns with independent modules
- **Test-Driven**: Comprehensive test coverage for all components
- **Type Safety**: Type hints throughout the codebase
- **Documentation**: Docstrings and inline comments
- **Configuration**: Centralized, environment-based configuration
- **Error Handling**: Robust error handling with fallback mechanisms

### Running Tests

```bash
# All tests
pytest

# Specific module with verbose output
pytest tests/xgboost/ -v

# With coverage report
pytest --cov=modules --cov-report=html
pytest --cov=modules --cov-report=term-missing

# Run tests matching pattern
pytest -k "test_signal"

# Run with memory optimization
pytest -c pytest_memory.ini

# Performance tests
pytest tests/performance/ -v
```

### Code Quality Tools

**Linting:**

```bash
# Using pylint
pylint modules/

# Using flake8
flake8 modules/
```

**Formatting:**

```bash
# Using black (recommended)
black modules/

# Check formatting without changes
black --check modules/
```

**Type Checking:**

```bash
# Using mypy (optional)
mypy modules/
```

### Development Workflow

1. **Create a feature branch:**

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes and add tests:**
   - Write tests first (TDD approach)
   - Implement the feature
   - Ensure all tests pass

3. **Run code quality checks:**

   ```bash
   pytest                    # Run tests
   black modules/            # Format code
   pylint modules/           # Lint code
   ```

4. **Commit and push:**

   ```bash
   git add .
   git commit -m "feat: add new feature"
   git push origin feature/your-feature-name
   ```

5. **Create a pull request**

### Adding New Modules

To add a new trading strategy or indicator module:

1. Create module directory in `modules/`
2. Implement core logic
3. Add configuration in `config/`
4. Create CLI interface in module's `cli/` directory
5. Add comprehensive tests in `tests/`
6. Update documentation
7. Register with signal calculators if applicable

See existing modules for reference implementation.

### Adding New Web Apps

To add a new web application:

1. Create app directory in `web/apps/`
2. Set up backend (FastAPI) and frontend (Vue.js)
3. Configure port in `web/scripts/`
4. Add documentation in `web/docs/`

See `web/docs/ADDING_NEW_APP.md` for detailed instructions.

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### Contribution Process

1. **Fork the repository**

   ```bash
   git clone https://github.com/your-username/crypto-probability.git
   cd crypto-probability
   ```

2. **Create a feature branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Write clean, documented code
   - Follow existing code style and conventions
   - Add type hints where applicable

4. **Add tests for new features**
   - Write comprehensive unit tests
   - Ensure test coverage for new code
   - Run all tests to verify nothing breaks

5. **Ensure all tests pass**

   ```bash
   pytest
   pytest --cov=modules --cov-report=term-missing
   ```

6. **Run code quality checks**

   ```bash
   black modules/           # Format code
   pylint modules/          # Lint code
   ```

7. **Commit your changes**

   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

8. **Push to your fork and submit a pull request**

   ```bash
   git push origin feature/your-feature-name
   ```

### Contribution Guidelines

- **Code Style**: Follow PEP 8 guidelines, use `black` for formatting
- **Documentation**: Add docstrings to all public functions and classes
- **Testing**: Maintain or improve test coverage
- **Commits**: Use clear, descriptive commit messages
- **Pull Requests**: Include description of changes and motivation

### Areas for Contribution

- 🚀 New trading strategies or indicators
- 🧪 Additional ML/DL models
- 📊 Enhanced visualization features
- 🌐 Web interface improvements
- 📚 Documentation and examples
- 🐛 Bug fixes and performance optimizations
- 🧪 Test coverage improvements

## 📞 Support

### Getting Help

- **Issues**: Report bugs or request features via [GitHub Issues](https://github.com/your-repo/crypto-probability/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/your-repo/crypto-probability/discussions)
- **Documentation**: Check the comprehensive documentation in the `docs/` directory and module READMEs

### Common Issues

**Installation Issues:**

- Ensure Python 3.9+ is installed
- Try creating a fresh virtual environment
- Check `requirements.txt` for dependency conflicts

**API Key Issues:**

- Verify environment variables are set correctly
- Run setup scripts again: `./setup/setup_api_keys.sh`
- See `setup/SECURITY.md` for troubleshooting

**Performance Issues:**

- Use Hybrid Analyzer for better performance
- Reduce symbol pool size
- Disable unnecessary indicators
- Check `tests/docs/test_memory_usage_guide.md` for optimization tips

**Data Fetching Issues:**

- Check exchange API status
- Verify API keys are correct
- System automatically falls back to other exchanges

## ⚠️ Disclaimer

**Important**: This software is provided for educational and research purposes only.

- **Not Financial Advice**: Nothing in this software constitutes financial, investment, trading, or any other type of advice.
- **High Risk**: Cryptocurrency trading involves substantial risk of loss and is not suitable for every investor.
- **No Guarantees**: Past performance is not indicative of future results. No guarantee of profit.
- **Your Responsibility**: Always conduct your own research (DYOR) and consult with a qualified financial advisor before making any investment decisions.
- **Liability**: The authors and contributors are not liable for any losses or damages resulting from the use of this software.
- **Use at Your Own Risk**: Never invest more than you can afford to lose.

By using this software, you acknowledge that you have read this disclaimer and agree to use the software at your own risk.

## 📄 License

This project is licensed under the terms specified in the `LICENSE` file.

---

## 📊 Project Status

**Version**: 3.0  
**Last Updated**: January 2025  
**Status**: Active Development  
**Python**: 3.9+ (3.10+ recommended)  

### Recent Updates

- ✅ Google Gemini AI integration for chart analysis
- ✅ Web interface with Vue.js + FastAPI
- ✅ LSTM with CNN-Attention architecture
- ✅ Enhanced quantitative metrics
- ✅ Comprehensive test suite with memory optimization
- ✅ Modular architecture refactoring

### Roadmap

- 🔜 Real-time trading integration
- 🔜 Advanced portfolio optimization
- 🔜 Mobile application
- 🔜 Multi-language support
- 🔜 Enhanced backtesting framework

---

**Built with ❤️ for the cryptocurrency trading community**

---

## 📖 Quick Start Guide

### For New Users

1. **Install basic dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API keys (optional, but recommended for authenticated endpoints):**

   ```bash
   # Windows (PowerShell)
   .\setup\setup_api_keys.ps1
   
   # Linux/Mac
   ./setup/setup_api_keys.sh
   ```

3. **Try the Hybrid Analyzer (recommended for beginners):**

   ```bash
   python main_complex_hybrid.py --timeframe 1h --enable-spc
   ```

   The Hybrid Analyzer uses sequential filtering to efficiently identify opportunities:
   - ATC scans for initial signals
   - Range Oscillator filters confirmation
   - SPC adds regime detection (if enabled)
   - Decision Matrix provides weighted voting (if enabled)

4. **Explore individual modules:**

   ```bash
   # XGBoost ML prediction
   python modules/xgboost/cli/main.py --symbol BTC/USDT --timeframe 1h
   
   # HMM state-based analysis
   python modules/hmm/cli/main.py --symbol BTC/USDT --timeframe 1h
   
   # Pairs trading analysis
   python modules/pairs_trading/cli/main.py --sort-by quantitative_score
   
   # AI-powered chart analysis
   python main_gemini_chart_analyzer.py --symbol BTC/USDT --timeframes 1h,4h,1d
   ```

5. **Launch web interface:**

   ```bash
   # Start all web applications
   python web/scripts/start_all.py
   
   # Access:
   # - Gemini Analyzer: http://localhost:8001
   # - ATC Visualizer: http://localhost:8002
   ```

### For Intermediate Users

**Install ML dependencies for advanced models:**

```bash
pip install -r requirements-ml.txt
```

**Try deep learning models:**

```bash
# LSTM model
python main_lstm.py --symbol BTC/USDT --timeframe 1h --epochs 50

# Temporal Fusion Transformer
python modules/deeplearning/cli/main.py --symbol BTC/USDT --timeframe 1h --epochs 10 --gpu
```

**Compare Hybrid vs Voting analyzers:**

```bash
# Hybrid (sequential filtering - faster)
python main_complex_hybrid.py --timeframe 1h --enable-spc --use-decision-matrix

# Voting (parallel evaluation - more comprehensive)
python main_complex_voting.py --timeframe 1h --enable-spc
```

### For Advanced Users

**Customize configuration:**

- Edit files in `config/` directory to adjust parameters
- Modify strategy weights and thresholds
- Configure indicator profiles in `IndicatorEngine`

**Add new strategies:**

- Follow module structure in `modules/`
- Implement strategy interface
- Register with signal calculators
- Add comprehensive tests

**Extend core analyzers:**

- Modify `core/hybrid_analyzer.py` for sequential filtering logic
- Modify `core/voting_analyzer.py` for voting mechanisms
- Add new signal calculators in `core/signal_calculators.py`

**Add custom indicators:**

- Extend `modules/common/IndicatorEngine.py`
- Implement indicator in `modules/common/indicators/`
- Register with appropriate profile (CORE, XGBOOST, DEEP_LEARNING)

### Workflow Selection Guide

**Choose Hybrid Analyzer when:**

- ✅ Limited computational resources
- ✅ Need faster results with early filtering
- ✅ Range Oscillator confirmation is important
- ✅ Prefer sequential, easy-to-debug workflow
- ✅ Want fallback mechanisms
- ✅ Analyzing large symbol pools (100+ symbols)

**Choose Voting Analyzer when:**

- ✅ Sufficient computational resources
- ✅ Want to consider all indicators simultaneously
- ✅ Prefer pure voting without sequential bias
- ✅ Need maximum flexibility in signal combination
- ✅ Analyzing smaller symbol pools (10-50 symbols)
- ✅ Want potentially higher accuracy

**Performance Comparison:**

- Hybrid: ~30-60 seconds for 100 symbols
- Voting: ~2-5 minutes for 100 symbols
- Hybrid: 70-90% fewer calculations through early filtering
- Voting: 100% of symbols get full indicator evaluation

See `core/README.md` for detailed workflow comparison and decision tree.

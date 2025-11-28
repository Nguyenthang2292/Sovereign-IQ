# Pairs Trading Module

Comprehensive pairs trading analysis toolkit for cryptocurrency markets.

## Overview

This module provides a complete framework for identifying, analyzing, and validating pairs trading opportunities based on statistical arbitrage principles.

## Architecture

```
pairs_trading/
├── core/                   # Core analysis components
│   ├── pairs_analyzer.py          # Main pairs trading analyzer
│   ├── pair_metrics_computer.py   # Quantitative metrics computation
│   └── opportunity_scorer.py      # Opportunity scoring logic
│
├── metrics/                # Statistical and quantitative metrics
│   ├── statistical_tests.py       # ADF, Johansen, half-life tests
│   ├── hedge_ratio.py             # OLS and Kalman hedge ratios
│   ├── zscore_metrics.py          # Z-score, Hurst exponent, direction metrics
│   └── risk_metrics.py            # Sharpe, max drawdown, Calmar ratio
│
├── analysis/               # Performance analysis
│   └── performance_analyzer.py    # Multi-timeframe performance analysis
│
├── utils/                  # Business logic utilities
│   ├── pairs_selector.py           # Pair selection algorithms
│   └── ensure_symbols_in_pools.py  # Candidate pool management
│
└── cli/                    # Command-line interface
    ├── argument_parser.py         # CLI argument parsing
    ├── interactive_prompts.py     # Interactive user prompts
    ├── input_parsers.py           # Input validation and parsing
    └── display/                   # Display utilities
        ├── display_performers.py
        └── display_pairs_opportunities.py
```

## Key Components

### Core Components

#### PairsTradingAnalyzer
Main class for analyzing pairs trading opportunities from best and worst performing symbols.

```python
from modules.pairs_trading import PairsTradingAnalyzer

analyzer = PairsTradingAnalyzer(
    min_spread=0.01,
    max_spread=0.50,
    min_correlation=0.3,
    max_correlation=0.9
)

pairs = analyzer.analyze_pairs_opportunity(
    best_performers=best_df,
    worst_performers=worst_df,
    data_fetcher=fetcher
)
```

#### PairMetricsComputer
Computes comprehensive quantitative metrics for trading pairs.

```python
from modules.pairs_trading import PairMetricsComputer

computer = PairMetricsComputer(
    adf_pvalue_threshold=0.05,
    zscore_lookback=60
)

metrics = computer.compute_pair_metrics(price1, price2)
# Returns: hedge_ratio, adf_pvalue, is_cointegrated, half_life,
#          zscore stats, Hurst exponent, Sharpe ratio, etc.
```

#### OpportunityScorer
Calculates opportunity scores based on spread, correlation, and quantitative metrics.

```python
from modules.pairs_trading import OpportunityScorer

scorer = OpportunityScorer(
    min_correlation=0.3,
    max_correlation=0.9
)

score = scorer.calculate_opportunity_score(
    spread=0.15,
    correlation=0.75,
    quant_metrics=metrics
)
```

### Metrics

#### Statistical Tests
- **ADF Test**: Augmented Dickey-Fuller test for stationarity
- **Johansen Test**: Cointegration test
- **Half-life**: Mean reversion half-life

```python
from modules.pairs_trading import (
    calculate_adf_test,
    calculate_johansen_test,
    calculate_half_life
)

adf_result = calculate_adf_test(spread_series)
johansen_result = calculate_johansen_test(price1, price2)
half_life = calculate_half_life(spread_series)
```

#### Z-score Metrics
- **Z-score Statistics**: Mean, std, skewness, kurtosis, current z-score
- **Hurst Exponent**: Mean reversion indicator (H < 0.5 = mean-reverting)
- **Direction Metrics**: Classification metrics for spread direction prediction

```python
from modules.pairs_trading import (
    calculate_zscore_stats,
    calculate_hurst_exponent,
    calculate_direction_metrics
)

zscore_stats = calculate_zscore_stats(spread_series)
hurst = calculate_hurst_exponent(spread_series)
direction_metrics = calculate_direction_metrics(spread_series)
```

#### Risk Metrics
- **Spread Sharpe Ratio**: Risk-adjusted return of the spread
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Calmar Ratio**: Return / max drawdown

```python
from modules.pairs_trading import (
    calculate_spread_sharpe,
    calculate_max_drawdown,
    calculate_calmar_ratio
)

sharpe = calculate_spread_sharpe(spread_series, periods_per_year=365*24)
max_dd = calculate_max_drawdown(spread_series)
calmar = calculate_calmar_ratio(spread_series, periods_per_year=365*24)
```

#### Hedge Ratios
- **OLS Hedge Ratio**: Ordinary Least Squares regression
- **Kalman Hedge Ratio**: Kalman filter for time-varying hedge ratio

```python
from modules.pairs_trading import (
    calculate_ols_hedge_ratio,
    calculate_kalman_hedge_ratio
)

ols_ratio = calculate_ols_hedge_ratio(price1, price2, fit_intercept=True)
kalman_ratio = calculate_kalman_hedge_ratio(price1, price2, delta=1e-5)
```

### Utilities

#### Pair Selection
```python
from modules.pairs_trading import (
    select_top_unique_pairs,
    select_pairs_for_symbols
)

# Select top N pairs with unique symbols
unique_pairs = select_top_unique_pairs(pairs_df, target_pairs=10)

# Select best pairs for specific symbols
symbol_pairs = select_pairs_for_symbols(
    pairs_df, 
    target_symbols=['BTC/USDT', 'ETH/USDT']
)
```

#### Candidate Pool Management
```python
from modules.pairs_trading import ensure_symbols_in_candidate_pools

# Ensure target symbols are in appropriate pools
best_df, worst_df = ensure_symbols_in_candidate_pools(
    performance_df,
    best_df,
    worst_df,
    target_symbols=['BTC/USDT', 'ETH/USDT']
)
```

### CLI Tools

#### Display Formatters
```python
from modules.pairs_trading import (
    display_performers,
    display_pairs_opportunities
)

# Display top/worst performers
display_performers(best_df, "Top Performers", color=Fore.GREEN)
display_performers(worst_df, "Worst Performers", color=Fore.RED)

# Display pairs opportunities
display_pairs_opportunities(pairs_df, max_display=10)
```

#### Argument Parsing
```python
from modules.pairs_trading import parse_args

args = parse_args()
# Parses all CLI arguments: --pairs-count, --weights, --min-spread, etc.
```

#### Interactive Prompts
```python
from modules.pairs_trading import (
    prompt_interactive_mode,
    prompt_weight_preset_selection,
    prompt_kalman_preset_selection,
    prompt_opportunity_preset_selection,
    prompt_target_pairs,
    prompt_candidate_depth
)

mode, symbols_str = prompt_interactive_mode()
preset = prompt_weight_preset_selection(current_preset='balanced')
kalman_preset = prompt_kalman_preset_selection()
opportunity_preset = prompt_opportunity_preset_selection()
target_pairs = prompt_target_pairs(default_count=10)
candidate_depth = prompt_candidate_depth(default=50)
```

#### Input Parsers
```python
from modules.pairs_trading import (
    parse_weights,
    parse_symbols,
    standardize_symbol_input
)

weights = parse_weights("1d:0.5,3d:0.3,1w:0.2")
symbols = parse_symbols("BTC/USDT,ETH/USDT")
standardized = standardize_symbol_input("btc/usdt")
```

## Workflow

### Typical Usage Flow

1. **Analyze Performance**
   ```python
   from modules.pairs_trading import PerformanceAnalyzer
   
   analyzer = PerformanceAnalyzer(weights={'1d': 0.5, '3d': 0.3, '1w': 0.2})
   performance_df = analyzer.analyze_all_symbols(symbols, data_fetcher)
   best_df = analyzer.get_top_performers(performance_df, top_n=50)
   worst_df = analyzer.get_worst_performers(performance_df, top_n=50)
   ```

2. **Analyze Pairs**
   ```python
   from modules.pairs_trading import PairsTradingAnalyzer
   
   pairs_analyzer = PairsTradingAnalyzer()
   pairs_df = pairs_analyzer.analyze_pairs_opportunity(
       best_performers=best_df,
       worst_performers=worst_df,
       data_fetcher=data_fetcher
   )
   ```

3. **Validate Pairs**
   ```python
   validated_pairs = pairs_analyzer.validate_pairs(
       pairs_df,
       data_fetcher=data_fetcher
   )
   ```

4. **Select and Display**
   ```python
   from modules.pairs_trading import (
       select_top_unique_pairs,
       display_pairs_opportunities
   )
   
   final_pairs = select_top_unique_pairs(validated_pairs, target_pairs=10)
   display_pairs_opportunities(final_pairs, max_display=10)
   ```

## Configuration

Key configuration parameters (from `modules.config`):

```python
# Spread thresholds
PAIRS_TRADING_MIN_SPREAD = 0.01  # 1%
PAIRS_TRADING_MAX_SPREAD = 0.50  # 50%

# Correlation thresholds
PAIRS_TRADING_MIN_CORRELATION = 0.3
PAIRS_TRADING_MAX_CORRELATION = 0.9

# Statistical tests
PAIRS_TRADING_ADF_PVALUE_THRESHOLD = 0.05
PAIRS_TRADING_MAX_HALF_LIFE = 50

# Z-score parameters
PAIRS_TRADING_ZSCORE_LOOKBACK = 60
PAIRS_TRADING_CLASSIFICATION_ZSCORE = 0.5

# Hedge ratio parameters
PAIRS_TRADING_OLS_FIT_INTERCEPT = True
PAIRS_TRADING_KALMAN_DELTA = 1e-5
PAIRS_TRADING_KALMAN_OBS_COV = 1.0
```

## Metrics Interpretation

### Opportunity Score
- **> 20%**: Excellent opportunity (green)
- **10-20%**: Good opportunity (yellow)
- **< 10%**: Weak opportunity (white)

### Quantitative Score (0-100)
- **>= 70**: Strong quantitative metrics (green)
- **50-70**: Moderate metrics (yellow)
- **< 50**: Weak metrics (red)

### Hurst Exponent
- **H < 0.5**: Mean-reverting (good for pairs trading)
- **H ≈ 0.5**: Random walk
- **H > 0.5**: Trending (less suitable)

### Correlation
- **|r| > 0.7**: Strong correlation (green)
- **0.4 < |r| <= 0.7**: Moderate correlation (yellow)
- **|r| <= 0.4**: Weak correlation (red)

## Best Practices

1. **Always validate pairs** with statistical tests (ADF, Johansen)
2. **Check Hurst exponent** - prefer H < 0.5 for mean reversion
3. **Monitor half-life** - shorter is better for faster mean reversion
4. **Use appropriate hedge ratios** - Kalman for time-varying relationships
5. **Diversify** - select pairs with unique symbols
6. **Review quantitative scores** - aim for >= 70 for robust opportunities

## Dependencies

- pandas
- numpy
- scipy (for statistical tests)
- sklearn (for classification metrics)
- statsmodels (for ADF, Johansen tests)
- pykalman (for Kalman filter)
- colorama (for CLI color output)

## License

Part of the crypto-probability project.

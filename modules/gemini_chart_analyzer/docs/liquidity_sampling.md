# Liquidity-Weighted Sampling with Volatility and Spread

## Overview

This document describes the enhanced **liquidity-weighted sampling** strategy for Stage 0 pre-filtering, which now incorporates:

1. **Volume** - Trading volume (quote volume)
2. **Volatility** - Average True Range (ATR) normalized by price
3. **Spread** - High-Low range as percentage of close

## Implementation

### Python Implementation

Location: `modules/gemini_chart_analyzer/core/prefilter/sampling_strategies.py`

**Key Functions:**

#### `_calculate_volatility_and_spread()`

Calculates ATR and spread metrics for a list of symbols.

- **Volatility (ATR%)**: `(ATR / current_price) × 100`
- **Spread (%)**: Average of `((high - low) / close) × 100` over lookback period
- **Lookback period**: 14 days (default)
- **Performance**:
  - Python: ~0.5-1s per symbol (sequential)
  - Rust: ~0.05-0.1s per symbol (parallel with Rayon)

#### `liquidity_weighted_sampling()`

Samples symbols based on composite liquidity score combining volume, volatility, and spread.

**Liquidity Score Formula:**

```formula
score = volume_weight × norm(volume)
      + spread_weight × (1 - norm(spread))     [lower spread = better]
      + volatility_weight × volatility_term    [depends on preference]
```

**Volatility Term:**

- If `prefer_low_volatility=True`: `1 - norm(volatility)` (prefer stable assets)
- If `prefer_low_volatility=False`: `1 - 4×(norm(volatility) - 0.5)²` (prefer moderate volatility, peaks at 0.5)

**Default Weights:**

- Volume: 40% (high volume = liquid)
- Spread: 20% (low spread = liquid)
- Volatility: 40% (moderate volatility = trading opportunity)

### Rust Implementation

Location: `modules/adaptive_trend_LTS/rust_extensions/src/liquidity_metrics.rs`

**Key Function:**

#### `compute_liquidity_metrics_batch()`

Batch processes multiple symbols in parallel using Rayon.

**Features:**

- **Parallel processing**: Uses Rayon's parallel iterators
- **True Range calculation**: Accurate ATR computation
- **Batch optimization**: Processes all symbols in one call
- **Error handling**: Gracefully handles missing/invalid data

**Performance Comparison:**

| Symbols | Python (Sequential) | Rust (Parallel) | Speedup |
|---------|---------------------|-----------------|---------|
| 100     | ~50-100s            | ~5-10s          | 10x     |
| 500     | ~250-500s           | ~25-50s         | 10x     |
| 1000    | ~500-1000s          | ~50-100s        | 10x     |

*Note: Times include OHLCV data fetching (network I/O bound)*

## Usage

### Basic Usage (Python only)

```python
from modules.gemini_chart_analyzer.core.prefilter.sampling_strategies import (
    SamplingStrategy,
    apply_sampling_strategy,
    get_symbol_volumes,
)
from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager

# Setup
exchange_manager = ExchangeManager()
data_fetcher = DataFetcher(exchange_manager)
symbols = ["BTC/USDT", "ETH/USDT", ...]  # Your symbol list

# Get volume data
volumes = get_symbol_volumes(symbols, data_fetcher)

# Sample using liquidity-weighted strategy (auto-calculates volatility/spread)
sampled = apply_sampling_strategy(
    symbols=symbols,
    sample_percentage=20.0,  # Sample 20% of symbols
    strategy=SamplingStrategy.LIQUIDITY_WEIGHTED,
    volumes=volumes,
    data_fetcher=data_fetcher,  # Required for metric calculation
    use_rust=True,  # Use Rust backend (10x faster)
)
```

### Advanced Usage (Custom Weights)

```python
from modules.gemini_chart_analyzer.core.prefilter.sampling_strategies import (
    liquidity_weighted_sampling,
)

# Custom liquidity scoring
sampled = liquidity_weighted_sampling(
    symbols=symbols,
    sample_percentage=20.0,
    volumes=volumes,
    data_fetcher=data_fetcher,
    calculate_metrics=True,  # Auto-calculate volatility/spread
    volume_weight=0.5,  # 50% weight on volume
    spread_weight=0.2,  # 20% weight on spread
    volatility_weight=0.3,  # 30% weight on volatility
    prefer_low_volatility=False,  # Prefer moderate volatility (trading opportunity)
    use_rust=True,
)
```

### Pre-computed Metrics

If you already have volatility/spread data:

```python
# Pre-computed metrics (e.g., from cache or database)
volatility_data = {"BTC/USDT": 3.5, "ETH/USDT": 4.2, ...}  # ATR%
spread_data = {"BTC/USDT": 0.8, "ETH/USDT": 1.1, ...}  # Spread%

sampled = liquidity_weighted_sampling(
    symbols=symbols,
    sample_percentage=20.0,
    volumes=volumes,
    volatility_data=volatility_data,  # Pre-computed
    spread_data=spread_data,  # Pre-computed
    calculate_metrics=False,  # Don't recalculate
)
```

### Integration with Stage 0 Workflow

The `liquidity_weighted` strategy is automatically available in the batch scanner:

```python
# In modules/gemini_chart_analyzer/core/prefilter/workflow.py
sampled_symbols = apply_sampling_strategy(
    symbols=all_symbols,
    sample_percentage=stage0_sample_percentage,
    strategy=SamplingStrategy.LIQUIDITY_WEIGHTED,
    volumes=volumes,
    data_fetcher=temp_data_fetcher,  # Automatically passed
)
```

## Metrics Explanation

### 1. Volume

- **What it measures**: Trading activity (quote volume in USDT)
- **Interpretation**: Higher volume = more liquid = easier to enter/exit positions
- **Source**: Binance exchange via `load_markets()` public API

### 2. Volatility (ATR%)

- **What it measures**: Average True Range normalized by price
- **Formula**: `ATR% = (ATR / current_price) × 100`
- **Interpretation**:
  - Low (0-2%): Stable, low volatility (good for conservative strategies)
  - Moderate (2-5%): Good for trading (balanced risk/reward)
  - High (5%+): Very volatile (high risk, high reward)
- **Source**: Calculated from daily OHLC data (14-day lookback)

### 3. Spread (%)

- **What it measures**: Average high-low range as percentage of close
- **Formula**: `Spread% = ((high - low) / close) × 100`
- **Interpretation**:
  - Low (0-1%): Tight spread, very liquid
  - Moderate (1-3%): Normal spread
  - High (3%+): Wide spread, less liquid
- **Source**: Calculated from daily OHLC data (14-day lookback)

## Performance Optimization

### When to Use Rust Backend

**Always use Rust (`use_rust=True`) when:**

- Processing 50+ symbols
- Running on server/production
- Need fastest possible execution

**Python fallback acceptable when:**

- Processing < 20 symbols
- Quick testing/development
- Rust not available (auto-fallback)

### Caching Strategy (Future Enhancement)

To further optimize, consider caching calculated metrics:

```python
# Pseudo-code for future enhancement
cache_key = f"liquidity_metrics_{date}_{lookback}"
cached_data = redis_client.get(cache_key)

if cached_data:
    volatility, spread = json.loads(cached_data)
else:
    volatility, spread = _calculate_volatility_and_spread(...)
    redis_client.setex(cache_key, 3600, json.dumps((volatility, spread)))  # 1 hour TTL
```

## Testing

### Unit Tests

Python tests are included in `_calculate_volatility_and_spread()`:

- Handles zero prices gracefully
- Validates ATR calculation
- Tests spread percentage calculation

Rust tests in `liquidity_metrics.rs`:

- `test_true_range()`: Validates TR formula
- `test_calculate_metrics_single()`: End-to-end metric calculation
- `test_zero_prices_handling()`: Edge case handling

### Integration Test Example

```python
import pytest
from modules.gemini_chart_analyzer.core.prefilter.sampling_strategies import (
    _calculate_volatility_and_spread,
    liquidity_weighted_sampling,
)

def test_liquidity_weighted_with_rust(data_fetcher):
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]

    # Calculate metrics
    volatility, spread = _calculate_volatility_and_spread(
        symbols, data_fetcher, timeframe="1d", lookback=14, use_rust=True
    )

    # Validate results
    assert len(volatility) > 0
    assert len(spread) > 0
    assert all(v >= 0 for v in volatility.values())
    assert all(s >= 0 for s in spread.values())

    # Test sampling
    volumes = {"BTC/USDT": 1000000, "ETH/USDT": 500000, "BNB/USDT": 200000}
    sampled = liquidity_weighted_sampling(
        symbols, 50.0, volumes, volatility, spread
    )

    assert len(sampled) > 0
    assert len(sampled) <= len(symbols)
```

## Building Rust Backend

To enable Rust acceleration:

```bash
# Windows
cd modules\adaptive_trend_LTS\rust_extensions
maturin develop --release

# Linux/Mac
cd modules/adaptive_trend_LTS/rust_extensions
maturin develop --release
```

Or use the automated build script:

```bash
# Windows
.\build_rust.bat

# Linux/Mac
./build_rust.sh
```

## Troubleshooting

### Rust Import Error

**Problem**: `ImportError: No module named 'atc_rust'`

**Solution**: Build Rust backend (see above) or set `use_rust=False`

### Slow Performance with Python

**Problem**: Metric calculation taking too long (> 2 minutes for 100 symbols)

**Possible causes:**

1. Network latency fetching OHLCV data
2. Using Python implementation instead of Rust
3. Large lookback period

**Solutions:**

- Enable Rust backend: `use_rust=True`
- Reduce lookback: `lookback=7` (default: 14)
- Cache OHLCV data
- Use faster timeframe: `timeframe="4h"` instead of `"1d"`

### Unexpected Sampling Results

**Problem**: Liquidity-weighted sampling returns unexpected symbols

**Debug steps:**

1. Check metric calculation:

   ```python
   vol, spread = _calculate_volatility_and_spread(symbols, data_fetcher)
   print(f"Volatility: {vol}")
   print(f"Spread: {spread}")
   ```

2. Verify volume data:

   ```python
   volumes = get_symbol_volumes(symbols, data_fetcher)
   print(f"Volumes: {volumes}")
   ```

3. Test with different weights:

   ```python
   # Heavily weight volume
   sampled = liquidity_weighted_sampling(
       symbols, 20.0, volumes, vol, spread,
       volume_weight=0.8, spread_weight=0.1, volatility_weight=0.1
   )
   ```

## Future Enhancements

### 1. Additional Metrics

- **Bid-Ask Spread**: Real-time order book spread (when available)
- **Market Depth**: Sum of bids/asks within X% of mid-price
- **Slippage Estimation**: Historical price impact analysis

### 2. Adaptive Weights

Automatically adjust weights based on market conditions:

```python
# Pseudo-code
if market_regime == "high_volatility":
    volatility_weight = 0.5  # Prefer stability
    prefer_low_volatility = True
else:
    volatility_weight = 0.3  # Allow more risk
    prefer_low_volatility = False
```

### 3. Machine Learning Scoring

Train ML model to predict liquidity quality:

```python
# Pseudo-code
liquidity_score = ml_model.predict([volume, volatility, spread, ...])
```

## References

- **ATR (Average True Range)**: <https://www.investopedia.com/terms/a/atr.asp>
- **Rayon (Rust Parallelism)**: <https://docs.rs/rayon/>
- **PyO3 (Python-Rust Bindings)**: <https://pyo3.rs/>

## Changelog

### v2.0.0 (Current)

- ✨ Added volatility (ATR%) and spread (%) metrics
- ✨ Implemented composite liquidity scoring
- ✨ Added Rust backend for 10x performance boost
- ✨ Support for custom weights and volatility preference
- ✨ Automatic fallback to Python if Rust unavailable
- 📝 Comprehensive documentation and testing

### v1.0.0 (Previous)

- Basic volume-weighted sampling only
- TODO comment for volatility/spread implementation

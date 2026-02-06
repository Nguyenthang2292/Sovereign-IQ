# Quick Start Guide - Adaptive Trend Classification (ATC) LTS Mini

Get started with ATC analysis in under 5 minutes!

## Prerequisites

Before you begin, ensure you have:

1. **Python 3.9+** installed (Python 3.10+ recommended)
2. **Rust toolchain** for building extensions (optional but recommended)
3. **Internet connection** for fetching market data

### Check Your Python Version

```bash
python --version
```

If you see Python 3.9 or higher, you're good to go!

## Installation

### Step 1: Install Core Dependencies

From the project root directory:

```bash
pip install pandas numpy
```

### Step 2: Install Exchange Dependencies

```bash
pip install ccxt
```

### Step 3: Build Rust Extensions (Recommended)

For optimal performance, build the Rust backend:

**Windows:**
```bash
cd modules\adaptive_trend_LTS_mini\rust_extensions
cargo build --release
cd ..\..\..
```

**Linux/Mac:**
```bash
cd modules/adaptive_trend_LTS_mini/rust_extensions
cargo build --release
cd ../../..
```

**Note:** If you don't have Rust installed:
- Visit [https://rustup.rs/](https://rustup.rs/) and follow the instructions
- After installing, restart your terminal
- If Rust is not available, the module will fall back to Numba (slower but functional)

### Step 4: Verify Installation

```bash
python -m modules.adaptive_trend_LTS_mini.cli.main --version
```

You should see: `main.py 1.0.0`

## Running Your First Analysis

### Option A: Analyze a Single Symbol (Fastest)

Run this command to analyze Bitcoin:

```bash
python -m modules.adaptive_trend_LTS_mini.cli.main --symbol BTC/USDT --timeframe 1h
```

**What happens:**
- Fetches 1500 1-hour candles for BTC/USDT from Binance
- Calculates ATC signals using 6 different moving averages
- Displays trend signals and current market state

**Expected runtime:** 2-5 seconds

### Option B: Scan Multiple Symbols (Auto Mode)

Find trading opportunities across all Binance futures:

```bash
python -m modules.adaptive_trend_LTS_mini.cli.main --auto --timeframe 1h --min-signal 0.5
```

**What happens:**
- Scans all available Binance futures symbols
- Filters symbols with strong signals (> 0.5)
- Displays top LONG and SHORT candidates

**Expected runtime:** 30-60 seconds for 100+ symbols

### Option C: Python API

Use ATC in your own scripts:

```python
from modules.adaptive_trend_LTS_mini.core.analyzer import analyze_symbol
from modules.adaptive_trend_LTS_mini.utils.config import ATCConfig
from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager

# Initialize components
exchange_manager = ExchangeManager()
data_fetcher = DataFetcher(exchange_manager)

# Configure ATC
config = ATCConfig(
    timeframe="1h",
    limit=1500,
    ema_len=28,
    robustness="Medium",
)

# Analyze symbol
result = analyze_symbol(
    symbol="BTC/USDT",
    data_fetcher=data_fetcher,
    config=config,
)

# Access results
if result:
    print(f"Current Price: {result['current_price']}")
    print(f"Average Signal: {result['atc_results']['Average_Signal'].iloc[-1]}")
```

## Expected Output

### Single Symbol Analysis

```
========================================
 Symbol: BTC/USDT
========================================
Exchange: Binance
Current Price: $45,234.56
Timeframe: 1h

ATC Signals (Latest):
  Average Signal:  0.75 [BULLISH]
  EMA Signal:      0.82
  HMA Signal:      0.71
  WMA Signal:      0.73
  DEMA Signal:     0.79
  LSMA Signal:     0.68
  KAMA Signal:     0.77

Interpretation:
  STRONG BUY - Multiple moving averages confirm bullish trend
```

### Auto Scan Mode

```
========================================
 ATC Auto Scan Results
========================================

TOP LONG SIGNALS (10):
┌────┬───────────┬─────────┬────────┐
│ #  │ Symbol    │ Signal  │ Price  │
├────┼───────────┼─────────┼────────┤
│ 1  │ ETH/USDT  │  0.89   │ $2,345 │
│ 2  │ BNB/USDT  │  0.82   │ $312   │
│ 3  │ SOL/USDT  │  0.78   │ $98    │
└────┴───────────┴─────────┴────────┘

TOP SHORT SIGNALS (5):
┌────┬───────────┬─────────┬────────┐
│ #  │ Symbol    │ Signal  │ Price  │
├────┼───────────┼─────────┼────────┤
│ 1  │ AVAX/USDT │ -0.76   │ $32    │
│ 2  │ ATOM/USDT │ -0.68   │ $9.45  │
└────┴───────────┴─────────┴────────┘
```

## Understanding the Signals

| Signal Value | Interpretation | Action |
|-------------|----------------|---------|
| **> 0.7**   | Strong Bullish | Consider LONG positions |
| **0.3 to 0.7** | Bullish | Moderate LONG opportunity |
| **-0.3 to 0.3** | Neutral | Wait for clearer signal |
| **-0.7 to -0.3** | Bearish | Moderate SHORT opportunity |
| **< -0.7**  | Strong Bearish | Consider SHORT positions |

## Common Command-Line Options

```bash
# Analyze specific timeframe
python -m modules.adaptive_trend_LTS_mini.cli.main --symbol ETH/USDT --timeframe 4h

# Scan with custom parameters
python -m modules.adaptive_trend_LTS_mini.cli.main --auto --max-symbols 50 --min-signal 0.6

# Adjust moving average lengths
python -m modules.adaptive_trend_LTS_mini.cli.main --symbol BTC/USDT --ema-len 21 --hma-len 21

# Change robustness (sensitivity)
python -m modules.adaptive_trend_LTS_mini.cli.main --symbol BTC/USDT --robustness Wide

# List all available symbols
python -m modules.adaptive_trend_LTS_mini.cli.main --list-symbols
```

## Troubleshooting

### Issue: "Module not found" error

**Solution:**
```bash
# Ensure you're in the project root directory
cd /path/to/crypto-probability

# Run with full module path
python -m modules.adaptive_trend_LTS_mini.cli.main --symbol BTC/USDT
```

### Issue: Slow performance

**Solution:**
1. Build Rust extensions (see Step 3 above)
2. Verify Rust backend is active:
   ```python
   from modules.adaptive_trend_LTS_mini.core.rust_backend import RUST_AVAILABLE
   print(f"Rust available: {RUST_AVAILABLE}")
   ```
3. Use smaller batch sizes for scanning:
   ```bash
   python -m modules.adaptive_trend_LTS_mini.cli.main --auto --batch-size 50
   ```

### Issue: No data fetched / Exchange errors

**Solution:**
- Check your internet connection
- Binance API might be rate-limiting. Wait 1-2 minutes and retry
- Try a different symbol: `--symbol ETH/USDT`

### Issue: Rust build fails

**Solution:**
- The module will automatically fall back to Numba (slower but works)
- Performance impact: ~3-5x slower, still usable
- No action required if you can accept slower performance

## Next Steps

Now that you've run your first analysis, explore more features:

1. **[Setting Guides](setting_guides.md)** - Detailed parameter tuning guide
2. **[README](../README.md)** - Complete documentation and architecture
3. **[Memory Optimizations](guide_memory_optimizations_usage.md)** - For large-scale backtesting
4. **[Dask Usage](guide_dask_usage.md)** - Parallel batch processing

### Python API Examples

**Batch scanning with custom symbols:**
```python
from modules.adaptive_trend_LTS_mini.core.scanner import scan_all_symbols
from modules.adaptive_trend_LTS_mini.utils.config import ATCConfig

symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
config = ATCConfig(timeframe="1h")

long_signals, short_signals = scan_all_symbols(
    data_fetcher=data_fetcher,
    atc_config=config,
    symbols=symbols,
    min_signal=0.5
)
```

**Access individual signals:**
```python
from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import compute_atc_signals
import pandas as pd

# Your price data
prices = pd.Series([...])  # Close prices

# Compute signals
result = compute_atc_signals(
    prices,
    use_rust_backend=True,
)

# Access individual MA signals
print(f"EMA Signal: {result['EMA_Signal'].iloc[-1]}")
print(f"HMA Signal: {result['HMA_Signal'].iloc[-1]}")
print(f"Average Signal: {result['Average_Signal'].iloc[-1]}")
```

## Performance Tips

1. **Enable Rust backend** (5-10x speedup): Build with `cargo build --release`
2. **Use appropriate timeframes**: Higher timeframes = less data = faster
3. **Limit symbol count**: Use `--max-symbols 50` for faster scans
4. **Adjust batch size**: Tune `--batch-size` based on your RAM (default: 100)
5. **Cache data**: The module automatically caches fetched data

## Getting Help

- **Documentation**: See `modules/adaptive_trend_LTS_mini/README.md`
- **Issues**: Check existing issues in the repository
- **Parameters**: Run `python -m modules.adaptive_trend_LTS_mini.cli.main --help`

## What's Next?

You've successfully run your first ATC analysis! Here are some ideas:

- Try different timeframes (5m, 15m, 1h, 4h, 1d)
- Experiment with robustness settings (Narrow, Medium, Wide)
- Adjust MA lengths to match your trading style
- Integrate ATC signals into your trading strategy
- Backtest signals using historical data

Happy trading!

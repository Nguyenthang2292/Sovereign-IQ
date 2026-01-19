# Range Oscillator Signal Explanation

## Overview

Range Oscillator signals are generated through 3 main steps:

1. **Calculate Range Oscillator indicator** (oscillator value from -100 to +100)
2. **Apply strategies** to generate signals (LONG/SHORT/NEUTRAL)
3. **Voting mechanism** to combine signals from multiple strategies

---

## Step 1: Calculate Range Oscillator Indicator

### 1.1. Calculate Weighted Moving Average (MA)

```python
# From close prices, calculate weighted MA based on price deltas
from modules.common.indicators.trend import calculate_weighted_ma
ma = calculate_weighted_ma(close, length=50)
```

**Formula:**

- For each bar, calculate `delta = |close[i] - close[i+1]|`
- Weight `w = delta / close[i+1]`
- Weighted MA = `Σ(close[i] * w) / Σ(w)`

**Purpose:** Emphasize bars with larger price movements, creating a more responsive MA.

### 1.2. Calculate ATR Range

```python
# Calculate ATR (Average True Range) and multiply by multiplier
from modules.common.indicators.volatility import calculate_atr_range
range_atr = calculate_atr_range(high, low, close, mult=2.0)
```

**Formula:**

- ATR = Average True Range with length 2000 (fallback 200)
- Range ATR = `ATR * mult` (default: 2.0)

**Purpose:** Determine the width of range bands, adapting to volatility.

### 1.3. Calculate Oscillator Value

```python
# For each bar:
osc_value = 100 * (close - MA) / RangeATR
```

**Values:**

- **+100**: Price at upper bound of range (breakout above)
- **0**: Price at equilibrium (MA)
- **-100**: Price at lower bound of range (breakout below)

**Example:**

- If `close = 50000`, `MA = 49000`, `RangeATR = 2000`
- `oscillator = 100 * (50000 - 49000) / 2000 = 50` (bullish, in middle of range)

### 1.4. Heatmap Visualization (Optional)

The heatmap feature provides color-coded visualization of oscillator "heat zones" based on value histograms:

```python
from modules.range_oscillator.core.oscillator import calculate_range_oscillator_with_heatmap
from modules.range_oscillator.config.heatmap_config import HeatmapConfig

oscillator, ma, range_atr, heat_colors, trend_direction = calculate_range_oscillator_with_heatmap(
    high=high,
    low=low,
    close=close,
    length=50,
    mult=2.0,
    heatmap_config=HeatmapConfig(
        levels_inp=5,      # Number of heat levels
        heat_thresh=3,      # Minimum touches per level
        lookback_bars=100,  # Lookback window
    ),
)
```

**How it works:**

1. **Divides oscillator range** into `levels_inp` horizontal levels
2. **Counts touches** per level in the last `lookback_bars` bars
3. **Applies gradient coloring** based on touch frequency:
   - Low touches → cold color (weak zones)
   - High touches → hot color (strong zones)
4. **Trend-based colors**:
   - Bullish trend: Green gradient (weak → strong bullish)
   - Bearish trend: Red gradient (weak → strong bearish)
   - Neutral/Transition: Blue

**Use cases:**

- **UI Visualization**: Color-code oscillator lines/charts
- **Signal Strength**: Heat intensity indicates zone strength
- **Zone Identification**: Identify frequently-touched price levels

**Color meanings:**

- **Strong Bullish** (`#09ff00`): High resistance in uptrends
- **Weak Bullish** (`#008000`): Pressure zones in uptrends
- **Strong Bearish** (`#ff0000`): High resistance in downtrends
- **Weak Bearish** (`#800000`): Pressure zones in downtrends
- **Transition** (`#0000ff`): Trend changes or neutral states

---

## Step 2: Apply Strategies to Generate Signals

Each strategy analyzes the oscillator and generates signals based on different logic:

### Strategy 5: Combined (Sustained + Crossover + Momentum)

```python
signals, signal_strength = generate_signals_combined_all_strategy(
    high=high, low=low, close=close,
    length=50, mult=2.0,
    use_sustained=True,    # Strategy 2
    use_crossover=True,    # Strategy 3
    use_momentum=True,     # Strategy 4
)
```

**Logic:**

- **Strategy 2 (Sustained)**: Oscillator above/below 0 for N bars → LONG/SHORT
- **Strategy 3 (Crossover)**: Oscillator crosses zero line → LONG/SHORT
- **Strategy 4 (Momentum)**: Rate of change of oscillator → LONG/SHORT
- **Voting**: Majority vote from 3 strategies

**Signal:**

- `1` = LONG (bullish)
- `-1` = SHORT (bearish)
- `0` = NEUTRAL

### Strategy 6: Breakout

```python
signals, signal_strength = generate_signals_strategy6_breakout(
    high=high, low=low, close=close,
    length=50, mult=2.0,
)
```

**Logic:**

- Detects when oscillator **breakouts** from extreme thresholds (±100)
- **LONG**: Oscillator breaks out above +100
- **SHORT**: Oscillator breaks out below -100
- Forward fill signal while oscillator remains in breakout zone

### Strategy 7: Divergence

```python
signals, signal_strength = generate_signals_divergence_strategy(
    high=high, low=low, close=close,
    length=50, mult=2.0,
)
```

**Logic:**

- Detects **divergence** between price and oscillator:
  - **Bearish Divergence**: Price creates higher high, but oscillator creates lower high → SHORT
  - **Bullish Divergence**: Price creates lower low, but oscillator creates higher low → LONG

### Strategy 8: Trend Following

```python
signals, signal_strength = generate_signals_trend_following_strategy(
    high=high, low=low, close=close,
    length=50, mult=2.0,
)
```

**Logic:**

- Follow trend with consistent oscillator position:
  - **LONG**: Oscillator above 0 and trend is bullish
  - **SHORT**: Oscillator below 0 and trend is bearish

### Strategy 9: Mean Reversion

```python
signals, signal_strength = generate_signals_mean_reversion_strategy(
    high=high, low=low, close=close,
    length=50, mult=2.0,
)
```

**Logic:**

- Detects mean reversion when oscillator moves from extreme back to zero:
  - **SHORT**: Oscillator at extreme positive (>+80), starting to return to zero
  - **LONG**: Oscillator at extreme negative (<-80), starting to return to zero

---

## Step 3: Get Latest Signal from Each Strategy

After each strategy generates a Series of signals (each bar has value 1/-1/0), we extract the **latest signal** (signal of the last bar):

```python
# Example with Strategy 5:
signals, signal_strength = generate_signals_combined_all_strategy(...)

# Get latest signal (skip NaN values)
non_nan_signals = signals.dropna()
if len(non_nan_signals) > 0:
    latest_signal = int(non_nan_signals.iloc[-1])  # Last bar
    # latest_signal = 1 (LONG), -1 (SHORT), or 0 (NEUTRAL)
```

**Example:**

```text
Bar 1: NaN
Bar 2: NaN
Bar 3: 0
Bar 4: 1
Bar 5: 1
Bar 6: 1  ← latest_signal = 1 (LONG)
```

---

## Step 4: Voting Mechanism (Combine Multiple Strategies)

After collecting signals from all strategies, we use a **voting mechanism**:

```python
# Collect signals from all strategies
strategy_signals = []  # Example: [1, 1, -1, 1, 0]

# Count votes
long_votes = sum(1 for s in strategy_signals if s == 1)   # 3 votes
short_votes = sum(1 for s in strategy_signals if s == -1)  # 1 vote
total_votes = len(strategy_signals)  # 5 strategies

# Calculate consensus
long_consensus = long_votes / total_votes   # 3/5 = 0.6 (60%)
short_consensus = short_votes / total_votes # 1/5 = 0.2 (20%)

# Check consensus threshold (default: 0.5 = 50%)
if long_consensus >= 0.5:
    return 1  # LONG (3/5 strategies agree)
elif short_consensus >= 0.5:
    return -1  # SHORT
else:
    return 0  # NEUTRAL (insufficient consensus)
```

**Specific Example:**

- **Strategies**: [5, 6, 7, 8, 9]
- **Signals from each strategy**: [1, 1, -1, 1, 0]
  - Strategy 5: LONG (1)
  - Strategy 6: LONG (1)
  - Strategy 7: SHORT (-1)
  - Strategy 8: LONG (1)
  - Strategy 9: NEUTRAL (0)
- **Votes**: 3 LONG, 1 SHORT, 1 NEUTRAL
- **Consensus**: 60% LONG → **Final Signal = 1 (LONG)**

---

## Complete Flow in Code

```python
def get_range_oscillator_signal(...):
    # 1. Fetch OHLCV data
    df = data_fetcher.fetch_ohlcv_with_fallback_exchange(...)
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    # 2. Run each strategy
    strategy_signals = []
    for strategy_num in [5, 6, 7, 8, 9]:
        signals, _ = generate_signals_strategyX(...)
        latest_signal = int(signals.dropna().iloc[-1])
        strategy_signals.append(latest_signal)
    
    # 3. Voting mechanism
    long_votes = sum(1 for s in strategy_signals if s == 1)
    short_votes = sum(1 for s in strategy_signals if s == -1)
    total_votes = len(strategy_signals)
    
    long_consensus = long_votes / total_votes
    short_consensus = short_votes / total_votes
    
    # 4. Check consensus threshold
    if long_consensus >= 0.5:
        return 1  # LONG
    elif short_consensus >= 0.5:
        return -1  # SHORT
    else:
        return 0  # NEUTRAL
```

---

## Summary

1. **Calculate Oscillator**: `oscillator = 100 * (close - MA) / RangeATR` → value from -100 to +100
2. **Apply Strategies**: Each strategy analyzes oscillator and generates signals (1/-1/0)
3. **Get Latest Signal**: Extract signal from the last bar of each strategy
4. **Voting**: Count votes and check consensus threshold
5. **Final Signal**: Return 1 (LONG), -1 (SHORT), or 0 (NEUTRAL)

**Benefits of voting mechanism:**

- ✅ Increases accuracy (multiple strategies confirm)
- ✅ Reduces false signals (requires consensus)
- ✅ Robust (if 1 strategy fails, other strategies still work)
- ✅ Flexible (can select strategies and threshold)

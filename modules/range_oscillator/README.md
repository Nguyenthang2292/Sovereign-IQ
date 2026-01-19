# Range Oscillator Module

Comprehensive Range Oscillator indicator implementation with heatmap visualization, ported from Pine Script (Zeiierman).

## Overview

The Range Oscillator is a technical indicator that measures price deviation from a weighted moving average, normalized by ATR-based range bands. It provides signals for trend identification, mean reversion, and momentum trading strategies.

## Features

- **Core Oscillator Calculation**: Weighted MA and ATR-based range calculation
- **Heatmap Visualization**: Color-coded heat zones based on oscillator value histograms
- **Multiple Trading Strategies**: 8+ signal generation strategies
- **Voting Mechanism**: Consensus-based signal combination
- **Performance Optimized**: Vectorized operations with numpy for speed

## Installation

The module is part of the crypto-probability project. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Oscillator Calculation

```python
import pandas as pd
from modules.range_oscillator.core.oscillator import calculate_range_oscillator

# Prepare price data
high = pd.Series([...])  # High prices
low = pd.Series([...])   # Low prices
close = pd.Series([...]) # Close prices

# Calculate oscillator
oscillator, ma, range_atr = calculate_range_oscillator(
    high=high,
    low=low,
    close=close,
    length=50,  # Minimum range length
    mult=2.0,   # Range width multiplier
)
```

### Oscillator with Heatmap Colors

```python
from modules.range_oscillator.core.oscillator import calculate_range_oscillator_with_heatmap
from modules.range_oscillator.config.heatmap_config import HeatmapConfig

# Calculate with heatmap
oscillator, ma, range_atr, heat_colors, trend_direction, osc_colors = calculate_range_oscillator_with_heatmap(
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

# heat_colors is a Series of hex color strings (e.g., "#09ff00")
# trend_direction is a Series with 1 (bullish), -1 (bearish), or 0 (neutral)
```

### Generate Trading Signals

```python
from modules.range_oscillator.strategies.combined import generate_signals_combined_all_strategy

signals, signal_strength = generate_signals_combined_all_strategy(
    high=high,
    low=low,
    close=close,
    length=50,
    mult=2.0,
)
```

## Core Components

### 1. Oscillator Calculation

The oscillator value is calculated as:

```text
oscillator = 100 * (close - weighted_ma) / range_atr
```

- **+100**: Price at upper bound (breakout above)
- **0**: Price at equilibrium (MA)
- **-100**: Price at lower bound (breakout below)

### 2. Heatmap Visualization

The heatmap feature visualizes "heat zones" based on oscillator value histograms:

- **Divides oscillator range** into configurable levels
- **Counts touches** per level in lookback window
- **Applies gradient coloring** based on touch frequency
- **Trend-based colors**: Different schemes for bullish vs bearish trends
- **Breakout Overrides**: Automatically applies strong trend colors when price breaks out of range bands
- **Trend-Flip Logic**: Resets to transition color immediately on trend direction change to avoid lag

#### Heatmap Configuration

```python
from modules.range_oscillator.config.heatmap_config import HeatmapConfig

config = HeatmapConfig(
    levels_inp=5,                    # Number of heat levels (2-100)
    heat_thresh=3,                   # Minimum touches per level
    lookback_bars=100,                # Lookback window for touch counting
    point_mode=True,                  # Use point mode (midpoint levels)
    weak_bullish_color="#008000",    # Weak bullish zones (green)
    strong_bullish_color="#09ff00",  # Strong bullish zones (bright green)
    weak_bearish_color="#800000",    # Weak bearish zones (maroon)
    strong_bearish_color="#ff0000",  # Strong bearish zones (red)
    transition_color="#0000ff",      # Transition/neutral color (blue)
)
```

#### Trend Direction

Trend direction is automatically calculated from close vs MA:

- **1 (Bullish)**: `close > ma`
- **-1 (Bearish)**: `close < ma`
- **0 (Neutral)**: `close == ma` (persists previous value)

### 3. Trading Strategies

The module includes multiple signal generation strategies:

- **Basic Strategy**: Simple threshold-based signals
- **Sustained Strategy**: Signals based on sustained oscillator levels
- **Crossover Strategy**: Zero-line and threshold crossovers
- **Momentum Strategy**: Rate of change in oscillator values
- **Breakout Strategy**: Extreme oscillator values
- **Divergence Strategy**: Price-oscillator divergence
- **Trend Following**: Trend-filtered signals
- **Mean Reversion**: Extreme value mean reversion
- **Combined Strategy**: Weighted voting of multiple strategies

See `OSCILLATOR_SIGNAL_EXPLANATION-en.md` for detailed strategy documentation.

## Performance

The module is optimized for performance with vectorized operations and memory-efficient data structures.

### Performance Optimizations

#### Trend Direction (`calculate_trend_direction`)

- **Memory**: Uses `int8` instead of `int64` (87.5% memory savings - 1 byte vs 8 bytes per value)
- **Speed**: Fully vectorized operations, no redundant reindexing
- **Efficiency**: `replace(0, np.nan).ffill()` pattern for trend persistence

#### Heatmap Colors (`calculate_heat_colors`)

- **Vectorization**: Numpy arrays instead of pandas Series in hot loops
- **Pre-computation**: Trend flips computed once (vectorized) instead of per bar
- **Touch Counting**: Matrix operations with numpy broadcasting (O(1) complexity)
- **Distance Calculation**: Vectorized nearest level finding using numpy argmin

### Performance Benchmarks

Run performance tests:

```bash
pytest tests/range_oscillator/test_heatmap_performance.py -v
```

**Performance Metrics** (on modern hardware):

#### Trend Direction Calculation

- **1000 bars**: < 0.1s
- **5000 bars**: < 0.5s
- **Memory**: 1 byte per value (int8)

#### Heatmap Color Calculation

- **1000 bars** (default config): < 5s
- **5000 bars** (default config): < 30s
- **1000 bars** (20 levels): < 10s
- **1000 bars** (500 lookback): < 8s

## API Reference

### Core Functions

#### `calculate_range_oscillator()`

Calculate Range Oscillator indicator.

```python
def calculate_range_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    length: int = 50,
    mult: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Returns: (oscillator, ma, range_atr)"""
```

#### `calculate_range_oscillator_with_heatmap()`

Calculate oscillator with heatmap colors.

```python
def calculate_range_oscillator_with_heatmap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    length: int = 50,
    mult: float = 2.0,
    heatmap_config: Optional[HeatmapConfig] = None,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Returns: (oscillator, ma, range_atr, heat_colors, trend_direction, osc_colors)"""
```

#### `calculate_trend_direction()`

Calculate trend direction for heatmap.

```python
def calculate_trend_direction(
    close: pd.Series,
    ma: pd.Series,
) -> pd.Series:
    """Returns: Series with 1 (bullish), -1 (bearish), or 0 (neutral)"""
```

#### `calculate_heat_colors()`

Calculate heatmap colors for oscillator values.

```python
def calculate_heat_colors(
    oscillator: pd.Series,
    trend_direction: pd.Series,
    config: Optional[HeatmapConfig] = None,
) -> pd.Series:
    """Returns: Series of hex color strings"""
```

## Configuration

### HeatmapConfig

```python
@dataclass
class HeatmapConfig:
    levels_inp: int = 2              # Number of heat levels (2-100)
    heat_thresh: int = 1             # Minimum touches per level (>= 1)
    lookback_bars: int = 100         # Lookback window (>= 1)
    point_mode: bool = True          # Use point mode
    weak_bullish_color: str = "#008000"
    strong_bullish_color: str = "#09ff00"
    weak_bearish_color: str = "#800000"
    strong_bearish_color: str = "#ff0000"
    transition_color: str = "#0000ff"
```

## Examples

### Example 1: Basic Usage

```python
import pandas as pd
from modules.range_oscillator.core.oscillator import calculate_range_oscillator

# Load your price data
df = pd.read_csv("price_data.csv")
high = df["high"]
low = df["low"]
close = df["close"]

# Calculate oscillator
oscillator, ma, range_atr = calculate_range_oscillator(
    high=high,
    low=low,
    close=close,
)

print(f"Current oscillator value: {oscillator.iloc[-1]:.2f}")
```

### Example 2: Heatmap Visualization

```python
from modules.range_oscillator.core.oscillator import calculate_range_oscillator_with_heatmap
from modules.range_oscillator.config.heatmap_config import HeatmapConfig

# Calculate with heatmap
oscillator, ma, range_atr, heat_colors, trend, osc_colors = calculate_range_oscillator_with_heatmap(
    high=high,
    low=low,
    close=close,
    heatmap_config=HeatmapConfig(levels_inp=5, heat_thresh=3),
)

# Use heat_colors for UI visualization
for i, (osc_val, color) in enumerate(zip(oscillator, heat_colors)):
    print(f"Bar {i}: Oscillator={osc_val:.2f}, Color={color}")
```

### Example 3: Signal Generation

```python
from modules.range_oscillator.strategies.combined import generate_signals_combined_all_strategy

signals, strength = generate_signals_combined_all_strategy(
    high=high,
    low=low,
    close=close,
    length=50,
    mult=2.0,
)

# Filter strong signals
strong_signals = signals[strength > 0.7]
print(f"Found {len(strong_signals)} strong signals")
```

## Testing

The module includes comprehensive test coverage with 50+ test cases for heatmap functionality.

### Test Suites

Run all tests:

```bash
pytest tests/range_oscillator/ -v
```

Run specific test suites:

```bash
# Core functionality
pytest tests/range_oscillator/test_core.py -v

# Heatmap functions (31 test cases)
pytest tests/range_oscillator/test_heatmap.py -v

# Performance benchmarks (8 test cases)
pytest tests/range_oscillator/test_heatmap_performance.py -v

# Validation tests (11 test cases)
pytest tests/range_oscillator/test_heatmap_validation.py -v

# Strategy tests
pytest tests/range_oscillator/test_strategy.py -v
```

### Test Coverage

#### Unit Tests (`test_heatmap.py`)

- ✅ 31 comprehensive test cases
- ✅ Configuration validation
- ✅ Function correctness
- ✅ Edge cases (NaN, empty, extreme values)
- ✅ Integration tests

#### Performance Benchmarks (`test_heatmap_performance.py`)

- ✅ 8 performance test cases
- ✅ Medium dataset (1000 bars)
- ✅ Large dataset (5000 bars)
- ✅ High levels configuration
- ✅ Long lookback configuration
- ✅ Memory efficiency validation
- ✅ Benchmark comparisons

#### Validation Tests (`test_heatmap_validation.py`)

- ✅ 11 validation test cases
- ✅ Pine Script logic matching
- ✅ Touch counting validation
- ✅ Gradient coloring validation
- ✅ Trend-based color validation
- ✅ Consistency checks
- ✅ Edge case handling

**Test Summary**: Total of 50 test cases, all passing

## Documentation

- **Signal Explanation**: `OSCILLATOR_SIGNAL_EXPLANATION-en.md` (English) / `OSCILLATOR_SIGNAL_EXPLANATION-vi.md` (Vietnamese)
- **Strategy Improvements**: `COMBINED_STRATEGY_IMPROVEMENTS-en.md`
- **Pine Script Source**: `source_pine.txt`

## Port Information

This module is a port of the Pine Script Range Oscillator indicator by Zeiierman.

- **Original License**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
- **Source**: See `source_pine.txt`
- **© Zeiierman**

### Pine Script Compatibility

The implementation closely follows the Pine Script logic from `source_pine.txt`:

- ✅ **Trend direction calculation**: Matches Pine Script `trendDir := close > ma ? 1 : close < ma ? -1 : nz(trendDir[1])`
- ✅ **Heatmap level calculation**: Divides range into `levelsInp` levels matching Pine Script
- ✅ **Touch counting logic**: Counts touches per level in lookback window (100 bars default)
- ✅ **Gradient coloring**: Maps touch count to color gradient (cold → hot) based on `heatThresh`
- ✅ **Color selection**: Finds nearest level to current oscillator value, matching Pine Script behavior

### Implementation Status

✅ **Complete**: All heatmap functionality has been implemented and tested.

#### Core Functions

- ✅ `calculate_trend_direction()`: Vectorized trend direction calculation
- ✅ `calculate_heat_colors()`: Heatmap color calculation with histogram analysis
- ✅ `calculate_range_oscillator_with_heatmap()`: Integrated oscillator + heatmap calculation
- ✅ `get_oscillator_data_with_heatmap()`: Utility function with heatmap support

#### Configuration

- ✅ `HeatmapConfig`: Comprehensive configuration class with validation
- ✅ Default values matching Pine Script implementation
- ✅ Color customization support

#### Code Quality

- ✅ **Linting**: No linter errors
- ✅ **Type Hints**: Full type annotations
- ✅ **Docstrings**: Comprehensive documentation
- ✅ **Error Handling**: Robust validation and error messages

## Future Enhancements

Optional improvements for future versions:

- [ ] Pine Script cross-validation tests (if Pine Script runtime available)
- [ ] Additional performance optimizations (numba JIT compilation)
- [ ] Caching for repeated calculations
- [ ] Parallel processing for large datasets

## Contributing

When adding new features:

1. Follow existing code style (PEP 8, type hints)
2. Add comprehensive tests
3. Update this README if adding new features
4. Run performance benchmarks for optimization changes

## Files Structure

### Core Modules

- `core/oscillator.py`: Main oscillator calculation
- `core/heatmap.py`: Heatmap color calculation
- `config/heatmap_config.py`: Heatmap configuration
- `utils/oscillator_data.py`: Utility functions

### Test Files

- `tests/range_oscillator/test_heatmap.py`: Unit tests (31 cases)
- `tests/range_oscillator/test_heatmap_performance.py`: Performance benchmarks (8 cases)
- `tests/range_oscillator/test_heatmap_validation.py`: Validation tests (11 cases)

## License

See project root LICENSE file.

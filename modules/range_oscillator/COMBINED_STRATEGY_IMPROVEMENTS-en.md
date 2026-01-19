# Strategy 5: Combined Strategy Improvements

## Overview

The file `modules/range_oscillator/strategies/combined.py` has been improved and extended with many new features.

**Note on Backward Compatibility:**

- ✅ Most features remain backward compatible (function signature, `use_*` flags, parameters, etc.)
- ⚠️ **Breaking Change**: `consensus_mode="majority"` and `"unanimous"` have been removed (see [Migration Guide](#migration-guide))

## New Features

### 1. **Support for All Strategies (2-9)**

- **Before**: Only supported 3 strategies (Sustained, Crossover, Momentum)
- **After**: Supports all 7 strategies:
- Strategy 2: Sustained Pressure
- Strategy 3: Zero Line Crossover
- Strategy 4: Momentum
- Strategy 6: Range Breakouts (NEW)
- Strategy 7: Divergence Detection (NEW)
- Strategy 8: Trend Following (NEW)
- Strategy 9: Mean Reversion (NEW)

### 2. **Multiple Consensus Modes**

- **"threshold"** (default): Requires a certain percentage of strategies to agree (via `consensus_threshold`)
- Default `consensus_threshold=0.5` (at least 50% of strategies must agree)
- Can be adjusted from 0.0 to 1.0 to require fewer/more strategies
- **"weighted"**: Weighted voting based on `strategy_weights`

### 3. **Weighting System**

- Allows setting different weights for each strategy
- Automatically normalizes weights
- Used with consensus mode "weighted"

### 4. **Signal Strength Filtering**
   - Parameter `min_signal_strength`: Only accept signals with strength >= threshold
   - Helps filter out weak signals, keeping only strong signals

### 5. **Flexible Strategy Selection**
   - Method 1: Use `enabled_strategies` list (e.g., `[2, 3, 4, 6]`)
   - Method 2: Use `use_*` flags (backward compatible)
   - Can combine both methods

### 6. **Strategy Contribution Statistics**
   - Parameter `return_strategy_stats=True`: Returns statistics about each strategy's contribution
   - Includes number of LONG/SHORT signals from each strategy
   - Useful for analysis and optimization

### 7. **Custom Parameters for Each Strategy**
   - All parameters of strategies can be customized
   - Examples: `breakout_upper_threshold`, `divergence_lookback_period`, etc.

### 8. **Dynamic Strategy Selection** (NEW)
   - Automatically selects strategies based on market conditions
   - Analyzes volatility, trend strength, range-bound vs trending
   - High volatility → Breakout, Divergence strategies
   - Trending market → Crossover, Momentum, Breakout, Trend Following
   - Range-bound market → Sustained, Divergence, Mean Reversion
   - ✅ **Can be used with Adaptive Weights**: Dynamic Selection chooses strategies, then Adaptive Weights adjusts their weights

### 9. **Adaptive Weights** (NEW - Improved)
   - Automatically adjusts weights based on **actual price movement accuracy** (not agreement with consensus)
   - Avoids circular logic and groupthink: Evaluates based on actual accuracy compared to the market
   - Logic:
     - **Accuracy (70%)**: If strategy generates LONG signal, check if price actually increases in next N bars
     - **Strength (30%)**: Average strength of correct signals
   - Only works with `consensus_mode="weighted"` and requires `close` prices
   - Automatically normalizes weights
   - ✅ **Can be used with Dynamic Selection**: Adaptive weights will adjust weights of strategies selected by Dynamic Selection

### 10. **Signal Confidence Score** (NEW)
   - Calculates confidence score (0.0 to 1.0) based on:
     - Agreement level: fraction of strategies that agree (60% weight)
     - Signal strength: average strength of agreeing strategies (40% weight)
   - Returns additional confidence score series when `return_confidence_score=True`

## Usage Examples

### Example 1: Basic Usage (Backward Compatible)
```python
signals, strength = generate_signals_combined_all_strategy(
    high=high, low=low, close=close,
    use_sustained=True,
    use_crossover=True,
    use_momentum=True
)
```

### Example 2: Use All Strategies (Threshold Mode - Default)
```python
signals, strength = generate_signals_combined_all_strategy(
    high=high, low=low, close=close,
    enabled_strategies=[2, 3, 4, 6, 7, 8, 9],  # All strategies
    # consensus_mode="threshold" is default, consensus_threshold=0.5
)
```

### Example 3: Threshold Mode with Custom Threshold
```python
signals, strength = generate_signals_combined_all_strategy(
    high=high, low=low, close=close,
    enabled_strategies=[2, 3, 4, 6, 7],
    consensus_mode="threshold",
    consensus_threshold=0.6  # At least 60% of strategies must agree
)
```

### Example 4: Weighted Voting
```python
signals, strength = generate_signals_combined_all_strategy(
    high=high, low=low, close=close,
    enabled_strategies=[2, 3, 4, 6],
    consensus_mode="weighted",
    strategy_weights={
        2: 1.0,  # Sustained: weight 1.0
        3: 1.5,  # Crossover: weight 1.5 (more important)
        4: 0.8,  # Momentum: weight 0.8
        6: 1.2,  # Breakout: weight 1.2
    }
)
```

### Example 5: Threshold Mode - Strict (Require Many Strategies)
```python
signals, strength = generate_signals_combined_all_strategy(
    high=high, low=low, close=close,
    enabled_strategies=[2, 3, 4, 6, 7, 8],
    consensus_mode="threshold",
    consensus_threshold=0.75  # At least 75% of strategies must agree (very strict)
)
```

### Example 6: Filter Signal Strength
```python
signals, strength = generate_signals_combined_all_strategy(
    high=high, low=low, close=close,
    enabled_strategies=[2, 3, 4],
    min_signal_strength=0.3  # Only accept signals with strength >= 0.3
)
```

### Example 7: Get Statistics
```python
signals, strength, stats = generate_signals_combined_all_strategy(
    high=high, low=low, close=close,
    enabled_strategies=[2, 3, 4, 6],
    return_strategy_stats=True
)

# stats will contain:
# {
#     2: {"name": "Sustained", "long_count": 10, "short_count": 5},
#     3: {"name": "Crossover", "long_count": 8, "short_count": 7},
#     ...
# }
```

### Example 8: Customize Parameters for Strategies
```python
signals, strength = generate_signals_combined_all_strategy(
    high=high, low=low, close=close,
    enabled_strategies=[2, 3, 4, 6],
    # Strategy 2 parameters
    min_bars_sustained=5,
    # Strategy 3 parameters
    confirmation_bars=3,
    # Strategy 4 parameters
    momentum_period=5,
    momentum_threshold=7.0,
    # Strategy 6 parameters
    breakout_upper_threshold=120.0,
    breakout_lower_threshold=-120.0,
)
```

### Example 9: Dynamic Strategy Selection
```python
signals, strength = generate_signals_combined_all_strategy(
    high=high, low=low, close=close,
    enabled_strategies=[2, 3, 4, 6, 7, 8, 9],  # All strategies
    enable_dynamic_selection=True,  # Enable dynamic selection
    dynamic_selection_lookback=20,  # Analyze last 20 bars
    dynamic_volatility_threshold=0.6,  # Threshold for high volatility
    dynamic_trend_threshold=0.5,  # Threshold for trending market
)
```

### Example 10: Adaptive Weights
```python
signals, strength = generate_signals_combined_all_strategy(
    high=high, low=low, close=close,
    enabled_strategies=[2, 3, 4, 6],
    consensus_mode="weighted",
    enable_adaptive_weights=True,  # Enable adaptive weights
    adaptive_performance_window=10,  # Calculate performance from last 10 bars
)
```

### Example 11: Confidence Score
```python
signals, strength, confidence = generate_signals_combined_all_strategy(
    high=high, low=low, close=close,
    enabled_strategies=[2, 3, 4, 6],
    return_confidence_score=True,  # Return confidence score
)

# confidence is a Series with values 0.0 to 1.0
# High value = many strategies agree and signal strength is high
```

### Example 12: Combine All Features (Including Dynamic Selection + Adaptive Weights)
```python
signals, strength, stats, confidence = generate_signals_combined_all_strategy(
    high=high, low=low, close=close,
    enabled_strategies=[2, 3, 4, 6, 7, 8, 9],
    # Dynamic selection: Automatically select strategies based on market conditions
    enable_dynamic_selection=True,
    dynamic_selection_lookback=20,
    # Adaptive weights: Automatically adjust weights of selected strategies
    consensus_mode="weighted",
    enable_adaptive_weights=True,
    adaptive_performance_window=10,
    # Confidence score
    return_confidence_score=True,
    # Stats
    return_strategy_stats=True,
)
```

**Important Note:**
- ✅ **Dynamic Selection and Adaptive Weights CAN be used together**
- Dynamic Selection chooses which strategies will be used (based on market conditions)
- Adaptive Weights adjusts weights of selected strategies (based on performance)
- Flow: Market Analysis → Dynamic Selection → Generate Signals → Adaptive Weights → Final Signals

## Backward Compatibility

### ✅ Features Still Backward Compatible

The following features remain **fully backward compatible** with no breaking changes:

- ✅ **Function signature**: Unchanged (except when using `return_strategy_stats=True`)
- ✅ **`use_*` flags**: `use_sustained`, `use_crossover`, `use_momentum` still work as before
- ✅ **Default parameters**: Maintains old behavior when not specified
- ✅ **Strategy parameters**: All strategy parameters (such as `min_bars_sustained`, `confirmation_bars`, etc.) remain compatible
- ✅ **Output format**: Signals and strength series still have the same format

### ⚠️ Breaking Changes

**⚠️ BREAKING CHANGE: `consensus_mode='majority'` and `'unanimous'` are no longer supported**

The following consensus modes have been **completely removed** and will raise `ValueError` if used:
- ❌ `consensus_mode="majority"` (no longer supported)
- ❌ `consensus_mode="unanimous"` (no longer supported)

**Consensus modes still supported:**
- ✅ `consensus_mode="threshold"` (default, with `consensus_threshold=0.5`)
- ✅ `consensus_mode="weighted"`

## Migration Guide

### Update Code Using Deprecated Consensus Modes

If your code uses `consensus_mode="majority"` or `"unanimous"`, you **must update** to avoid `ValueError`:

**Migration for `consensus_mode="majority"`:**
```python
# Old code (will raise ValueError)
consensus_mode="majority"  # ❌ No longer works

# New code (recommended)
consensus_mode="threshold"  # default
consensus_threshold=0.5     # = 50% of strategies must agree (equivalent to majority)
```

**Migration for `consensus_mode="unanimous"`:**
```python
# Old code (will raise ValueError)
consensus_mode="unanimous"  # ❌ No longer works

# New code (recommended)
consensus_mode="threshold"
consensus_threshold=1.0     # = 100% of strategies must agree (equivalent to unanimous)
```

**Note:**
- Code using `consensus_mode="majority"` or `"unanimous"` will raise `ValueError` immediately
- Must update code to use `consensus_mode="threshold"` with appropriate `consensus_threshold`
- If `consensus_mode` is not specified, default is `"threshold"` with `consensus_threshold=0.5`

## Performance Improvements

- Uses vectorized operations with NumPy
- Optimized memory usage
- Better error handling (graceful fallback when strategy fails)

## Error Handling

- Better parameter validation
- Graceful handling when strategy fails (skip and continue)
- Fallback to basic strategy if all strategies fail

## Documentation

- Detailed and complete docstrings
- Comments explaining logic
- Complete type hints

## Recent Improvements (Latest Updates)

### 1. **Removed Deprecated Consensus Modes**
   - ✅ Completely removed `"majority"` and `"unanimous"` (no backward compatibility)
   - ✅ Only supports `"threshold"` and `"weighted"`
   - ✅ Clear validation with error messages when using invalid values
   - ✅ Code using deprecated values will raise `ValueError` immediately

### 2. **Improved Threshold Voting Logic**
   - ✅ Uses `ceil(n * threshold)` to calculate min_agreement
   - ✅ With 4 strategies and threshold=0.5: requires >= 2 votes
   - ✅ Keeps check `long_votes > short_votes` to ensure NO_SIGNAL when votes are equal

### 3. **Improved Python Compatibility**
   - ✅ Removed `strict=True` in `zip()` for compatibility with Python < 3.10
   - ✅ Code now runs on Python 3.9+

### 4. **Improved Error Handling and Validation**
   - ✅ **adaptive_trend/equity.py**: Added full validation, logging, NaN handling
   - ✅ **adaptive_trend/layer1.py**: 
     - Fixed `weighted_signal` to preserve all indices (union instead of intersection)
     - Added validation, logging, edge case handling
   - ✅ **adaptive_trend/moving_averages.py**: 
     - Raise error immediately when MA calculation fails (don't return partial tuple)
     - Added validation, logging, error handling
   - ✅ **adaptive_trend/signals.py**: Added validation, logging, NaN and index alignment handling
   - ✅ **adaptive_trend/utils.py**: Added validation, logging, overflow handling
   - ✅ **adaptive_trend/scanner.py**: Added full validation, error tracking, summary logging

### 5. **Improved Code Quality**
   - ✅ All modules have complete input validation
   - ✅ Consistent logging from `modules.common.utils`
   - ✅ Clear error messages in English
   - ✅ Complete documentation with `Raises` sections
   - ✅ Better edge case handling (NaN, empty series, index mismatches)

### 6. **Improved Performance and Reliability**
   - ✅ Optimized code with list comprehensions instead of duplication
   - ✅ Overflow handling in exponential calculations
   - ✅ Automatic index alignment when needed
   - ✅ Early error detection and reporting

## Bug Fixes and Improvements

### Fixed Issues

1. **Weighted Signal Index Preservation** (`layer1.py`)
   - ✅ Fixed logic to preserve all indices from all pairs (union instead of intersection)
   - ✅ Avoids losing valid indices when pairs have different indices

2. **Threshold Voting Logic** (`combined.py`)
   - ✅ Uses `ceil(n * threshold)` to calculate min_agreement
   - ✅ Clearer and simpler logic

3. **Partial MA Tuple Handling** (`moving_averages.py`)
   - ✅ Raise error immediately when MA calculation fails
   - ✅ Avoids returning tuple with None values causing TypeError downstream

4. **Python Version Compatibility**
   - ✅ Removed `strict=True` for compatibility with Python < 3.10

5. **Removed Deprecated Values**
   - ✅ Completely removed `"majority"` and `"unanimous"`
   - ✅ Code using these values will raise `ValueError` immediately

## Planned Enhancements (Not Yet Implemented)

The following features are planned but not yet implemented:

1. **Strategy Performance Tracking**: Track performance of each strategy over time to evaluate long-term effectiveness
2. **Strategy Ensembles**: Combine multiple consensus modes simultaneously to create more complex voting methods

## Implemented Enhancements (Already Implemented)

The following features have been implemented and can be used immediately:

- ✅ **Dynamic Strategy Selection** (see section 8, lines 53-59)
- ✅ **Adaptive Weights** (see section 9, lines 61-69)
- ✅ **Signal Confidence Score** (see section 10, lines 71-75)

## Improved Modules

### Adaptive Trend Classification (ATC) Modules

#### 1. **equity.py**
- ✅ Full validation for all parameters
- ✅ Logging when NaN values, floor hits occur
- ✅ Automatic index alignment handling
- ✅ Error handling with try-except

#### 2. **layer1.py**
- ✅ Fixed `weighted_signal`: preserve all indices (union)
- ✅ Validation for all functions
- ✅ Logging for warnings and errors
- ✅ NaN and edge case handling

#### 3. **moving_averages.py**
- ✅ Raise error immediately when MA calculation fails
- ✅ Validation for lengths, robustness, ma_type
- ✅ Logging for warnings and errors
- ✅ Optimized code with list comprehensions

#### 4. **signals.py**
- ✅ Validation and index alignment
- ✅ NaN value handling
- ✅ Logging for warnings
- ✅ Conflict handling (both crossover and crossunder True)

#### 5. **utils.py**
- ✅ Validation for rate_of_change, diflen, exp_growth
- ✅ Overflow handling in exp_growth
- ✅ Ensures diflen does not return length <= 0
- ✅ Logging for warnings and errors

#### 6. **scanner.py**
- ✅ Full validation for all parameters
- ✅ Error tracking and skipped symbols
- ✅ Summary logging at end
- ✅ Data quality issue handling

### Range Oscillator Strategy Modules

#### 1. **combined.py**
- ✅ Completely removed "majority" and "unanimous" (no backward compatibility)
- ✅ Improved threshold voting logic
- ✅ Better validation and error handling
- ✅ Python compatibility (removed strict=True)

## Technical Details

### Code Quality Improvements

- **Validation**: All functions have complete input validation
- **Error Handling**: Try-except blocks with detailed logging
- **Type Safety**: Complete type hints and validation
- **Documentation**: Docstrings with `Raises` sections
- **Logging**: Consistent logging from `modules.common.utils`
- **Edge Cases**: Handling NaN, empty series, index mismatches

### Performance Optimizations

- **Vectorization**: Uses NumPy operations
- **Memory**: Optimized memory usage with proper dtype
- **Code Duplication**: Reduced duplication with list comprehensions
- **Early Validation**: Fail fast with early validation

### Compatibility

- **Python Version**: Compatible with Python 3.9+ (removed strict=True)
- **Breaking Changes**: `consensus_mode="majority"` and `"unanimous"` are no longer supported
- **Migration**: Use `consensus_mode="threshold"` with appropriate `consensus_threshold`

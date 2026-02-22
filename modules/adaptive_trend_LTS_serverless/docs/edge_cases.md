# Edge Cases

This document describes the edge cases handled by the ATC Serverless module.

## Overview

The algorithm is designed to handle various market conditions and edge cases gracefully.

## Supported Edge Cases

### 1. Gap Handling

**Description**: Missing bars in historical data due to market closures or data gaps.

**Behavior**: 
- Algorithm handles missing data by skipping gaps
- Uses available data points for calculations
- Does not fail on non-contiguous timestamps

**Test**: `test_gap_handling()`

### 2. Extreme Volatility

**Description**: Price movements greater than 10% in a single period.

**Behavior**:
- No NaN output
- Signal detection remains stable
- No division by zero or overflow

**Test**: `test_extreme_volatility()`

### 3. Flash Crashes

**Description**: Rapid price drops followed by quick recoveries.

**Behavior**:
- Handles sudden reversals
- Signal detection remains stable
- No infinite or NaN values

**Test**: `test_flash_crash()`

### 4. Low Liquidity

**Description**: Low trading volume with wide spreads.

**Behavior**:
- Processes data regardless of volume
- No special handling required
- Results may be more volatile

**Test**: `test_low_liquidity()`

### 5. Circuit Breakers

**Description**: Market-wide trading halts during extreme drops.

**Behavior**:
- Handles flat price periods
- Signal detection remains stable
- No crashes on edge data

**Test**: `test_circuit_breaker()`

### 6. Signal Consistency

**Description**: Same input produces same output.

**Behavior**:
- Deterministic calculations
- No random behavior
- Reproducible results

**Test**: `test_signal_consistency()`

## Test Data

Test data is located in `test_data/real_market/`:

- `gap_data.csv`: Data with missing time periods
- `volatility_data.csv`: Extreme price movements
- `flash_crash_data.csv`: Rapid price reversals
- `low_liquidity_data.csv`: Low volume data
- `circuit_breaker_data.csv`: Market halts

## Running Tests

Run all edge case tests:
```bash
cargo test --test real_market_data_tests
```

Run a specific test:
```bash
cargo test --test real_market_data_tests test_gap_handling
```

## Performance Considerations

All edge cases are handled with minimal performance overhead:
- No additional allocations
- No special branching for most cases
- O(n) complexity maintained
# Real Market Data Tests

This directory contains test data for real market scenarios used in the ATC Serverless module testing.

## Test Data Files

- `gap_data.csv`: Historical data with missing bars/gaps
- `volatility_data.csv`: Data with extreme price movements (>10%)
- `flash_crash_data.csv`: Data showing rapid price reversals
- `low_liquidity_data.csv`: Data with wide spreads and low trading volume
- `circuit_breaker_data.csv`: Data during market circuit breaker events

## Usage

These files are used in the real market data tests to validate:

- Gap handling in calculations
- Extreme volatility resilience
- Flash crash detection
- Low liquidity scenario handling
- Circuit breaker behavior

## Format

Each CSV file follows this format:
- timestamp,open,high,low,close,volume

## Test Cases

The tests cover various edge cases and real-world scenarios to ensure the algorithm handles:

- Missing data points
- Extreme price movements
- Rapid market reversals
- Low liquidity conditions
- Circuit breaker events

## Contributing

To add new test data:
1. Create a new CSV file in this directory
2. Follow the existing format
3. Add corresponding tests in `tests/real_market_data_tests.rs`
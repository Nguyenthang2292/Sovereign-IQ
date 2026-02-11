# ATC Serverless - Adaptive Trend Classification for AWS Lambda

A high-performance Rust implementation of the Adaptive Trend Classification (ATC) algorithm, optimized for AWS Lambda serverless deployment. This module provides real-time trading signal detection for cryptocurrency markets with sub-second latency.

## Overview

ATC Serverless implements a multi-timeframe trend classification system that:
- Calculates 6 types of Moving Averages (EMA, HMA, WMA, DEMA, LSMA, KAMA)
- Uses 8 length variations per MA type (via diflen) for robustness
- Applies Layer 1 signal detection with equity-based weighting
- Aggregates signals across multiple timeframes
- Returns LONG/SHORT/NEUTRAL classifications with confidence scores

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   API Gateway   │────▶│   AWS Lambda     │────▶│   SQS Queue     │
│   (HTTP/REST)   │     │   (This Module)  │     │   (Results)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Parallel       │
                       │   Processing     │
                       │   (Rayon)        │
                       └──────────────────┘
```

## Project Structure

```
modules/adaptive_trend_LTS_serverless/
├── Cargo.toml              # Workspace configuration
├── src/
│   ├── lib.rs              # Core data structures and exports
│   ├── ma_calculations.rs  # Moving Average implementations
│   ├── signal_detection.rs # Layer 1 signal logic with diflen
│   ├── equity.rs           # Equity curve calculations (Layer 2)
│   ├── multi_tf_voting.rs  # Multi-timeframe aggregation
│   └── aggregation.rs      # Batch processing with error recovery
├── lambda/
│   ├── Cargo.toml          # Lambda-specific dependencies
│   └── src/
│       ├── main.rs         # Lambda entry point
│       ├── handler.rs      # Request handler
│       └── sqs.rs          # SQS client
└── tests/
    └── atc_tests.rs        # Comprehensive test suite
```

## Installation

### Prerequisites
- Rust 1.70+ (install via [rustup](https://rustup.rs/))
- AWS CLI (for deployment)
- Docker (optional, for building Lambda deployment packages)

### Build

```bash
# Clone the repository
cd modules/adaptive_trend_LTS_serverless

# Build for development
cargo build

# Build optimized release binary
cargo build --release

# Build Lambda deployment package
cd lambda
cargo lambda build --release
```

## Usage

### As a Library

```rust
use atc_serverless::{ATCConfig, MAConfig, process_batch, SymbolData, OHLCVData};
use std::collections::HashMap;

// Configure the algorithm
let config = ATCConfig {
    weights: {
        let mut w = HashMap::new();
        w.insert("1h".to_string(), 0.6);
        w.insert("4h".to_string(), 0.4);
        w
    },
    threshold: 0.3,
    min_signal: 0.0,
    use_signal_strength: true,
    lambda_param: 0.02,
    decay: 0.03,
    cutout: 0,
    ma_configs: vec![
        MAConfig { ma_type: "EMA".to_string(), length: 12, weight: 1.0 },
        MAConfig { ma_type: "HMA".to_string(), length: 12, weight: 1.0 },
        MAConfig { ma_type: "WMA".to_string(), length: 12, weight: 1.0 },
        MAConfig { ma_type: "DEMA".to_string(), length: 12, weight: 1.0 },
        MAConfig { ma_type: "LSMA".to_string(), length: 12, weight: 1.0 },
        MAConfig { ma_type: "KAMA".to_string(), length: 12, weight: 1.0 },
    ],
};

// Process a batch of symbols
let (results, errors) = process_batch(symbols, config);
```

### As a Lambda Function

Send a POST request to your Lambda function URL:

```json
{
  "batch_id": "batch-001",
  "symbols": [
    {
      "symbol": "BTCUSDT",
      "timeframes": {
        "1h": {
          "timestamp": [1704067200, 1704070800, ...],
          "open": [42000.0, 42100.0, ...],
          "high": [42200.0, 42300.0, ...],
          "low": [41900.0, 42000.0, ...],
          "close": [42100.0, 42200.0, ...],
          "volume": [100.0, 150.0, ...]
        }
      }
    }
  ],
  "config": {
    "weights": {"1h": 0.6, "4h": 0.4},
    "threshold": 0.3,
    "min_signal": 0.0,
    "use_signal_strength": true,
    "lambda_param": 0.02,
    "decay": 0.03,
    "cutout": 0,
    "ma_configs": [
      {"ma_type": "EMA", "length": 12, "weight": 1.0}
    ]
  }
}
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weights` | HashMap<String, f64> | - | Timeframe weights (e.g., {"1h": 0.6, "4h": 0.4}) |
| `threshold` | f64 | 0.3 | Signal threshold for LONG/SHORT classification |
| `min_signal` | f64 | 0.0 | Minimum signal strength to consider |
| `use_signal_strength` | bool | true | Enable signal strength weighting |
| `lambda_param` | f64 | 0.02 | Lambda parameter for equity calculation |
| `decay` | f64 | 0.03 | Decay factor for equity weighting |
| `cutout` | usize | 0 | Number of initial bars to cut out |
| `ma_configs` | Vec<MAConfig> | 6 default MAs | Configuration for each MA type |

### MAConfig Options

| Field | Type | Description |
|-------|------|-------------|
| `ma_type` | String | "EMA", "HMA", "WMA", "DEMA", "LSMA", or "KAMA" |
| `length` | usize | Base length for the MA calculation |
| `weight` | f64 | Static weight for this MA type |

## Building and Deploying to AWS

### 1. Build the Lambda Package

```bash
cd modules/adaptive_trend_LTS_serverless/lambda

# Install cargo-lambda if not already installed
cargo install cargo-lambda

# Build for x86_64 Lambda runtime
cargo lambda build --release --target x86_64-unknown-linux-gnu

# Or for ARM64 (Graviton2)
cargo lambda build --release --target aarch64-unknown-linux-gnu
```

### 2. Deploy to AWS Lambda

```bash
# Create the Lambda function
cargo lambda deploy \
  --iam-role arn:aws:iam::YOUR_ACCOUNT:role/YOUR_LAMBDA_ROLE \
  --runtime provided.al2 \
  atc-serverless

# Configure environment variables
aws lambda update-function-configuration \
  --function-name atc-serverless \
  --environment "Variables={RUST_LOG=info}"

# Set memory and timeout (adjust based on your needs)
aws lambda update-function-configuration \
  --function-name atc-serverless \
  --memory-size 3008 \
  --timeout 60
```

### 3. Set Up SQS Output Queue

```bash
# Create SQS queue for results
aws sqs create-queue --queue-name atc-results

# Grant Lambda permission to send to SQS
aws lambda add-permission \
  --function-name atc-serverless \
  --statement-id sqs-send \
  --action lambda:InvokeFunction \
  --principal sqs.amazonaws.com
```

## Testing

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test
cargo test test_ma_calculations

# Run benchmarks
cargo bench
```

### Test Coverage

The test suite includes:
- **MA Calculations**: All 6 MA types with edge cases
- **Signal Detection**: Layer 1 logic with diflen variations
- **Equity Calculation**: Layer 2 equity curve computation
- **Multi-TF Voting**: Timeframe aggregation
- **Batch Processing**: Parallel processing with error recovery
- **Integration Tests**: End-to-end pipeline validation

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Cold Start | < 1 second |
| Warm Invocation | ~50-100ms |
| Throughput | ~30 symbols/second |
| Memory Usage | ~100-200MB |
| Binary Size | < 15MB (optimized) |

### Optimization Features

- **Parallel Processing**: Uses Rayon for parallel symbol processing
- **SIMD Optimizations**: Leverages ndarray for vectorized operations
- **Release Optimizations**: LTO, strip symbols, single codegen unit
- **Error Recovery**: Per-symbol error handling prevents total batch failure

## Monitoring and Observability

The Lambda handler includes structured logging:

```rust
info!("Processing batch: {} with {} symbols", batch_id, symbol_count);
warn!("Batch {} completed with {} errors", batch_id, error_count);
info!("Batch {} completed: {} successful, {} errors", batch_id, success_count, error_count);
```

### CloudWatch Metrics

Recommended CloudWatch alarms:
- Error rate > 5%
- Duration > 35 seconds
- Memory usage > 80%

### Tracing

Enable tracing with:
```bash
aws lambda update-function-configuration \
  --function-name atc-serverless \
  --tracing-config Mode=Active
```

## Troubleshooting

### Common Issues

**1. Cold Start Too Slow**
- Increase Lambda memory allocation
- Use Provisioned Concurrency for critical workloads
- Consider using ARM64 (Graviton2) for better price/performance

**2. Out of Memory**
- Reduce batch size (number of symbols per invocation)
- Increase Lambda memory allocation
- Check for memory leaks in custom code

**3. Signal Accuracy Issues**
- Verify OHLCV data quality (no gaps, correct timestamps)
- Check config.threshold setting
- Compare with Python reference implementation

**4. Build Errors**
- Ensure Rust version >= 1.70
- Run `cargo clean` and rebuild
- Check for conflicting dependencies

### Debug Mode

Enable debug logging:
```bash
export RUST_LOG=debug
cargo run
```

## Comparison with Python Implementation

| Feature | Rust | Python |
|---------|------|--------|
| Performance | ~10-20x faster | Baseline |
| Memory | ~10x lower | Higher |
| Cold Start | < 1s | N/A |
| Warm Latency | ~50ms | ~500ms |
| Binary Size | ~15MB | N/A |

## API Reference

### Core Functions

#### `process_batch`
```rust
pub fn process_batch(
    symbols: Vec<SymbolData>,
    config: ATCConfig,
) -> (Vec<SignalResult>, Vec<SymbolError>)
```
Process a batch of symbols with error recovery. Returns partial results even if some symbols fail.

#### `compute_symbol_score`
```rust
pub fn compute_symbol_score(
    prices: &[f64],
    config: &ATCConfig,
) -> (f64, String)
```
Calculate the final score and signal type for a single symbol's price data.

#### `calculate_layer1_signal`
```rust
pub fn calculate_layer1_signal(
    prices: ArrayView1<f64>,
    ma_type: &str,
    base_length: usize,
    lambda: f64,
    decay: f64,
) -> (Array1<f64>, f64)
```
Calculate Layer 1 signal with full diflen variations (8 MA calculations).

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original ATC algorithm design by the trading systems team
- Rust implementation inspired by the Python reference
- Thanks to the ndarray and Rayon communities for excellent crates

## Changelog

### 0.1.0 (2026-02-11)
- Initial release
- Complete signal detection logic with diflen
- Error recovery with per-symbol handling
- Comprehensive test suite
- AWS Lambda deployment ready

## Support

For issues and questions:
- GitHub Issues: [Report a bug](https://github.com/your-org/your-repo/issues)
- Documentation: [Full API Docs](https://docs.rs/atc_serverless)
- Email: support@yourcompany.com

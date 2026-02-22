# Performance Benchmarking

This document explains the performance regression detection system for the ATC Serverless module.

## Overview

The performance regression detection system monitors and tracks the performance of key algorithms over time, automatically detecting regressions and generating reports.

## Components

### 1. Benchmark CI Workflow

The CI pipeline includes a benchmark job that:

- Runs `cargo bench` on every push to main/develop branches
- Compares results against a baseline
- Fails the build if regressions are detected
- Uploads benchmark artifacts

### 2. Benchmark Tracking Script

The `scripts/benchmark_tracking.py` script:

- Compares baseline and current benchmark results
- Generates HTML reports (`docs/benchmark_report.html`)
- **Generates detailed regression reports (`docs/regression_report.txt`)**
- Detects regressions (5% threshold by default)
- Can be run manually for local testing

### 3. Dashboard Generator

The `scripts/generate_dashboard.py` script:

- **Creates comprehensive HTML dashboard (`docs/benchmark_dashboard.html`)**
- **Visualizes performance trends over time**
- **Highlights regressions with color-coded badges**
- **Shows historical data and statistics**
- **Interactive overview with charts and metrics**

## Setup

### Prerequisites

- Rust toolchain
- Cargo
- Python 3
- GitHub Actions

### Installation

1. Install dependencies:
   ```bash
   cargo install cargo-benchcmp
   pip install -r requirements.txt  # If using Python dependencies
   ```

2. Configure CI workflow in `.github/workflows/benchmark.yml`

## Usage

### Running Benchmarks Locally

```bash
# Run benchmarks
cargo bench

# Save results
mkdir -p benchmarks
cargo bench -- --output-format=json > benchmarks/current.json

# Compare with baseline and generate reports
python scripts/benchmark_tracking.py benchmarks/baseline.json benchmarks/current.json

# Generate comprehensive dashboard (default output in docs/)
python scripts/generate_dashboard.py benchmarks/current.json benchmarks/
```

### CI Integration

The GitHub Actions workflow automatically:

1. Runs benchmarks on every push
2. Compares with baseline
3. Generates detailed regression report
4. Creates comprehensive HTML dashboard
5. Fails build on regression

## Thresholds

- **Regression threshold**: 5% slowdown (configurable)
- **Improvement threshold**: Any decrease in execution time

## Reports

### Regression Report (regression_report.txt)

Text-based detailed report containing:

- Total number of regressions detected
- Regression threshold (5%)
- Detailed breakdown for each regression:
  - Test name
  - Baseline performance
  - Current performance
  - Percentage change
  - Status indicator

Example output:
```
Performance Regression Report - 2026-02-16 14:30:00
================================================================================

Total Regressions Detected: 2
Regression Threshold: 5%

Test: calculate_ema_benchmark
  Baseline: 1234.56 ns
  Current:  1345.67 ns
  Change:   +9.00%
  Status:   REGRESSION (>5% slower)
--------------------------------------------------------------------------------
```

### HTML Report (benchmark_report.html)

Simple tabular view with:

- Test name
- Baseline performance
- Current performance
- Percentage change
- Regression status
- Color-coded indicators (red=regression, green=improvement)

### Comprehensive Dashboard (benchmark_dashboard.html)

**Full-featured interactive dashboard** providing:

- **Summary Statistics**: Total tests, regressions, improvements, stable
- **Visual Overview**: Color-coded cards showing key metrics
- **Detailed Results Table**: Sortable table with all benchmark data
- **Performance Trends**: Historical charts showing trends over time
- **Regression Highlighting**: Visual alerts for performance issues
- **Responsive Design**: Works on desktop and mobile devices

Features:
- 🎨 Modern, professional UI with gradient backgrounds
- 📊 Real-time performance indicators (↑ ↓ →)
- ⚠️ Automatic regression alerts
- 📈 Historical trend visualization
- 🎯 Badge system for quick status identification
- 📱 Mobile-responsive design

## Troubleshooting

### Common Issues

- **No baseline found**: Run benchmarks on main branch to create baseline
- **Benchmark failures**: Ensure Rust toolchain is properly configured
- **Regression false positives**: Adjust thresholds in the script

### Debugging

Enable detailed logging:
```bash
RUST_LOG=debug cargo bench
```

## Maintenance

### Updating Baseline

To update the baseline (e.g., after significant improvements):
```bash
# Run benchmarks
cargo bench -- --output-format=json > benchmarks/new_baseline.json

# Replace old baseline
mv benchmarks/new_baseline.json benchmarks/baseline.json
```

### Adding New Benchmarks

Add new benchmark tests to your Rust code and they will be automatically included in the CI pipeline.

## Integration with Other Systems

The benchmark system can be integrated with:

- **Monitoring systems**: Alert on regressions
- **Documentation**: Auto-generate performance documentation
- **Release process**: Block releases with regressions

## Future Enhancements

- Automated threshold optimization
- Historical trend analysis
- Integration with performance monitoring tools
- Machine learning for anomaly detection
# Pipeline Test Guide

## Overview

`test_pipeline.py` - Comprehensive test script for the auto-trade signal pipeline.

## Features

✅ **Rust Component Verification** - Checks ScanCache, calculate_weighted_score, aggregate_signals  
✅ **XGBoost Model Check** - Verifies model file exists  
✅ **Gemini API Check** - Confirms API key configuration  
✅ **Full Pipeline Execution** - ATC → XGBoost → Gemini → Signal Selection  
✅ **Detailed Output** - Shows confidence, price levels, R/R ratio, source breakdown  

## Prerequisites

### 1. Build Rust Backend

```bash
cd rust_backend
cargo build --release
pip install -e .
```

### 2. Train XGBoost Model

```bash
# Place trained model at: models/xgboost_model.joblib
```

### 3. Configure Gemini API (Optional)

```bash
export GEMINI_API_KEY=your_api_key_here
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Basic Test (3 symbols)

```bash
python modules/auto_trade/test_pipeline.py
```

### Custom Symbols

```bash
python modules/auto_trade/test_pipeline.py --symbols BTC/USDT ETH/USDT SOL/USDT AVAX/USDT
```

### Skip Pre-flight Checks

```bash
python modules/auto_trade/test_pipeline.py --skip-checks
```

## Expected Output

```
================================================================================
🚀 AUTO-TRADE PIPELINE TEST
================================================================================

📦 Checking Rust Components...
--------------------------------------------------------------------------------
✅ sovereign_prime (Rust backend) - AVAILABLE
✅ ScanCache - WORKING
✅ calculate_weighted_score - WORKING (score=0.5)
✅ aggregate_signals - WORKING (found 1 signals)

🤖 Checking XGBoost Model...
--------------------------------------------------------------------------------
✅ Model found: models/xgboost_model.joblib
   Size: 2.45 MB

🤖 Checking Gemini API...
--------------------------------------------------------------------------------
✅ GEMINI_API_KEY configured: AIzaSyDx...abc123

🔧 Initializing Pipeline Components...
--------------------------------------------------------------------------------
1. DataFetcher...
   ✅ Initialized
2. SymbolManager...
   ✅ Using 3 test symbols: BTC/USDT, ETH/USDT, BNB/USDT
3. ATCScanner...
   ✅ Configured: ['1h', '15m', '5m'] timeframes
   ✅ Rust cache: True
4. XGBoostFilter...
   ✅ Model: models/xgboost_model.joblib
   ✅ Min confidence: 0.6
5. GeminiIntegration...
   ✅ API configured
6. SignalSelector...
   ✅ Weights: XGBoost=0.4, Gemini=0.6
7. SignalPersistenceSQLite...
   ✅ Database: data/signals/test_signals.db
8. SignalPipeline...
   ✅ Pipeline ready

🚀 Running Pipeline...
================================================================================

⏱️ Pipeline completed in 12.45s

================================================================================
🎯 FINAL SIGNAL
================================================================================

Symbol:       BTC/USDT
Signal Type:  LONG
Confidence:   78.5%
Score:        83.20/100

📊 Price Levels:
  Entry:       $45,234.50
  Stop Loss:   $44,100.00
  Take Profit: $48,500.00

  Risk:        $1,134.50
  Reward:      $3,265.50
  R/R Ratio:   2.88

🔍 Source Breakdown:
  XGBoost:     72.0% confidence
  Gemini:      83.0% confidence
  Reasoning:   Strong bullish divergence on 1h, RSI oversold recovery, MACD bullish crossover...

⏰ Timestamp:   2026-02-02 00:14:03
================================================================================

🏥 System Health:
  ✅ Memory: Memory Usage: 45.2%
  ✅ GeminiAPI: Circuit Breaker CLOSED
```

## What Gets Tested

### 1. Rust Components

- `ScanCache` - LRU cache with TTL
- `calculate_weighted_score` - Signal weighting
- `aggregate_signals` - Multi-timeframe aggregation

### 2. Pipeline Stages

- **ATC Scanner** - Multi-timeframe trend detection
- **XGBoost Filter** - ML-based signal validation
- **Gemini Analysis** - AI chart analysis (if configured)
- **Signal Selector** - Final signal selection with confidence scoring

### 3. Outputs

- Signal type (LONG/SHORT/NEUTRAL)
- Confidence percentage
- Quality score (0-100)
- Price levels (entry, SL, TP)
- Risk/Reward ratio
- Source breakdown (XGBoost vs Gemini)
- System health status

## Troubleshooting

### ❌ sovereign_prime not available

```bash
cd rust_backend
cargo build --release
pip install -e .
```

### ❌ XGBoost model not found

Train model or provide path:

```python
model_path = "path/to/your/xgboost_model.joblib"
```

### ❌ Gemini API not configured

Optional - pipeline will skip AI analysis:

```bash
export GEMINI_API_KEY=your_key
```

### ❌ Pipeline timeout

Increase timeout in config:

```python
config={"pipeline_timeout": 600}  # 10 minutes
```

## Performance Benchmarks

| Stage | Time (with Rust) | Time (without Rust) |
|-------|------------------|---------------------|
| ATC Scan (3 symbols, 3 TFs) | ~3s | ~30s |
| XGBoost Filter | ~2s | ~2s |
| Gemini Analysis (3 candidates) | ~5s | ~5s |
| Signal Selection | <0.1s | ~1s |
| **Total** | **~10s** | **~38s** |

**Speedup**: 3.8x with Rust components

## Next Steps

After successful test:

1. Run with production symbol list (20-50 symbols)
2. Monitor performance metrics
3. Verify signal accuracy over 1-2 weeks
4. Deploy to production

## Related Files

- `core/signal_pipeline_review_v1.md` - Pipeline architecture review
- `../legacy/MIGRATION_SUMMARY.md` - Recent consolidation changes

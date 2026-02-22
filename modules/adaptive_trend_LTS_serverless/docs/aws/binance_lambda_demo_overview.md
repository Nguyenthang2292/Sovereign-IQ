# Binance Lambda Demo - Tổng Quan

Tài liệu tổng hợp cho việc triển khai và testing ATC Serverless trên AWS Lambda với dữ liệu Binance.

---

## 📦 Deliverables

Dự án bao gồm các file sau:

### 1. Scripts

#### **`scripts/binance_lambda_demo.py`**

Script Python chính để:

- Fetch dữ liệu OHLCV từ Binance API
- Load tất cả symbols USDT trên Binance
- Invoke Lambda function với batch data
- Display kết quả signals với formatting đẹp
- Tính toán performance metrics

**Features:**

- ✅ Hỗ trợ multiple timeframes (1m, 5m, 15m, 1h, 4h, 1d)
- ✅ Configurable batch size
- ✅ Mock mode để test local không cần Lambda
- ✅ Retry logic với exponential backoff
- ✅ Detailed error handling
- ✅ Color-coded output
- ✅ Custom ATC configuration support

**Usage:**

```bash
python binance_lambda_demo.py --endpoint YOUR_URL --symbols 50 --timeframes 1h 4h
```

#### **`scripts/requirements.txt`**

Python dependencies:

- `requests>=2.31.0` - HTTP client cho Binance API và Lambda
- `boto3>=1.28.0` - AWS SDK (nếu dùng SQS)
- `urllib3>=2.0.0` - HTTP connection pooling

### 2. Documentation

#### **`../QUICK_START.md`**

Hướng dẫn nhanh 15 phút:

- Setup môi trường
- Build và deploy Lambda
- Run demo script
- Troubleshooting common issues
- Cost estimates

**Ideal cho:** Người mới bắt đầu, cần chạy nhanh demo

#### **`AWS_SETUP_DEPLOYMENT_GUIDE.md`**

Hướng dẫn chi tiết đầy đủ:

- Yêu cầu hệ thống
- Cài đặt tools
- Cấu hình AWS (IAM, SQS, CloudWatch)
- Build options và optimization
- Multiple deployment methods (cargo-lambda, AWS CLI, SAM)
- API Gateway setup
- Monitoring & logging setup
- Performance tuning
- Cost optimization
- CI/CD setup

**Ideal cho:** Production deployment, advanced users

---

## 🚀 Quick Start

### Bước 1: Install Dependencies

```powershell
# Install cargo-lambda
cargo install cargo-lambda

# Install Python deps
cd modules\adaptive_trend_LTS_serverless\scripts
pip install -r requirements.txt
```

### Bước 2: Configure AWS

```powershell
# Configure credentials
aws configure

# Create IAM role
aws iam create-role --role-name ATC-Lambda-Role --assume-role-policy-document file://trust-policy.json

# Note the Role ARN
```

### Bước 3: Build & Deploy

```powershell
cd modules\adaptive_trend_LTS_serverless\lambda

# Build
cargo lambda build --release --target x86_64-unknown-linux-musl

# Deploy
cargo lambda deploy atc-lambda --iam-role YOUR_ROLE_ARN
```

### Bước 4: Get Function URL

```powershell
# Create public URL
aws lambda create-function-url-config --function-name atc-lambda --auth-type NONE

# Add permission
aws lambda add-permission --function-name atc-lambda --statement-id FunctionURLAllowPublicAccess --action lambda:InvokeFunctionUrl --principal "*" --function-url-auth-type NONE

# Get URL
aws lambda get-function-url-config --function-name atc-lambda
```

### Bước 5: Run Demo

```powershell
cd ..\scripts

# Test with 10 symbols
python binance_lambda_demo.py --endpoint YOUR_FUNCTION_URL --symbols 10
```

---

## 📊 Demo Script Features

### Command Line Arguments

```bash
python binance_lambda_demo.py [OPTIONS]

Options:
  --endpoint URL          Lambda function URL (required for real invocation)
  --symbols N             Number of symbols to process (default: 10)
  --all-symbols           Process ALL Binance USDT pairs (~2000+)
  --timeframes TF [TF..] Timeframes to fetch (default: 1h 4h)
  --limit N               Candles per timeframe (default: 100)
  --details               Show detailed per-timeframe results
  --config PATH           Custom ATC configuration JSON
  --mock                  Use mock responses (test without Lambda)
```

### Example Outputs

#### Simple Run (10 symbols)

```powershell
python binance_lambda_demo.py --endpoint https://xxx.lambda-url.us-east-1.on.aws/ --symbols 10
```

Output:

```
2026-02-16 23:00:00 - INFO - Fetching symbols from Binance...
2026-02-16 23:00:01 - INFO - Found 2147 USDT trading pairs
2026-02-16 23:00:01 - INFO - Limited to 10 symbols
...
====================================================================================================
ATC SIGNAL RESULTS
====================================================================================================
#    Symbol       Signal     Score      Confidence  
----------------------------------------------------------------------------------------------------
1    BTCUSDT      LONG       0.6234     62.34%
2    ETHUSDT      SHORT      -0.5421    54.21%
...
====================================================================================================

Summary:
  Total Signals: 10
  LONG:    4 (40.0%)
  SHORT:   3 (30.0%)
  NEUTRAL: 3 (30.0%)

Performance:
  Processing Time: 2.14s
  Symbols/Second:  4.67
```

#### Detailed Output

```powershell
python binance_lambda_demo.py --endpoint YOUR_URL --symbols 5 --details
```

Shows per-timeframe signal strengths:

```
1    BTCUSDT      LONG       0.6234     62.34%
     └─ 1h: 0.6500
     └─ 4h: 0.5800
```

#### Custom Configuration

```powershell
# Create custom config
echo '{
  "weights": {"1h": 0.7, "4h": 0.3},
  "threshold": 0.5,
  "robustness": "Wide",
  "ma_configs": [
    {"ma_type": "EMA", "length": 20, "weight": 1.5},
    {"ma_type": "KAMA", "length": 20, "weight": 1.0}
  ]
}' > custom_config.json

# Use it
python binance_lambda_demo.py --endpoint YOUR_URL --symbols 10 --config custom_config.json
```

---

## 🏗️ Architecture

```
┌─────────────────────┐
│   Binance API       │
│   (OHLCV Data)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────┐
│   binance_lambda_demo.py        │
│   - Fetch symbols               │
│   - Fetch OHLCV data            │
│   - Format batch request        │
└──────────┬──────────────────────┘
           │ HTTP POST
           ▼
┌─────────────────────────────────┐
│   AWS Lambda Function           │
│   (atc-lambda)                  │
│   - Process batch (Rust)        │
│   - Calculate signals           │
│   - Return results              │
└──────────┬──────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│   Console Output                │
│   - Formatted table             │
│   - Performance metrics         │
│   - Summary statistics          │
└─────────────────────────────────┘
```

---

## 📈 Performance Characteristics

### Lambda Function

| Metric | Value | Notes |
|--------|-------|-------|
| Cold Start | < 1s | First invocation |
| Warm Latency | ~50-100ms | Subsequent invocations |
| Throughput | ~30-50 symbols/sec | @ 1024MB memory |
| Memory Usage | ~200-400MB | Depends on batch size |
| Max Batch Size | ~200 symbols | Before timeout risk |

### Binance API Fetching

| Metric | Value | Notes |
|--------|-------|-------|
| Rate Limit | 1200 requests/min | Weight-based |
| Fetch Time | ~0.2s/symbol/timeframe | With delays |
| Recommended Batch | 50-100 symbols | Good balance |

### End-to-End Demo

**Example: 50 symbols, 2 timeframes (1h, 4h)**

- Binance fetch: ~20s (100 requests)
- Lambda processing: ~2s
- Total: ~22s
- **Throughput: ~2.3 symbols/second**

---

## 💰 Cost Analysis

### AWS Lambda Free Tier

- **Requests:** 1 million/month
- **Compute:** 400,000 GB-seconds/month

**Typical testing usage:**

- 100 requests × 2s @ 1024MB = 200 GB-seconds
- **Cost: $0.00** (well within free tier)

### Production Usage Estimate

**Scenario:** 1 million requests/month, 50 symbols/request, 2s duration @ 1024MB

| Component | Cost |
|-----------|------|
| Lambda requests | $0.20 |
| Lambda compute | $8.00 |
| Data transfer | $1.00 |
| **Total** | **~$9.20/month** |

**Cost optimization tips:**

- Use ARM64 (Graviton2): ~20% cheaper
- Increase memory for faster execution: often cheaper overall
- Batch efficiently: 50-100 symbols optimal

---

## 🔍 Monitoring & Debugging

### CloudWatch Logs

```powershell
# View logs in real-time
aws logs tail /aws/lambda/atc-lambda --follow

# Filter errors
aws logs filter-log-events `
  --log-group-name /aws/lambda/atc-lambda `
  --filter-pattern "ERROR"
```

### Custom Metrics

Lambda code emits custom CloudWatch metrics:

- `SymbolsPerSecond` - Processing throughput
- `MemoryUsageMB` - Peak memory usage
- `MemoryDeltaMB` - Memory growth during processing
- `ErrorRate` - Percentage of failed symbols
- `ThreadCount` - Parallelism level

**View metrics:**

```powershell
aws cloudwatch get-metric-statistics `
  --namespace ATC/Serverless `
  --metric-name SymbolsPerSecond `
  --start-time 2026-02-16T00:00:00Z `
  --end-time 2026-02-16T23:59:59Z `
  --period 3600 `
  --statistics Average,Maximum
```

### Performance Dashboard

Generate HTML dashboard:

```powershell
cd modules\adaptive_trend_LTS_serverless
python scripts\generate_dashboard.py
```

Opens `docs/benchmark_dashboard.html` with:

- Processing time trends
- Memory usage patterns
- Error rate graphs
- Throughput analysis

---

## 🧪 Testing Scenarios

### 1. Smoke Test (Quick Validation)

```powershell
# Mock mode - no AWS needed
python binance_lambda_demo.py --mock --symbols 5

# Expected: Mock results in ~1s
```

### 2. Basic Test (Small Batch)

```powershell
# 10 symbols, 2 timeframes
python binance_lambda_demo.py --endpoint YOUR_URL --symbols 10 --timeframes 1h 4h

# Expected: Results in ~5-10s
```

### 3. Medium Load Test

```powershell
# 50 symbols, 2 timeframes
python binance_lambda_demo.py --endpoint YOUR_URL --symbols 50

# Expected: Results in ~20-30s
```

### 4. Heavy Load Test

```powershell
# 100 symbols, 3 timeframes
python binance_lambda_demo.py --endpoint YOUR_URL --symbols 100 --timeframes 1h 4h 1d

# Expected: Results in ~60-90s
```

### 5. Stress Test (All Symbols)

```powershell
# ALL Binance USDT pairs (~2000+)
python binance_lambda_demo.py --endpoint YOUR_URL --all-symbols --timeframes 1h

# Expected: 5-10 minutes, costs ~$0.50-$1.00
# WARNING: Only run this 1-2 times for testing!
```

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. `No module named 'requests'`

```powershell
pip install -r scripts\requirements.txt
```

#### 2. `Rate limit exceeded` (Binance)

```powershell
# Reduce batch size or add delays
python binance_lambda_demo.py --endpoint YOUR_URL --symbols 20  # Instead of 100
```

#### 3. Lambda timeout

```powershell
# Increase timeout and memory
aws lambda update-function-configuration `
  --function-name atc-lambda `
  --timeout 90 `
  --memory-size 2048
```

#### 4. `Unable to import module 'bootstrap'`

```powershell
# Re-build with correct target
cd lambda
cargo lambda build --release --target x86_64-unknown-linux-musl
cargo lambda deploy atc-lambda
```

#### 5. Cold start too slow

```powershell
# Option 1: Increase memory (faster CPU)
aws lambda update-function-configuration --function-name atc-lambda --memory-size 3008

# Option 2: Use Provisioned Concurrency (keeps function warm)
aws lambda put-provisioned-concurrency-config `
  --function-name atc-lambda `
  --provisioned-concurrent-executions 1
```

---

## 📚 Additional Resources

### Documentation Files

| File | Purpose | Audience |
|------|---------|----------|
| `QUICK_START.md` | 15-min setup guide | Beginners |
| `AWS_SETUP_DEPLOYMENT_GUIDE.md` | Comprehensive guide | Advanced users |
| `README.md` | Module overview | All users |
| `TESTING.md` | Test suite guide | Developers |
| `BENCHMARKING.md` | Performance analysis | Performance tuning |

### External Links

- [AWS Lambda Documentation](https://docs.aws.amazon.com/lambda/)
- [Cargo Lambda](https://www.cargo-lambda.info/)
- [Binance API Docs](https://binance-docs.github.io/apidocs/spot/en/)
- [Rust Lang Book](https://doc.rust-lang.org/book/)

---

## ✅ Checklist

Before running the demo, verify:

- [ ] Rust installed (`cargo --version`)
- [ ] cargo-lambda installed (`cargo lambda --version`)
- [ ] AWS CLI configured (`aws sts get-caller-identity`)
- [ ] Python 3.8+ installed (`python --version`)
- [ ] Python deps installed (`pip list | grep requests`)
- [ ] Lambda function deployed (`aws lambda get-function --function-name atc-lambda`)
- [ ] Function URL created (`aws lambda get-function-url-config --function-name atc-lambda`)

---

## 🎯 Next Steps

### For Learning

1. Read `QUICK_START.md`
2. Run smoke test with `--mock`
3. Deploy to AWS
4. Run basic test with 10 symbols
5. Experiment with different configurations

### For Development

- [AWS Setup & Deployment Guide](docs/aws/AWS_SETUP_DEPLOYMENT_GUIDE.md)
- [Module README](../../README.md)
- Setup CloudWatch monitoring
- Test with different batch sizes
- Optimize Lambda configuration
- Implement CI/CD

### For Production

1. Add authentication (API Gateway + API keys)
2. Setup rate limiting
3. Implement request queuing (SQS)
4. Configure auto-scaling
5. Setup alerting (SNS)
6. Multi-region deployment

---

## 📞 Support

For issues or questions:

- Check `docs/TROUBLESHOOTING.md`
- Review CloudWatch logs
- Check Lambda configuration
- Verify AWS permissions

---

**Built with ❤️ for high-performance serverless crypto trading**

*Last updated: 2026-02-16*

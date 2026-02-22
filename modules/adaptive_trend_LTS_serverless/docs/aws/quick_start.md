# Quick Start Guide - ATC Serverless Lambda Demo

Hướng dẫn nhanh để chạy demo ATC Serverless với Binance data.

---

## Tóm Tắt Nhanh

Dưới đây là các bước tối thiểu để chạy demo từ đầu đến cuối trong **15 phút**.

---

## Prerequisites

Đảm bảo đã cài đặt:

- ✅ Rust (1.70+): `cargo --version`
- ✅ AWS CLI v2: `aws --version`
- ✅ Python 3.8+: `python --version`
- ✅ AWS Account với credentials đã configure

---

## Bước 1: Install Tools (2 phút)

```powershell
# Install cargo-lambda
cargo install cargo-lambda

# Install Python dependencies
cd modules\adaptive_trend_LTS_serverless\scripts
pip install -r requirements.txt
```

---

## Bước 2: Setup AWS (3 phút)

```powershell
# Configure AWS credentials (chỉ cần làm 1 lần)
aws configure
# Nhập: Access Key ID, Secret Access Key, Region (us-east-1), Format (json)

# Tạo IAM role cho Lambda
aws iam create-role `
  --role-name ATC-Lambda-Role `
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "lambda.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

# Attach basic lambda policy
aws iam attach-role-policy `
  --role-name ATC-Lambda-Role `
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

# Lấy role ARN (save this!)
aws iam get-role --role-name ATC-Lambda-Role --query 'Role.Arn' --output text
```

**Lưu Role ARN**, ví dụ: `arn:aws:iam::123456789012:role/ATC-Lambda-Role`

---

## Bước 3: Build Lambda (3 phút)

```powershell
# Navigate to lambda directory
cd modules\adaptive_trend_LTS_serverless\lambda

# Build for Lambda
cargo lambda build --release --target x86_64-unknown-linux-musl
```

**Build time:** ~2-3 phút lần đầu tiên (sau đó nhanh hơn)

---

## Bước 4: Deploy Lambda (2 phút)

```powershell
# Deploy (thay YOUR_ROLE_ARN bằng ARN từ bước 2)
cargo lambda deploy atc-lambda `
  --iam-role YOUR_ROLE_ARN `
  --memory 1024 `
  --timeout 60 `
  --env-var RUST_LOG=info

# Tạo public function URL
aws lambda create-function-url-config `
  --function-name atc-lambda `
  --auth-type NONE

# Add permission
aws lambda add-permission `
  --function-name atc-lambda `
  --statement-id FunctionURLAllowPublicAccess `
  --action lambda:InvokeFunctionUrl `
  --principal "*" `
  --function-url-auth-type NONE

# Lấy function URL (save this!)
aws lambda get-function-url-config --function-name atc-lambda --query FunctionUrl --output text
```

**Lưu Function URL**, ví dụ: `https://abc123.lambda-url.us-east-1.on.aws/`

---

## Bước 5: Run Demo (1 phút)

```powershell
# Navigate to scripts directory
cd ..\scripts

# Test với mock data (không cần Lambda)
python binance_lambda_demo.py --mock --symbols 5

# Test với real Lambda (thay YOUR_FUNCTION_URL)
python binance_lambda_demo.py `
  --endpoint YOUR_FUNCTION_URL `
  --symbols 10 `
  --timeframes 1h 4h
```

---

## Expected Output

```
2026-02-16 23:00:00 - INFO - Fetching symbols from Binance...
2026-02-16 23:00:01 - INFO - Found 2000+ USDT trading pairs
2026-02-16 23:00:01 - INFO - Limited to 10 symbols
2026-02-16 23:00:02 - INFO - Fetching OHLCV data for ['1h', '4h'] timeframes...
2026-02-16 23:00:10 - INFO - Successfully fetched data for 10 symbols
2026-02-16 23:00:10 - INFO - Invoking Lambda function...
2026-02-16 23:00:12 - INFO - Lambda invocation successful: 10 succeeded, 0 errors

====================================================================================================
ATC SIGNAL RESULTS
====================================================================================================
#    Symbol       Signal     Score      Confidence  
----------------------------------------------------------------------------------------------------
1    BTCUSDT      LONG       0.6234     62.34%
2    ETHUSDT      SHORT      -0.5421    54.21%
3    BNBUSDT      LONG       0.4852     48.52%
4    SOLUSDT      NEUTRAL    0.1234     12.34%
5    ADAUSDT      LONG       0.3421     34.21%
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

---

## Common Issues & Solutions

### Issue: `cargo lambda: command not found`

```powershell
cargo install cargo-lambda
# Restart terminal sau khi install
```

### Issue: `No module named 'requests'`

```powershell
pip install -r requirements.txt
```

### Issue: `An error occurred (AccessDeniedException)`

AWS credentials chưa đúng hoặc thiếu quyền.

```powershell
# Re-configure
aws configure

# Verify
aws sts get-caller-identity
```

### Issue: Lambda timeout

```powershell
# Increase timeout và memory
aws lambda update-function-configuration `
  --function-name atc-lambda `
  --timeout 90 `
  --memory-size 2048
```

---

## Advanced Usage Examples

### Test với nhiều symbols hơn

```powershell
# 50 symbols
python binance_lambda_demo.py --endpoint YOUR_URL --symbols 50

# 100 symbols
python binance_lambda_demo.py --endpoint YOUR_URL --symbols 100
```

### Test với nhiều timeframes

```powershell
python binance_lambda_demo.py `
  --endpoint YOUR_URL `
  --symbols 20 `
  --timeframes 1h 4h 1d
```

### Show detailed results

```powershell
python binance_lambda_demo.py `
  --endpoint YOUR_URL `
  --symbols 10 `
  --details
```

### Custom configuration

```powershell
# Tạo config file
echo '{
  "weights": {"1h": 0.7, "4h": 0.3},
  "threshold": 0.4,
  "robustness": "Wide"
}' > custom_config.json

# Use custom config
python binance_lambda_demo.py `
  --endpoint YOUR_URL `
  --symbols 10 `
  --config custom_config.json
```

---

## Test ALL Binance Symbols (Cẩn Thận!)

⚠️ **WARNING:** Binance có ~2000 USDT pairs. Processing tất cả có thể tốn:

- Lambda execution time: ~5-10 phút
- Cost: ~$0.50 - $1.00 (vượt free tier)
- Binance rate limit risk

```powershell
# Process ALL symbols (chỉ nên test 1-2 lần)
python binance_lambda_demo.py `
  --endpoint YOUR_URL `
  --all-symbols `
  --timeframes 1h
```

**Khuyến nghị:**

- Test với 10-50 symbols trước
- Tăng dần lên 100, 200
- Production: batch 50-100 symbols/request

---

## Next Steps

1. **Read Full Documentation:**
   - [AWS Setup & Deployment Guide](../docs/aws/AWS_SETUP_DEPLOYMENT_GUIDE.md)
   - [Module README](../README.md)

2. **Setup Monitoring:**
   - CloudWatch Logs
   - CloudWatch Alarms
   - Performance Dashboard

3. **Production Hardening:**
   - Add authentication
   - Setup API Gateway với rate limiting
   - Implement error handling & retries

4. **Integration:**
   - Connect với trading bot
   - Setup automated scanning
   - Build notification system

---

## Cleanup (Xóa Resources)

Nếu muốn xóa tất cả để không tốn phí:

```powershell
# Delete Lambda function
aws lambda delete-function --function-name atc-lambda

# Delete IAM role (phải detach policies trước)
aws iam detach-role-policy `
  --role-name ATC-Lambda-Role `
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam delete-role --role-name ATC-Lambda-Role
```

---

## Cost Estimate

**Free Tier** (đủ cho testing):

- 1 million Lambda requests/month
- 400,000 GB-seconds compute time/month

**Test 100 requests với 50 symbols:**

- Lambda invocations: 100
- Duration: ~2s @ 1024MB = 200 GB-seconds
- **Cost: $0.00** (trong free tier)

**Production usage (1M requests/month, 50 symbols/request):**

- Lambda cost: ~$5-10/month
- Data transfer: ~$1-2/month
- **Total: ~$6-12/month**

---

**Happy Trading! 🚀**

# AWS Setup & Deployment Guide - ATC Serverless

Hướng dẫn chi tiết triển khai module ATC Serverless lên AWS Lambda, từ setup môi trường đến deployment và testing.

---

## Mục Lục

1. [Yêu Cầu Hệ Thống](#yêu-cầu-hệ-thống)
2. [Cài Đặt Môi Trường](#cài-đặt-môi-trường)
3. [Cấu Hình AWS](#cấu-hình-aws)
4. [Build Lambda Function](#build-lambda-function)
5. [Automated Deployment (Recommended)](#automated-deployment-recommended)
6. [Deploy to AWS Lambda (Manual)](#deploy-to-aws-lambda-manual)
7. [Cấu Hình API Gateway](#cấu-hình-api-gateway)
8. [Testing với Binance Data](#testing-với-binance-data)
9. [Monitoring & Logging](#monitoring--logging)
10. [Troubleshooting](#troubleshooting)
11. [Chi Phí & Tối Ưu Hóa](#chi-phí--tối-ưu-hóa)

---

## Yêu Cầu Hệ Thống

### Phần Mềm Cần Thiết

- **Rust** 1.70 hoặc mới hơn ([rustup](https://rustup.rs/))
- **AWS CLI** v2 ([hướng dẫn cài đặt](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html))
- **Python** 3.8+ (cho demo script)
- **Docker** (optional, để build trong container)
- **cargo-lambda** (CLI tool cho Rust Lambda)

### AWS Account Requirements

- AWS Account với quyền admin hoặc quyền:
  - Lambda: CreateFunction, UpdateFunctionCode, InvokeFunction
  - IAM: CreateRole, AttachRolePolicy
  - SQS: CreateQueue, SendMessage
  - CloudWatch: PutMetricData, CreateLogGroup
  - API Gateway: CreateRestApi (nếu dùng API Gateway)

---

## Cài Đặt Môi Trường

### 1. Install Rust và Cargo Lambda

```powershell
# Install Rust (nếu chưa có)
# Tải từ https://rustup.rs/ và chạy installer

# Verify installation
rustc --version
cargo --version

# Install cargo-lambda
cargo install cargo-lambda

# Verify cargo-lambda
cargo lambda --version
```

### 2. Install AWS CLI

```powershell
# Download AWS CLI v2 installer
# https://awscli.amazonaws.com/AWSCLIV2.msi

# Sau khi install, verify
aws --version

# Nên thấy: aws-cli/2.x.x ...
```

### 3. Install Python Dependencies

```powershell
# Navigate to module directory
cd modules\adaptive_trend_LTS_serverless

# Install Python dependencies cho demo script
pip install requests boto3

# Hoặc sử dụng requirements file
pip install -r scripts\requirements.txt
```

**File `scripts/requirements.txt`:**

```
requests>=2.31.0
boto3>=1.28.0
urllib3>=2.0.0
```

---

## Cấu Hình AWS

### 1. Configure AWS Credentials

```powershell
# Configure AWS CLI với credentials
aws configure

# Nhập thông tin:
# AWS Access Key ID: YOUR_ACCESS_KEY
# AWS Secret Access Key: YOUR_SECRET_KEY
# Default region: us-east-1
# Default output format: json
```

**Kiểm tra cấu hình:**

```powershell
aws sts get-caller-identity
```

Expected output:

```json
{
    "UserId": "AIDACKCEVSQ6C2EXAMPLE",
    "Account": "123456789012",
    "Arn": "arn:aws:iam::123456789012:user/YourUser"
}
```

### 2. Tạo IAM Role cho Lambda

Lambda cần IAM role với các quyền cần thiết.

**Tạo trust policy file** (`lambda-trust-policy.json`):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

**Tạo role:**

```powershell
# Tạo IAM role
aws iam create-role `
  --role-name ATC-Lambda-ExecutionRole `
  --assume-role-policy-document file://lambda-trust-policy.json

# Attach basic Lambda execution policy
aws iam attach-role-policy `
  --role-name ATC-Lambda-ExecutionRole `
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

# Attach SQS policy (để gửi kết quả)
aws iam attach-role-policy `
  --role-name ATC-Lambda-ExecutionRole `
  --policy-arn arn:aws:iam::aws:policy/AmazonSQSFullAccess
```

**Lấy ARN của role (cần dùng sau):**

```powershell
aws iam get-role --role-name ATC-Lambda-ExecutionRole --query 'Role.Arn' --output text
```

Lưu ARN này (dạng: `arn:aws:iam::123456789012:role/ATC-Lambda-ExecutionRole`)

### 3. Tạo SQS Queue cho Results

```powershell
# Tạo queue để nhận kết quả
aws sqs create-queue --queue-name atc-results

# Lấy queue URL
aws sqs get-queue-url --queue-name atc-results
```

**Output:**

```json
{
    "QueueUrl": "https://sqs.us-east-1.amazonaws.com/123456789012/atc-results"
}
```

Lưu Queue URL này.

---

## Build Lambda Function

### 1. Build với Cargo Lambda

```powershell
# Navigate to lambda directory
cd modules\adaptive_trend_LTS_serverless\lambda

# Build for x86_64 (Intel processors)
cargo lambda build --release --target x86_64-unknown-linux-musl

# Hoặc build for ARM64 (Graviton2 - rẻ hơn ~20%)
cargo lambda build --release --target aarch64-unknown-linux-gnu
```

**Build output** sẽ nằm trong:

- `target/lambda/atc-lambda/bootstrap.zip` (x86_64)
- Hoặc architecture tương ứng

### 2. Build Options và Optimization

**Để build tối ưu nhất:**

```powershell
# Full optimization build
cargo lambda build --release `
  --target x86_64-unknown-linux-musl `
  --features simd

# Nếu build bị lỗi với SIMD, bỏ feature này
cargo lambda build --release `
  --target x86_64-unknown-linux-musl
```

**Binary size sau build:** ~10-15MB (compressed)

### 3. Test Local (Optional)

```powershell
# Start local Lambda runtime
cargo lambda watch

# Trong terminal khác, invoke local function
cargo lambda invoke atc-lambda --data-file ../test_data_120.json
```

---

## Automated Deployment (Recommended)

Module cung cấp script tự động hóa toàn bộ quá trình build và deploy:

```powershell
# Chạy script deploy
python scripts/deploy_lambda.py

# Script sẽ tự động:
# 1. Kiểm tra môi trường (Rust, Cargo Lambda)
# 2. Tạo IAM Role và SQS Queue (nếu chưa có)
# 3. Build Lambda binaries (release mode)
# 4. Deploy function lên AWS
```

If you encounter build errors on Windows (e.g., linker not found), follow these steps:

1. Ensure **Visual Studio Build Tools** with "Desktop development with C++" is installed.
2. Open **"x64 Native Tools Command Prompt for VS 2022"** (search in Start Menu).
3. Run the deployment script from that terminal.

---

## Deploy to AWS Lambda (Manual)

### Phương Án 1: Deploy với Cargo Lambda (Khuyên Dùng)

```powershell
# Deploy trực tiếp
cargo lambda deploy atc-lambda `
  --iam-role arn:aws:iam::YOUR_ACCOUNT_ID:role/ATC-Lambda-ExecutionRole `
  --env-var RUST_LOG=info `
  --env-var SQS_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/YOUR_ACCOUNT_ID/atc-results `
  --memory 1024 `
  --timeout 60

# Thay YOUR_ACCOUNT_ID bằng AWS account ID của bạn
```

### Phương Án 2: Deploy với AWS CLI

**Nếu đã build ở bước trước:**

```powershell
# Tạo function
aws lambda create-function `
  --function-name atc-lambda `
  --runtime provided.al2 `
  --handler bootstrap `
  --architectures x86_64 `
  --role arn:aws:iam::YOUR_ACCOUNT_ID:role/ATC-Lambda-ExecutionRole `
  --zip-file fileb://target/lambda/atc-lambda/bootstrap.zip `
  --timeout 60 `
  --memory-size 1024 `
  --environment Variables={RUST_LOG=info,SQS_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/YOUR_ACCOUNT_ID/atc-results}
```

### Phương Án 3: Deploy với SAM (Infrastructure as Code)

Module đã có sẵn `template.yaml` cho AWS SAM deployment.

```powershell
# Install AWS SAM CLI (nếu chưa có)
# https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html

# Build và deploy
cd modules\adaptive_trend_LTS_serverless

sam build

sam deploy --guided
```

**SAM sẽ hỏi:**

- Stack name: `atc-serverless-stack`
- AWS Region: `us-east-1`
- Confirm changes: `Y`
- Allow SAM CLI IAM role creation: `Y`
- Save arguments to config file: `Y`

### Verify Deployment

```powershell
# Check function exists
aws lambda get-function --function-name atc-lambda

# Expected output: Function configuration details
```

---

## Cấu Hình API Gateway

Để có thể gọi Lambda qua HTTP endpoint.

### 1. Tạo Function URL (Cách Đơn Giản Nhất)

```powershell
# Tạo public function URL
aws lambda create-function-url-config `
  --function-name atc-lambda `
  --auth-type NONE `
  --cors "AllowOrigins=*,AllowMethods=POST,AllowHeaders=content-type"

# Thêm quyền invoke public
aws lambda add-permission `
  --function-name atc-lambda `
  --statement-id FunctionURLAllowPublicAccess `
  --action lambda:InvokeFunctionUrl `
  --principal "*" `
  --function-url-auth-type NONE
```

**Lấy Function URL:**

```powershell
aws lambda get-function-url-config --function-name atc-lambda
```

**Output:**

```json
{
    "FunctionUrl": "https://abcdefg12345.lambda-url.us-east-1.on.aws/",
    "AuthType": "NONE",
    ...
}
```

Lưu URL này để test.

### 2. Tạo API Gateway REST API (Production Setup)

Cho phép rate limiting, custom domains, API keys, etc.

```powershell
# Tạo REST API
aws apigateway create-rest-api `
  --name "ATC Serverless API" `
  --description "API for ATC signal processing"

# Lấy API ID từ output
$API_ID = "your-api-id"

# Lấy root resource ID
aws apigateway get-resources --rest-api-id $API_ID

# Create resource /scan
aws apigateway create-resource `
  --rest-api-id $API_ID `
  --parent-id ROOT_RESOURCE_ID `
  --path-part scan

# Tạo POST method
aws apigateway put-method `
  --rest-api-id $API_ID `
  --resource-id RESOURCE_ID `
  --http-method POST `
  --authorization-type NONE

# Integrate với Lambda
aws apigateway put-integration `
  --rest-api-id $API_ID `
  --resource-id RESOURCE_ID `
  --http-method POST `
  --type AWS_PROXY `
  --integration-http-method POST `
  --uri arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-1:YOUR_ACCOUNT_ID:function:atc-lambda/invocations

# Deploy API
aws apigateway create-deployment `
  --rest-api-id $API_ID `
  --stage-name prod
```

**Endpoint URL:** `https://{API_ID}.execute-api.us-east-1.amazonaws.com/prod/scan`

---

## Testing với Binance Data

### 1. Chạy Demo Script

```powershell
# Navigate to scripts directory
cd modules\adaptive_trend_LTS_serverless\scripts

# Test với 10 symbols (mock mode - không cần Lambda)
python binance_lambda_demo.py --mock --symbols 10

# Test với Lambda endpoint (sau khi deploy)
python binance_lambda_demo.py `
  --endpoint https://YOUR_FUNCTION_URL.lambda-url.us-east-1.on.aws/ `
  --symbols 20 `
  --timeframes 1h 4h
```

### 2. Test Parameters

```powershell
# Xử lý TẤT CẢ symbols trên Binance (cẩn thận - rất nhiều!)
python binance_lambda_demo.py `
  --endpoint YOUR_ENDPOINT `
  --all-symbols

# Chỉ xử lý các symbols cụ thể
python binance_lambda_demo.py `
  --endpoint YOUR_ENDPOINT `
  --symbols 50 `
  --timeframes 1h 4h 1d

# Show detailed results per timeframe
python binance_lambda_demo.py `
  --endpoint YOUR_ENDPOINT `
  --symbols 10 `
  --details

# Sử dụng custom config
python binance_lambda_demo.py `
  --endpoint YOUR_ENDPOINT `
  --symbols 10 `
  --config custom_config.json
```

### 3. Custom Configuration File

Tạo `custom_config.json`:

```json
{
  "weights": {
    "1h": 0.5,
    "4h": 0.3,
    "1d": 0.2
  },
  "threshold": 0.4,
  "min_signal": 0.1,
  "use_signal_strength": true,
  "lambda_param": 0.025,
  "decay": 0.035,
  "robustness": "Wide",
  "ma_configs": [
    { "ma_type": "EMA", "length": 20, "weight": 1.5 },
    { "ma_type": "HMA", "length": 20, "weight": 1.0 },
    { "ma_type": "KAMA", "length": 20, "weight": 1.2 }
  ]
}
```

### 4. Expected Output

```
================================================================================
ATC SIGNAL RESULTS
================================================================================
#    Symbol       Signal     Score      Confidence  
--------------------------------------------------------------------------------
1    BTCUSDT      LONG       0.6234     62.34%
2    ETHUSDT      SHORT      -0.5421    54.21%
3    BNBUSDT      LONG       0.4852     48.52%
4    SOLUSDT      NEUTRAL    0.1234     12.34%
...
================================================================================

Summary:
  Total Signals: 20
  LONG:    8 (40.0%)
  SHORT:   6 (30.0%)
  NEUTRAL: 6 (30.0%)

Performance:
  Processing Time: 2.34s
  Symbols/Second:  8.55
```

---

## Monitoring & Logging

### 1. CloudWatch Logs

```powershell
# View recent logs
aws logs tail /aws/lambda/atc-lambda --follow

# View logs for specific time range
aws logs filter-log-events `
  --log-group-name /aws/lambda/atc-lambda `
  --start-time (Get-Date).AddHours(-1).ToUniversalTime().ToString("s") `
  --filter-pattern "ERROR"
```

### 2. CloudWatch Metrics

Lambda tự động gửi metrics:

- **Invocations**: Số lần invoke
- **Duration**: Thời gian xử lý
- **Errors**: Số lỗi
- **Throttles**: Số lần bị throttle

**Custom Metrics** từ code:

```powershell
# View custom metrics
aws cloudwatch get-metric-statistics `
  --namespace ATC/Serverless `
  --metric-name SymbolsPerSecond `
  --start-time (Get-Date).AddHours(-1).ToUniversalTime().ToString("s") `
  --end-time (Get-Date).ToUniversalTime().ToString("s") `
  --period 300 `
  --statistics Average
```

### 3. Setup CloudWatch Alarms

Module đã có script tự động tạo alarms:

```powershell
# Run alarm setup script
.\scripts\setup_cloudwatch_alarms.ps1
```

Hoặc tạo manual:

```powershell
# Alarm cho error rate cao
aws cloudwatch put-metric-alarm `
  --alarm-name atc-lambda-high-errors `
  --alarm-description "Alert when error rate > 5%" `
  --metric-name Errors `
  --namespace AWS/Lambda `
  --statistic Sum `
  --period 300 `
  --threshold 5 `
  --comparison-operator GreaterThanThreshold `
  --evaluation-periods 1
```

### 4. Performance Dashboard

Generate dashboard HTML:

```powershell
cd modules\adaptive_trend_LTS_serverless

python scripts\generate_dashboard.py
```

Mở `docs/benchmark_dashboard.html` trong browser.

---

## Troubleshooting

### Lỗi: "No module named 'requests'"

```powershell
pip install requests boto3
```

### Lỗi: "Unable to import module 'bootstrap'"

Lambda runtime không tìm thấy binary.

**Giải pháp:**

1. Verify build target architecture match với Lambda config
2. Re-build với correct target:

   ```powershell
   cargo lambda build --release --target x86_64-unknown-linux-musl
   ```

### Lỗi: "Task timed out after 3.00 seconds"

Default timeout quá ngắn.

**Giải pháp:**

```powershell
aws lambda update-function-configuration `
  --function-name atc-lambda `
  --timeout 60
```

### Lỗi: "Memory Size: 128 MB Max Memory Used: 512 MB"

Out of memory error.

**Giải pháp:**

```powershell
aws lambda update-function-configuration `
  --function-name atc-lambda `
  --memory-size 1024
```

### Lỗi: "Rate exceeded" từ Binance

Binance API có rate limit. Demo script đã có delay 0.1s giữa các requests.

**Giải pháp:**

- Reduce --symbols count
- Add longer delays trong code

### Lambda Cold Start Quá Lâu

**Giải pháp:**

1. **Tăng memory** (CPU scale theo memory):

   ```powershell
   aws lambda update-function-configuration `
     --function-name atc-lambda `
     --memory-size 3008
   ```

2. **Sử dụng Provisioned Concurrency**:

   ```powershell
   aws lambda put-provisioned-concurrency-config `
     --function-name atc-lambda `
     --provisioned-concurrent-executions 1 `
     --qualifier prod
   ```

3. **Chuyển sang ARM64 (Graviton2)**:
   - Faster performance
   - Lower cost (~20% cheaper)

   ```powershell
   cargo lambda build --release --target aarch64-unknown-linux-gnu
   
   aws lambda update-function-configuration `
     --function-name atc-lambda `
     --architectures arm64
   ```

---

## Chi Phí & Tối Ưu Hóa

### Chi Phí Ước Tính (US Region)

**Lambda Pricing (Monthly, 1 million requests):**

| Memory | Avg Duration | Monthly Cost | Assumptions |
|--------|-------------|--------------|-------------|
| 1024 MB | 100ms | ~$2.08 | Free tier: 1M requests/month, 400K GB-seconds |
| 2048 MB | 50ms | ~$2.08 | Faster but more expensive per GB-second |
| 3008 MB | 35ms | ~$2.12 | Maximum performance |

**Free Tier:**

- 1 million requests per month
- 400,000 GB-seconds of compute time per month

**Lưu ý:** Vượt free tier mới tính phí.

### Optimization Tips

#### 1. Memory vs. Performance

**Test để tìm memory optimal:**

```powershell
# Test với different memory sizes
foreach ($mem in @(512, 1024, 2048, 3008)) {
    Write-Host "Testing memory: $mem MB"
    
    aws lambda update-function-configuration `
      --function-name atc-lambda `
      --memory-size $mem
    
    # Run benchmark
    python scripts\binance_lambda_demo.py `
      --endpoint YOUR_ENDPOINT `
      --symbols 100
}
```

**Rule of thumb:**

- 1024 MB: Good balance cho most cases
- 2048 MB: Nếu xử lý > 100 symbols/batch
- 3008 MB: Maximum performance, cold start nhanh nhất

#### 2. Batch Size Optimization

```python
# Tối ưu batch size
# 50-100 symbols/batch thường optimal
# Quá nhiều → timeout
# Quá ít → overhead cao

symbols_per_batch = 75  # Recommended
```

#### 3. Use ARM64 (Graviton2)

**~20% rẻ hơn x86_64, performance tương đương hoặc tốt hơn.**

```powershell
# Build for ARM64
cargo lambda build --release --target aarch64-unknown-linux-gnu

# Update function architecture
aws lambda update-function-configuration `
  --function-name atc-lambda `
  --architectures arm64
```

#### 4. Reduce Binary Size

Module đã optimize với:

- LTO (Link Time Optimization)
- Strip symbols
- Exclude unused features

**Nếu cần optimize thêm:**

```toml
# Cargo.toml
[profile.release]
opt-level = 'z'     # Optimize for size
lto = true
codegen-units = 1
strip = true
panic = 'abort'
```

#### 5. Reuse Lambda Containers

Lambda có thể reuse containers giữa invocations.

**Best practices:**

- Initialize reusable resources outside handler
- Use connection pooling
- Cache configuration

---

## Advanced: Continuous Deployment

### Setup CI/CD với GitHub Actions

Create `.github/workflows/deploy-lambda.yml`:

```yaml
name: Deploy to AWS Lambda

on:
  push:
    branches: [main]
    paths:
      - 'modules/adaptive_trend_LTS_serverless/**'

env:
  AWS_REGION: us-east-1
  FUNCTION_NAME: atc-lambda

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        target: x86_64-unknown-linux-musl
    
    - name: Install cargo-lambda
      run: cargo install cargo-lambda
    
    - name: Build Lambda
      working-directory: modules/adaptive_trend_LTS_serverless/lambda
      run: cargo lambda build --release --target x86_64-unknown-linux-musl
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: ${{ env.AWS_REGION }}
    
    - name: Deploy to Lambda
      working-directory: modules/adaptive_trend_LTS_serverless/lambda
      run: |
        cargo lambda deploy ${{ env.FUNCTION_NAME }} \
          --iam-role ${{ secrets.LAMBDA_ROLE_ARN }}
```

---

## Kết Luận

Bạn đã hoàn thành:

✅ Setup môi trường development  
✅ Build Lambda function từ Rust code  
✅ Deploy lên AWS Lambda  
✅ Cấu hình API Gateway/Function URL  
✅ Testing với real Binance data  
✅ Setup monitoring & alerting  
✅ Tối ưu hóa performance & cost  

### Next Steps

1. **Production Hardening:**
   - Add authentication (API keys, IAM auth)
   - Implement rate limiting
   - Setup DDoS protection (AWS WAF)

2. **Scaling:**
   - Setup auto-scaling rules
   - Implement request queuing (SQS)
   - Multi-region deployment

3. **Integration:**
   - Connect với trading bot
   - Setup webhook callbacks
   - Build notification system

### Resources

- [AWS Lambda Documentation](https://docs.aws.amazon.com/lambda/)
- [Cargo Lambda Guide](https://www.cargo-lambda.info/)
- [Binance API Documentation](https://binance-docs.github.io/apidocs/)
- [Module README](../../README.md)

---

**Designed with ❤️ for high-performance serverless trading systems.**

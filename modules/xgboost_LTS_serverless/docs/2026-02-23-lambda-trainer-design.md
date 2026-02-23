# Design: Lambda Python Trainer — XGBoost Cloud Training

**Date:** 2026-02-23  
**Status:** Approved — Ready for Implementation  
**Scope:** `modules/xgboost_LTS_serverless` + `modules/auto_trade/core/xgboost_auto_trainer.py`

---

## 1. Understanding Summary

|               |                                                                                                                                                   |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Cái gì**    | Chuyển training XGBoost per-symbol từ local daemon thread lên AWS Lambda Python                                                                   |
| **Tại sao**   | Pipeline hiện tại (`xgboost_auto_trainer.py`) spawn daemon thread trên máy local — máy local phải online, phải đủ RAM, là single point of failure |
| **Cho ai**    | `XGBoostServerlessFilter` tự trigger khi ATC scan phát hiện symbol chưa có model trong S3                                                         |
| **Ràng buộc** | Tái dùng 100% `xgboost_LTS/core/model.py`; model JSON format phải khớp với Lambda Rust (`Booster.load()`)                                         |
| **Non-goal**  | Không viết Rust training; không thay đổi Lambda Rust inference; không dùng GPU                                                                    |
| **Mục tiêu**  | A: Giảm time-to-model sau khi data mới có; B: Loại bỏ dependency vào máy local                                                                    |

---

## 2. Kiến Trúc Tổng Quan

### Luồng hiện tại (Local Training — BỎ)

```text
[XGBoostServerlessFilter]
    │  model missing in S3
    ▼
xgboost_auto_trainer.request_training()
    │
    └─ threading.Thread(target=_train_and_upload).start()
           │  chạy trên MÁY LOCAL
           ├── DataFetcher → Binance
           ├── train_and_predict()
           └── boto3.upload → S3
```

### Luồng mới (Lambda Cloud Training — ĐỀ XUẤT)

```text
[XGBoostServerlessFilter]
    │  model missing in S3
    ▼
xgboost_auto_trainer.request_training()
    │
    └─ lambda_client.invoke(
           FunctionName="xgboost-trainer",
           InvocationType="Event"   ← async fire-and-forget
       )
           │  chạy trên AWS LAMBDA
           ├── DataFetcher → Binance REST API
           ├── IndicatorEngine → 92 features
           ├── apply_directional_labels()
           ├── train_and_predict()   ← tái dùng xgboost_LTS/core/model.py
           ├── booster.save_model("/tmp/MODEL.json")
           └── boto3.upload → S3
                     │
                     ▼
           [xgboost-serverless-predict Lambda — Rust]  ← không đổi
```

### Sơ đồ AWS Resources

```text
┌─────────────────────────────────────────────────────────────┐
│                        AWS Account                          │
│                                                             │
│  ┌──────────────────────┐      ┌─────────────────────────┐  │
│  │  xgboost-trainer     │      │  xgboost-models-store   │  │
│  │  (Lambda Python 3.12)│────▶│  (S3 Bucket)            │  │
│  │  RAM: 3008 MB        │      │  BTCUSDT_15m_v1.json    │  │
│  │  Timeout: 900s       │      │  ETHUSDT_15m_v1.json    │  │
│  │  Container Image     │      │  ...                    │  │
│  └──────────────────────┘      └──────────┬──────────────┘  │
│           ▲                               │                 │
│           │ invoke async                  │ download model  │
│           │                               ▼                 │
│  ┌──────────────────────┐      ┌─────────────────────────┐  │
│  │  Auto Trade          │      │  xgboost-serverless-    │  │
│  │  (Local Python)      │      │  predict (Lambda Rust)  │  │
│  │  XGBoostServerless   │────▶│  Booster.load()         │  │
│  │  Filter              │      │  predict() → JSON       │  │
│  └──────────────────────┘      └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Thay Đổi Cần Thiết

### 3.1 File mới: `lambda/trainer/` trong `xgboost_LTS_serverless`

```text
modules/xgboost_LTS_serverless/
└── lambda/
    ├── src/          ← Rust inference (hiện có, không đổi)
    └── trainer/      ← MỚI: Python trainer Lambda
        ├── handler.py
        ├── requirements.txt
        └── Dockerfile
```

#### `lambda/trainer/handler.py`

```python
"""
XGBoost Trainer Lambda Handler

Triggered async by XGBoostServerlessFilter khi model thiếu trong S3.
Tái dùng toàn bộ xgboost_LTS training pipeline.

Event format:
{
    "symbol": "BTC/USDT",
    "timeframe": "15m",
    "model_version": "v1",
    "s3_bucket": "xgboost-models-store",
    "fetch_limit": 1000
}
"""

import json
import os
import time
from pathlib import Path

# Lazy import — tránh cold start nặng
def _imports():
    from modules.common.core.data_fetcher import DataFetcher
    from modules.common.core.exchange_manager import ExchangeManager
    from modules.common.core.indicator_engine import (
        IndicatorConfig, IndicatorEngine, IndicatorProfile,
    )
    from modules.xgboost_LTS.core.labeling import apply_directional_labels
    from modules.xgboost_LTS.core.model import train_and_predict
    from modules.xgboost_LTS.utils.features import add_advanced_features
    return (DataFetcher, ExchangeManager, IndicatorConfig,
            IndicatorEngine, IndicatorProfile,
            apply_directional_labels, train_and_predict, add_advanced_features)


def _normalize(symbol: str) -> str:
    return "".join(ch for ch in symbol.upper() if ch.isalnum())


def handler(event, context):
    """Lambda entrypoint."""
    t0 = time.perf_counter()

    symbol       = event["symbol"]
    timeframe    = event.get("timeframe", "15m")
    version      = event.get("model_version", "v1")
    s3_bucket    = event["s3_bucket"]
    fetch_limit  = int(event.get("fetch_limit", 1000))

    print(f"[trainer] START symbol={symbol} tf={timeframe} ver={version}")

    # ── 1. Import heavy deps ───────────────────────────────────────────────────
    (DataFetcher, ExchangeManager, IndicatorConfig,
     IndicatorEngine, IndicatorProfile,
     apply_directional_labels, train_and_predict,
     add_advanced_features) = _imports()

    # ── 2. Exchange + DataFetcher (dùng env vars từ Lambda env) ───────────────
    exchange = ExchangeManager()
    fetcher = DataFetcher(exchange)

    df = fetcher.fetch_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        limit=fetch_limit,
        check_freshness=False,
    )
    if df is None or df.empty:
        raise ValueError(f"fetch_ohlcv returned empty data for {symbol}")
    print(f"[trainer] fetched {len(df)} candles")

    # ── 3. Indicators ──────────────────────────────────────────────────────────
    engine = IndicatorEngine(IndicatorConfig.for_profile(IndicatorProfile.XGBOOST))
    result = engine.compute_features(df)
    df = result[0] if isinstance(result, tuple) else result
    df = add_advanced_features(df)

    # ── 4. Labels ──────────────────────────────────────────────────────────────
    df = apply_directional_labels(df, use_cache=False)
    df = df.dropna(subset=["Target"])

    # ── 5. Train ───────────────────────────────────────────────────────────────
    model = train_and_predict(df, use_cache=False)
    print(f"[trainer] training done in {time.perf_counter() - t0:.1f}s")

    # ── 6. Save model JSON ─────────────────────────────────────────────────────
    normalized = _normalize(symbol)
    filename = f"{normalized}_{timeframe}_{version}.json"
    local_path = Path("/tmp") / filename

    booster = model.get_booster()
    booster.save_model(str(local_path))

    # ── 7. Upload to S3 ────────────────────────────────────────────────────────
    import boto3
    s3 = boto3.client("s3")
    s3.upload_file(
        str(local_path),
        s3_bucket,
        filename,   # bare key — khớp với Lambda Rust handler
        ExtraArgs={
            "ContentType": "application/json",
            "Metadata": {
                "symbol": symbol,
                "timeframe": timeframe,
                "version": version,
                "trained_at": str(int(time.time())),
            },
        },
    )
    elapsed = time.perf_counter() - t0
    print(f"[trainer] uploaded s3://{s3_bucket}/{filename} in {elapsed:.1f}s total")

    return {
        "status": "ok",
        "symbol": symbol,
        "s3_key": filename,
        "elapsed_s": round(elapsed, 1),
    }
```

#### `lambda/trainer/Dockerfile`

> ⚠️ **Bug đã phát hiện (2026-02-23):** Thư mục có tên `lambda/` là **từ khoá Python**.
> `import modules.xgboost_LTS_serverless.lambda.trainer.handler` sẽ gây `SyntaxError` khi Python parser đọc.
> **Fix:** Copy handler.py lên root dưới tên `trainer_handler.py` và dùng CMD trỏ vào đó.
> `importlib.import_module()` hoạt động với đường dẫn này vì dùng string—không qua parser.

```dockerfile
# Dùng container image vì package size > 250MB (xgboost + numpy + pandas)
FROM public.ecr.aws/lambda/python:3.12

WORKDIR /var/task

# System deps cho xgboost (libgomp = OpenMP runtime)
RUN dnf install -y libgomp && dnf clean all

# Copy project source (modules cần cho training pipeline)
COPY modules/ ./modules/
COPY config.py ./
COPY requirements_trainer.txt ./

# ⚠️ FIX: Copy handler lên root tránh 'lambda' Python keyword trong import path
# AWS Lambda Python runtime dùng importlib nhưng để an toàn, đặt handler ở root
COPY modules/xgboost_LTS_serverless/lambda/trainer/handler.py ./trainer_handler.py

RUN pip install --no-cache-dir -r requirements_trainer.txt

# CMD format: "module.function" — trainer_handler = tên file tại /var/task root
CMD ["trainer_handler.handler"]
```

#### `requirements_trainer.txt` (root level, cho Lambda image)

```text
xgboost>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
boto3>=1.34.0
ccxt>=4.0.0
python-dotenv>=1.0.0
```

---

### 3.2 Sửa `template.yaml` — Thêm TrainerFunction

```yaml
# Thêm vào Resources section trong template.yaml

  XGBoostTrainerFunction:
    Type: AWS::Serverless::Function
    Properties:
      FunctionName: xgboost-trainer
      PackageType: Image
      ImageUri: !Sub "${AWS::AccountId}.dkr.ecr.${AWS::Region}.amazonaws.com/xgboost-trainer:latest"
      MemorySize: 3008        # 3GB — xgboost training cần RAM
      Timeout: 900            # 15 phút maximum Lambda timeout
      EphemeralStorage:
        Size: 1024            # 1GB /tmp cho model files
      Policies:
        - S3FullAccessPolicy:
            BucketName: !Ref ModelBucket
        - Statement:
            - Effect: Allow
              Action:
                - logs:CreateLogGroup
                - logs:CreateLogStream
                - logs:PutLogEvents
              Resource: "*"
      Environment:
        Variables:
          MODEL_BUCKET: !Ref ModelBucket
          # Binance API keys — inject từ AWS Secrets Manager
          BINANCE_API_KEY: !Sub "{{resolve:secretsmanager:auto-trade/binance:SecretString:api_key}}"
          BINANCE_API_SECRET: !Sub "{{resolve:secretsmanager:auto-trade/binance:SecretString:api_secret}}"

Outputs:
  TrainerFunctionName:
    Description: XGBoost Trainer Lambda function name
    Value: !Ref XGBoostTrainerFunction

  TrainerFunctionArn:
    Description: XGBoost Trainer Lambda ARN
    Value: !GetAtt XGBoostTrainerFunction.Arn
```

---

### 3.3 Sửa `xgboost_auto_trainer.py` — Thay thread bằng Lambda invoke

**Thay đổi duy nhất:** hàm `request_training()` invoke Lambda async thay vì spawn thread.

```python
# Thêm constant ở đầu file
_TRAINER_FUNCTION_NAME = os.environ.get(
    "XGBOOST_TRAINER_FUNCTION_NAME", "xgboost-trainer"
)
_TRAINER_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")


def _invoke_lambda_trainer(
    symbol: str,
    timeframe: str,
    model_version: str,
    s3_bucket: str,
    cache_key: str,
) -> None:
    """Invoke Lambda trainer async. Runs in a short-lived thread chỉ để gọi invoke."""
    try:
        import boto3
        lambda_client = boto3.client("lambda", region_name=_TRAINER_REGION)
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "model_version": model_version,
            "s3_bucket": s3_bucket,
            "fetch_limit": _FETCH_LIMIT,
        }
        lambda_client.invoke(
            FunctionName=_TRAINER_FUNCTION_NAME,
            InvocationType="Event",          # async — không chờ response
            Payload=json.dumps(payload).encode("utf-8"),
        )
        log_info(
            "XGBoostAutoTrainer: [%s] Lambda trainer invoked async (key=%s)",
            symbol, cache_key,
        )
    except Exception as exc:
        log_error(
            "XGBoostAutoTrainer: [%s] Lambda invoke failed, falling back to local thread: %s",
            symbol, exc,
        )
        # Fallback: chạy local nếu Lambda không available
        _train_and_upload(symbol, timeframe, model_version, s3_bucket,
                          data_fetcher, cache_key)


# Trong request_training() — thay thread.start() bằng:
#
# TRƯỚC:
#   thread = threading.Thread(target=_train_and_upload, ...)
#   thread.start()
#
# SAU:
#   thread = threading.Thread(target=_invoke_lambda_trainer, ...)
#   thread.start()   # thread này chỉ mất <1s để gọi invoke
```

---

### 3.4 Thêm script build & deploy

#### `scripts/build_trainer.sh`

```bash
#!/bin/bash
set -euo pipefail

AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=${AWS_DEFAULT_REGION:-us-east-1}
ECR_REPO="xgboost-trainer"
IMAGE_TAG="latest"
IMAGE_URI="${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"

echo "==> Building trainer Docker image..."
# Build từ project root để COPY modules/ hoạt động
docker build \
  -f modules/xgboost_LTS_serverless/lambda/trainer/Dockerfile \
  -t "${ECR_REPO}:${IMAGE_TAG}" \
  .

echo "==> Pushing to ECR..."
aws ecr get-login-password --region "${AWS_REGION}" | \
  docker login --username AWS --password-stdin \
  "${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com"

aws ecr describe-repositories --repository-names "${ECR_REPO}" 2>/dev/null || \
  aws ecr create-repository --repository-name "${ECR_REPO}"

docker tag "${ECR_REPO}:${IMAGE_TAG}" "${IMAGE_URI}"
docker push "${IMAGE_URI}"

echo "==> Done: ${IMAGE_URI}"
```

#### `scripts/deploy_trainer.sh`

```bash
#!/bin/bash
set -euo pipefail
STAGE=${1:-staging}
REGION=${2:-us-east-1}
BUCKET=${3:-xgboost-models-store}

AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
IMAGE_URI="${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/xgboost-trainer:latest"

echo "==> Deploying trainer to stage=${STAGE} region=${REGION}"
sam deploy \
  --template-file modules/xgboost_LTS_serverless/template.yaml \
  --stack-name "xgboost-serverless-${STAGE}" \
  --parameter-overrides \
      "ModelBucketName=${BUCKET}" \
  --image-repositories "XGBoostTrainerFunction=${IMAGE_URI}" \
  --capabilities CAPABILITY_IAM \
  --region "${REGION}" \
  --no-fail-on-empty-changeset

echo "==> Trainer deployed successfully"
```

---

## 4. Data Flow Chi Tiết

```text
ATC Scan Signal (e.g. BTC/USDT LONG)
    │
    ▼
XGBoostServerlessFilter.filter_signals()
    │
    ├─ [model có trong S3]
    │       │
    │       └─ invoke Lambda Rust → predict → confidence check → pass/reject signal
    │
    └─ [model THIẾU — "Failed to download model from S3"]
            │
            ▼
        _handle_missing_models()
            │
            ▼
        xgboost_auto_trainer.request_training()
            │
            ├─ status == "pending" → pass signal through (non-blocking)
            ├─ status == "ready"   → retry Lambda Rust ngay
            └─ status == None      → set "pending" → invoke Lambda Trainer async
                                            │
                                            │  (cloud, ~2-5 phút)
                                            ▼
                                    [xgboost-trainer Lambda]
                                        fetch OHLCV → features → labels → train
                                        → save /tmp/BTCUSDT_15m_v1.json
                                        → upload S3
                                        │
                                        ► [Next pipeline cycle]
                                        xgboost_auto_trainer status = "ready"
                                        → retry Lambda Rust → signal filtered
```

---

## 5. Vấn đề Cần Giải Quyết Khi Implement

### 5.1 Status Sync — Lambda vs Local State

**Vấn đề:** `_STATUS` dict hiện tại chỉ sống trong memory của process local. Khi Lambda trainer hoàn thành (trên cloud), local process không biết.

**Giải pháp đề xuất:** Lambda trainer ghi kết quả vào S3 metadata — local filter kiểm tra S3 object existence thay vì in-memory status.

```python
# Thay vì _STATUS dict, kiểm tra S3:
def _model_exists_in_s3(symbol, timeframe, version, bucket) -> bool:
    import boto3
    s3 = boto3.client("s3")
    key = f"{_normalize(symbol)}_{timeframe}_{version}.json"
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False
```

> ⚠️ **TODO:** Quyết định trong implementation: giữ in-memory `_STATUS` để tránh S3 API call mỗi cycle, hay switch sang S3-backed status. Gợi ý: TTL 5 phút cho in-memory cache + fallback check S3.

### 5.2 API Keys cho Lambda Trainer

**Vấn đề:** Lambda Trainer cần Binance API key để fetch OHLCV.

**Giải pháp:** Inject qua AWS Secrets Manager vào Lambda environment variables (đã có trong `template.yaml` trên). Không hardcode trong code.

### 5.3 Package Size

**Vấn đề:** `xgboost + pandas + numpy + scikit-learn` > 250MB → vượt Lambda ZIP limit.

**Giải pháp:** Dùng **container image** (đã design trong Dockerfile trên). AWS Lambda hỗ trợ container image lên đến 10GB.

### 5.4 `config.py` Dependencies

**Vấn đề:** `xgboost_LTS/core/model.py` import từ `config` ở project root — Lambda cần `config.py` accessible.

**Giải pháp:** Copy `config.py` vào Docker image (đã có `COPY config.py ./` trong Dockerfile).

### 5.5 `lambda` là Python Keyword — Handler Import Path

**Vấn đề phát hiện lúc implement (2026-02-23):**

Thư mục `modules/xgboost_LTS_serverless/lambda/trainer/` chứa handler, nhưng `lambda` là **từ khoá Python**:

```python
# Câu này gây SyntaxError — Python parser thấy 'lambda' là keyword:
from modules.xgboost_LTS_serverless.lambda.trainer.handler import handler

# Câu này HOẠT ĐỘNG — importlib dùng string, không qua parser:
importlib.import_module('modules.xgboost_LTS_serverless.lambda.trainer.handler')
```

AWS Lambda Python runtime dùng `importlib` nên về lý thuyết hoạt động, **nhưng rủi ro cao** với các tool khác (linters, IDE, test runners).

**Giải pháp đã áp dụng:** Trong `Dockerfile`, copy handler lên root `/var/task/trainer_handler.py` và trỏ CMD vào đó:

```dockerfile
COPY modules/xgboost_LTS_serverless/lambda/trainer/handler.py ./trainer_handler.py
CMD ["trainer_handler.handler"]
```

**Alternative nếu muốn refactor hoàn toàn:** Đổi tên thư mục `lambda/` → `lambda_fn/` trong toàn bộ project (nhưng cần cẩn thận vì Rust code trong `lambda/src/` cũng sẽ bị ảnh hưởng về Cargo.toml paths).

---

## 6. Decision Log

| #   | Quyết định                                      | Thay thế đã xem xét      | Lý do chọn                                                                             |
| --- | ----------------------------------------------- | ------------------------ | -------------------------------------------------------------------------------------- |
| D1  | Lambda Python (Option 1) thay vì SageMaker      | SageMaker Training Job   | SageMaker startup 3-5 phút quá chậm cho per-symbol trigger; Lambda đủ cho 1000 candles |
| D2  | SageMaker Training Job loại (Option 2)          | —                        | Overkill, cost cao, startup chậm                                                       |
| D3  | ECS Fargate loại (Option 3)                     | —                        | Phức tạp hơn Lambda, startup 1-2 phút, không có GPU benefit                            |
| D4  | Không viết lại training bằng Rust               | Rust 100% (Option C)     | Rust thiếu TimeSeriesCV, labeling logic; xgboost-rs API rất hạn chế; rủi ro cao        |
| D5  | Tái dùng `train_and_predict()` từ `xgboost_LTS` | Viết lại script mới      | Code đã battle-tested với 20 test cases, class diversity checks, CV                    |
| D6  | Container image thay vì ZIP deployment          | Lambda Layer             | Package size > 250MB vượt ZIP limit; image linh hoạt hơn                               |
| D7  | `InvocationType="Event"` (async)                | `RequestResponse` (sync) | Training mất 2-5 phút; pipeline không thể block chờ                                    |
| D8  | Fallback về local thread nếu Lambda unavailable | Hard fail                | Pipeline phải non-blocking; graceful degradation quan trọng                            |

---

## 7. Implementation Plan

### Phase 1 — Lambda Trainer Function

- [ ] Tạo `lambda/trainer/handler.py`
- [ ] Tạo `lambda/trainer/Dockerfile`
- [ ] Tạo `requirements_trainer.txt`
- [ ] Test handler locally với Docker: `docker run --env-file .env <image> '{"symbol":"BTC/USDT",...}'`

### Phase 2 — SAM Template Update

- [ ] Thêm `XGBoostTrainerFunction` vào `template.yaml`
- [ ] Thêm IAM policy cho trainer: S3 write + Secrets Manager read
- [ ] Test `sam local invoke XGBoostTrainerFunction`

### Phase 3 — Auto Trainer Modification

- [ ] Thêm `_invoke_lambda_trainer()` vào `xgboost_auto_trainer.py`
- [ ] Sửa `request_training()` để gọi Lambda thay vì thread
- [ ] Giữ local thread làm fallback khi Lambda unavailable
- [ ] Update unit tests

### Phase 4 — Build & Deploy Scripts

- [ ] Tạo `scripts/build_trainer.sh`
- [ ] Tạo `scripts/deploy_trainer.sh`
- [ ] Setup ECR repository
- [ ] Deploy to staging, smoke test

### Phase 5 — Status Sync (Optional Enhancement)

- [ ] Implement S3-backed status check thay vì in-memory only
- [ ] TTL cache để giảm S3 API calls

---

## 8. Assumptions

1. S3 bucket `xgboost-models-store` đã tồn tại và có versioning enabled
2. Model S3 key format `{SYMBOL}_{timeframe}_{version}.json` không thay đổi
3. Lambda Rust inference (`xgboost-serverless-predict`) không cần sửa
4. Binance API credentials sẽ được store trong AWS Secrets Manager
5. Training với 1000 candles hoàn thành trong < 15 phút (Lambda timeout)
6. AWS account có ECR access để push container images

---

## 9. Risks & Mitigations

| Risk                                                        | Likelihood |   Impact   | Mitigation                                                        |
| ----------------------------------------------------------- | :--------: | :--------: | ----------------------------------------------------------------- |
| Training vượt 15 phút timeout                               |    Thấp    |    Cao     | Monitor với CloudWatch; giảm `fetch_limit` nếu cần                |
| Lambda cold start chậm                                      | Trung bình |    Thấp    | Provisioned concurrency nếu cần; cold start chỉ ảnh hưởng lần đầu |
| Binance API rate limit trong Lambda                         |    Thấp    | Trung bình | Reuse `check_freshness=False`; add retry logic                    |
| `config.py` import path issues                              | Trung bình |    Cao     | Test Docker image locally trước khi deploy                        |
| Status sync lag (local biết "pending" nhưng Lambda đã xong) | Trung bình |    Thấp    | TTL 5 phút tự reset; S3 head_object check                         |

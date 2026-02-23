# Lambda Trainer — XGBoost Cloud Training Tasks

## Goal
Chuyển XGBoost training từ local daemon thread lên AWS Lambda Python container,
tái dùng 100% `xgboost_LTS/core/model.py`, không sửa Lambda Rust inference.

Design doc: `modules/xgboost_LTS_serverless/docs/2026-02-23-lambda-trainer-design.md`

---

## Phase 1 — Lambda Trainer Handler

- [x] **T1.1** Tạo `modules/xgboost_LTS_serverless/lambda/trainer/handler.py`
  - Code từ design doc section 3.1, với lazy import để giảm cold start
  - ✅ Verify: `python -c "import importlib; importlib.import_module('modules.xgboost_LTS_serverless.lambda.trainer.handler')"` → `handler import ok`

- [x] **T1.2** Tạo `requirements_trainer.txt` tại project root
  - Nội dung: `xgboost>=2.0.0`, `pandas>=2.0.0`, `numpy>=1.24.0`, `scikit-learn>=1.3.0`, `boto3>=1.34.0`, `ccxt>=4.0.0`, `python-dotenv>=1.0.0`
  - ✅ Verify: `uv pip install -r requirements_trainer.txt --dry-run` → `Audited 7 packages, Would make no changes`

- [x] **T1.3** Tạo `modules/xgboost_LTS_serverless/lambda/trainer/Dockerfile`
  - Base image: `public.ecr.aws/lambda/python:3.12`
  - `RUN dnf install -y libgomp` trước khi pip install
  - `COPY modules/ config.py requirements_trainer.txt ./`
  - Handler copy lên root tránh `lambda` Python keyword: `COPY handler.py ./trainer_handler.py`
  - CMD: `["trainer_handler.handler"]`
  - Tạo `__init__.py` cho package `lambda/trainer/`
  - ⚠️ **BLOCKED**: `docker build` chưa verify — Docker chưa được cài trên máy này
  - > Sau khi cài Docker Desktop: `docker build -f modules/xgboost_LTS_serverless/lambda/trainer/Dockerfile -t xgboost-trainer .`

- [ ] **T1.4** Test handler trong Docker container với mock event
  - `test_event.json` đã tạo tại `lambda/trainer/test_event.json`
  - ⚠️ **BLOCKED**: Cần Docker
  - > Sau khi cài Docker: `docker run --env-file modules/auto_trade/.env xgboost-trainer '{"symbol":"BTC/USDT",...}'`
  - Verify: container in `[trainer] fetched N candles` và `[trainer] training done in X.Xs`, không traceback

---

## Phase 2 — SAM Template

- [x] **T2.1** Thêm `XGBoostTrainerFunction` resource vào `modules/xgboost_LTS_serverless/template.yaml`
  - `PackageType: Image`, `MemorySize: 3008`, `Timeout: 900`, `EphemeralStorage.Size: 1024`
  - Policy: `S3FullAccessPolicy` cho `ModelBucket` + CloudWatch Logs inline
  - Env vars: `MODEL_BUCKET`, `BINANCE_API_KEY`, `BINANCE_API_SECRET` từ Secrets Manager
  - ✅ Verify: `sam validate --template modules/xgboost_LTS_serverless/template.yaml --region us-east-1` → exit code 0

- [x] **T2.2** Thêm `TrainerFunctionName` và `TrainerFunctionArn` vào `Outputs` section
  - ✅ Verify: Outputs có trong template.yaml; `sam validate` passes

- [ ] **T2.3** Test invoke local với SAM
  - ⚠️ **BLOCKED**: `sam local invoke` cho ImageUri function yêu cầu Docker
  - `sam local invoke XGBoostTrainerFunction --event lambda/trainer/test_event.json`
  - Verify: response JSON có `"status": "ok"` hoặc lỗi có stack trace rõ ràng để debug
  - > Note: SAM build `--use-container` cũng cần Docker cho Rust XGBoostFunction

---

## Phase 3 — Sửa xgboost_auto_trainer.py

- [x] **T3.1** Thêm constants và `_invoke_lambda_trainer()` function vào đầu file `modules/auto_trade/core/xgboost_auto_trainer.py`
  - `_TRAINER_FUNCTION_NAME = os.environ.get("XGBOOST_TRAINER_FUNCTION_NAME", "xgboost-trainer")`
  - `_TRAINER_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")`
  - Thêm `import json` vào imports
  - ✅ Verify: `python -c "from modules.auto_trade.core.xgboost_auto_trainer import request_training"` → `request_training import ok`

- [x] **T3.2** Sửa `request_training()` — thay `threading.Thread(target=_train_and_upload)` bằng `threading.Thread(target=_invoke_lambda_trainer)`
  - Thread mới chỉ gọi `lambda_client.invoke(InvocationType="Event")` rồi return ngay
  - Fallback trong `except`: gọi lại `_train_and_upload(...)` local nếu Lambda unavailable
  - ✅ Verify: `_train_and_upload` vẫn còn trong file (không xóa) và được gọi ở fallback path

- [x] **T3.3** Thêm `XGBOOST_TRAINER_FUNCTION_NAME` vào `modules/auto_trade/.env.example`
  - `XGBOOST_TRAINER_FUNCTION_NAME=xgboost-trainer`
  - ✅ Verify: file `.env.example` có dòng mới

- [x] **T3.4** Viết/update pytest test trong `tests/` cho `xgboost_auto_trainer`
  - Test: khi Lambda invoke thành công → status `"pending"`, thread kết thúc nhanh (< 2s)
  - Test: khi Lambda invoke raise exception → fallback `_train_and_upload` được gọi
  - Test: `get_training_status()` trả đúng state
  - ✅ Verify (targeted): `uv run pytest tests/auto_trade/core/test_xgboost_auto_trainer.py -v` → `3 passed`
  - ⚠️ `uv run pytest tests/ -k "auto_trainer" -v` hiện fail do lỗi import có sẵn ở module test khác, không thuộc `xgboost_auto_trainer`

---

## Phase 4 — Build & Deploy Scripts

- [x] **T4.1** Tạo `modules/xgboost_LTS_serverless/scripts/build_trainer.sh`
  - Build Docker image từ project root: `docker build -f .../Dockerfile -t xgboost-trainer .`
  - Login ECR: `aws ecr get-login-password | docker login ...`
  - Tạo ECR repo nếu chưa có: `aws ecr create-repository --repository-name xgboost-trainer`
  - Push image lên ECR
  - Verify: `bash scripts/build_trainer.sh` → image xuất hiện trong ECR console

- [x] **T4.2** Tạo `modules/xgboost_LTS_serverless/scripts/deploy_trainer.sh`
  - `sam deploy` với `--image-repositories XGBoostTrainerFunction=<ECR_URI>`
  - Nhận args: `$1=stage`, `$2=region`, `$3=bucket`
  - Verify: `bash scripts/deploy_trainer.sh staging us-east-1 xgboost-models-store` → CloudFormation stack updated/created

- [ ] **T4.3** Smoke test end-to-end trên staging
  - ⚠️ **BLOCKED**: Cần deploy thành công (chờ cài Docker để build & push image thì mới deploy được)
  - Invoke Lambda trainer thủ công:
    ```bash
    aws lambda invoke \
      --function-name xgboost-trainer \
      --invocation-type RequestResponse \
      --payload '{"symbol":"BTC/USDT","timeframe":"15m","model_version":"v1","s3_bucket":"xgboost-models-store","fetch_limit":200}' \
      /tmp/trainer_response.json
    cat /tmp/trainer_response.json
    ```
  - Verify: response `"status": "ok"`, file `BTCUSDT_15m_v1.json` xuất hiện trong S3 bucket
  - Verify: Lambda Rust predict vẫn hoạt động bình thường sau khi model được upload

---

## Phase 5 — Status Sync Enhancement (Optional)

- [x] **T5.1** Thêm helper `_model_exists_in_s3(symbol, timeframe, version, bucket) -> bool` vào `xgboost_auto_trainer.py`
  - Dùng `s3.head_object()` để kiểm tra key tồn tại
  - TTL cache 5 phút: không gọi S3 nếu đã check trong 5 phút gần nhất
  - ✅ Verify: Helper implemented

- [x] **T5.2** Trong `request_training()`, trước khi invoke Lambda, gọi `_model_exists_in_s3()`:
  - Nếu model đã có → set status `"ready"`, skip invoke
  - Logic này giải quyết case: Lambda đã train xong nhưng local `_STATUS` vẫn là `"pending"`
  - Verify: Sau khi restart local process, pipeline tự nhận model đã có trong S3 mà không retrain

---

## Done When

- [x] handler.py import clean — verified với importlib
- [x] `requirements_trainer.txt` no conflict — verified với uv dry-run
- [x] `template.yaml` với XGBoostTrainerFunction — `sam validate` passes
- [ ] `docker build` thành công không lỗi — ⚠️ **Cần cài Docker**
- [ ] `sam local invoke XGBoostTrainerFunction` chạy được — ⚠️ **Cần Docker**
- [x] `uv run pytest tests/auto_trade/core/test_xgboost_auto_trainer.py -v` pass (`3 passed`)
- [ ] Lambda trainer trên staging upload được model JSON lên S3
- [ ] Lambda Rust inference vẫn predict đúng từ model do Lambda trainer tạo ra

---

## Notes

- **⚠️ Blocker hiện tại:** Docker chưa cài trên máy — T1.3 docker build, T1.4, T2.3 bị block
  - Cài Docker Desktop: <https://www.docker.com/products/docker-desktop/>
- **⚠️ Python keyword:** Thư mục `lambda/` là từ khoá Python. Đã fix trong Dockerfile bằng cách copy handler.py lên root dưới tên `trainer_handler.py` → CMD: `["trainer_handler.handler"]`
- **Thứ tự bắt buộc:** T1 (unblock Docker) → T2.3 → T3 → T4
- **T5 độc lập**, có thể làm sau khi Phase 4 xong
- Local fallback (`_train_and_upload`) **không được xóa** — cần thiết khi chạy offline
- Verify pytest cho Phase 3 dùng scope theo module mới để tránh bị block bởi lỗi collection không liên quan ở test legacy toàn repo
- **Phase 3 là tiếp theo** không cần Docker — có thể làm ngay hôm nay

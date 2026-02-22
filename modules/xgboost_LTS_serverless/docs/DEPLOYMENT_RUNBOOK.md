# Deployment Runbook

## 1) Pre-Deploy Checklist

- `cargo test --workspace`
- `cargo build --workspace`
- `cargo check --features xgboost`
- `python scripts/train_and_upload.py --help`

## 2) Build Artifacts

```bash
bash scripts/build.sh
```

## 3) Staging Deploy

```bash
bash scripts/deploy.sh staging us-east-1
```

## 4) Smoke Test

```bash
python scripts/lambda_demo.py \
  --endpoint https://<api-id>.execute-api.us-east-1.amazonaws.com/staging/predict \
  --symbol BTC/USDT \
  --timeframe 1h \
  --model-version v1
```

## 5) Load Test (Baseline)

Use concurrent requests against staging endpoint and capture p50/p95 latency + error rate.
Suggested command (example):

```bash
python scripts/load_test.py --endpoint https://<api-id>.execute-api.us-east-1.amazonaws.com/staging/predict --concurrency 20 --requests 200
```

## 6) Production Deploy

```bash
bash scripts/deploy.sh production us-east-1
```

## 7) Rollback

- Re-deploy previous template/model version
- Point traffic to previous API stage if needed
- Validate smoke test again

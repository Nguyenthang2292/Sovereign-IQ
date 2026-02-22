# Model Export Guide

## Goal

Export a Python-trained XGBoost classifier to JSON so the Rust Lambda can load it.

## Expected File Naming

`{SYMBOL}_{TIMEFRAME}_{VERSION}.json`

Example:

- `BTC_USDT_1h_v1.json`

## Export Command

From module root:

```bash
python scripts/train_and_upload.py \
  --symbol BTC/USDT \
  --timeframe 1h \
  --version v1 \
  --bucket xgboost-models-production
```

## S3 Key Convention

Current training script uploads with prefix:

- `models/xgboost/{SYMBOL}_{TIMEFRAME}_{VERSION}.json`

If your Lambda request uses `model_s3_key`, pass the full key exactly.

## Runtime Compatibility

- Rust expects 92-feature vector ordering in `docs/FEATURE_REFERENCE.md`
- Ensure model was trained with same feature order and preprocessing assumptions

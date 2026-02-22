#!/bin/bash
set -e

echo "Building XGBoost Serverless Lambda..."

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "Building release binary..."
RUSTFLAGS="-C target-cpu=haswell -C target-feature=+avx2" cargo lambda build --release --target x86_64-unknown-linux-gnu

echo "Build complete!"
echo "Output: target/lambda/xgboost_lambda/bootstrap"

# XGBoost LTS Serverless Module - Design Document

**Date:** February 21, 2026  
**Author:** Design Session  
**Status:** Approved for Implementation

## Overview

This document describes the design for XGBoost LTS (Long-Term Support) Serverless module - a high-performance Rust implementation of XGBoost inference optimized for AWS Lambda deployment. The module provides real-time cryptocurrency price prediction with sub-second latency.

### Key Decisions

- **Scope:** Inference Only (Option A)
- **Model Storage:** S3 Model Registry (Option A)
- **Feature Set:** Full Feature Parity - All 90+ features (Option B)
- **Input Format:** Raw OHLCV Arrays (Option A)

---

## Section 1: High-Level Architecture

### System Overview

The XGBoost LTS serverless module is a Rust-based Lambda function that performs real-time cryptocurrency price prediction with the following data flow:

**Request Flow:**

1. Client sends OHLCV data (500+ candles) + symbol + timeframe to Lambda
2. Lambda handler validates input and extracts parameters
3. Model Manager checks `/tmp` cache for trained model
4. If not cached, downloads from S3 (`s3://models/xgboost/{symbol}_{timeframe}_v{version}.json`)
5. Feature Engine calculates all 90+ features from OHLCV data
6. XGBoost inference engine runs prediction
7. Returns classification (UP/DOWN/NEUTRAL) with probabilities

**AWS Components:**

- **Lambda Function**: Rust binary with 3008MB memory (max CPU), 30s timeout
- **S3 Bucket**: Model storage with versioning enabled
- **API Gateway**: Optional REST endpoint for HTTP access
- **CloudWatch**: Logging and performance metrics
- **SQS Queue**: Optional async result delivery (like ATC module)

**Performance Targets:**

- Cold start: <2s (including S3 model download)
- Warm inference: <100ms per prediction
- Batch processing: 50+ symbols in parallel via Rayon

### Architecture Diagram

```text
┌─────────────────┐      ┌──────────────────┐     ┌─────────────────┐
│   API Gateway   │────▶│   AWS Lambda     │────▶│   SQS Queue     │
│   (HTTP/REST)   │      │   (Rust Module)  │     │   (Results)     │
└─────────────────┘      └──────────────────┘     └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   S3 Bucket      │
                       │   (Models)       │
                       └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Feature Engine │
                       │   (90+ features) │
                       └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   XGBoost        │
                       │   Inference      │
                       └──────────────────┘
```

---

## Section 2: Rust Module Structure

### Directory Layout

```text
modules/xgboost_LTS_serverless/
├── Cargo.toml                    # Workspace configuration
├── rust-toolchain.toml           # Rust version pin (1.70+)
├── README.md                     # Documentation
├── CHANGELOG.md
├── template.yaml                 # AWS SAM template
├── lambda-trust-policy.json      # IAM policies
│
├── src/                          # Core library
│   ├── lib.rs                    # Public API & data structures
│   ├── ohlcv.rs                  # OHLCV data structures & validation
│   ├── feature_engine.rs         # Feature calculation orchestrator
│   ├── features/                 # Feature implementations
│   │   ├── mod.rs
│   │   ├── price_derived.rs     # returns, log_volume, ranges
│   │   ├── indicators.rs        # RSI, MACD, ATR, BBands, Stochastic
│   │   ├── moving_averages.rs   # SMA family
│   │   ├── candlestick.rs       # 48 candlestick patterns
│   │   ├── advanced.rs          # ROC, volatility ratios, rolling stats
│   │   └── lag_features.rs      # Lag features for returns & RSI
│   ├── xgboost_inference.rs     # XGBoost model wrapper
│   ├── model_manager.rs         # S3 download & caching
│   └── error.rs                 # Error types
│
├── lambda/                       # Lambda binary
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs               # Lambda entry point
│       ├── handler.rs            # Request/response handling
│       └── s3_client.rs          # AWS S3 operations
│
├── scripts/                      # Utilities
│   ├── requirements.txt
│   ├── train_and_upload.py      # Train model & upload to S3
│   ├── lambda_demo.py           # Test client
│   └── deploy.sh                # Deployment script
│
├── tests/                        # Integration tests
│   ├── feature_tests.rs         # Feature calculation tests
│   ├── inference_tests.rs       # Model inference tests
│   └── test_data/
│       ├── btc_usdt_1h.json    # Sample OHLCV data
│       └── test_model.json      # Sample XGBoost model
│
└── docs/
    ├── QUICK_START.md
    ├── FEATURE_REFERENCE.md     # All 90+ features documented
    ├── AWS_SETUP.md
    └── MODEL_EXPORT.md          # How to export Python models
```

### Key Dependencies

**Cargo.toml (Workspace Root):**

```toml
[workspace]
members = [".", "lambda"]

[package]
name = "xgboost_serverless"
version = "0.1.0"
edition = "2021"

[dependencies]
# XGBoost inference
xgboost = "0.3"  # or custom JSON parser

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Numerical computing
ndarray = "0.15"
ta = "0.5"  # Technical analysis indicators (or implement from scratch)

# Parallel processing
rayon = "1.8"

# Error handling
thiserror = "1.0"
anyhow = "1.0"

# Utilities
once_cell = "1.19"
smallvec = { version = "1.13", features = ["const_generics"] }

[dev-dependencies]
proptest = "1.4"
approx = "0.5"  # For floating-point comparisons

[profile.release]
opt-level = 3
lto = "thin"
strip = true
codegen-units = 1
```

**lambda/Cargo.toml:**

```toml
[package]
name = "xgboost_lambda"
version = "0.1.0"
edition = "2021"

[dependencies]
xgboost_serverless = { path = "../" }
lambda_runtime = "0.8"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
aws-config = "1.0"
aws-sdk-s3 = "1.0"
aws-sdk-sqs = "1.0"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["json"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

[profile.release]
opt-level = 3
lto = "thin"
strip = true
codegen-units = 1
```

---

## Section 3: Feature Calculation Engine

### Feature Orchestration Strategy

The Feature Engine calculates all 90+ features in optimized groups to minimize redundant calculations.

**Feature Categories (from config/shared/model_features.py):**

1. **Price-Derived Features (5)**: returns_1, returns_5, log_volume, high_low_range, close_open_diff
2. **Moving Averages (3)**: SMA_20, SMA_50, SMA_200
3. **RSI Family (3)**: RSI_9, RSI_14, RSI_25
4. **MACD (3)**: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
5. **Volatility (1)**: ATR_14
6. **Bollinger Bands (1)**: BBP_5_2.0
7. **Stochastic RSI (2)**: STOCHRSIk_14_14_3_3, STOCHRSId_14_14_3_3
8. **Volume (1)**: OBV
9. **Candlestick Patterns (48)**: All 48 patterns from CANDLESTICK_PATTERN_NAMES
10. **Advanced Momentum (4)**: roc_3, roc_5, roc_10, roc_20
11. **Volatility Ratios (1)**: atr_ratio
12. **Price/MA Ratios (3)**: price_to_SMA_20, price_to_SMA_50, price_to_SMA_200
13. **Rolling Statistics (4)**: rolling_std_10, rolling_std_20, rolling_skew_10, rolling_skew_20
14. **Lag Features (9)**: returns_1_lag_1/2/3, RSI_14_lag_1/2/3, MACD_lag_1/2/3

### Total: 92 features

### Calculation Pipeline

```rust
pub struct FeatureEngine {
    ohlcv: OHLCVData,
    cache: FeatureCache,  // Memoize expensive calculations
}

// Execution order (dependencies flow downward):
impl FeatureEngine {
    pub fn calculate_all(&mut self) -> Result<Vec<f64>> {
        // 1. Basic OHLCV arrays (open, high, low, close, volume)
        let (open, high, low, close, volume) = self.extract_ohlcv();
        
        // 2. Returns & Price Differences
        let returns_1 = self.calculate_returns(&close, 1);
        let returns_5 = self.calculate_returns(&close, 5);
        let close_open_diff = self.calc_close_open_diff(&open, &close);
        let high_low_range = self.calc_high_low_range(&high, &low, &close);
        
        // 3. Moving Averages (cached for reuse)
        let sma_20 = self.cache_or_calc("SMA_20", || sma(&close, 20));
        let sma_50 = self.cache_or_calc("SMA_50", || sma(&close, 50));
        let sma_200 = self.cache_or_calc("SMA_200", || sma(&close, 200));
        
        // 4. Price/SMA Ratios (uses cached SMAs)
        let price_to_sma_20 = &close / &sma_20;
        let price_to_sma_50 = &close / &sma_50;
        let price_to_sma_200 = &close / &sma_200;
        
        // 5. Volatility (ATR cached for reuse)
        let atr_14 = self.cache_or_calc("ATR_14", || atr(&high, &low, &close, 14));
        let atr_ratio = &atr_14 / &close;
        
        // 6. Momentum Oscillators
        let rsi_9 = rsi(&close, 9);
        let rsi_14 = rsi(&close, 14);
        let rsi_25 = rsi(&close, 25);
        let (macd, macd_signal, macd_hist) = macd(&close, 12, 26, 9);
        
        // 7. Stochastic RSI
        let (stoch_k, stoch_d) = stochastic_rsi(&close, 14, 14, 3, 3);
        
        // 8. Volume Indicators
        let log_volume = volume.mapv(|v| v.ln());
        let obv = on_balance_volume(&close, &volume);
        
        // 9. Bollinger Bands
        let bbp = bollinger_band_percent(&close, 5, 2.0);
        
        // 10. Advanced Features
        let roc_3 = rate_of_change(&close, 3);
        let roc_5 = rate_of_change(&close, 5);
        let roc_10 = rate_of_change(&close, 10);
        let roc_20 = rate_of_change(&close, 20);
        
        // 11. Rolling Statistics
        let rolling_std_10 = rolling_std(&returns_1, 10);
        let rolling_std_20 = rolling_std(&returns_1, 20);
        let rolling_skew_10 = rolling_skewness(&returns_1, 10);
        let rolling_skew_20 = rolling_skewness(&returns_1, 20);
        
        // 12. Candlestick Patterns (parallel computation)
        let patterns = self.detect_candlestick_patterns_parallel(&open, &high, &low, &close);
        
        // 13. Lag Features
        let returns_1_lags = create_lags(&returns_1, &[1, 2, 3]);
        let rsi_14_lags = create_lags(&rsi_14, &[1, 2, 3]);
        let macd_lags = create_lags(&macd, &[1, 2, 3]);
        
        // 14. Assemble feature vector (latest values only)
        let features = self.assemble_feature_vector(/* all calculated features */);
        
        Ok(features)
    }
}
```

### Performance Optimizations

1. **Vectorized Operations**: Use ndarray for SIMD-accelerated calculations
2. **Parallel Pattern Detection**: Rayon parallelizes 48 candlestick patterns across candles
3. **Smart Caching**: Cache intermediate results (e.g., SMA used by multiple features)
4. **Rolling Windows**: Efficient rolling statistics with pre-allocated buffers
5. **Feature Groups**: Calculate related features together to share computations

### Memory Management

- Pre-allocate all feature arrays upfront (90 x N size)
- Use stack allocation for small intermediate calculations
- Pool OHLCV buffers for batch processing
- Reference counting (Arc) for shared data between features

### Candlestick Pattern Implementation

All 48 patterns from Python implementation:

```rust
pub struct CandlestickPatterns {
    // Single-candle patterns
    pub doji: Vec<bool>,
    pub hammer: Vec<bool>,
    pub inverted_hammer: Vec<bool>,
    pub shooting_star: Vec<bool>,
    pub marubozu_bull: Vec<bool>,
    pub marubozu_bear: Vec<bool>,
    // ... (48 total patterns)
}

impl CandlestickPatterns {
    pub fn detect_all(ohlcv: &OHLCVData) -> Self {
        // Use Rayon for parallel detection
        let patterns: Vec<_> = (0..ohlcv.len())
            .into_par_iter()
            .map(|i| detect_patterns_at_index(i, ohlcv))
            .collect();
        
        Self::from_parallel_results(patterns)
    }
}
```

---

## Section 4: XGBoost Model Integration

### Model Format & Loading

**Python Model Export Strategy:**

The Python training script exports models in XGBoost's JSON format for Rust consumption:

```python
# In scripts/train_and_upload.py
import xgboost as xgb
import boto3

def export_model_for_lambda(model: xgb.XGBClassifier, symbol: str, timeframe: str, version: str = "v1"):
    """Export trained XGBoost model to JSON format and upload to S3"""
    
    # 1. Save as JSON (includes tree structure, feature names, class labels)
    model_path = f"/tmp/{symbol.replace('/', '_')}_{timeframe}_{version}.json"
    model.save_model(model_path)
    
    # 2. Upload to S3
    s3 = boto3.client('s3')
    bucket = 'xgboost-models-production'
    key = f"{symbol.replace('/', '_')}_{timeframe}_{version}.json"
    
    s3.upload_file(model_path, bucket, key)
    print(f"Model uploaded: s3://{bucket}/{key}")
    
    return f"s3://{bucket}/{key}"
```

### Rust Model Loading

**Implementation using xgboost-rs crate:**

```rust
use xgboost::{Booster, DMatrix};
use std::path::Path;

pub struct XGBoostModel {
    booster: Booster,
    feature_names: Vec<String>,
    num_classes: usize,
}

impl XGBoostModel {
    pub fn from_json_file(path: &Path) -> Result<Self> {
        let booster = Booster::load(path)?;
        
        // Extract metadata from model
        let feature_names = booster.feature_names()?;
        let num_classes = booster.num_classes()?;
        
        Ok(Self {
            booster,
            feature_names,
            num_classes,
        })
    }
    
    pub fn predict(&self, features: &[f64]) -> Result<PredictionResult> {
        // Convert features to DMatrix
        let dmat = DMatrix::from_dense(features, 1)?;
        
        // Run inference
        let predictions = self.booster.predict(&dmat)?;
        
        // Parse output (probabilities for each class)
        let probabilities: [f64; 3] = [
            predictions[0],  // DOWN
            predictions[1],  // NEUTRAL
            predictions[2],  // UP
        ];
        
        // Determine predicted class
        let (predicted_idx, confidence) = probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        
        let label = match predicted_idx {
            0 => "DOWN",
            1 => "NEUTRAL",
            2 => "UP",
            _ => unreachable!(),
        }.to_string();
        
        Ok(PredictionResult {
            label,
            probabilities,
            confidence: *confidence,
        })
    }
}
```

### Model Cache Strategy

```rust
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct ModelManager {
    cache: Arc<RwLock<HashMap<String, Arc<XGBoostModel>>>>,
    s3_client: S3Client,
    cache_dir: PathBuf,  // /tmp for Lambda
}

impl ModelManager {
    pub async fn get_or_load(
        &self,
        symbol: &str,
        timeframe: &str,
        version: &str,
    ) -> Result<Arc<XGBoostModel>> {
        let cache_key = format!("{symbol}_{timeframe}_{version}");
        
        // 1. Check in-memory cache
        {
            let cache = self.cache.read().await;
            if let Some(model) = cache.get(&cache_key) {
                return Ok(Arc::clone(model));
            }
        }
        
        // 2. Check /tmp filesystem cache
        let tmp_path = self.cache_dir.join(&cache_key).with_extension("json");
        if tmp_path.exists() {
            let model = Arc::new(XGBoostModel::from_json_file(&tmp_path)?);
            
            // Store in memory cache
            let mut cache = self.cache.write().await;
            cache.insert(cache_key.clone(), Arc::clone(&model));
            
            return Ok(model);
        }
        
        // 3. Download from S3
        let s3_key = format!("{symbol}_{timeframe}_{version}.json");
        self.s3_client.download_file(
            "xgboost-models-production",
            &s3_key,
            &tmp_path,
        ).await?;
        
        // 4. Load and cache
        let model = Arc::new(XGBoostModel::from_json_file(&tmp_path)?);
        
        let mut cache = self.cache.write().await;
        cache.insert(cache_key, Arc::clone(&model));
        
        Ok(model)
    }
}
```

### Inference API

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    pub label: String,              // "UP", "DOWN", "NEUTRAL"
    pub probabilities: [f64; 3],    // [prob_down, prob_neutral, prob_up]
    pub confidence: f64,            // Max probability
}

pub fn predict(features: &[f64], model: &XGBoostModel) -> Result<PredictionResult> {
    // Validate feature count
    if features.len() != 92 {
        return Err(XGBoostError::InvalidFeatureCount {
            expected: 92,
            got: features.len(),
        });
    }
    
    // Run inference
    model.predict(features)
}
```

---

## Section 5: Lambda Handler & API Design

### Request/Response Format

**Lambda Input Payload:**

```json
{
  "version": "1.0",
  "mode": "single",
  "requests": [
    {
      "symbol": "BTC/USDT",
      "timeframe": "1h",
      "ohlcv": [
        [1708531200000, 51234.5, 51500.0, 51100.0, 51450.0, 1234.56],
        [1708534800000, 51450.0, 51600.0, 51300.0, 51550.0, 1456.78]
      ],
      "model_version": "v1"
    }
  ],
  "options": {
    "return_features": false,
    "sqs_result_queue": null
  }
}
```

**Lambda Response:**

```json
{
  "success": true,
  "predictions": [
    {
      "symbol": "BTC/USDT",
      "timeframe": "1h",
      "prediction": {
        "label": "UP",
        "probabilities": {
          "DOWN": 0.15,
          "NEUTRAL": 0.25,
          "UP": 0.60
        },
        "confidence": 0.60,
        "model_version": "v1"
      },
      "metadata": {
        "candles_processed": 500,
        "features_calculated": 92,
        "inference_time_ms": 45
      }
    }
  ],
  "timing": {
    "total_ms": 150,
    "model_load_ms": 80,
    "feature_calc_ms": 50,
    "inference_ms": 20
  }
}
```

### Handler Logic Flow

```rust
use lambda_runtime::{service_fn, Error, LambdaEvent};

async fn handler(event: LambdaEvent<XGBoostRequest>) -> Result<XGBoostResponse, Error> {
    let start_time = Instant::now();
    
    // 1. Validate input
    let request = validate_request(event.payload)?;
    tracing::info!("Processing request: mode={}, symbols={}", 
        request.mode, request.requests.len());
    
    // 2. Load model from cache or S3
    let model_start = Instant::now();
    let model = MODEL_MANAGER.get_or_load(
        &request.requests[0].symbol,
        &request.requests[0].timeframe,
        &request.requests[0].model_version.as_deref().unwrap_or("v1"),
    ).await?;
    let model_load_ms = model_start.elapsed().as_millis() as u64;
    
    // 3. Calculate features
    let feature_start = Instant::now();
    let features = if request.mode == "batch" {
        // Parallel processing with Rayon
        request.requests
            .par_iter()
            .map(|req| {
                let engine = FeatureEngine::new(&req.ohlcv)?;
                engine.calculate_all()
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        let engine = FeatureEngine::new(&request.requests[0].ohlcv)?;
        vec![engine.calculate_all()?]
    };
    let feature_calc_ms = feature_start.elapsed().as_millis() as u64;
    
    // 4. Run inference
    let inference_start = Instant::now();
    let predictions = features
        .iter()
        .zip(request.requests.iter())
        .map(|(features, req)| {
            let prediction = model.predict(features)?;
            Ok(PredictionResponse {
                symbol: req.symbol.clone(),
                timeframe: req.timeframe.clone(),
                prediction,
                metadata: ResponseMetadata {
                    candles_processed: req.ohlcv.len(),
                    features_calculated: features.len(),
                    inference_time_ms: 0, // Updated below
                },
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let inference_ms = inference_start.elapsed().as_millis() as u64;
    
    // 5. Optional: Send to SQS
    if let Some(queue_url) = request.options.sqs_result_queue {
        let sqs_client = SQS_CLIENT.get().unwrap();
        sqs_client.send_results(&queue_url, &predictions).await?;
    }
    
    // 6. Return response
    let total_ms = start_time.elapsed().as_millis() as u64;
    
    Ok(XGBoostResponse {
        success: true,
        predictions,
        timing: TimingInfo {
            total_ms,
            model_load_ms,
            feature_calc_ms,
            inference_ms,
        },
    })
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .json()
        .init();
    
    // Initialize global state
    initialize_model_manager().await;
    initialize_sqs_client().await;
    
    // Run handler
    lambda_runtime::run(service_fn(handler)).await
}
```

### Error Handling Strategy

```rust
#[derive(Debug, thiserror::Error)]
pub enum XGBoostError {
    #[error("Invalid input: {0}")]
    ValidationError(String),  // 400 Bad Request
    
    #[error("Model not found: {symbol}_{timeframe}_{version}")]
    ModelNotFoundError {      // 404 Not Found
        symbol: String,
        timeframe: String,
        version: String,
    },
    
    #[error("Feature calculation failed: {0}")]
    FeatureCalculationError(String),  // 422 Unprocessable Entity
    
    #[error("Insufficient OHLCV data: need {required}, got {actual}")]
    InsufficientDataError {
        required: usize,
        actual: usize,
    },
    
    #[error("XGBoost inference failed: {0}")]
    InferenceError(String),  // 500 Internal Server Error
    
    #[error("S3 operation failed: {0}")]
    S3Error(String),  // 503 Service Unavailable
    
    #[error("SQS operation failed: {0}")]
    SQSError(String),
}

// Convert to Lambda-compatible response
impl From<XGBoostError> for lambda_runtime::Error {
    fn from(err: XGBoostError) -> Self {
        Box::new(err)
    }
}
```

---

## Section 6: Deployment & AWS Configuration

### AWS SAM Template (template.yaml)

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Description: XGBoost Serverless Inference Lambda

Globals:
  Function:
    Timeout: 30
    MemorySize: 3008  # Max Lambda memory = max CPU
    Runtime: provided.al2  # Custom Rust runtime
    Architectures:
      - x86_64  # or arm64 for Graviton2 (20% cost savings)
    Environment:
      Variables:
        RUST_LOG: info
        RUST_BACKTRACE: 1

Parameters:
  ModelBucketName:
    Type: String
    Default: xgboost-models-production
    Description: S3 bucket for model storage
  
  Environment:
    Type: String
    Default: production
    AllowedValues:
      - development
      - staging
      - production
    Description: Deployment environment

Resources:
  XGBoostFunction:
    Type: AWS::Serverless::Function
    Properties:
      FunctionName: !Sub "xgboost-inference-${Environment}"
      CodeUri: ./target/lambda/xgboost_lambda/
      Handler: bootstrap  # Rust Lambda custom runtime
      Environment:
        Variables:
          MODEL_BUCKET: !Ref ModelBucket
          ENVIRONMENT: !Ref Environment
      Policies:
        - S3ReadPolicy:
            BucketName: !Ref ModelBucket
        - SQSSendMessagePolicy:
            QueueName: !GetAtt ResultQueue.QueueName
      Events:
        ApiEvent:
          Type: Api
          Properties:
            Path: /predict
            Method: post
            RestApiId: !Ref XGBoostApi
      Tags:
        Environment: !Ref Environment
        Project: crypto-probability

  XGBoostApi:
    Type: AWS::Serverless::Api
    Properties:
      StageName: !Ref Environment
      Cors:
        AllowMethods: "'POST, OPTIONS'"
        AllowHeaders: "'Content-Type'"
        AllowOrigin: "'*'"

  ResultQueue:
    Type: AWS::SQS::Queue
    Properties:
      QueueName: !Sub "xgboost-results-${Environment}"
      MessageRetentionPeriod: 86400  # 24 hours
      VisibilityTimeout: 120
      Tags:
        - Key: Environment
          Value: !Ref Environment

  ModelBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Ref ModelBucketName
      VersioningConfiguration:
        Status: Enabled
      LifecycleConfiguration:
        Rules:
          - Id: DeleteOldVersions
            Status: Enabled
            NoncurrentVersionExpirationInDays: 30
          - Id: TransitionToIA
            Status: Enabled
            Transitions:
              - TransitionInDays: 90
                StorageClass: STANDARD_IA
      PublicAccessBlockConfiguration:
        BlockPublicAcls: true
        BlockPublicPolicy: true
        IgnorePublicAcls: true
        RestrictPublicBuckets: true
      Tags:
        - Key: Environment
          Value: !Ref Environment

  # CloudWatch Log Group
  XGBoostLogGroup:
    Type: AWS::Logs::LogGroup
    Properties:
      LogGroupName: !Sub "/aws/lambda/xgboost-inference-${Environment}"
      RetentionInDays: 30

  # CloudWatch Dashboard
  XGBoostDashboard:
    Type: AWS::CloudWatch::Dashboard
    Properties:
      DashboardName: !Sub "XGBoost-${Environment}"
      DashboardBody: !Sub |
        {
          "widgets": [
            {
              "type": "metric",
              "properties": {
                "metrics": [
                  ["AWS/Lambda", "Invocations", {"stat": "Sum"}],
                  [".", "Errors", {"stat": "Sum"}],
                  [".", "Duration", {"stat": "Average"}]
                ],
                "period": 300,
                "stat": "Average",
                "region": "${AWS::Region}",
                "title": "Lambda Metrics"
              }
            }
          ]
        }

Outputs:
  ApiUrl:
    Description: API Gateway endpoint
    Value: !Sub "https://${XGBoostApi}.execute-api.${AWS::Region}.amazonaws.com/${Environment}/predict"
    Export:
      Name: !Sub "XGBoostApiUrl-${Environment}"
  
  FunctionArn:
    Description: Lambda function ARN
    Value: !GetAtt XGBoostFunction.Arn
    Export:
      Name: !Sub "XGBoostFunctionArn-${Environment}"
  
  QueueUrl:
    Description: SQS result queue URL
    Value: !Ref ResultQueue
    Export:
      Name: !Sub "XGBoostQueueUrl-${Environment}"
  
  ModelBucketName:
    Description: S3 bucket for models
    Value: !Ref ModelBucket
    Export:
      Name: !Sub "XGBoostModelBucket-${Environment}"
```

### Build & Deployment Scripts

**scripts/deploy.sh:**

```bash
#!/bin/bash
set -e

ENVIRONMENT=${1:-development}
REGION=${2:-us-east-1}

echo "Deploying XGBoost Serverless to $ENVIRONMENT in $REGION..."

# 1. Build Rust Lambda
echo "Building Rust Lambda..."
cd lambda
cargo lambda build --release --arm64  # Use --x86-64 for x86_64

# 2. Run tests
echo "Running tests..."
cd ..
cargo test --release

# 3. Deploy with SAM
echo "Deploying with SAM..."
sam deploy \
  --stack-name "xgboost-serverless-${ENVIRONMENT}" \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
    Environment=$ENVIRONMENT \
    ModelBucketName="xgboost-models-${ENVIRONMENT}" \
  --region $REGION \
  --no-fail-on-empty-changeset

echo ""
echo "Deployment complete!"
echo ""
sam list endpoints --stack-name "xgboost-serverless-${ENVIRONMENT}" --region $REGION
```

**scripts/build.sh:**

```bash
#!/bin/bash
set -e

# Build for Lambda (Amazon Linux 2)
cd lambda
cargo lambda build --release --arm64

# Optional: Build with optimizations
RUSTFLAGS="-C target-cpu=native" cargo lambda build --release --arm64

echo "Build complete: target/lambda/xgboost_lambda/bootstrap"
```

### Python Training & Upload Script

**scripts/train_and_upload.py:**

```python
"""
Train XGBoost model in Python and upload to S3 for Lambda inference.
"""

import argparse
import boto3
import os
from pathlib import Path

from modules.xgboost.core.model import train_model_with_cv
from modules.common.core.data_fetcher import DataFetcher
from config import XGBOOST_PARAMS


def train_and_upload(
    symbol: str,
    timeframe: str,
    version: str = "v1",
    bucket: str = "xgboost-models-production",
    epochs: int = 200,
):
    """
    Train XGBoost model and upload to S3 for Lambda inference.
    
    Args:
        symbol: Trading pair (e.g., "BTC/USDT")
        timeframe: Timeframe (e.g., "1h")
        version: Model version (e.g., "v1")
        bucket: S3 bucket name
        epochs: Number of training epochs
    """
    print(f"Training XGBoost model: {symbol} {timeframe}")
    
    # 1. Fetch data
    fetcher = DataFetcher()
    df = fetcher.fetch_ohlcv(symbol, timeframe, limit=5000)
    
    # 2. Train model
    print("Training model...")
    model, metrics = train_model_with_cv(df, symbol, timeframe)
    
    print(f"Training complete. Accuracy: {metrics['accuracy']:.4f}")
    
    # 3. Export as JSON
    model_filename = f"{symbol.replace('/', '_')}_{timeframe}_{version}.json"
    model_path = Path("/tmp") / model_filename
    
    print(f"Exporting model to {model_path}")
    model.save_model(str(model_path))
    
    # 4. Upload to S3
    print(f"Uploading to S3: s3://{bucket}/{model_filename}")
    s3_client = boto3.client('s3')
    
    s3_client.upload_file(
        str(model_path),
        bucket,
        model_filename,
        ExtraArgs={'Metadata': {
            'symbol': symbol,
            'timeframe': timeframe,
            'version': version,
            'accuracy': str(metrics['accuracy']),
            'features': str(len(model.feature_names_in_)),
        }}
    )
    
    print(f"✓ Model uploaded successfully!")
    print(f"  S3 URI: s3://{bucket}/{model_filename}")
    print(f"  Features: {len(model.feature_names_in_)}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    
    # Clean up
    model_path.unlink()
    
    return f"s3://{bucket}/{model_filename}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and upload XGBoost model")
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair")
    parser.add_argument("--timeframe", default="1h", help="Timeframe")
    parser.add_argument("--version", default="v1", help="Model version")
    parser.add_argument("--bucket", default="xgboost-models-production", help="S3 bucket")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    
    args = parser.parse_args()
    
    train_and_upload(
        args.symbol,
        args.timeframe,
        args.version,
        args.bucket,
        args.epochs,
    )
```

---

## Section 7: Testing Strategy & Validation

### Testing Layers

#### 1. Unit Tests (Rust)

**tests/feature_tests.rs:**

```rust
use xgboost_serverless::features::*;
use approx::assert_relative_eq;

#[test]
fn test_rsi_calculation() {
    let close = vec![
        44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10,
        45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28,
    ];
    
    let rsi = calculate_rsi(&close, 14);
    
    // Expected values from Python pandas_ta
    let expected = 70.53;  // Last value
    assert_relative_eq!(rsi.last().unwrap(), expected, epsilon = 0.5);
}

#[test]
fn test_all_90_features() {
    let ohlcv = load_test_data("tests/test_data/btc_usdt_1h.json");
    let mut engine = FeatureEngine::new(ohlcv);
    
    let features = engine.calculate_all().unwrap();
    
    assert_eq!(features.len(), 92, "Should calculate all 92 features");
    assert!(
        !features.iter().any(|f| f.is_nan()),
        "No features should be NaN"
    );
}

#[test]
fn test_candlestick_patterns() {
    // Test DOJI pattern
    let ohlcv = OHLCVData {
        timestamp: vec![1],
        open: vec![100.0],
        high: vec![101.0],
        low: vec![99.0],
        close: vec![100.1],  // Very close to open
        volume: vec![1000.0],
    };
    
    let patterns = detect_candlestick_patterns(&ohlcv);
    assert!(patterns.doji[0], "Should detect DOJI pattern");
}

#[test]
fn test_moving_average() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let sma = calculate_sma(&data, 3);
    
    // Expected: [NaN, NaN, 2.0, 3.0, 4.0]
    assert!(sma[0].is_nan());
    assert!(sma[1].is_nan());
    assert_relative_eq!(sma[2], 2.0, epsilon = 0.001);
    assert_relative_eq!(sma[3], 3.0, epsilon = 0.001);
    assert_relative_eq!(sma[4], 4.0, epsilon = 0.001);
}

#[test]
fn test_returns_calculation() {
    let close = vec![100.0, 102.0, 101.0, 103.0];
    let returns = calculate_returns(&close, 1);
    
    // Expected: [NaN, 0.02, -0.0098, 0.0198]
    assert!(returns[0].is_nan());
    assert_relative_eq!(returns[1], 0.02, epsilon = 0.0001);
    assert_relative_eq!(returns[2], -0.0098, epsilon = 0.0001);
    assert_relative_eq!(returns[3], 0.0198, epsilon = 0.0001);
}
```

#### 2. Feature Parity Tests (Python ↔ Rust)

**scripts/validate_feature_parity.py:**

```python
"""
Validate that Rust feature calculations match Python exactly.
"""

import numpy as np
import pandas as pd
import subprocess
import json
from pathlib import Path

from modules.common.indicators import IndicatorEngine
from modules.xgboost.core.model import calculate_features


def test_feature_parity():
    """Ensure Rust features match Python within tolerance"""
    
    # 1. Load test data
    data_path = Path("tests/test_data/btc_usdt_1h.json")
    with open(data_path) as f:
        ohlcv_data = json.load(f)
    
    df = pd.DataFrame(ohlcv_data['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # 2. Calculate Python features
    print("Calculating Python features...")
    indicator_engine = IndicatorEngine()
    df = indicator_engine.calculate_indicators(df, profile="XGBOOST")
    python_features = calculate_features(df)
    
    # 3. Calculate Rust features
    print("Calculating Rust features...")
    result = subprocess.run(
        ["cargo", "run", "--bin", "calculate_features", "--", str(data_path)],
        cwd="modules/xgboost_LTS_serverless",
        capture_output=True,
        text=True,
    )
    rust_features = json.loads(result.stdout)
    
    # 4. Compare features
    print("Comparing features...")
    mismatches = []
    
    for feature_name in python_features.keys():
        if feature_name not in rust_features:
            mismatches.append(f"Missing in Rust: {feature_name}")
            continue
        
        py_val = python_features[feature_name]
        rust_val = rust_features[feature_name]
        
        # Handle NaN values
        if pd.isna(py_val) and pd.isna(rust_val):
            continue
        
        # Compare with tolerance
        if not np.allclose(py_val, rust_val, rtol=1e-4, atol=1e-6):
            diff = abs(py_val - rust_val)
            rel_diff = diff / abs(py_val) if py_val != 0 else diff
            mismatches.append(
                f"{feature_name}: Python={py_val:.6f}, Rust={rust_val:.6f}, "
                f"diff={diff:.6f} ({rel_diff*100:.2f}%)"
            )
    
    # 5. Report results
    if mismatches:
        print(f"\n❌ Feature parity check FAILED ({len(mismatches)} mismatches):")
        for mismatch in mismatches[:10]:  # Show first 10
            print(f"  - {mismatch}")
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches)-10} more")
        return False
    else:
        print(f"\n✓ Feature parity check PASSED (all {len(python_features)} features match)")
        return True


if __name__ == "__main__":
    success = test_feature_parity()
    exit(0 if success else 1)
```

#### 3. Model Inference Tests

**tests/inference_tests.rs:**

```rust
use xgboost_serverless::*;

#[tokio::test]
async fn test_model_loading_from_s3() {
    let manager = ModelManager::new("test-bucket", "/tmp");
    
    let model = manager
        .get_or_load("BTC_USDT", "1h", "v1")
        .await
        .expect("Failed to load model");
    
    assert!(model.is_loaded());
    assert_eq!(model.num_features(), 92);
}

#[test]
fn test_prediction_output_format() {
    let model = load_test_model("tests/test_data/test_model.json");
    
    // Create dummy features (92 values)
    let features: Vec<f64> = (0..92).map(|i| i as f64 * 0.01).collect();
    
    let result = model.predict(&features).unwrap();
    
    // Validate output
    assert_eq!(result.probabilities.len(), 3);
    assert!(
        (result.probabilities.iter().sum::<f64>() - 1.0).abs() < 0.01,
        "Probabilities should sum to 1.0"
    );
    assert!(
        ["UP", "DOWN", "NEUTRAL"].contains(&result.label.as_str()),
        "Label should be one of UP/DOWN/NEUTRAL"
    );
    assert!(
        result.confidence >= 0.0 && result.confidence <= 1.0,
        "Confidence should be in [0, 1]"
    );
}

#[test]
fn test_invalid_feature_count() {
    let model = load_test_model("tests/test_data/test_model.json");
    
    // Too few features
    let features = vec![0.0; 50];
    let result = model.predict(&features);
    
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        XGBoostError::InvalidFeatureCount { .. }
    ));
}
```

#### 4. Integration Tests (End-to-End)

**scripts/lambda_demo.py:**

```python
"""
End-to-end integration test for Lambda deployment.
"""

import json
import boto3
import requests
from typing import Dict, Any

from modules.common.core.data_fetcher import DataFetcher


class XGBoostLambdaClient:
    def __init__(self, endpoint_url: str):
        self.endpoint_url = endpoint_url
        self.lambda_client = boto3.client('lambda')
    
    def predict(
        self,
        symbol: str,
        timeframe: str,
        model_version: str = "v1",
    ) -> Dict[str, Any]:
        """Send prediction request to Lambda"""
        
        # Fetch recent OHLCV data
        fetcher = DataFetcher()
        df = fetcher.fetch_ohlcv(symbol, timeframe, limit=500)
        
        # Convert to Lambda format
        ohlcv = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.tolist()
        
        payload = {
            "version": "1.0",
            "mode": "single",
            "requests": [{
                "symbol": symbol,
                "timeframe": timeframe,
                "ohlcv": ohlcv,
                "model_version": model_version,
            }],
            "options": {
                "return_features": False,
            }
        }
        
        # Invoke Lambda
        response = requests.post(
            self.endpoint_url,
            json=payload,
            timeout=30,
        )
        
        return response.json()


def test_lambda_integration():
    """Full end-to-end Lambda test"""
    
    # 1. Train and upload model
    print("Training model...")
    from scripts.train_and_upload import train_and_upload
    model_uri = train_and_upload("BTC/USDT", "1h", "test_v1")
    print(f"Model uploaded: {model_uri}")
    
    # 2. Get Lambda endpoint
    endpoint = "https://your-api-id.execute-api.us-east-1.amazonaws.com/Prod/predict"
    
    # 3. Test prediction
    print("\nTesting prediction...")
    client = XGBoostLambdaClient(endpoint)
    result = client.predict("BTC/USDT", "1h", "test_v1")
    
    # 4. Validate response
    assert result['success'], "Request should succeed"
    assert len(result['predictions']) == 1, "Should have 1 prediction"
    
    prediction = result['predictions'][0]
    assert prediction['symbol'] == "BTC/USDT"
    assert prediction['prediction']['label'] in ['UP', 'DOWN', 'NEUTRAL']
    assert 0 <= prediction['prediction']['confidence'] <= 1
    
    # Check timing
    timing = result['timing']
    assert timing['total_ms'] < 5000, "Should complete within 5s (cold start)"
    print(f"\n✓ Test passed! Total time: {timing['total_ms']}ms")
    print(f"  Prediction: {prediction['prediction']['label']} "
          f"(confidence: {prediction['prediction']['confidence']:.2%})")
    
    return True


if __name__ == "__main__":
    test_lambda_integration()
```

#### 5. Performance Benchmarks

**benches/inference_benchmark.rs:**

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use xgboost_serverless::*;

fn bench_feature_calculation(c: &mut Criterion) {
    let ohlcv = load_test_data("tests/test_data/btc_1h_500.json");
    
    c.bench_function("calculate_all_features", |b| {
        b.iter(|| {
            let mut engine = FeatureEngine::new(&ohlcv);
            black_box(engine.calculate_all().unwrap())
        });
    });
}

fn bench_single_prediction(c: &mut Criterion) {
    let model = load_test_model("tests/test_data/test_model.json");
    let features: Vec<f64> = (0..92).map(|i| i as f64 * 0.01).collect();
    
    c.bench_function("single_prediction", |b| {
        b.iter(|| {
            black_box(model.predict(&features).unwrap())
        });
    });
}

fn bench_batch_prediction(c: &mut Criterion) {
    let requests = generate_batch_requests(50);  // 50 symbols
    
    c.bench_function("batch_50_predictions", |b| {
        b.iter(|| {
            black_box(process_batch_parallel(&requests).unwrap())
        });
    });
}

criterion_group!(
    benches,
    bench_feature_calculation,
    bench_single_prediction,
    bench_batch_prediction
);
criterion_main!(benches);
```

### Quality Gates

Before deployment to production:

1. ✅ **All unit tests pass** (100% pass rate)
2. ✅ **Feature parity validated** (< 0.1% difference from Python)
3. ✅ **Model predictions match Python** (within 1% for same input)
4. ✅ **Performance benchmarks met**:
   - Cold start < 2s
   - Warm inference < 100ms
   - Batch 50 symbols < 500ms
5. ✅ **No memory leaks** (valgrind/miri checks)
6. ✅ **Integration test passes** (real Lambda invocation)
7. ✅ **Load testing** (handle 100 concurrent requests)

---

## Implementation Checklist

### Phase 1: Core Infrastructure (Week 1)

- [x] Set up Rust workspace structure
- [x] Implement OHLCV data structures
- [x] Create error types
- [x] Set up testing framework
- [x] Implement basic S3 client

### Phase 2: Feature Engine (Week 2-3)

- [x] Implement price-derived features (5 features)
- [x] Implement moving averages (3 features)
- [x] Implement RSI family (3 features)
- [x] Implement MACD (3 features)
- [x] Implement volatility indicators (2 features)
- [x] Implement volume indicators (2 features)
- [x] Implement advanced features (12 features)
- [x] Implement all 48 candlestick patterns
- [x] Implement lag features (9 features)
- [x] Write feature parity tests

### Phase 3: XGBoost Integration (Week 4)

- [x] Integrate xgboost-rs crate
- [x] Implement model loading from JSON
- [x] Implement inference API
- [x] Implement model caching
- [x] Write inference tests

### Phase 4: Lambda Handler (Week 5)

- [x] Implement request/response structures
- [x] Implement handler logic
- [x] Implement batch processing
- [x] Implement SQS integration
- [x] Write integration tests

### Phase 5: Deployment (Week 6)

- [x] Create SAM template
- [x] Write deployment scripts
- [x] Write Python training script
- [x] Set up CI/CD pipeline
- [ ] Deploy to staging
- [x] Load testing
- [ ] Deploy to production

### Phase 6: Documentation (Week 7)

- [x] Write README with quick start
- [x] Document all 92 features
- [x] Write AWS setup guide
- [x] Write model export guide
- [x] Create deployment runbook

### Additional Completed Work (Post-Checklist)

- [x] Eliminated Rust build warnings across related workspaces and verified clean `cargo check`
- [x] Updated Lambda handler to use real `model.predict(&features)` flow instead of dummy prediction values
- [x] Added `timeframe` and `model_version` request handling with defaults and cache-key alignment
- [x] Added S3 fallback flow: download model on cache miss and load into model cache
- [x] Added `PredictionResult` serialization for Lambda JSON responses
- [x] Added Windows-safe feature gating so `cargo check --features xgboost` passes on Windows while preserving non-Windows real backend path
- [x] Verified module status with `cargo test` (10/10 pass) and `cargo build --workspace` (clean)
- [x] Refactored Lambda request schema to batch envelope (`mode`, `requests`, `options`) with timing metadata in response
- [x] Added `lambda/src/sqs_client.rs` and optional SQS result publishing via `options.sqs_result_queue`
- [x] Added `scripts/lambda_demo.py` end-to-end integration script for deployed API validation
- [x] Added `src/bin/calculate_features.rs` and `scripts/validate_feature_parity.py` for Python↔Rust feature parity workflow
- [x] Added module CI workflow at `modules/xgboost_LTS_serverless/.github/workflows/ci.yml`
- [x] Added `scripts/load_test.py` baseline concurrent load-testing utility
- [x] Added `docs/AWS_SETUP.md`, `docs/MODEL_EXPORT.md`, and `docs/DEPLOYMENT_RUNBOOK.md`
- [x] Expanded Lambda handler validation tests to cover batch and SQS option scenarios (7/7 passing)

---

## Performance Estimates

Based on similar Rust Lambda implementations:

- **Cold Start**: 1.5-2s (including S3 model download)
- **Warm Invocation**: 50-80ms per prediction
- **Feature Calculation**: 20-30ms for 500 candles
- **XGBoost Inference**: 10-20ms per prediction
- **Batch 50 Symbols**: 300-400ms (parallel processing)

**Cost Estimates (AWS Lambda):**

- Memory: 3008MB
- Duration: ~80ms average (warm)
- Requests: 1M/month
- Cost: ~$30/month (with 1M free tier requests)

---

## Future Enhancements

1. **Multi-Model Ensemble**: Support loading multiple models and ensemble voting
2. **Feature Selection API**: Allow clients to specify subset of features
3. **Real-time Data Fetching**: Integrate with exchange WebSocket for live data
4. **Model A/B Testing**: Support serving multiple model versions simultaneously
5. **Auto-Retraining Pipeline**: Trigger retraining on performance degradation
6. **ONNX Support**: Export models to ONNX for broader runtime support
7. **GPU Acceleration**: Use CUDA for faster inference if needed
8. **Streaming Inference**: WebSocket endpoint for real-time continuous predictions

---

## References

- **XGBoost Rust**: <https://github.com/davechallis/rust-xgboost>
- **AWS Lambda Rust Runtime**: <https://github.com/awslabs/aws-lambda-rust-runtime>
- **Cargo Lambda**: <https://github.com/cargo-lambda/cargo-lambda>
- **Technical Analysis Library**: <https://github.com/greyblake/ta-rs>
- **ATC Serverless Reference**: `modules/adaptive_trend_LTS_serverless/`
- **XGBoost Python Module**: `modules/xgboost/`

---

## End of Design Document

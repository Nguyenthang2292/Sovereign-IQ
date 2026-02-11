# AWS Lambda Serverless ATC Batch Scanning Implementation Plan

## Context

### Problem
The current ATC batch scanning system runs locally in the GUI application, which limits scalability when scanning 1000+ symbols across multiple timeframes. For large symbol sets, this can consume significant local CPU resources and take several minutes to complete.

### Solution
Migrate the ATC calculation workload to AWS Lambda using standalone Rust functions. This enables:
- **Massive parallelization**: 34+ Lambda functions processing batches simultaneously
- **No local resource consumption**: Heavy calculations offloaded to AWS
- **Fast execution**: 1000 symbols processed in ~1-2 minutes
- **Cost-effective**: ~$0.001 per full scan
- **Scalable**: Auto-scaling Lambda instances handle variable loads

### Architecture Overview

```
Local GUI App → Fetch OHLCV data (1000+ symbols × 3 TFs)
              ↓
              → Create 34 batches (30 symbols each)
              ↓
              → Invoke 34 Lambda functions (parallel, async)
                              ↓
AWS Lambda (Rust) → Process batch (30 symbols × 3 TFs)
                  → Calculate ATC signals
                  → Send results to SQS
                              ↓
              ← Poll SQS queue for results
              ↓
              → Aggregate and display in GUI
```

**Key Design Decisions** (from user answers):
- **Lambda Runtime**: Standalone Rust (no Python in Lambda)
- **Scale**: 1000+ symbols processing capability
- **Result Aggregation**: SQS queue (local app polls)
- **Orchestration**: Local Python app via boto3

---

## Implementation Phases

### Phase 1: Create ATC_LTS_serverless Rust Module

**Objective**: Extract ATC calculation logic from ATC_LTS_mini into a standalone Rust library (no PyO3 dependencies).

**Create new module**: `modules/adaptive_trend_LTS_serverless/`

**Directory structure**:
```
adaptive_trend_LTS_serverless/
├── Cargo.toml           # Workspace: library + Lambda binary
├── README.md
├── src/
│   ├── lib.rs           # Core ATC logic (MA, equity, signals)
│   ├── ma_calculations.rs
│   ├── equity.rs
│   ├── signal_detection.rs
│   ├── aggregation.rs
│   └── multi_tf_voting.rs
├── lambda/              # Lambda-specific code
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs      # Lambda handler
│       ├── handler.rs   # Request processing
│       ├── models.rs    # Serde data structures
│       └── sqs.rs       # SQS client
└── tests/
    └── atc_tests.rs
```

**Key tasks**:

1. **Extract MA calculations** from `modules/adaptive_trend_LTS_mini/rust_extensions/src/ma_calculations.rs`:
   - Remove PyO3 wrappers (`#[pyfunction]`, `PyReadonlyArray1`, `PyArray1`)
   - Keep internal calculation logic (`calculate_ema_internal`, `calculate_hma_internal`, etc.)
   - Convert input/output: `PyReadonlyArray1<f64>` → `&[f64]`, `PyArray1<f64>` → `Vec<f64>`
   - Functions to extract: `calculate_ema`, `calculate_hma`, `calculate_wma`, `calculate_dema`, `calculate_lsma`, `calculate_kama`

2. **Extract equity calculation** from `modules/adaptive_trend_LTS_mini/rust_extensions/src/equity.rs`:
   - Remove PyO3 bindings
   - Keep `equity_series()` core logic

3. **Extract signal detection** from `modules/adaptive_trend_LTS_mini/core/compute_atc_signals/`:
   - Port Layer 1 signal detection (trend classification)
   - Port Layer 2 equity weighting
   - Port Average_Signal aggregation

4. **Implement multi-TF voting**:
   - Port logic from `modules/auto_trade/core/atc_scanner.py` (`_aggregate_signals_python`)
   - Weighted voting across timeframes (15m: 0.5, 1h: 0.3, 4h: 0.2)
   - Threshold-based classification (LONG/SHORT/NEUTRAL)

5. **Define data models** with serde:
   ```rust
   // Input
   pub struct BatchRequest {
       pub batch_id: String,
       pub symbols: Vec<SymbolData>,
       pub config: ATCConfig,
   }

   pub struct SymbolData {
       pub symbol: String,
       pub timeframes: HashMap<String, OHLCVData>,
   }

   // Output
   pub struct SignalResult {
       pub symbol: String,
       pub score: f64,
       pub signal_type: String,  // "LONG", "SHORT", "NEUTRAL"
       pub details: HashMap<String, String>,
       pub strengths: HashMap<String, f64>,
   }
   ```

**Dependencies** (`Cargo.toml`):
```toml
[dependencies]
ndarray = "0.15"
rayon = "1.8"        # Parallelism
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
thiserror = "1.0"

# Lambda-specific (in lambda/Cargo.toml)
lambda_runtime = "0.8"
tokio = { version = "1", features = ["macros"] }
aws-config = "1.0"
aws-sdk-sqs = "1.0"
tracing = "0.1"
tracing-subscriber = "0.3"
```

**Critical files to reference**:
- [modules/adaptive_trend_LTS_mini/rust_extensions/src/ma_calculations.rs](modules/adaptive_trend_LTS_mini/rust_extensions/src/ma_calculations.rs) - MA functions
- [modules/adaptive_trend_LTS_mini/rust_extensions/src/equity.rs](modules/adaptive_trend_LTS_mini/rust_extensions/src/equity.rs) - Equity calculations
- [modules/auto_trade/core/atc_scanner.py](modules/auto_trade/core/atc_scanner.py) - Multi-TF voting logic

---

### Phase 2: Build Lambda Handler

**Objective**: Create Lambda function handler with SQS integration.

**Create Lambda binary** in `modules/adaptive_trend_LTS_serverless/lambda/src/`

**Key files**:

1. **main.rs** - Lambda entry point:
   ```rust
   use lambda_runtime::{run, service_fn, Error};

   #[tokio::main]
   async fn main() -> Result<(), Error> {
       tracing_subscriber::fmt()
           .with_max_level(tracing::Level::INFO)
           .json()
           .init();

       let sqs_client = sqs::SQSClient::new().await;
       let func = service_fn(|event| handler::handle_request(event, &sqs_client));
       run(func).await
   }
   ```

2. **handler.rs** - Request processing:
   ```rust
   pub async fn handle_request(
       event: LambdaEvent<BatchRequest>,
       sqs_client: &SQSClient,
   ) -> Result<(), Error> {
       let request = event.payload;

       // Process batch (Rayon parallelism)
       let results = compute_batch_signals(&request)?;

       // Send to SQS
       let scan_result = ScanResult { /* ... */ };
       sqs_client.send_batch_result(&scan_result).await?;

       Ok(())
   }
   ```

3. **sqs.rs** - SQS client:
   ```rust
   pub struct SQSClient {
       client: aws_sdk_sqs::Client,
       queue_url: String,
   }

   impl SQSClient {
       pub async fn new() -> Self {
           let config = aws_config::load_from_env().await;
           let client = aws_sdk_sqs::Client::new(&config);
           let queue_url = env::var("SQS_QUEUE_URL").unwrap();
           Self { client, queue_url }
       }

       pub async fn send_batch_result(&self, result: &ScanResult) -> Result<()> {
           let body = serde_json::to_string(result)?;
           self.client
               .send_message()
               .queue_url(&self.queue_url)
               .message_body(body)
               .send()
               .await?;
           Ok(())
       }
   }
   ```

**Performance optimizations**:
- Use Rayon for symbol-level parallelism (30 symbols processed concurrently)
- Pre-allocate arrays to reduce memory allocations
- Enable release profile optimizations: `opt-level = 3`, `lto = "thin"`, `strip = true`
- Target binary size: <15MB for fast cold starts (<1s)

---

### Phase 3: Implement Local Python Orchestrator

**Objective**: Create Python orchestrator to coordinate data fetching, Lambda invocations, and result collection.

**Create**: `modules/auto_trade/core/serverless_orchestrator.py`

**Class: ServerlessATCOrchestrator**

**Key methods**:

1. **scan_symbols()** - Main entry point:
   - Phase 1: Prefetch OHLCV data (reuse existing DataFetcher)
   - Phase 2: Create batches (30 symbols per batch)
   - Phase 3: Invoke Lambdas (async via boto3)
   - Phase 4: Poll SQS for results
   - Phase 5: Aggregate results into DataFrame

2. **_prefetch_ohlcv_data()** - Data fetching:
   ```python
   async def _prefetch_ohlcv_data(
       self, symbols: List[str], timeframes: List[str], limit: int
   ) -> Dict[str, Dict[str, pd.DataFrame]]:
       """Fetch OHLCV data for all symbols and timeframes."""
       data = {}

       for tf in timeframes:
           tf_data = self.data_fetcher.fetch_ohlcv_batch_parallel(
               symbols=symbols,
               timeframe=tf,
               limit=limit,
               max_workers=min(32, len(symbols)),
           )

           # Organize by symbol
           for symbol, (df, exchange_id) in tf_data.items():
               if df is not None:
                   if symbol not in data:
                       data[symbol] = {}
                   data[symbol][tf] = df

       return data
   ```

3. **_create_batches()** - Batch creation:
   ```python
   def _create_batches(
       self, ohlcv_data: Dict[str, Dict[str, pd.DataFrame]], atc_config: Dict
   ) -> List[Dict]:
       """Create Lambda payload batches (30 symbols each)."""
       batches = []
       symbols = list(ohlcv_data.keys())

       for i in range(0, len(symbols), self.batch_size):
           batch_symbols = symbols[i : i + self.batch_size]

           payload = {
               "batch_id": f"batch_{i // self.batch_size:04d}",
               "symbols": [
                   {
                       "symbol": symbol,
                       "timeframes": {
                           tf: {
                               "timestamp": df.index.astype(int).tolist(),
                               "open": df["open"].tolist(),
                               "high": df["high"].tolist(),
                               "low": df["low"].tolist(),
                               "close": df["close"].tolist(),
                               "volume": df["volume"].tolist(),
                           }
                           for tf, df in ohlcv_data[symbol].items()
                       }
                   }
                   for symbol in batch_symbols
               ],
               "config": atc_config,
           }
           batches.append(payload)

       return batches
   ```

4. **_invoke_lambda_batch()** - Lambda invocation:
   ```python
   async def _invoke_lambda_batch(self, batches: List[Dict]) -> None:
       """Invoke Lambda functions asynchronously for all batches."""
       tasks = [
           asyncio.create_task(self._invoke_lambda_async(batch))
           for batch in batches
       ]
       await asyncio.gather(*tasks)
   ```

5. **_collect_sqs_results()** - SQS polling:
   ```python
   async def _collect_sqs_results(self, expected_batches: int) -> List[Dict]:
       """Poll SQS queue and collect all batch results."""
       results = []
       received_batch_ids = set()

       while len(received_batch_ids) < expected_batches:
           # Long polling (20s wait)
           response = self.sqs_client.receive_message(
               QueueUrl=self.sqs_queue_url,
               MaxNumberOfMessages=10,
               WaitTimeSeconds=20,
           )

           for msg in response.get("Messages", []):
               body = json.loads(msg["Body"])
               batch_id = body.get("batch_id")

               if batch_id not in received_batch_ids:
                   results.append(body)
                   received_batch_ids.add(batch_id)

               # Delete message
               self.sqs_client.delete_message(
                   QueueUrl=self.sqs_queue_url,
                   ReceiptHandle=msg["ReceiptHandle"]
               )

       return results
   ```

**Dependencies**:
- boto3 (AWS SDK)
- Existing DataFetcher from `modules/common/core/data_fetcher/`

**Integration point**:
- Modify `modules/auto_trade/core/atc_scanner.py` to add serverless mode toggle
- Add configuration option: `use_serverless: bool`

**Critical files to modify**:
- [modules/auto_trade/core/atc_scanner.py](modules/auto_trade/core/atc_scanner.py) - Add serverless routing

---

### Phase 4: AWS Deployment

**Objective**: Deploy Lambda function and create AWS resources.

**Build Lambda binary**:
```bash
cd modules/adaptive_trend_LTS_serverless/lambda

# Build for AWS Lambda (x86_64)
cargo build --release --target x86_64-unknown-linux-musl

# Strip symbols to reduce size
strip target/x86_64-unknown-linux-musl/release/bootstrap

# Package for deployment
cd target/x86_64-unknown-linux-musl/release
zip function.zip bootstrap
```

**AWS Resources to create**:

1. **IAM Role**: `atc-lambda-execution-role`
   - Policies:
     - `AWSLambdaBasicExecutionRole` (CloudWatch Logs)
     - Custom policy: `SQS:SendMessage` on target queue

2. **SQS Queue**: `atc-scan-results`
   - Type: Standard (FIFO not required)
   - Message retention: 1 hour
   - Visibility timeout: 30s
   - Dead Letter Queue: `atc-scan-dlq`

3. **Lambda Function**: `atc-batch-scanner`
   - Runtime: `provided.al2023` (custom runtime)
   - Handler: `bootstrap`
   - Memory: 2048 MB
   - Timeout: 60s
   - Environment variables:
     - `SQS_QUEUE_URL`: SQS queue URL
     - `AWS_REGION`: us-east-1

**Deployment command**:
```bash
aws lambda create-function \
  --function-name atc-batch-scanner \
  --runtime provided.al2023 \
  --handler bootstrap \
  --zip-file fileb://function.zip \
  --role arn:aws:iam::ACCOUNT:role/atc-lambda-execution-role \
  --memory-size 2048 \
  --timeout 60 \
  --environment Variables="{SQS_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/ACCOUNT/atc-scan-results}"
```

**Monitoring**:
- CloudWatch Logs: `/aws/lambda/atc-batch-scanner` (retention: 7 days)
- Metrics: Invocations, Duration, Errors, Throttles
- Alarms:
  - Error rate > 5%
  - Average duration > 35s

---

### Phase 5: GUI Integration

**Objective**: Add serverless mode to GUI with configuration options.

**Modify**: `modules/auto_trade/settings.yaml`
```yaml
scanner:
  # Existing settings
  scan_interval: 5
  timeframe: 15m
  symbol_list: Top 20
  auto_start: true

  # New serverless settings
  use_serverless: false
  serverless_config:
    lambda_function_name: atc-batch-scanner
    sqs_queue_url: https://sqs.us-east-1.amazonaws.com/ACCOUNT/atc-scan-results
    aws_region: us-east-1
    batch_size: 30
```

**Update GUI**:
- Add toggle: "Use Serverless Scanning"
- Add AWS credentials configuration (or use ~/.aws/credentials)
- Add progress indicators:
  - Data fetching progress
  - Lambda invocation progress (X/34 batches invoked)
  - SQS polling progress (X/34 batches received)
- Add fallback: If serverless fails, fall back to local scanning

**Modified files**:
- [modules/auto_trade/settings.yaml](modules/auto_trade/settings.yaml) - Add serverless config
- [modules/auto_trade/gui/main_window.py](modules/auto_trade/gui/main_window.py) - Add UI toggle
- [modules/auto_trade/core/atc_scanner.py](modules/auto_trade/core/atc_scanner.py) - Route to serverless orchestrator

---

### Phase 6: Testing & Validation

**Unit tests**:
```bash
# Test Rust ATC logic
cd modules/adaptive_trend_LTS_serverless
cargo test

# Test Lambda handler locally (SAM CLI)
cd lambda
sam local invoke -e test_event.json
```

**Integration tests**:
1. **Local end-to-end test**:
   - Start with 10 symbols
   - Verify: Data fetch → Lambda invoke → SQS receive → Aggregation
   - Expected time: <30s

2. **Scale test**:
   - 100 symbols: Expected ~1 minute
   - 500 symbols: Expected ~1.5 minutes
   - 1000 symbols: Expected ~2 minutes

3. **Error handling test**:
   - Lambda failure: Verify retry + DLQ
   - SQS timeout: Verify timeout handling
   - Missing data: Verify graceful degradation

**Performance validation**:
- Cold start: <1s ✓
- Batch processing: <30s (30 symbols × 3 TFs) ✓
- Total scan (1000 symbols): <5 minutes ✓
- Cost per scan: <$0.01 ✓

---

## Critical Files Summary

### Files to Create (New)

1. **modules/adaptive_trend_LTS_serverless/src/lib.rs**
   - Core ATC logic extracted from ATC_LTS_mini (no PyO3)

2. **modules/adaptive_trend_LTS_serverless/lambda/src/main.rs**
   - Lambda handler entry point

3. **modules/auto_trade/core/serverless_orchestrator.py**
   - Local Python orchestrator for Lambda coordination

### Files to Modify (Existing)

1. **modules/auto_trade/core/atc_scanner.py**
   - Add serverless mode routing

2. **modules/auto_trade/settings.yaml**
   - Add serverless configuration options

3. **config/modules/auto_trade.py**
   - Add serverless config schema

### Files to Reference (Source)

1. **modules/adaptive_trend_LTS_mini/rust_extensions/src/ma_calculations.rs**
   - Source for MA calculations

2. **modules/adaptive_trend_LTS_mini/rust_extensions/src/equity.rs**
   - Source for equity calculations

3. **modules/common/core/data_fetcher/batch_parallel.py**
   - Reference for batch parallel fetching pattern

---

## Verification Steps

### Phase 1 Verification
```bash
cd modules/adaptive_trend_LTS_serverless
cargo test --lib
# All ATC calculation tests should pass
```

### Phase 2 Verification
```bash
cd modules/adaptive_trend_LTS_serverless/lambda
cargo build --release
# Binary size should be <15MB
# Test with SAM CLI: sam local invoke -e test_event.json
```

### Phase 3 Verification
```python
# Test orchestrator locally
from modules.auto_trade.core.serverless_orchestrator import ServerlessATCOrchestrator
from modules.common.core.exchange_manager import ExchangeManager
from modules.common.core.data_fetcher import DataFetcher

exchange_mgr = ExchangeManager()
data_fetcher = DataFetcher(exchange_mgr)
orchestrator = ServerlessATCOrchestrator(data_fetcher, ...)

# Mock Lambda/SQS for local testing
import asyncio
results = asyncio.run(orchestrator.scan_symbols(["BTC/USDT", "ETH/USDT"]))
assert len(results) > 0
```

### Phase 4 Verification
```bash
# Deploy to AWS
aws lambda invoke --function-name atc-batch-scanner \
  --payload file://test_payload.json \
  response.json

# Check CloudWatch Logs
aws logs tail /aws/lambda/atc-batch-scanner --follow

# Verify SQS message
aws sqs receive-message --queue-url $SQS_QUEUE_URL
```

### Phase 5 Verification
- Start GUI
- Enable "Use Serverless Scanning"
- Configure AWS credentials
- Run scan with 100 symbols
- Verify:
  - Progress indicators update correctly
  - Results appear in GUI tables
  - Execution time <2 minutes

### End-to-End Verification
1. **Full 1000-symbol scan**:
   - Start from GUI
   - Monitor CloudWatch Logs
   - Verify all 34 batches complete
   - Check cost (should be ~$0.001)
   - Total time: <5 minutes

2. **Error scenarios**:
   - Disconnect network during scan
   - Kill Lambda function mid-execution
   - Verify fallback to local scanning

---

## Performance Targets

| Metric | Target | Expected |
|--------|--------|----------|
| Cold start | <1s | 800-1200ms |
| Batch processing (30 symbols × 3 TFs) | <30s | 0.8-1.1s |
| Total scan (1000 symbols) | <5 min | 1-2 min |
| Cost per scan | <$1 | ~$0.001 |
| Lambda memory | 2048 MB | ~500-800 MB used |
| Binary size | <15 MB | 10-12 MB |

---

## Rollback Plan

If serverless implementation encounters issues:

1. **Immediate**: Disable serverless mode in GUI (toggle to local scanning)
2. **Short-term**: Keep existing local ATC scanning fully functional
3. **No breaking changes**: Serverless is opt-in, default to local mode

The implementation is designed to be **additive** - existing functionality remains unchanged.

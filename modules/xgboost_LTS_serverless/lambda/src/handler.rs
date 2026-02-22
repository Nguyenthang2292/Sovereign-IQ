use std::sync::OnceLock;
use std::time::Instant;
use tokio::task::JoinSet;

use lambda_runtime::{Error, LambdaEvent};
use serde::{Deserialize, Serialize};
use xgboost_serverless::{FeatureEngine, ModelManager, OHLCVData, PredictionResult, XGBoostError};

use crate::s3_client::S3Client;
use crate::sqs_client::SqsClient;

static MODEL_MANAGER: OnceLock<ModelManager> = OnceLock::new();
static AWS_CONFIG: OnceLock<aws_config::SdkConfig> = OnceLock::new();

async fn get_aws_config() -> &'static aws_config::SdkConfig {
    if let Some(config) = AWS_CONFIG.get() {
        return config;
    }

    let loaded = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
    let _ = AWS_CONFIG.set(loaded);
    AWS_CONFIG
        .get()
        .expect("AWS config should be initialized")
}

#[derive(Debug, Deserialize)]
pub struct PredictionItem {
    pub symbol: String,
    pub timeframe: Option<String>,
    pub model_version: Option<String>,
    pub timestamp: Option<i64>,
    pub data: OHLCVData,
    pub model_s3_key: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct PredictionMetadata {
    pub candles_processed: usize,
    pub features_calculated: usize,
    pub inference_time_ms: u64,
}

#[derive(Debug, Serialize)]
pub struct PredictionEntry {
    pub symbol: String,
    pub timeframe: String,
    pub prediction: PredictionResult,
    pub metadata: PredictionMetadata,
}

#[derive(Debug, Deserialize)]
pub struct RequestOptions {
    pub return_features: Option<bool>,
    pub sqs_result_queue: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct XGBoostRequest {
    pub version: Option<String>,
    pub mode: Option<String>,
    pub requests: Vec<PredictionItem>,
    pub options: Option<RequestOptions>,
}

#[derive(Debug, Serialize)]
pub struct TimingInfo {
    pub total_ms: u64,
    pub model_load_ms: u64,
    pub feature_calc_ms: u64,
    pub inference_ms: u64,
}

#[derive(Debug, Serialize)]
pub struct XGBoostResponse {
    pub success: bool,
    pub predictions: Vec<PredictionEntry>,
    pub timing: TimingInfo,
}

/// Fast JSON parser using SIMD acceleration (simd-json).
/// Reserved for future direct-invocation path; Lambda runtime currently
/// deserializes the event payload automatically via serde.
#[allow(dead_code)]
pub fn parse_request_simd(raw_json: &str) -> Result<XGBoostRequest, XGBoostError> {
    let mut bytes = raw_json.as_bytes().to_vec();
    simd_json::serde::from_slice(&mut bytes)
        .map_err(|error| XGBoostError::ValidationError(format!("Invalid JSON payload: {error}")))
}


pub async fn handle_request(event: LambdaEvent<XGBoostRequest>) -> Result<XGBoostResponse, Error> {
    let start_time = Instant::now();
    let request = event.payload;
    validate_request(&request)?;
    let request_version = request.version.as_deref().unwrap_or("1.0");
    let return_features = request
        .options
        .as_ref()
        .and_then(|options| options.return_features)
        .unwrap_or(false);
    tracing::debug!(
        request_version,
        return_features,
        "Processing request envelope"
    );

    MODEL_MANAGER.get_or_init(ModelManager::new);

    let mut model_load_ms: u64 = 0;
    let mut feature_calc_ms: u64 = 0;
    let mut inference_ms: u64 = 0;
    let mut predictions: Vec<PredictionEntry> = Vec::with_capacity(request.requests.len());

    let mut set = JoinSet::new();

    for item in request.requests {
        let item_timestamp = item.timestamp.unwrap_or(0);
        let timeframe = item.timeframe.clone().unwrap_or_else(|| "15m".to_string());
        let model_version = item
            .model_version
            .clone()
            .unwrap_or_else(|| "v1".to_string());
        
        set.spawn(async move {
            tracing::debug!(symbol = %item.symbol, timeframe, model_version, item_timestamp, "Processing prediction item");
            let model_manager = MODEL_MANAGER.get().unwrap();

            let model_start = Instant::now();
            let model = get_or_load_model(model_manager, &item, &timeframe, &model_version).await?;
            let ml_ms = model_start.elapsed().as_millis() as u64;

            let feature_start = Instant::now();
            let symbol = item.symbol;
            let data = item.data;
            let candles_processed = data.len();

            let features = tokio::task::spawn_blocking(move || {
                let mut feature_engine = FeatureEngine::new();
                feature_engine.calculate_all(&data)
            })
            .await
            .map_err(|e| Error::from(format!("Tokio spawn blocking error: {e}")))?
            .map_err(|e| Error::from(e.to_string()))?;
            
            let fc_ms = feature_start.elapsed().as_millis() as u64;

            let inference_start = Instant::now();
            let prediction = model
                .predict(&features)
                .map_err(|e| Error::from(format!("Inference failed: {e}")))?;
            let in_ms = inference_start.elapsed().as_millis() as u64;

            Ok::<_, Error>((
                PredictionEntry {
                    symbol,
                    timeframe,
                    prediction,
                    metadata: PredictionMetadata {
                        candles_processed,
                        features_calculated: features.len(),
                        inference_time_ms: in_ms,
                    },
                },
                ml_ms,
                fc_ms,
                in_ms,
            ))
        });
    }

    while let Some(res) = set.join_next().await {
        let (entry, ml_ms, fc_ms, in_ms) = res.map_err(|e| Error::from(format!("Task panic: {e}")))??;
        predictions.push(entry);
        model_load_ms += ml_ms;
        feature_calc_ms += fc_ms;
        inference_ms += in_ms;
    }

    let total_ms = start_time.elapsed().as_millis() as u64;
    let response = XGBoostResponse {
        success: true,
        predictions,
        timing: TimingInfo {
            total_ms,
            model_load_ms,
            feature_calc_ms,
            inference_ms,
        },
    };

    if let Some(options) = &request.options {
        if let Some(queue_url) = &options.sqs_result_queue {
            let aws_config = get_aws_config().await;
            let sqs_client = SqsClient::new(aws_config);
            sqs_client
                .send_json(queue_url, &response)
                .await
                .map_err(|e| Error::from(format!("Failed to send result to SQS: {e}")))?;
        }
    }

    Ok(response)
}

async fn get_or_load_model(
    model_manager: &ModelManager,
    item: &PredictionItem,
    timeframe: &str,
    model_version: &str,
) -> Result<std::sync::Arc<xgboost_serverless::XGBoostModel>, Error> {
    let mut model = model_manager.get_or_load(&item.symbol, timeframe, model_version);

    if let Err(XGBoostError::ModelNotFoundError(_)) = &model {
        if let Some(model_key) = &item.model_s3_key {
            let model_bucket = std::env::var("MODEL_BUCKET").map_err(|_| {
                Error::from("MODEL_BUCKET environment variable is not set".to_string())
            })?;
            let aws_config = get_aws_config().await;
            let s3_client = S3Client::new(aws_config, model_bucket);
            let bytes = s3_client
                .download_model(model_key)
                .await
                .map_err(|e| Error::from(format!("Failed to download model from S3: {e}")))?;

            let cache_key = format!("{}_{}_{}", item.symbol, timeframe, model_version);
            let tmp_path = std::path::PathBuf::from("/tmp")
                .join(cache_key.clone())
                .with_extension("json");
            std::fs::write(&tmp_path, &bytes)?;

            model_manager
                .load_into_cache(&cache_key, &tmp_path)
                .map_err(|e| Error::from(format!("Failed to load into cache: {e}")))?;

            model = model_manager.get_or_load(&item.symbol, timeframe, model_version);
        }
    }

    model.map_err(|e| Error::from(format!("Failed to get or load model: {e}")))
}

fn validate_request(request: &XGBoostRequest) -> Result<(), XGBoostError> {
    const MAX_BATCH_SIZE: usize = 50;

    if request.requests.is_empty() {
        return Err(XGBoostError::ValidationError(
            "Requests cannot be empty".to_string(),
        ));
    }

    if request.requests.len() > MAX_BATCH_SIZE {
        return Err(XGBoostError::ValidationError(format!(
            "Batch size exceeds maximum of {}",
            MAX_BATCH_SIZE
        )));
    }

    if let Some(mode) = &request.mode {
        if mode != "single" && mode != "batch" {
            return Err(XGBoostError::ValidationError(
                "Mode must be either 'single' or 'batch'".to_string(),
            ));
        }
    }

    for item in &request.requests {
        if item.symbol.is_empty() {
            return Err(XGBoostError::ValidationError(
                "Symbol cannot be empty".to_string(),
            ));
        }

        if item.data.is_empty() {
            return Err(XGBoostError::ValidationError(
                "Data cannot be empty".to_string(),
            ));
        }

        if item.data.len() < 50 {
            return Err(XGBoostError::ValidationError(
                "Need at least 50 data points".to_string(),
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        parse_request_simd, validate_request, PredictionItem, RequestOptions, XGBoostRequest,
    };
    use xgboost_serverless::OHLCVData;

    fn build_ohlcv(len: usize) -> OHLCVData {
        let timestamps: Vec<i64> = (0..len).map(|index| index as i64).collect();
        let open: Vec<f64> = (0..len).map(|index| 100.0 + index as f64).collect();
        let high: Vec<f64> = (0..len).map(|index| 101.0 + index as f64).collect();
        let low: Vec<f64> = (0..len).map(|index| 99.0 + index as f64).collect();
        let close: Vec<f64> = (0..len).map(|index| 100.5 + index as f64).collect();
        let volume: Vec<f64> = (0..len).map(|index| 1000.0 + index as f64).collect();

        OHLCVData::new(timestamps, open, high, low, close, volume)
            .expect("test OHLCV vectors should be valid")
    }

    fn build_item(symbol: &str, len: usize) -> PredictionItem {
        PredictionItem {
            symbol: symbol.to_string(),
            timeframe: Some("1h".to_string()),
            model_version: Some("v1".to_string()),
            timestamp: Some(1),
            data: build_ohlcv(len),
            model_s3_key: None,
        }
    }

    #[test]
    fn validate_single_request_success() {
        let request = XGBoostRequest {
            version: Some("1.0".to_string()),
            mode: Some("single".to_string()),
            requests: vec![build_item("BTC/USDT", 60)],
            options: Some(RequestOptions {
                return_features: Some(false),
                sqs_result_queue: None,
            }),
        };

        assert!(validate_request(&request).is_ok());
    }

    #[test]
    fn validate_batch_request_success() {
        let request = XGBoostRequest {
            version: Some("1.0".to_string()),
            mode: Some("batch".to_string()),
            requests: vec![build_item("BTC/USDT", 60), build_item("ETH/USDT", 60)],
            options: None,
        };

        assert!(validate_request(&request).is_ok());
    }

    #[test]
    fn validate_rejects_invalid_mode() {
        let request = XGBoostRequest {
            version: Some("1.0".to_string()),
            mode: Some("invalid".to_string()),
            requests: vec![build_item("BTC/USDT", 60)],
            options: None,
        };

        assert!(validate_request(&request).is_err());
    }

    #[test]
    fn validate_rejects_empty_requests() {
        let request = XGBoostRequest {
            version: Some("1.0".to_string()),
            mode: Some("batch".to_string()),
            requests: vec![],
            options: None,
        };

        assert!(validate_request(&request).is_err());
    }

    #[test]
    fn validate_rejects_too_few_candles() {
        let request = XGBoostRequest {
            version: Some("1.0".to_string()),
            mode: Some("single".to_string()),
            requests: vec![build_item("BTC/USDT", 49)],
            options: None,
        };

        assert!(validate_request(&request).is_err());
    }

    #[test]
    fn validate_accepts_sqs_queue_option() {
        let request = XGBoostRequest {
            version: Some("1.0".to_string()),
            mode: Some("batch".to_string()),
            requests: vec![build_item("BTC/USDT", 60), build_item("ETH/USDT", 60)],
            options: Some(RequestOptions {
                return_features: Some(false),
                sqs_result_queue: Some(
                    "https://sqs.us-east-1.amazonaws.com/123456789012/xgboost-results".to_string(),
                ),
            }),
        };

        assert!(validate_request(&request).is_ok());
    }

    #[test]
    fn validate_rejects_batch_when_any_symbol_empty() {
        let request = XGBoostRequest {
            version: Some("1.0".to_string()),
            mode: Some("batch".to_string()),
            requests: vec![build_item("BTC/USDT", 60), build_item("", 60)],
            options: Some(RequestOptions {
                return_features: Some(false),
                sqs_result_queue: Some(
                    "https://sqs.us-east-1.amazonaws.com/123456789012/xgboost-results".to_string(),
                ),
            }),
        };

        assert!(validate_request(&request).is_err());
    }

    #[test]
    fn validate_rejects_batch_over_50_requests() {
        let requests = (0..51)
            .map(|index| build_item(&format!("SYM{}", index), 60))
            .collect();

        let request = XGBoostRequest {
            version: Some("1.0".to_string()),
            mode: Some("batch".to_string()),
            requests,
            options: None,
        };

        assert!(validate_request(&request).is_err());
    }

    #[test]
    fn parse_request_simd_deserializes_valid_payload() {
        let payload = serde_json::json!({
            "version": "1.0",
            "mode": "single",
            "requests": [{
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "model_version": "v1",
                "timestamp": 1,
                "data": {
                    "timestamp": (0..50).collect::<Vec<i64>>(),
                    "open": vec![1.0; 50],
                    "high": vec![2.0; 50],
                    "low": vec![0.5; 50],
                    "close": vec![1.5; 50],
                    "volume": vec![100.0; 50]
                }
            }]
        })
        .to_string();

        let request =
            parse_request_simd(&payload).expect("SIMD parser should parse a valid payload");
        assert_eq!(request.requests.len(), 1);
        assert_eq!(request.requests[0].symbol, "BTC/USDT");
        assert_eq!(request.requests[0].data.close.len(), 50);
    }
}

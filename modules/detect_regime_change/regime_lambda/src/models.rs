use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Default)]
pub struct RegimeAnalysisConfig {
    pub pelt_model: Option<String>,
    pub pelt_min_segment: Option<usize>,
    pub hmm_train_ratio: Option<f64>,
    pub hmm_high_confidence_threshold: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct OhlcvData {
    pub timestamps: Vec<String>,
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Vec<f64>,
}

#[derive(Debug, Deserialize)]
pub struct RegimeAnalysisRequest {
    pub symbol: String,
    pub timeframe: String,
    pub lookback_days: u32,
    pub ohlcv: OhlcvData,
    #[serde(default)]
    pub config: Option<RegimeAnalysisConfig>,
}

#[derive(Debug, Serialize, PartialEq)]
pub struct RegimeAnalysisResponse {
    pub is_valid: bool,
    pub recommended_duration_hours: Option<f64>,
    pub pelt_avg_duration_hours: Option<f64>,
    pub hmm_next_state_duration_hours: Option<f64>,
    pub error: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialization() {
        let response = RegimeAnalysisResponse {
            is_valid: true,
            recommended_duration_hours: Some(4.0),
            pelt_avg_duration_hours: Some(4.5),
            hmm_next_state_duration_hours: Some(2.2),
            error: None,
        };
        let js = serde_json::to_string(&response).unwrap();
        
        let expected = r#"{"is_valid":true,"recommended_duration_hours":4.0,"pelt_avg_duration_hours":4.5,"hmm_next_state_duration_hours":2.2,"error":null}"#;
        assert_eq!(js, expected);
        
        // And deserialization side:
        let req_js = r#"{
            "symbol": "BTC/USDT",
            "timeframe": "15m",
            "lookback_days": 60,
            "ohlcv": {
                "timestamps": ["2026-03-09T00:00:00Z"],
                "open": [100.0],
                "high": [102.0],
                "low": [99.0],
                "close": [101.0],
                "volume": [10.0]
            }
        }"#;
        let req: RegimeAnalysisRequest = serde_json::from_str(req_js).unwrap();
        assert_eq!(req.symbol, "BTC/USDT");
        assert_eq!(req.ohlcv.close.len(), 1);
        assert_eq!(req.ohlcv.close[0], 101.0);
    }
}

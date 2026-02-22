use serde::{Deserialize, Serialize};

use crate::error::XGBoostError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OHLCVData {
    pub timestamp: Vec<i64>,
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Vec<f64>,
}

impl OHLCVData {
    pub fn new(
        timestamp: Vec<i64>,
        open: Vec<f64>,
        high: Vec<f64>,
        low: Vec<f64>,
        close: Vec<f64>,
        volume: Vec<f64>,
    ) -> Result<Self, XGBoostError> {
        let expected_len = timestamp.len();
        if open.len() != expected_len
            || high.len() != expected_len
            || low.len() != expected_len
            || close.len() != expected_len
            || volume.len() != expected_len
        {
            return Err(XGBoostError::ValidationError(format!(
                "OHLCV vector length mismatch: timestamp={}, open={}, high={}, low={}, close={}, volume={}",
                expected_len,
                open.len(),
                high.len(),
                low.len(),
                close.len(),
                volume.len()
            )));
        }

        Ok(Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        })
    }

    pub fn len(&self) -> usize {
        self.close.len()
    }

    pub fn is_empty(&self) -> bool {
        self.close.is_empty()
    }
}

use crate::error::XGBoostError;
use crate::EXPECTED_FEATURE_COUNT;
use serde::{Deserialize, Serialize};
#[cfg(all(feature = "xgboost", not(windows)))]
use std::cmp::Ordering;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    pub label: String,
    pub probabilities: [f64; 3],
    pub confidence: f64,
}

pub struct XGBoostModel {
    #[cfg(all(feature = "xgboost", not(windows)))]
    booster: xgboost::Booster,
}

impl XGBoostModel {
    pub fn from_json_file(_path: &Path) -> Result<Self, XGBoostError> {
        #[cfg(all(feature = "xgboost", not(windows)))]
        {
            let booster = xgboost::Booster::load(_path)
                .map_err(|e| XGBoostError::InferenceError(e.to_string()))?;
            Ok(Self { booster })
        }
        #[cfg(any(not(feature = "xgboost"), windows))]
        {
            Ok(Self {})
        }
    }

    pub fn predict(&self, features: &[f64]) -> Result<PredictionResult, XGBoostError> {
        if features.len() != EXPECTED_FEATURE_COUNT {
            return Err(XGBoostError::InvalidFeatureCount {
                expected: EXPECTED_FEATURE_COUNT,
                got: features.len(),
            });
        }

        #[cfg(all(feature = "xgboost", not(windows)))]
        {
            // Real inference
            let dmat = xgboost::DMatrix::from_dense(features, 1)
                .map_err(|e| XGBoostError::InferenceError(e.to_string()))?;

            let preds = self
                .booster
                .predict(&dmat)
                .map_err(|e| XGBoostError::InferenceError(e.to_string()))?;

            let mut probabilities = [0.0; 3];
            if preds.len() >= 3 {
                probabilities[0] = preds[0];
                probabilities[1] = preds[1];
                probabilities[2] = preds[2];
            } else if !preds.is_empty() {
                probabilities[0] = preds[0];
            }

            let (idx, &confidence) = probabilities
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
                .unwrap();

            let label = match idx {
                0 => "DOWN".to_string(),
                1 => "NEUTRAL".to_string(),
                2 => "UP".to_string(),
                _ => "NEUTRAL".to_string(),
            };

            Ok(PredictionResult {
                label,
                probabilities,
                confidence,
            })
        }
        #[cfg(any(not(feature = "xgboost"), windows))]
        {
            Ok(PredictionResult {
                label: "NEUTRAL".to_string(),
                probabilities: [0.1, 0.8, 0.1],
                confidence: 0.8,
            })
        }
    }
}

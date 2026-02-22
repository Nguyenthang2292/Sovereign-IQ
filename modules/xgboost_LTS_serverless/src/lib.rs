pub mod error;
pub mod feature_engine;
pub mod features;
pub mod ohlcv;

pub use error::XGBoostError;
pub use feature_engine::FeatureEngine;
pub use ohlcv::OHLCVData;

pub mod model_manager;
pub mod xgboost_inference;

pub use model_manager::ModelManager;
pub use xgboost_inference::{PredictionResult, XGBoostModel};

pub const EXPECTED_FEATURE_COUNT: usize = 92;

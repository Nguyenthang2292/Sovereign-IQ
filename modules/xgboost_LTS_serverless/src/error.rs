use thiserror::Error;

#[derive(Debug, Error)]
pub enum XGBoostError {
    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("Model not found: {0}")]
    ModelNotFoundError(String),

    #[error("Feature engineering error: {0}")]
    FeatureEngineeringError(String),

    #[error("Inference error: {0}")]
    InferenceError(String),

    #[error("Invalid feature count: expected {expected}, got {got}")]
    InvalidFeatureCount { expected: usize, got: usize },

    #[error("Training error: {0}")]
    TrainingError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(String),
}

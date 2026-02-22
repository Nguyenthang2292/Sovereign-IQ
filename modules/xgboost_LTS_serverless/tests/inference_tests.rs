use xgboost_serverless::error::XGBoostError;
use xgboost_serverless::xgboost_inference::XGBoostModel;
use xgboost_serverless::EXPECTED_FEATURE_COUNT;

#[cfg(not(feature = "xgboost"))]
#[test]
fn test_prediction_output_format() {
    let model = XGBoostModel::from_json_file(std::path::Path::new("dummy.json")).unwrap();
    let features: Vec<f64> = vec![0.5; EXPECTED_FEATURE_COUNT];
    let result = model.predict(&features).expect("Prediction should succeed");

    assert!(["UP", "DOWN", "NEUTRAL"].contains(&result.label.as_str()));
    assert_eq!(result.probabilities.len(), 3);
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
}

#[cfg(not(feature = "xgboost"))]
#[test]
fn test_invalid_feature_count_returns_error() {
    let model = XGBoostModel::from_json_file(std::path::Path::new("dummy.json")).unwrap();
    let features: Vec<f64> = vec![0.5; 50];

    let result = model.predict(&features);
    assert!(matches!(
        result,
        Err(XGBoostError::InvalidFeatureCount {
            expected: EXPECTED_FEATURE_COUNT,
            got: 50
        })
    ));
}

#[cfg(not(feature = "xgboost"))]
#[test]
fn test_probabilities_sum_to_one() {
    let model = XGBoostModel::from_json_file(std::path::Path::new("dummy.json")).unwrap();
    let features: Vec<f64> = vec![0.5; EXPECTED_FEATURE_COUNT];
    let result = model.predict(&features).unwrap();

    let sum: f64 = result.probabilities.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

#[cfg(not(feature = "xgboost"))]
#[test]
fn test_prediction_label_matches_max_probability() {
    let model = XGBoostModel::from_json_file(std::path::Path::new("dummy.json")).unwrap();
    let features: Vec<f64> = vec![0.5; EXPECTED_FEATURE_COUNT];
    let result = model.predict(&features).unwrap();

    let (max_idx, _) = result
        .probabilities
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();

    let expected_label = match max_idx {
        0 => "DOWN",
        1 => "NEUTRAL",
        2 => "UP",
        _ => "NEUTRAL",
    };

    assert_eq!(result.label, expected_label);
}

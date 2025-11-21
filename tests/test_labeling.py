import pandas as pd

from modules import xgboost_prediction_labeling as labeling


def test_apply_directional_labels_assigns_expected_classes(monkeypatch):
    monkeypatch.setattr(labeling, "TARGET_HORIZON", 2)
    monkeypatch.setattr(labeling, "TARGET_BASE_THRESHOLD", 0.05)

    df = pd.DataFrame(
        {
            "close": [100.0, 101.0, 120.0, 104.0, 90.0],
            "ATR_RATIO_14_50": [1.0] * 5,
        }
    )

    result = labeling.apply_directional_labels(df.copy())

    assert result.loc[0, "TargetLabel"] == "UP"
    assert result.loc[2, "TargetLabel"] == "DOWN"
    assert result.loc[1, "TargetLabel"] == "NEUTRAL"
    assert result.loc[0, "DynamicThreshold"] == 0.05
    assert result.loc[0, "Target"] == labeling.LABEL_TO_ID["UP"]

import numpy as np

from modules.xgboost_LTS.utils import display as display_module


def test_print_classification_report_calls_metrics(monkeypatch):
    called = {"report": False, "cm": False}

    def fake_report(y_true, y_pred, target_names, output_dict):
        called["report"] = True
        assert output_dict is False
        return "fake report"

    def fake_cm(y_true, y_pred):
        called["cm"] = True
        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    monkeypatch.setattr(display_module, "classification_report", fake_report)
    monkeypatch.setattr(display_module, "confusion_matrix", fake_cm)
    monkeypatch.setattr(display_module, "color_text", lambda text, color: text)
    monkeypatch.setattr(display_module, "log_analysis", lambda msg: None)
    monkeypatch.setattr(display_module, "log_info", lambda msg: None)
    monkeypatch.setattr(display_module, "log_model", lambda msg: None)

    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])

    display_module.print_classification_report(y_true, y_pred, title="Test")

    assert called["report"] is True
    assert called["cm"] is True

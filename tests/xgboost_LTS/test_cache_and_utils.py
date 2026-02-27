import time
import uuid

import numpy as np
import optuna
import pandas as pd
import pytest

from modules.xgboost_LTS.core.optimization import StudyManager
from modules.xgboost_LTS.utils.cache_manager import CacheManager
from modules.xgboost_LTS.utils.cv_utils import apply_cv_gap
from modules.xgboost_LTS.utils.utils import get_prediction_window


@pytest.mark.unit
def test_compute_df_hash_stable_and_sensitive_to_changes():
    manager = CacheManager(subsystem=f"xgboost_test_{uuid.uuid4().hex}")

    df = pd.DataFrame({"close": [100, 101, 102], "volume": [1, 2, 3]})
    same_df = df.copy()
    changed_df = df.copy()
    changed_df.loc[2, "close"] = 999

    hash_1 = manager._compute_df_hash(df)
    hash_2 = manager._compute_df_hash(same_df)
    hash_3 = manager._compute_df_hash(changed_df)

    assert hash_1 == hash_2
    assert hash_1 != hash_3


@pytest.mark.unit
def test_cache_manager_evict_oldest_when_limit_exceeded():
    manager = CacheManager(subsystem=f"xgboost_test_{uuid.uuid4().hex}", max_cache_entries=1)

    old_file = manager.models_dir / "old.json"
    new_file = manager.models_dir / "new.json"

    old_file.write_text("old", encoding="utf-8")
    time.sleep(0.01)
    new_file.write_text("new", encoding="utf-8")

    manager._evict_oldest(manager.models_dir)

    files = list(manager.models_dir.glob("*.json"))
    assert len(files) == 1
    assert files[0].name == "new.json"


@pytest.mark.unit
def test_get_prediction_window_returns_expected_mapping_and_fallback():
    assert get_prediction_window("1h") == "24h"
    assert get_prediction_window("unknown") == "next sessions"


@pytest.mark.unit
def test_study_manager_save_and_load_roundtrip(tmp_path):
    manager = StudyManager(storage_dir=str(tmp_path))

    study = optuna.create_study(direction="maximize")
    trial = study.ask({"max_depth": optuna.distributions.IntDistribution(3, 8)})
    study.tell(trial, 0.77)

    best_params = {"max_depth": 5, "learning_rate": 0.05}
    manager.save_study(study, symbol="BTCUSDT", timeframe="1h", best_params=best_params, best_score=0.77)

    loaded = manager.load_best_params("BTCUSDT", "1h", max_age_days=30)
    assert loaded == best_params


@pytest.mark.unit
def test_apply_cv_gap_filters_train_tail_and_overlapping_test_indices():
    train_idx = np.arange(0, 20)
    test_idx = np.arange(15, 30)

    filtered_train, filtered_test = apply_cv_gap(train_idx, test_idx, gap=3)

    assert filtered_train.tolist() == list(range(0, 17))
    assert filtered_test.tolist() == list(range(20, 30))

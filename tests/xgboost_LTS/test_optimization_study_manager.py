import optuna

from modules.xgboost_LTS.core.optimization import StudyManager


def test_study_manager_save_and_load(tmp_path):
    storage_dir = tmp_path / "optuna_storage"
    manager = StudyManager(storage_dir=str(storage_dir))

    study = optuna.create_study(direction="maximize")
    study.add_trial(
        optuna.trial.create_trial(
            value=0.42,
            params={"n_estimators": 100},
            distributions={"n_estimators": optuna.distributions.IntDistribution(50, 500)},
        )
    )

    manager.save_study(
        study=study,
        symbol="BTCUSDT",
        timeframe="1h",
        best_params={"n_estimators": 100},
        best_score=0.42,
    )

    loaded_params = manager.load_best_params("BTCUSDT", "1h", max_age_days=30)
    assert loaded_params == {"n_estimators": 100}

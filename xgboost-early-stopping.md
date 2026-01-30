# XGBoost Early Stopping (10-30% Training Reduction)

## Goal

Add early stopping to XGBoost training so training stops when validation metric stops improving for N rounds, reducing training time by 10-30% while preventing overfitting.

## Tasks

- [x] **Task 1**: Add `early_stopping_rounds` to `config/xgboost.py` XGBOOST_PARAMS (e.g. `20`) → Verify: `grep early_stopping config/xgboost.py` shows the key

- [x] **Task 2**: Update `model.fit()` at line 277 in `modules/xgboost_LTS/core/model.py` to pass `eval_set=[(X_test, y_test)]` and `verbose=False` → Verify: Holdout training uses eval_set for early stopping

- [x] **Task 3**: Update `model.fit()` at line 360 (sequential CV) in `model.py` to pass `eval_set` with the fold's test indices and `verbose=False` → Verify: CV folds use early stopping when validation data exists

- [x] **Task 4**: Update final `model.fit(X, y)` at line 395 in `model.py` to use temporal holdout (e.g. last 15–20% of rows) as `eval_set` if `early_stopping_rounds` is set → Verify: Final model training can stop early

- [x] **Task 5**: Update `run_parallel_cv` in `utils/cv_parallel.py` line 81 to pass `eval_set=[(X_test, y_test)]` and `verbose=False` for each fold → Verify: Parallel CV folds use early stopping

- [x] **Task 6**: Update `optimization.py` line 307 to pass `eval_set` (fold test) and `verbose=False` to `model.fit()` for Optuna trials → Verify: Optimization trials use early stopping

- [x] **Task 7**: Add unit test that trains with early stopping and asserts `best_iteration` (or equivalent) is set when applicable → Verify: `pytest tests/xgboost_LTS/ -v -k early` passes

- [x] **Task 8**: Run benchmark to confirm training time reduction → Verify: `pytest modules/xgboost_LTS/benchmarks/ -v` passes; optional: compare before/after timings

## Done When

- [x] Early stopping is configurable via `config/xgboost.py`
- [x] All `model.fit()` calls that have validation data use `eval_set` and early stopping
- [x] Tests pass and no regressions in accuracy

## Notes

- **XGBoost API**: `early_stopping_rounds` can be in constructor or `fit()`. Use `fit(..., eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)` for compatibility.
- **Fallback**: The `GradientBoostingWrapper` does not support `eval_set`. Only pass `eval_set`/`early_stopping_rounds` when the model is real XGBClassifier (e.g. check `'eval_set' in inspect.signature(model.fit).parameters` or try/except).
- **Final fit (line 395)**: Use last 15–20% of `X, y` as eval_set for early stopping; ensure sufficient samples.
- **config**: `early_stopping_rounds` is a training parameter; keep it in `XGBOOST_PARAMS` per roadmap, but extract before constructor if XGBClassifier rejects it in older versions.

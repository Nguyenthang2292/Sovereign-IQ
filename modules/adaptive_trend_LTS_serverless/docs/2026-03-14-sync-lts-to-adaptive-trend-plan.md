# Sync LTS Modules To adaptive_trend

## Goal
Make `modules/adaptive_trend_LTS_mini` and `modules/adaptive_trend_LTS_serverless` produce the same signal behavior as `modules/adaptive_trend` for Layer 1, Layer 2, and final classification.

## Tasks
- [x] Task 1: Lock source-of-truth behavior from `modules/adaptive_trend` into a short parity spec (signal state persistence, 9 MA set, Layer1 weighted_signal, Layer2 weighting, thresholds) -> Verify: `docs/parity_contract_adaptive_trend.md` exists with explicit formula and function/file references.
- [x] Task 2: Build deterministic golden datasets (trend up, trend down, sideways, noisy, NaN gaps) and expected outputs from `modules/adaptive_trend` (EMA/HMA/WMA/DEMA/LSMA/KAMA signals and `Average_Signal`) -> Verify: `tests/parity_fixtures/*.json` generated and reproducible with one command.
- [x] Task 3: Add cross-implementation parity harness that runs all 3 modules on the same fixture input and reports per-bar deltas -> Verify: one test command prints per-scenario max abs diff and pass/fail summary.
- [x] Task 4: Align `adaptive_trend_LTS_mini` signal generation to source-of-truth (`crossover/crossunder` event detection plus persistent state, not instant price>MA classification) -> Verify: mini outputs exact per-bar signal state on fixture cases (no drift after crossing bars).
- [x] Task 5: Align `adaptive_trend_LTS_mini` Layer1/Layer2 pipeline to source-of-truth (include base length MA + 8 diflen variants, equity-weighted Layer1, same cut_signal thresholds) -> Verify: mini `Average_Signal` matches source module within agreed tolerance on all fixtures.
- [x] Task 6: Align `adaptive_trend_LTS_serverless` Layer1 signal engine to source-of-truth (same 9 MA set and same event/persistence signal semantics) -> Verify: Rust per-MA Layer1 series matches source reference on fixtures.
- [x] Task 7: Align `adaptive_trend_LTS_serverless` final scoring path to source-of-truth (same threshold semantics, same cut_signal timing, same Layer2 weighting intent before final vote/classification) -> Verify: Rust final LONG/SHORT/NEUTRAL and score sign match source across all fixtures.
- [x] Task 8: Normalize MA numeric behavior and constants where needed (especially KAMA params/init and any MA warmup differences) to reduce structural drift -> Verify: MA series delta report shows only expected floating-point tolerance noise.
- [x] Task 9: Add regression tests in both LTS modules that fail on parity break (bar-level and final classification checks) and wire them into CI -> Verify: CI runs parity test jobs for Python mini and Rust serverless.
- [x] Task 10: Final verification run (must be last): run source module tests, mini tests, serverless tests, then run parity harness end-to-end -> Verify: all test commands pass and parity report shows no behavioral mismatches.

## Done When
- [x] All fixture scenarios produce matching LONG/SHORT/NEUTRAL classification across source, mini, and serverless.
- [x] `Average_Signal` direction and threshold crossing points match source behavior.
- [x] No remaining known logic mismatch is listed in parity report.
- [x] CI blocks merge if parity tests fail.

## Notes
Critical path: Task 1 -> Task 2 -> Task 3 -> Task 4/6 -> Task 5/7 -> Task 8 -> Task 9 -> Task 10.
Keep threshold and scaling parity explicit: lambda `/1000`, decay `/100`, long `0.1`, short `-0.1` unless source module changes.
Rust MSVC toolchain is now usable via `VsDevCmd.bat` (Build Tools 2022), but shell PATH still does not include `link.exe` by default.
Latest parity verification status:
- `report_adaptive_trend_ma_deltas.py --strict`: `total=30 passed=30 failed=0`.
- `run_adaptive_trend_parity_harness.py --strict --verbose`: `total=10 passed=10 failed=0`.
- Final verification run completed:
  - `pytest modules/adaptive_trend/signal_atc_test.py`: `65 passed`.
  - `pytest modules/adaptive_trend_LTS_mini/tests/test_parity_harness_regression.py`: `1 passed`.
  - `pytest modules/adaptive_trend_LTS_serverless/tests/test_parity_harness_regression.py`: `1 passed`.
- CI parity workflow added: `.github/workflows/adaptive_trend_parity.yml` (jobs: mini parity + serverless parity).

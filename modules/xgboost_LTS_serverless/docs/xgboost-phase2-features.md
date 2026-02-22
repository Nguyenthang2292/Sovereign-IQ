# XGBoost Phase 2: Feature Calculation Engine

## Goal

Implement the full 92-feature pipeline defined in Section 3 of the design document — from filling gaps in individual feature modules to a complete, cached `FeatureEngine::calculate_all()` orchestrator.

## Tasks

- [x] **Task 1 — Complete `price_derived.rs`** (4 functions)
  - Add `high_low_range(high, low, close)` → `(high - low) / close`
  - Add `close_open_diff(open, close)` → `(close - open) / open`
  - Add `log_volume(volume)` → `volume.map(|v| v.ln())`
  - Rename `returns_1`/`returns_5` calls to align with feature naming (`returns_n(close, n)`)
  - → Verify: unit test in `tests/feature_tests.rs` checks output length == input length

- [x] **Task 2 — Complete `indicators.rs`** (4 fixes/additions)
  - Fix `atr()`: replace stub `vec![0.0; period]` with full Wilder's smoothed ATR rolling calculation
  - Add `stochastic_rsi(close, rsi_period, stoch_period, smooth_k, smooth_d) -> (Vec<f64>, Vec<f64>)`
  - Add `on_balance_volume(close, volume) -> Vec<f64>`
  - Add `bollinger_band_percent(close, period, num_std) -> Vec<f64>` → `(close - lower) / (upper - lower)`
  - → Verify: `cargo test -p xgboost_serverless -- indicators` passes, ATR for known BTC data ≈ expected

- [x] **Task 3 — Expand `candlestick.rs` from 6 → 48 patterns**
  - Keep existing 6: doji, hammer, engulfing_bullish/bearish, morning_star, evening_star
  - Add 42 remaining patterns matching `CANDLESTICK_PATTERN_NAMES` in Python config:
    - Single-candle: inverted_hammer, shooting_star, marubozu_bull, marubozu_bear, spinning_top, gravestone_doji, dragonfly_doji, long_legged_doji
    - Two-candle: bullish_harami, bearish_harami, piercing, dark_cloud_cover, tweezer_top, tweezer_bottom, bullish_belt_hold, bearish_belt_hold
    - Three-candle: three_white_soldiers, three_black_crows, bullish_abandoned_baby, bearish_abandoned_baby, bullish_tri_star, bearish_tri_star, rising_three_methods, falling_three_methods, three_inside_up, three_inside_down, three_outside_up, three_outside_down
    - Up to 48 total — add remaining via proportional body/shadow ratio logic
  - Add `to_feature_vec(&self) -> Vec<f64>` (0.0/1.0 encoding per pattern) on `CandlestickPatterns`
  - → Verify: `CandlestickPatterns::detect()` returns struct with exactly 48 `Vec<bool>` fields, `to_feature_vec()` length == 48

- [x] **Task 4 — Add `FeatureCache` + complete `feature_engine.rs`**
  - Add `FeatureCache` struct (`HashMap<&'static str, Vec<f64>>`) with `get_or_insert(key, f)` method
  - Rewrite `FeatureEngine::calculate_all(&mut self, data: &OHLCVData) -> Result<Vec<f64>>` following pipeline in design Section 3 (steps 1–13)
  - Cache ATR, SMAs (20/50/200), RSI_14 for reuse across ratio/lag calculations
  - Add `assemble_feature_vector(n: usize, all_series: &[&[f64]]) -> Vec<f64>` — extracts row `n` (latest candle) from each feature series into a single flat `Vec<f64>` of length 92
  - → Verify: `engine.calculate_all(&ohlcv_500_candles)` returns `Ok(vec)` where `vec.len() == 92`

- [x] **Task 5 — Create `src/model_manager.rs` + `src/xgboost_inference.rs`**
  - `xgboost_inference.rs`: stub `XGBoostModel` that loads from JSON path using `xgboost-rs` feature flag; `predict(&[f64]) -> Result<PredictionResult>` returning `label`, `probabilities [f64;3]`, `confidence`
  - `model_manager.rs`: `ModelManager` with in-memory `HashMap` cache + `/tmp` filesystem cache; `get_or_load(symbol, timeframe, version) -> Result<Arc<XGBoostModel>>` (S3 download delegated to `s3_client.rs`)
  - Export both from `lib.rs`
  - → Verify: `cargo check` passes; `ModelManager::new()` constructs without panic

- [x] **Task 6 — Write integration tests + add test data**
  - Create `tests/test_data/btc_usdt_1h.json` — 600 rows of sample OHLCV (can be synthetic, fixed seed)
  - Create `tests/feature_tests.rs`:
    - `test_price_derived_lengths()` — each series length == input length
    - `test_atr_positive()` — all ATR values > 0 for valid data
    - `test_rsi_bounds()` — RSI values in [0, 100]
    - `test_candlestick_48_fields()` — struct has 48 pattern vecs
    - `test_feature_engine_92_features()` — `calculate_all()` returns 92 values
  - → Verify: `cargo test --manifest-path modules/xgboost_LTS_serverless/Cargo.toml` — all 5 tests pass

- [x] **Task 7 — Final build + docs update**
  - `cargo build --workspace` — zero warnings, zero errors
  - Update `docs/FEATURE_REFERENCE.md` with table of all 92 features, function name, category
  - Mark Section 3 tasks complete in `docs/xgboost-serverless-phase1-setup.md`
  - → Verify: `cargo build --workspace` outputs only `Finished` line; `FEATURE_REFERENCE.md` has 92 rows

## Done When

- [x] `calculate_all()` on 500-candle OHLCV returns exactly 92 features
- [x] All 6 tests in `tests/feature_tests.rs` pass
- [x] `cargo build --workspace` zero warnings

## Notes

- `xgboost-rs` crate requires CMake + libxgboost system library; guard behind `#[cfg(feature = "xgboost")]` so Tasks 1–6 compile without it
- Candlestick patterns use float threshold constants (e.g., `DOJI_RATIO = 0.1`) — define as `const` in `candlestick.rs`
- Per design, feature vector always represents the **latest candle** (index `n-1`); earlier rows only used for rolling window calculations
- Source of truth for 48 pattern names: `config/shared/model_features.py` → `CANDLESTICK_PATTERN_NAMES`

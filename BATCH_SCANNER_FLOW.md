# Quy trình Pre-filter (3-Stage Sequential Filtering)

## Entry Point

- **Module:** `modules/gemini_chart_analyzer/core/scanners/market_batch_scanner.py`
- **Method:** `MarketBatchScanner.scan_market()` → `_run_pre_filter()`
- **Workflow Module:** `modules/gemini_chart_analyzer/core/prefilter/workflow.py`
- **Workflow Function:** `run_prefilter_worker()`
- **Điều kiện:** `enable_pre_filter=True` và có `all_symbols`

---

## Flow Diagram

```

┌─────────────────────────────────────────────────────────────┐
│ 1. MarketBatchScanner.scan_market()                         │
│    - Step 1.5: Apply internal pre-filter if enabled         │
│    - Gọi: self._run_pre_filter()                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────────────── ┐
│ 2. MarketBatchScanner._run_pre_filter()                      │
│    File: market_batch_scanner.py (line ~769)                 │
│    - Validate percentage (default: 10%)                      │
│    - Gọi: run_prefilter_worker()                             │
└──────────────────────┬────────────────────────────────────── ┘
                       │
                       ▼
┌───────────────────────────────────────────────────────────       ──┐
│ 3. run_prefilter_worker()                                          │
│    File: workflow.py                                               │
│    Module: modules/gemini_chart_analyzer/core/prefilter/workflow.py│
│                                                                    │
│    Bước 3.1: Validate & Setup                                      │
│    - Validate all_symbols                                          │
│    - Tạo argparse.Namespace với config                             │
│    - Khởi tạo: ExchangeManager, DataFetcher, VotingAnalyzer        │
│    - Configure ATC performance settings (nếu có atc_performance)   │
└──────────────────────┬────────────────────────────────────       ──┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 0: Random Sampling (Optional)                          │
│    Function: run_prefilter_worker() - Stage 0 logic         │
│                                                              │
│    Bước 3.2: Random Sampling (nếu stage0_sample_percentage) │
│    - Nếu stage0_sample_percentage > 0 và < 100:            │
│      • Tính sample_count = total_symbols * percentage / 100│
│      • Random sample symbols từ all_symbols                │
│      • Kết quả: symbols_to_process (sampled symbols)       │
│    - Nếu không có sampling:                                  │
│      • symbols_to_process = all_symbols                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────   ┐
│ STAGE 1: ATC Filter (Filter lần 1)                             │
│    Function: _filter_stage_1_atc()                             │
│    File: workflow.py                                           │
│                                                                │
│    Bước 4.1: Run ATC Scan                                      │
│    - Gọi: analyzer.run_atc_scan()                              │
│    - Module: modules/adaptive_trend/signal_atc.py              │
│    - Quét symbols_to_process (sau Stage 0) để tìm ATC signals  │
│    - Kết quả:                                                  │
│      • long_signals_atc (DataFrame)                            │
│      • short_signals_atc (DataFrame)                           │
│                                                                │
│    Bước 4.2: Extract Symbols                                   │
│    - Lấy 100% symbols từ long_signals_atc và short_signals_atc │
│    - Chỉ lấy symbols có trong symbols_to_process               │
│    - Kết quả: stage1_symbols (100% của ATC results)            │
│    - Fallback: Nếu không có symbols, trả về all_symbols        │
└──────────────────────┬──────────────────────────────────────   ┘
                       │
                       ▼
┌───────────────────────────────────────────────────────────── ┐
│ STAGE 2: Range Oscillator + SPC Filter (Filter lần 2)        │
│    Function: _filter_stage_2_osc_spc()                       │
│    File: workflow.py                                         │
│                                                              │
│    Bước 5.1: Get ATC Signals for Stage 1 Symbols             │
│    - Lọc ATC signals chỉ cho symbols từ Stage 1              │
│    - Combine long_signals_atc và short_signals_atc           │
│    - Tách thành long_signals và short_signals                │
│                                                              │
│    Bước 5.2: Calculate Range Oscillator + SPC Signals        │
│    - Gọi: analyzer.calculate_signals_for_all_indicators()    │
│      với indicators_to_calculate=["oscillator", "spc"]       │
│    - Xử lý riêng cho LONG và SHORT signals                   │
│                                                              │
│      ┌──────────────────────────────────────────┐            │
│      │ 5.2.1: Range Oscillator                  │            │
│      │ Module: modules/range_oscillator/        │            │
│      │ - Tính oscillator signals                │            │
│      │ - Kết quả: osc_signal, osc_confidence    │            │
│      └──────────────────────────────────────────┘            │
│                                                              │
│      ┌──────────────────────────────────────────┐            │
│      │ 5.2.2: SPC (Simplified Percentile Clustering)│        │
│      │ Module: modules/simplified_percentile_clustering/│    │
│      │ - Tính SPC signals từ 3 strategies:       │           │
│      │   • Cluster Transition                    │           │
│      │   • Regime Following                      │           │
│      │   • Mean Reversion                        │           │
│      │ - Aggregator: SPCVoteAggregator           │           │
│      │ - Kết quả: spc_vote, spc_strength         │           │
│      └────────────────────────────────────────── ┘           │
│                                                              │
│    Bước 5.3: Apply Voting System                             │
│    - Gọi: analyzer.apply_voting_system()                     │
│      với indicators_to_include=["atc", "oscillator", "spc"]  │
│    - Module: core/voting_analyzer.py                         │
│    - Tính votes từ: ATC + Range Osc + SPC                    │
│    - Xử lý riêng cho LONG và SHORT                           │
│                                                              │
│    Bước 5.4: Decision Matrix                                 │
│    - Module: modules/decision_matrix/                        │
│    - Tính weighted_score từ votes                            │
│    - Filter theo voting_threshold và min_votes               │
│    - Chỉ giữ symbols có cumulative_vote = 1                  │
│    - Kết quả: stage2_symbols (100% của voting results)       │
│                stage2_signals (dict với 'long' và 'short'    │
│                DataFrames chứa signals)                      │
│    - Fallback: Nếu không có symbols, dùng stage1_symbols     │
└──────────────────────┬────────────────────────────────────── ┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 3: ML Models Filter (Filter lần 3)                    │
│    Function: _filter_stage_3_ml_models()                   │
│    File: workflow.py                                          │
│    Note: Trong fast mode, Stage 3 vẫn tính tất cả ML models │
│                                                              │
│    Bước 6.1: Enable ML Models                               │
│    - Lưu original flags: enable_xgboost, enable_hmm,       │
│      enable_random_forest                                    │
│    - Tạm thời enable: enable_xgboost=True, enable_hmm=True  │
│    - Enable random_forest nếu có rf_model_path              │
│                                                              │
│    Bước 6.2: Get ATC Signals for Stage 2 Symbols            │
│    - Lọc ATC signals chỉ cho symbols từ Stage 2             │
│    - Combine long_signals_atc và short_signals_atc          │
│    - Tách thành long_signals và short_signals                │
│                                                              │
│    Bước 6.3: Calculate ML Models Only                      │
│    - Gọi: analyzer.calculate_signals_for_all_indicators()   │
│      với indicators_to_calculate=["xgboost", "hmm", "random_forest"]│
│    - Xử lý riêng cho LONG và SHORT signals                 │
│    - Chỉ tính ML models, KHÔNG tính lại Range Osc và SPC    │
│                                                              │
│      ┌──────────────────────────────────────────┐          │
│      │ 6.3.1: XGBoost                           │          │
│      │ Module: modules/xgboost/                  │          │
│      │ - Predict signals từ trained model        │          │
│      │ - Kết quả: xgboost_signal, confidence    │          │
│      └──────────────────────────────────────────┘          │
│                                                              │
│      ┌──────────────────────────────────────────┐          │
│      │ 6.3.2: HMM                               │          │
│      │ Module: modules/hmm/                       │          │
│      │ - Hidden Markov Model analysis            │          │
│      │ - Kết quả: hmm_signal, hmm_confidence     │          │
│      └──────────────────────────────────────────┘          │
│                                                              │
│      ┌──────────────────────────────────────────┐          │
│      │ 6.3.3: Random Forest (nếu có model)        │          │
│      │ Module: modules/random_forest/             │          │
│      │ - Predict từ RF model                     │          │
│      │ - Kết quả: rf_signal, rf_confidence       │          │
│      └──────────────────────────────────────────┘          │
│                                                              │
│    Bước 6.4: Apply Voting System                            │
│    - Gọi: analyzer.apply_voting_system()                     │
│      với indicators_to_include=["atc", "xgboost", "hmm", "random_forest"]│
│    - Tính votes từ: ATC + XGBoost + HMM + RF                │
│      (KHÔNG include Range Osc và SPC)                        │
│    - Xử lý riêng cho LONG và SHORT                          │
│                                                              │
│    Bước 6.5: Decision Matrix                                │
│    - Module: modules/decision_matrix/                        │
│    - Tính weighted_score từ votes                           │
│    - Filter theo voting_threshold và min_votes               │
│    - Chỉ giữ symbols có cumulative_vote = 1                 │
│    - Kết quả: stage3_symbols (100% của voting results)      │
│                stage3_scores (dict {symbol: weighted_score})│
│    - Fallback: Nếu không có symbols, dùng stage2_symbols    │
│                                                              │
│    Bước 6.6: Restore ML Model Flags                         │
│    - Khôi phục lại original flags                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. Apply Percentage Filter (Optional)                       │
│    File: workflow.py - run_prefilter_worker()                │
│                                                              │
│    Bước 7.1: Check Auto-Skip Threshold                      │
│    - Nếu len(stage3_symbols) < auto_skip_threshold (default: 10):│
│      • Skip percentage filter, dùng tất cả stage3_symbols   │
│                                                              │
│    Bước 7.2: Apply Percentage Filter                        │
│    - Nếu percentage > 0 và < 100:                            │
│      • Tính target_count = len(stage3_symbols) * percentage / 100│
│      • Sort symbols theo weighted_score (từ stage3_scores)   │
│      • Lấy top target_count symbols                         │
│    - Nếu percentage = 100 hoặc không áp dụng:                │
│      • Trả về 100% stage3_symbols                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 8. Return Filtered Symbols                                  │
│    - Trả về: List[str] (filtered symbols)                   │
│    - Được sử dụng trong scan_market()                       │
│    - Thay thế all_symbols ban đầu                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Modules Tham Gia

1. **`market_batch_scanner.py`** - Entry point, gọi pre-filter
   - Location: `modules/gemini_chart_analyzer/core/scanners/market_batch_scanner.py`
   - Method: `MarketBatchScanner._run_pre_filter()`
2. **`workflow.py`** - Orchestrator, 4-stage filtering workflow
   - Location: `modules/gemini_chart_analyzer/core/prefilter/workflow.py`
   - Function: `run_prefilter_worker()`
   - Functions: `_filter_stage_1_atc()`, `_filter_stage_2_osc_spc()`, `_filter_stage_3_ml_models()`
3. **`core/voting_analyzer.py`** - Main analyzer, voting logic với subset indicators support
4. **`modules/adaptive_trend/signal_atc.py`** - ATC scan
5. **`modules/range_oscillator/`** - Range Oscillator signals
6. **`modules/simplified_percentile_clustering/`** - SPC analysis
7. **`modules/xgboost/`** - XGBoost predictions
8. **`modules/hmm/`** - HMM analysis
9. **`modules/random_forest/`** - RF predictions (nếu có model)
10. **`modules/decision_matrix/`** - Voting & decision logic
11. **`modules/common/core/data_fetcher.py`** - Fetch OHLCV data
12. **`modules/common/core/exchange_manager.py`** - Exchange connections

---

## 4-Stage Filtering Workflow

### Stage 0: Random Sampling (Optional)

- **Input:** Tất cả symbols từ market scan
- **Process:** Random sampling nếu `stage0_sample_percentage` được cung cấp
- **Output:** Sampled symbols (hoặc 100% nếu không có sampling)
- **Parameters:** `stage0_sample_percentage` (1-100, None để skip)
- **Use Case:** Giảm workload cho testing hoặc quick scans

### Stage 1: ATC Filter (Filter lần 1)

- **Input:** Symbols từ Stage 0 (hoặc tất cả symbols nếu không có Stage 0)
- **Process:** ATC scan để tìm LONG/SHORT signals
- **Output:** 100% symbols đã pass ATC scan
- **Indicators:** ATC only
- **Fallback:** Trả về all_symbols nếu không có symbols pass

### Stage 2: Range Oscillator + SPC Filter (Filter lần 2)

- **Input:** 100% symbols từ Stage 1
- **Process:**
  - Lọc ATC signals cho Stage 1 symbols
  - Tính Range Oscillator signals (riêng cho LONG và SHORT)
  - Tính SPC signals (3 strategies) (riêng cho LONG và SHORT)
  - Voting system với ATC + Range Osc + SPC
  - Decision Matrix filter
- **Output:** 100% symbols đã pass voting (cumulative_vote = 1)
- **Indicators:** ATC + Range Oscillator + SPC
- **Fallback:** Dùng stage1_symbols nếu không có symbols pass

### Stage 3: ML Models Filter (Filter lần 3)

- **Input:** 100% symbols từ Stage 2
- **Process:**
  - Enable tất cả ML models (ignore fast_mode flag)
  - Lọc ATC signals cho Stage 2 symbols
  - Tính chỉ ML models (XGBoost, HMM, RF) - riêng cho LONG và SHORT
  - KHÔNG tính lại Range Osc và SPC
  - Voting system với ATC + ML models only (exclude Range Osc và SPC)
  - Decision Matrix filter
- **Output:** 100% symbols đã pass voting (cumulative_vote = 1)
- **Indicators in Voting:** ATC + XGBoost + HMM + RF (Range Osc và SPC excluded)
- **Fallback:** Dùng stage2_symbols nếu không có symbols pass

### Final: Percentage Filter (Optional)

- **Input:** 100% symbols từ Stage 3
- **Process:** 
  - Auto-skip nếu len(stage3_symbols) < auto_skip_threshold (default: 10)
  - Áp dụng percentage filter nếu percentage > 0 và < 100
  - Sort theo weighted_score (từ stage3_scores) trước khi lấy top percentage
- **Output:** Top percentage symbols (hoặc 100% nếu percentage = 100 hoặc auto-skipped)

---

## Fast Mode vs Full Mode

### Fast Mode (`fast_mode=True`)

**Stage 1:**

- ✅ **ATC:** có

**Stage 2:**

- ✅ **ATC:** có (từ Stage 1)
- ✅ **Range Oscillator:** có
- ✅ **SPC:** có

**Stage 3:**

- ✅ **ATC:** có (từ Stage 1, dùng trong voting)
- ❌ **Range Oscillator:** không (đã dùng ở Stage 2, excluded khỏi Stage 3 voting)
- ❌ **SPC:** không (đã dùng ở Stage 2, excluded khỏi Stage 3 voting)
- ✅ **XGBoost:** có (ignore fast_mode flag)
- ✅ **HMM:** có (ignore fast_mode flag)
- ✅ **Random Forest:** có (nếu có model)

### Full Mode (`fast_mode=False`)

**Tất cả stages:**

- ✅ **Tất cả indicators:** ATC, Range Osc, SPC, XGBoost, HMM, RF

**Lưu ý:** Trong Full Mode, tất cả indicators đều được tính ở Stage 3, nhưng workflow vẫn giữ 3-stage để đảm bảo tính nhất quán.

---

## Kết Quả

- **Input:** `all_symbols` (ví dụ: 443 symbols)
- **Stage 0 Output (Optional):** `symbols_to_process` (ví dụ: 100 symbols nếu stage0_sample_percentage = 22.5%)
- **Stage 1 Output:** `stage1_symbols` (ví dụ: 200 symbols - 100% của ATC results)
- **Stage 2 Output:** `stage2_symbols` (ví dụ: 100 symbols - 100% của voting results)
- **Stage 3 Output:** `stage3_symbols` (ví dụ: 50 symbols - 100% của voting results)
- **Final Output:** `filtered_symbols` (ví dụ: 5 symbols nếu percentage = 10%)
- **Sorted by:** `weighted_score` (descending) - nếu áp dụng percentage filter
- **Fallback Behavior:** Mỗi stage có fallback về stage trước nếu không có symbols pass

---

## Đặc Điểm

### Ưu Điểm

- ✅ **Sequential Filtering:** Mỗi stage lọc dần, giảm workload cho stage sau
- ✅ **100% Processing:** Mỗi stage xử lý 100% kết quả của stage trước (không mất dữ liệu)
- ✅ **Comprehensive Voting:** Stage 3 sử dụng tất cả indicators cho voting toàn diện
- ✅ **Fast Mode Override:** Stage 3 vẫn tính ML models ngay cả trong fast mode
- ✅ **Flexible:** Có thể áp dụng percentage filter ở cuối nếu cần
- ✅ **Random Sampling:** Stage 0 cho phép quick testing với subset symbols
- ✅ **Fallback Safety:** Mỗi stage có fallback về stage trước nếu không có kết quả
- ✅ **Auto-Skip Logic:** Tự động skip percentage filter nếu đã có ít symbols

### Lưu Ý

- ✅ **Performance:** Stage 3 chỉ tính ML models, không tính lại Range Osc và SPC
- ✅ **Efficiency:** Mỗi stage chỉ tính indicators cần thiết cho voting của stage đó
- ⚠️ **Memory:** Lưu trữ signals từ Stage 2 (nhưng không dùng Range Osc và SPC trong Stage 3 voting)
- ⚠️ **Complexity:** Workflow phức tạp hơn nhưng linh hoạt và có thể mở rộng
- ⚠️ **Stage 0:** Random sampling có thể làm mất symbols quan trọng, chỉ dùng cho testing

---

## Parameters

### `run_prefilter_worker()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `all_symbols` | `List[str]` | Required | List of all symbols to filter |
| `percentage` | `float` | Required | Percentage of symbols to select (0-100) |
| `timeframe` | `str` | Required | Timeframe string for analysis |
| `limit` | `int` | Required | Number of candles per symbol |
| `mode` | `str` | `"voting"` | Pre-filter mode ('voting' or 'hybrid') |
| `fast_mode` | `bool` | `True` | Fast mode flag (Stage 3 still calculates all ML models) |
| `spc_config` | `Dict[str, Any] \| None` | `None` | Optional SPC configuration |
| `rf_model_path` | `str \| None` | `None` | Optional path to Random Forest model |
| `stage0_sample_percentage` | `float \| None` | `None` | Optional random sampling percentage (1-100) |
| `atc_performance` | `Dict[str, Any] \| None` | `None` | Optional ATC performance configuration |
| `auto_skip_threshold` | `int` | `10` | Auto-skip percentage filter if Stage 3 returns fewer symbols |

### ATC Performance Configuration

Khi `atc_performance` được cung cấp, các settings sau được áp dụng:

| Config Key | Type | Description |
|------------|------|-------------|
| `batch_processing` | `bool` | Enable batch processing |
| `use_cuda` | `bool` | Use CUDA for GPU acceleration |
| `parallel_l1` | `bool` | Parallel processing for Layer 1 |
| `parallel_l2` | `bool` | Parallel processing for Layer 2 |
| `use_dask` | `bool` | Use Dask for out-of-core processing |
| `npartitions` | `int \| None` | Number of Dask partitions |

---

## Tổng Kết

Quy trình 4-stage filtering này giúp:

1. **Giảm chi phí:** Lọc dần qua các stages, chỉ tính ML models cho symbols đã pass Stage 2
2. **Tăng chất lượng:** Mỗi stage đảm bảo chỉ giữ lại symbols có signals mạnh
3. **Linh hoạt:** Có thể điều chỉnh percentage filter ở cuối nếu cần
4. **Comprehensive:** Stage 3 sử dụng tất cả indicators để đánh giá toàn diện
5. **Testing Support:** Stage 0 random sampling cho phép quick testing với subset nhỏ
6. **Robust:** Fallback logic đảm bảo luôn có kết quả ngay cả khi stage nào đó fail
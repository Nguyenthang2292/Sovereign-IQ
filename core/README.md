# Core Analyzers - Workflow Comparison

Thư mục `core` chứa hai analyzer chính để kết hợp signals từ nhiều indicators:
- **HybridAnalyzer** (Phương án 1): Sequential filtering + voting system
- **VotingAnalyzer** (Phương án 2): Pure voting system

## Tổng quan

Cả hai analyzer đều sử dụng các indicators sau:
1. **ATC (Adaptive Trend Classification)**: Phân loại xu hướng thị trường
2. **Range Oscillator**: Xác định vùng quá mua/quá bán
3. **SPC (Simplified Percentile Clustering)**: Phân cụm dựa trên percentile (3 strategies)
4. **XGBoost** (optional): Machine learning prediction
5. **HMM** (optional): Hidden Markov Model signal prediction
6. **Random Forest** (optional): Random Forest model prediction

## HybridAnalyzer (Phương án 1)

**File**: `core/hybrid_analyzer.py`

### Triết lý
Kết hợp **sequential filtering** và **voting system**. Lọc từng bước để giảm số lượng symbols trước khi áp dụng voting.

### Workflow chi tiết

```
1. Determine timeframe
   └─ Chọn timeframe từ args hoặc interactive menu

2. Display configuration
   └─ Hiển thị cấu hình cho tất cả indicators

3. Run ATC auto scan
   └─ Quét tất cả symbols để tìm LONG/SHORT signals từ ATC
   └─ Kết quả: long_signals_atc, short_signals_atc

4. Filter by Range Oscillator confirmation ⭐
   └─ Với mỗi symbol từ ATC:
      ├─ Tính Range Oscillator signal
      ├─ Kiểm tra xem có khớp với ATC signal không
      └─ Chỉ giữ lại symbols có Range Oscillator xác nhận
   └─ Kết quả: long_signals_confirmed, short_signals_confirmed
   └─ Fallback: Nếu không có symbol nào pass, dùng lại ATC signals

5. Calculate SPC signals (if enabled)
   └─ Với mỗi symbol đã được xác nhận:
      ├─ Tính Cluster Transition signal
      ├─ Tính Regime Following signal
      └─ Tính Mean Reversion signal
   └─ Tính XGBoost signal (if enabled)
   └─ Tính HMM signal (if enabled)
   └─ Tính Random Forest signal (if enabled)
   └─ Kết quả: Thêm SPC, XGBoost, HMM và Random Forest signals vào DataFrame

6. Apply Decision Matrix voting (if enabled) ⭐
   └─ Với mỗi symbol:
      ├─ Tính votes từ tất cả indicators:
      │  ├─ ATC vote (luôn = 1 vì đã pass ATC scan)
      │  ├─ Oscillator vote (1 nếu khớp, 0 nếu không)
      │  ├─ SPC vote (aggregated từ 3 strategies)
      │  ├─ XGBoost vote (nếu enabled)
      │  ├─ HMM vote (nếu enabled)
      │  └─ Random Forest vote (nếu enabled)
      ├─ Áp dụng Decision Matrix với weighted voting
      └─ Chỉ giữ lại symbols có cumulative_vote = 1
   └─ Kết quả: long_signals_confirmed, short_signals_confirmed (đã được filter)

7. Display final results
   └─ Hiển thị kết quả cuối cùng với metadata
```

### Đặc điểm

✅ **Ưu điểm:**
- Giảm số lượng symbols sớm, tiết kiệm tài nguyên
- Range Oscillator filter loại bỏ nhiều false positives
- Sequential approach dễ debug và theo dõi
- Có fallback mechanism khi không có signal nào pass

❌ **Nhược điểm:**
- Có thể loại bỏ symbols tốt nếu Range Oscillator không khớp
- Phải chờ Range Oscillator xong mới tính SPC
- Phụ thuộc vào thứ tự filtering

### Code Flow

```python
analyzer = HybridAnalyzer(args, data_fetcher)
analyzer.run()
  ├─ run_atc_scan()                    # Step 1: ATC scan
  ├─ filter_by_oscillator()            # Step 2: Filter by Oscillator
  │  └─ filter_signals_by_range_oscillator()
  │     └─ _process_symbol_for_oscillator()  # Parallel processing
  ├─ calculate_spc_signals_for_all()   # Step 3: Calculate SPC (if enabled)
  │  └─ calculate_spc_signals()
  │     └─ _process_symbol_for_spc()  # Parallel processing
  ├─ filter_by_decision_matrix()       # Step 4: Apply voting (if enabled)
  │  └─ apply_decision_matrix()
  │     ├─ calculate_indicator_votes()
  │     └─ _aggregate_spc_votes()
  └─ display_results()                 # Step 5: Display
```

---

## VotingAnalyzer (Phương án 2)

**File**: `core/voting_analyzer.py`

### Triết lý
**Pure voting system**: Tính tất cả signals song song, sau đó áp dụng voting để quyết định. Không có sequential filtering.

### Workflow chi tiết

```
1. Determine timeframe
   └─ Chọn timeframe từ args hoặc interactive menu

2. Display configuration
   └─ Hiển thị cấu hình cho tất cả indicators

3. Run ATC auto scan
   └─ Quét tất cả symbols để tìm LONG/SHORT signals từ ATC
   └─ Kết quả: long_signals_atc, short_signals_atc

4. Calculate signals from all indicators in parallel ⭐
   └─ Với mỗi symbol từ ATC (song song):
      ├─ Tính ATC vote và strength (từ signal đã có)
      ├─ Tính Range Oscillator signal, vote, confidence
      ├─ Tính SPC signals (3 strategies) nếu enabled:
      │  ├─ Cluster Transition signal
      │  ├─ Regime Following signal
      │  └─ Mean Reversion signal
      ├─ Tính XGBoost signal, vote, confidence (nếu enabled)
      ├─ Tính HMM signal, vote, confidence (nếu enabled)
      └─ Tính Random Forest signal, vote, confidence (nếu enabled)
   └─ Kết quả: DataFrame với tất cả signals và votes

5. Apply voting system ⭐
   └─ Với mỗi symbol:
      ├─ Tạo DecisionMatrixClassifier với tất cả indicators
      ├─ Thêm votes từ tất cả indicators:
      │  ├─ ATC vote + strength
      │  ├─ Oscillator vote + confidence
      │  ├─ SPC vote (aggregated từ 3 strategies) + strength
      │  ├─ XGBoost vote + confidence (nếu enabled)
      │  ├─ HMM vote + confidence (nếu enabled)
      │  └─ Random Forest vote + confidence (nếu enabled)
      ├─ Tính weighted impact và cumulative vote
      └─ Chỉ giữ lại symbols có cumulative_vote = 1
   └─ Kết quả: long_signals_final, short_signals_final

6. Display final results
   └─ Hiển thị kết quả cuối cùng với voting metadata
```

### Đặc điểm

✅ **Ưu điểm:**
- Tính tất cả signals song song, nhanh hơn
- Không loại bỏ symbols sớm, giữ lại tất cả thông tin
- Voting system cân nhắc tất cả indicators cùng lúc
- Linh hoạt hơn, không phụ thuộc vào thứ tự

❌ **Nhược điểm:**
- Phải tính signals cho tất cả symbols, tốn tài nguyên hơn
- Không có early filtering để giảm workload
- Phức tạp hơn trong việc debug

### Code Flow

```python
analyzer = VotingAnalyzer(args, data_fetcher)
analyzer.run()
  ├─ run_atc_scan()                          # Step 1: ATC scan
  ├─ calculate_and_vote()                    # Step 2: Calculate & Vote
  │  ├─ calculate_signals_for_all_indicators()
  │  │  └─ _process_symbol_for_all_indicators()  # Parallel: tất cả signals
  │  └─ apply_voting_system()
  │     ├─ _aggregate_spc_votes()
  │     └─ DecisionMatrixClassifier
  └─ display_results()                       # Step 3: Display
```

---

## So sánh chi tiết

| Tiêu chí | HybridAnalyzer | VotingAnalyzer |
|----------|----------------|----------------|
| **Approach** | Sequential filtering + voting | Pure voting |
| **Filtering** | Có (Range Oscillator trước) | Không |
| **Signal Calculation** | Tuần tự (Oscillator → SPC) | Song song (tất cả cùng lúc) |
| **Performance** | Nhanh hơn (ít symbols hơn) | Chậm hơn (nhiều symbols hơn) |
| **Resource Usage** | Thấp hơn | Cao hơn |
| **Accuracy** | Có thể loại bỏ signals tốt | Giữ lại tất cả thông tin |
| **Flexibility** | Phụ thuộc thứ tự | Linh hoạt hơn |
| **Fallback** | Có (khi Oscillator không match) | Không |
| **Use Case** | Khi muốn filter sớm | Khi muốn xem xét tất cả |

## Khi nào dùng cái nào?

### Dùng HybridAnalyzer khi:
- Bạn muốn giảm số lượng symbols sớm để tiết kiệm tài nguyên
- Range Oscillator là indicator quan trọng và bạn muốn nó filter trước
- Bạn muốn có fallback mechanism
- Bạn cần workflow dễ debug và theo dõi

### Dùng VotingAnalyzer khi:
- Bạn muốn xem xét tất cả indicators cùng lúc
- Bạn không muốn loại bỏ symbols sớm
- Bạn có đủ tài nguyên để tính tất cả signals
- Bạn muốn voting system quyết định hoàn toàn

## Shared Components

Cả hai analyzer đều sử dụng:

- **`core/signal_calculators.py`**:
  - `get_range_oscillator_signal()`: Tính Range Oscillator signal
  - `get_spc_signal()`: Tính SPC signal cho một strategy
  - `get_xgboost_signal()`: Tính XGBoost prediction
  - `get_hmm_signal()`: Tính HMM signal prediction
  - `get_random_forest_signal()`: Tính Random Forest prediction

- **`modules.decision_matrix.classifier.DecisionMatrixClassifier`**:
  - Voting system với weighted impact và cumulative vote

- **`modules.simplified_percentile_clustering.aggregation.SPCVoteAggregator`**:
  - Aggregate 3 SPC strategies thành một vote

## Entry Points

- **HybridAnalyzer**: `main_hybrid.py`
- **VotingAnalyzer**: `main_voting.py`

Cả hai đều sử dụng `modules.common.utils.initialize_components()` để khởi tạo ExchangeManager và DataFetcher.


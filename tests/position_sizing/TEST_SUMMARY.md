# Test Cases Summary cho Position Sizing và Backtester Modules

## Tổng quan

Các test cases này bao phủ các tính năng mới đã được implement:
- Multiprocessing batch processing
- Multithreading cho indicators
- GPU support cho XGBoost
- Caching improvements
- Performance monitoring
- **DataFrame parameter optimization** (giảm số lần gọi API)

## Test Files

### 1. `test_position_sizer.py`
**Mục đích**: Test PositionSizer class - orchestrator chính

**Test Cases**:
- `test_calculate_position_size_structure`: Kiểm tra cấu trúc kết quả
- `test_calculate_position_size_with_custom_params`: Test với custom parameters
- `test_calculate_portfolio_allocation`: Test portfolio allocation
- `test_calculate_portfolio_allocation_normalization`: Test normalization khi vượt max exposure
- `test_empty_result_structure`: Test empty result structure
- `test_position_size_bounds`: Test min/max bounds

### 2. `test_hybrid_signal_calculator.py`
**Mục đích**: Test HybridSignalCalculator - tính toán hybrid signals

**Test Cases**:
- `test_hybrid_signal_calculator_initialization`: Test initialization
- `test_hybrid_signal_calculator_custom_indicators`: Test với custom indicators
- `test_calculate_hybrid_signal_returns_tuple`: Test return type
- `test_calculate_hybrid_signal_caching`: Test caching mechanism
- `test_combine_signals_majority_vote`: Test signal combination
- `test_combine_signals_insufficient_agreement`: Test insufficient agreement
- `test_clear_cache`: Test cache clearing
- `test_get_cache_stats`: Test cache statistics

### 3. `test_multithreading.py`
**Mục đích**: Test multithreading cho indicator calculations

**Test Cases**:
- `test_multithreading_enabled`: Test khi multithreading enabled
- `test_multithreading_disabled`: Test khi multithreading disabled
- `test_indicator_caching`: Test indicator result caching
- `test_indicator_cache_eviction`: Test cache eviction
- `test_calc_indicator_methods`: Test individual indicator methods
- `test_parallel_indicator_calculation_timeout`: Test timeout handling

### 4. `test_integration.py`
**Mục đích**: Integration tests cho toàn bộ workflow

**Test Cases**:
- `test_full_position_sizing_workflow`: Test complete workflow
- `test_portfolio_allocation_workflow`: Test portfolio allocation workflow
- `test_regime_adjustment_affects_position_size`: Test regime adjustment
- `test_kelly_calculator_integration`: Test Kelly calculator integration
- `test_error_handling_in_position_sizing`: Test error handling
- `test_position_size_bounds_enforcement`: Test bounds enforcement
- `test_portfolio_normalization`: Test portfolio normalization
- `test_different_signal_types`: Test với different signal types

### 5. `test_gpu_support.py`
**Mục đích**: Test GPU support cho XGBoost

**Test Cases**:
- `test_gpu_detection`: Test GPU detection logic
- `test_gpu_detection_no_gpu`: Test khi không có GPU
- `test_xgboost_with_gpu_config`: Test XGBoost với GPU config
- `test_xgboost_fallback_to_cpu`: Test fallback to CPU
- `test_xgboost_gpu_params_in_model`: Test GPU parameters
- `test_xgboost_cpu_mode`: Test CPU-only mode

## Backtester Tests

### 6. `test_parallel_processing.py`
**Mục đích**: Test multiprocessing batch processing

**Test Cases**:
- `test_parallel_processing_enabled`: Test khi parallel processing enabled
- `test_parallel_processing_fallback`: Test fallback to sequential
- `test_sequential_processing_for_small_datasets`: Test sequential cho small datasets
- `test_batch_processing_worker_function`: Test worker function
- `test_parallel_vs_sequential_consistency`: Test consistency giữa parallel và sequential

### 7. `test_performance.py`
**Mục đích**: Test performance monitoring và profiling

**Test Cases**:
- `test_performance_logging_enabled`: Test performance logging
- `test_performance_profiling_disabled`: Test khi profiling disabled
- `test_backtest_handles_empty_data`: Test với empty data
- `test_backtest_handles_missing_columns`: Test với missing columns
- `test_backtest_metrics_calculation`: Test metrics calculation

### 8. `test_edge_cases.py`
**Mục đích**: Test edge cases và error handling

**Test Cases**:
- `test_backtest_with_no_signals`: Test khi không có signals
- `test_backtest_with_very_small_dataset`: Test với very small dataset
- `test_backtest_with_extreme_price_movements`: Test với extreme volatility
- `test_backtest_with_constant_price`: Test với constant price
- `test_equity_curve_calculation`: Test equity curve calculation
- `test_metrics_calculation_with_no_trades`: Test metrics với no trades
- `test_metrics_calculation_with_only_winning_trades`: Test với only wins
- `test_metrics_calculation_with_only_losing_trades`: Test với only losses

## DataFrame Optimization Tests

### 9. `test_dataframe_optimization.py`
**Mục đích**: Test DataFrame parameter optimization trong PositionSizing module

**Test Cases**:
- `test_position_sizer_fetches_data_once`: Verify PositionSizer fetch data một lần và chia sẻ
- `test_regime_detector_with_dataframe_parameter`: Verify RegimeDetector nhận DataFrame parameter
- `test_regime_detector_without_dataframe_parameter`: Verify backward compatibility cho RegimeDetector
- `test_hybrid_signal_calculator_with_dataframe`: Verify HybridSignalCalculator sử dụng DataFrame
- `test_position_sizer_dataframe_sharing`: Verify DataFrame được chia sẻ giữa RegimeDetector và Backtester
- `test_backward_compatibility_position_sizer`: Verify backward compatibility cho PositionSizer

### 10. `test_dataframe_integration.py`
**Mục đích**: Integration tests cho DataFrame optimization flow

**Test Cases**:
- `test_end_to_end_dataframe_sharing`: Test end-to-end data sharing
- `test_regime_detector_and_backtester_independent_dataframe`: Test các components hoạt động độc lập với DataFrame
- `test_dataframe_consistency_across_components`: Verify consistency khi dùng cùng DataFrame
- `test_performance_improvement_with_dataframe`: Verify performance improvement (ít API calls hơn)

### 11. `tests/backtester/test_dataframe_parameter.py`
**Mục đích**: Test DataFrame parameter trong FullBacktester

**Test Cases**:
- `test_backtest_with_dataframe_parameter`: Verify rằng khi truyền DataFrame, không fetch từ API
- `test_backtest_without_dataframe_parameter`: Verify backward compatibility (không truyền df vẫn hoạt động)
- `test_backtest_dataframe_vs_fetch_same_result`: Verify kết quả giống nhau khi dùng DataFrame vs fetch
- `test_backtest_with_empty_dataframe`: Test với empty DataFrame
- `test_backtest_dataframe_preserves_index`: Verify DataFrame index được preserve

### 12. `tests/core/test_signal_calculators.py` (updated)
**Mục đích**: Test indicator functions với DataFrame parameter

**Test Cases** (thêm mới):
- `test_get_range_oscillator_signal_with_dataframe_parameter`: Verify DataFrame parameter hoạt động
- `test_get_range_oscillator_signal_backward_compatibility`: Verify backward compatibility

## Cách chạy tests

```bash
# Chạy tất cả tests cho position_sizing
pytest tests/position_sizing/ -v

# Chạy tất cả tests cho backtester
pytest tests/backtester/ -v

# Chạy một test file cụ thể
pytest tests/position_sizing/test_position_sizer.py -v

# Chạy tests cho DataFrame optimization
pytest tests/backtester/test_dataframe_parameter.py -v
pytest tests/position_sizing/test_dataframe_optimization.py -v
pytest tests/position_sizing/test_dataframe_integration.py -v

# Chạy tests cho signal calculators (bao gồm DataFrame tests)
pytest tests/core/test_signal_calculators.py -v

# Chạy tất cả tests liên quan đến DataFrame optimization
pytest tests/backtester/test_dataframe_parameter.py tests/position_sizing/test_dataframe_optimization.py tests/position_sizing/test_dataframe_integration.py -v

# Chạy với coverage
pytest tests/position_sizing/ --cov=modules.position_sizing --cov-report=html
```

## Coverage

Các test cases này bao phủ:
- ✅ Core functionality của PositionSizer
- ✅ Hybrid signal calculation với multithreading
- ✅ Multiprocessing batch processing
- ✅ GPU support detection và fallback
- ✅ Caching mechanisms
- ✅ Error handling và edge cases
- ✅ Performance monitoring
- ✅ Integration workflows
- ✅ **DataFrame parameter optimization**:
  - ✅ Optional DataFrame parameter hoạt động đúng
  - ✅ Backward compatibility (code cũ vẫn hoạt động)
  - ✅ DataFrame được sử dụng thay vì fetch API khi được truyền vào
  - ✅ PositionSizer fetch data một lần và chia sẻ
  - ✅ RegimeDetector và Backtester nhận optional DataFrame
  - ✅ Performance improvement (giảm số lần gọi API)
  - ✅ Integration giữa các components với DataFrame

## Notes

- Một số tests sử dụng mocking để tránh phụ thuộc vào external services
- GPU tests có thể không chạy được nếu không có GPU, nhưng vẫn test logic
- Parallel processing tests có thể chạy chậm hơn do overhead của multiprocessing
- Một số tests yêu cầu HMM module được setup đúng cách
- **DataFrame optimization tests**:
  - Tất cả tests sử dụng mocks để tránh gọi API thực sự
  - Tests verify cả behavior mới (với DataFrame) và backward compatibility (không có DataFrame)
  - Performance tests verify rằng số lần gọi API giảm khi sử dụng DataFrame


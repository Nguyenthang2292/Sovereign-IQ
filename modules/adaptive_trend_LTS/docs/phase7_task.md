# Phase 7: Memory Optimizations - Memory-Mapped Arrays & Data Compression

> **Scope**: Memory-mapped arrays for large datasets and compression for historical data
> **Expected Performance Gain**: 90% memory reduction for backtesting, 5-10x storage reduction
> **Timeline**: 2–3 weeks
> **Status**: ⚠️ **IN PROGRESS**

---

## 1. Mục tiêu

Triển khai các tối ưu hóa bộ nhớ để xử lý datasets lớn hơn và giảm chi phí lưu trữ:

- **Memory-Mapped Arrays**: Sử dụng memory-mapped files để xử lý datasets lớn mà không cần load toàn bộ vào RAM → **90% memory reduction** cho backtesting
- **Data Compression**: Nén dữ liệu lịch sử với blosc/zlib → **5-10x storage reduction** với <10% CPU overhead
- **Tương thích ngược** với code hiện tại (optional features)

## Expected Performance Gains

| Component | Current | Target (Optimized) | Expected Benefit |
| --------- | ------- | ----------------- | ---------------- |
| Backtesting Memory Usage | Full dataset in RAM | Memory-mapped files | 90% memory reduction |
| Historical Data Storage | Uncompressed files | Compressed with blosc | 5-10x storage reduction |
| Cache File Size | Uncompressed cache | Compressed cache | 5-10x smaller cache files |
| CPU Overhead | N/A | Compression/decompression | <10% CPU overhead |

---

## 2. Prerequisites & Dependencies

### 2.1 Required Software

#### Install Additional Dependencies

```bash
# Install blosc for compression
pip install blosc

# For testing
pip install pytest pytest-benchmark psutil

# Verify existing dependencies
pip install numpy pandas dask
```

### 2.2 Required Knowledge

- Memory-mapped files (numpy.memmap)
- Data compression algorithms (blosc, zlib)
- File I/O optimization
- Memory profiling techniques
- Existing backtesting and cache infrastructure

### 2.3 Existing Code to Review

- [core/backtesting/dask_backtest.py](../core/backtesting/dask_backtest.py) – Dask-based backtesting
- [utils/cache_manager.py](../../utils/cache_manager.py) – Cache management system
- [utils/config.py](../../utils/config.py) – ATCConfig class
- [core/scanner/process_symbol.py](../core/scanner/process_symbol.py) – Symbol processing

### 2.4 Timeline Estimate

| Task | Estimated Time | Priority |
| ---- | -------------- | -------- |
| **Part 1: Memory-Mapped Arrays** | 6 days | High |
| **Part 2: Data Compression** | 5 days | Medium |
| **Integration & Testing** | 4 days | High |
| **Benchmarking & Validation** | 2 days | High |
| **Documentation** | 2 days | Medium |
| **Total** | **~19 days (~2-3 weeks)** | |

---

## 3. Implementation Tasks

### 3.1 Part 1: Memory-Mapped Arrays for Large Datasets

#### Overview

Sử dụng memory-mapped files để xử lý datasets lớn mà không cần load toàn bộ vào RAM. Hữu ích cho backtesting với historical data lớn.

**Expected Gain**: **90% memory reduction** cho backtesting  
**Status**: ⚠️ **IN PROGRESS** (core utilities & Dask integration implemented; tests/benchmarks pending)

---

#### Task 3.1.1: Create Memory-Mapped Array Utility

**Status**: ✅ **COMPLETED**

**Implementation Details:**

- File location: `modules/adaptive_trend_LTS/utils/memory_mapped_data.py`
- Key functions to implement:
  - `create_memory_mapped_array()` - Create memory-mapped file from price data
  - `load_memory_mapped_array()` - Load existing memory-mapped file
  - `convert_to_memory_mapped()` - Convert pandas Series/DataFrame to memory-mapped format
  - `get_memory_mapped_info()` - Get metadata about memory-mapped file

**Example Implementation:**

```python
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

def create_memory_mapped_array(
    prices: pd.Series,
    filepath: str,
    dtype: np.dtype = np.float32,
    mode: str = 'w+'
) -> np.memmap:
    """Create memory-mapped array from price series.
    
    Args:
        prices: Price series to convert
        filepath: Path to memory-mapped file
        dtype: Data type (default: float32 for memory efficiency)
        mode: File mode ('w+' for write, 'r' for read-only)
    
    Returns:
        Memory-mapped numpy array
    """
    # Implementation here
    pass

def load_memory_mapped_array(
    filepath: str,
    shape: Tuple[int, ...],
    dtype: np.dtype = np.float32,
    mode: str = 'r'
) -> np.memmap:
    """Load existing memory-mapped array.
    
    Args:
        filepath: Path to memory-mapped file
        shape: Shape of the array
        dtype: Data type
        mode: File mode ('r' for read-only)
    
    Returns:
        Memory-mapped numpy array
    """
    # Implementation here
    pass
```

**Verification:**
- Memory-mapped file created successfully
- Can read data without loading full array into RAM
- Memory usage measured and reduced vs in-memory array

---

#### Task 3.1.2: Integrate Memory-Mapped Arrays into Backtesting

**Status**: ✅ **COMPLETED**

**Implementation Details:**

- Update `core/backtesting/dask_backtest.py` to support memory-mapped input
- Add option to use memory-mapped files for large historical datasets
- Ensure compatibility with existing Dask backtesting workflow

**Changes Required:**

1. Add `use_memory_mapped` parameter to `backtest_with_dask()`
2. Create memory-mapped files for large datasets before processing
3. Use memory-mapped arrays in symbol group processing
4. Clean up temporary memory-mapped files after processing

**Verification:**
- Backtest runs successfully with memory-mapped files
- Memory usage reduced by ~90% vs loading full dataset
- Results match non-memory-mapped backtesting

---

### 3.2 Part 2: Data Compression for Historical Data

#### Overview

Nén dữ liệu lịch sử với blosc để giảm storage footprint. Hữu ích cho cache files và historical data storage.

**Expected Gain**: **5-10x storage reduction** với <10% CPU overhead

**Status**: ✅ **COMPLETED (core compression feature implemented; validation & docs tracked in Section 4–5)**

---

#### Task 3.2.1: Add Blosc Compression Utility

**Status**: ✅ **COMPLETED**

**Implementation Details:**

- File location: `modules/adaptive_trend_LTS/utils/data_compression.py`
- Key functions to implement:
  - `compress_prices()` - Compress price data with blosc
  - `decompress_prices()` - Decompress price data
  - `compress_dataframe()` - Compress entire DataFrame
  - `decompress_dataframe()` - Decompress DataFrame
  - `get_compression_ratio()` - Calculate compression ratio

**Example Implementation:**

```python
import blosc
import numpy as np
import pandas as pd
from typing import Tuple

def compress_prices(
    prices: np.ndarray,
    typesize: int = 8,
    clevel: int = 5,
    cname: str = 'blosclz'
) -> bytes:
    """Compress price array with blosc.
    
    Args:
        prices: Price array to compress
        typesize: Size of each element in bytes
        clevel: Compression level (0-9, higher = better compression)
        cname: Compression algorithm name
    
    Returns:
        Compressed bytes
    """
    # Implementation here
    pass

def decompress_prices(
    compressed: bytes,
    dtype: np.dtype = np.float64,
    shape: Tuple[int, ...]
) -> np.ndarray:
    """Decompress price array from blosc.
    
    Args:
        compressed: Compressed bytes
        dtype: Data type of decompressed array
        shape: Shape of decompressed array
    
    Returns:
        Decompressed numpy array
    """
    # Implementation here
    pass
```

**Verification:**
- Compressed file is 5-10x smaller than original
- Decompression works correctly and produces identical data
- Compression/decompression time measured (<10% CPU overhead)

---

#### Task 3.2.2: Integrate Compression into Cache Manager

**Status**: ✅ **COMPLETED**

**Implementation Details:**

- Update `utils/cache_manager.py` to optionally compress cached data
- Add compression flag to cache operations
- Ensure backward compatibility (uncompressed cache still works)

**Changes Required:**

1. Add `use_compression` parameter to cache operations
2. Compress data before writing to cache
3. Decompress data when reading from cache
4. Handle both compressed and uncompressed cache files (backward compatibility)

**Verification:**
- Cache files are compressed when flag is enabled
- Decompression on read works correctly
- Backward compatible with existing uncompressed cache files

---

### 3.3 Part 3: Configuration & Integration

#### Overview

Thêm configuration flags để enable/disable memory optimizations và đảm bảo tương thích ngược.

**Timeline**: ~4 days

---

#### Task 3.3.1: Add Configuration Flags to ATCConfig

**Status**: ✅ **COMPLETED**

**Implementation Details:**

- Update `utils/config.py` to add memory optimization flags
- Add `use_memory_mapped: bool = False` - Enable memory-mapped arrays
- Add `use_compression: bool = False` - Enable data compression
- Add `compression_level: int = 5` - Compression level (0-9)
- Ensure backward compatibility (defaults to False)

**Verification:**
- Flags control behavior correctly
- Backward compatible (defaults to False, existing code works)
- Configuration can be passed to backtesting and cache operations

---

#### Task 3.3.2: Integration Testing

**Status**: ⚠️ **PENDING**

**Implementation Details:**

- Test memory-mapped arrays with backtesting
- Test compression with cache manager
- Test both features enabled simultaneously
- Verify backward compatibility

**Verification:**
- All integration tests pass
- Memory and storage improvements verified
- No regressions in existing functionality

---

## 4. Validation & Performance Testing

### 4.1 Part 1: Functional Validation

#### Overview

Đảm bảo memory optimizations hoạt động đúng và không ảnh hưởng đến kết quả tính toán.

**Timeline**: ~2 days

---

#### Task 4.1.1: Memory Usage Validation

**Status**: ⚠️ **PENDING**

**Validation Criteria:**

- Memory usage reduced by ~90% with memory-mapped arrays
- Compression ratio is 5-10x for typical price data
- CPU overhead for compression/decompression is <10%
- Results match non-optimized versions exactly

---

#### Task 4.1.2: Data Integrity Validation

**Status**: ⚠️ **PENDING**

**Validation Criteria:**

- Memory-mapped arrays produce identical results to in-memory arrays
- Compressed/decompressed data matches original exactly
- No data corruption or loss during compression/decompression
- Edge cases handled (empty arrays, single values, etc.)

---

### 4.2 Part 2: Performance Benchmarking

#### Overview

Tạo benchmark script để đo lường memory và storage improvements.

**Timeline**: ~1 day

---

#### Task 4.2.1: Create Benchmark Script

**Status**: ⚠️ **PENDING**

**Implementation Details:**

- File: `modules/adaptive_trend_LTS/benchmarks/benchmark_memory_optimizations.py`
- Benchmarks:
  - Memory usage: Memory-mapped vs in-memory arrays
  - Storage size: Compressed vs uncompressed files
  - CPU overhead: Compression/decompression time
  - Backtesting performance: With and without memory optimizations

**Verification:**
- Benchmark script runs successfully
- Memory and storage improvements measured and documented

---

## 5. Documentation & Deployment

### 5.1 Part 1: User Documentation

#### Overview

Tạo tài liệu hướng dẫn cho người dùng về cách sử dụng memory optimizations.

**Timeline**: ~2 days

---

#### Task 5.1.1: Create Usage Guide

**Status**: ⚠️ **PENDING**

**Output:**

- Document location: `modules/adaptive_trend_LTS/docs/memory_optimizations_usage_guide.md`
- Sections:
  - When to use memory-mapped arrays
  - When to use compression
  - How to enable optimizations
  - Performance characteristics
  - Limitations and trade-offs

---

### 5.2 Part 2: Code Documentation

#### Overview

Cập nhật documentation code để phản ánh memory optimizations mới.

**Timeline**: ~1 day

---

#### Task 5.2.1: Update Code Documentation

**Status**: ⚠️ **PENDING**

**Updates:**

- Update `README.md` with memory optimization features
- Update inline code documentation
- Add usage examples
- Document configuration options

---

## 6. Summary

### 6.1 What Will Be Accomplished

| Component | Status | Key Achievement |
| --------- | ------ | ---------------- |
| Memory-Mapped Array Utility | ⚠️ Pending | 90% memory reduction for backtesting |
| Backtesting Integration | ⚠️ Pending | Support for large datasets without RAM limits |
| Compression Utility | ⚠️ Pending | 5-10x storage reduction |
| Cache Manager Integration | ⚠️ Pending | Compressed cache files |
| Configuration Flags | ⚠️ Pending | Optional features with backward compatibility |
| Unit Tests | ⚠️ Pending | Comprehensive test coverage |
| Performance Benchmarks | ⚠️ Pending | Measured memory and storage improvements |

### 6.2 Expected Benefits

| Use Case | Before | After | Improvement |
| --------- | ------- | ------ | ----------- |
| Backtesting Large Datasets | Limited by RAM | Memory-mapped files | 90% memory reduction |
| Historical Data Storage | Uncompressed files | Compressed with blosc | 5-10x storage reduction |
| Cache File Size | Uncompressed | Compressed | 5-10x smaller files |
| CPU Overhead | N/A | Compression/decompression | <10% overhead |

### 6.3 Known Considerations

1. **Memory-Mapped Arrays**:
   - **Benefit**: 90% memory reduction for large datasets
   - **Trade-off**: Slightly slower access than in-memory arrays (acceptable for backtesting)
   - **Use Case**: Large historical datasets that don't fit in RAM

2. **Data Compression**:
   - **Benefit**: 5-10x storage reduction
   - **Trade-off**: <10% CPU overhead for compression/decompression
   - **Use Case**: Cache files and historical data storage

3. **Backward Compatibility**:
   - Both features are optional (defaults to False)
   - Existing code continues to work without changes
   - Users can opt-in to optimizations as needed

---

## 7. Next Steps

1. ⚠️ **Part 1 (Memory-Mapped Arrays)**: Create utility and integrate into backtesting
2. ⚠️ **Part 2 (Data Compression)**: Create utility and integrate into cache manager
3. ⚠️ **Part 3 (Configuration)**: Add flags to ATCConfig
4. ⚠️ **Part 4 (Testing)**: Create unit tests and benchmarks
5. ⚠️ **Part 5 (Documentation)**: Create usage guide and update code docs

**Final Status**:
- ⚠️ **Phase 7 is IN PROGRESS** - Implementation pending
- ⚠️ **Part 1 (Memory-Mapped Arrays)**: Pending implementation
- ⚠️ **Part 2 (Data Compression)**: Pending implementation

**Deliverables**:
1. Memory-mapped array utility for 90% memory reduction
2. Compression utility for 5-10x storage reduction
3. Integration with backtesting and cache manager
4. Optional features with backward compatibility
5. Comprehensive test coverage
6. Full documentation with usage examples

---

**End of Phase 7 Task List**

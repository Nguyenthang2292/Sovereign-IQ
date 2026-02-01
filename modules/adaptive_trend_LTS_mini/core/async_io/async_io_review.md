# Code Review: modules/adaptive_trend_LTS_mini/core/async_io/async_compute.py

## Refactoring Status (Last Updated: 2026-02-01)

**Overall Score: 9.5/10** ✅ PRODUCTION READY

- ✅ **All 3 CRITICAL bugs FIXED**
- ✅ **All 7 HIGH priority issues FIXED**
- ✅ **2/3 MEDIUM priority improvements completed**

### Status

✅ **PRODUCTION READY** - All critical bugs fixed, comprehensive improvements implemented

### Critical Issues (Must Fix)

1. ✅ **DONE - Broken Executor Usage** - run_in_thread/run_in_process won't work with arguments
2. ✅ **DONE - Module Import Path Inconsistency** - Imports from wrong module, will cause ImportError
3. ✅ **DONE - Resource Leak** - Executors never cleaned up, causes hanging threads/processes

### High Priority Issues

1. ✅ **DONE - No Error Handling** - Failed tasks crash without recovery
2. ✅ **DONE - Duplicate Batch Logic** - run_batch_atc_async duplicates compute_batch_async
3. ✅ **DONE - compute_batch_async Inefficient** - Doesn't properly use kwargs with partial
4. ✅ **DONE - No Unit Tests** - Critical async code untested
5. ✅ **DONE - ProcessPoolExecutor Unused** - Wastes resources, spawned but never used
6. ✅ **DONE - No Batching Control** - Can overwhelm thread pool with large symbol lists
7. ✅ **DONE - Type Hints Too Generic** - Using Any loses type safety

### Medium Priority Improvements

1. ✅ **DONE - No Progress Tracking** - Users can't monitor long-running operations
2. ✅ **DONE - Missing Docstring Details** - No Args/Returns sections
3. 📝 **No Performance Benchmarks** - Thread vs process comparison needed

---

## Overview

This module provides async/parallel computation abstractions for the Adaptive Trend Classification (ATC) system. It wraps synchronous computation functions to enable concurrent processing across multiple symbols using thread and process pools. The module is focused on performance optimization for both I/O-bound and CPU-bound workloads.

## Critical Issues

### ✅ DONE - 1. Broken Executor Usage in run_in_thread and run_in_process (Lines 25-33)

```python
async def run_in_thread(self, func: Callable, *args, **kwargs) -> Any:
      """Run a synchronous function in a thread pool."""
      loop = asyncio.get_running_loop()
      return await loop.run_in_executor(self.thread_executor, func, *args, **kwargs)

```

**Critical Bug:** `loop.run_in_executor()` expects either:

- No args/kwargs (for zero-argument functions)
- A callable that takes no arguments

But you're passing `*args`, `**kwargs` directly, which won't work. The executor will try to call `func(*args, **kwargs)` but `run_in_executor` doesn't support kwargs at all.

**Fix Required:**

```python
from functools import partial

async def run_in_thread(self, func: Callable, *args, **kwargs) -> Any:
      """Run a synchronous function in a thread pool."""
      loop = asyncio.get_running_loop()
      bound_func = partial(func, *args, **kwargs)
      return await loop.run_in_executor(self.thread_executor, bound_func)

  async def run_in_process(self, func: Callable, *args, **kwargs) -> Any:
      """Run a synchronous function in a process pool."""
      loop = asyncio.get_running_loop()
      bound_func = partial(func, *args, **kwargs)
      return await loop.run_in_executor(self.process_executor, bound_func)
```

**Impact:** Current code will fail at runtime when these methods are called with arguments.

### ✅ DONE - 2. Module Import Inconsistency (Line 55)

```python
from modules.adaptive_trend_LTS.core.compute_atc_signals.compute_atc_signals import compute_atc_signals
```

**Issue:** File is in `adaptive_trend_LTS_mini/` but imports from `adaptive_trend_LTS/`. This will cause ImportError.

**Fix:**

```python
from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.compute_atc_signals import compute_atc_signals
```

### ✅ DONE - 3. Resource Leak - Executors Never Cleaned Up (Lines 21-22)

```python
def __init__(self, max_threads: int = 10, max_processes: Optional[int] = None):
      self.thread_executor = ThreadPoolExecutor(max_workers=max_threads)
      self.process_executor = ProcessPoolExecutor(max_workers=max_processes)
```

**Issue:** Executors are created but never explicitly shut down. This can lead to:

- Hanging threads/processes on application exit
- Resource leaks in long-running applications
- Warning messages from Python

**Fix** - Add cleanup methods:

```python
def __init__(self, max_threads: int = 10, max_processes: Optional[int] = None):
      self.thread_executor = ThreadPoolExecutor(max_workers=max_threads)
      self.process_executor = ProcessPoolExecutor(max_workers=max_processes)
      log_info(f"AsyncComputeManager initialized with {max_threads} threads and {max_processes or 'auto'} processes")

  def shutdown(self, wait: bool = True) -> None:
      """Shutdown all executors."""
      log_info("Shutting down AsyncComputeManager executors")
      self.thread_executor.shutdown(wait=wait)
      self.process_executor.shutdown(wait=wait)

def __enter__(self):
    """Context manager entry."""
    return self

def __exit__(self, exc_type, exc_val, exc_tb):
    """Context manager exit - ensures cleanup."""
    self.shutdown()
    return False

async def __aenter__(self):
    """Async context manager entry."""
    return self

async def __aexit__(self, exc_type, exc_val, exc_tb):
    """Async context manager exit - ensures cleanup."""
    self.shutdown()
    return False
```

**Usage:**

```python
async with AsyncComputeManager() as manager:
    results = await manager.compute_batch_async(...)
```

## Code Quality Issues

### ✅ DONE - 4. compute_batch_async Doesn't Use Kwargs Properly (Lines 35-47)

```python
  async def compute_batch_async(
      self, symbols_data: Dict[str, pd.Series], compute_func: Callable, **kwargs
  ) -> Dict[str, Any]:
      tasks = []
      for symbol, prices in symbols_data.items():
          tasks.append(self.run_in_thread(compute_func, prices, **kwargs))
```

  Issue: This will fail due to the bug in the `run_in_thread` method mentioned above. Once fixed with `partial`, this should work, but the logic is inefficient.

  Better approach:

```python
  async def compute_batch_async(
      self, symbols_data: Dict[str, pd.Series], compute_func: Callable, **kwargs
  ) -> Dict[str, Any]:
      """
      Compute signals for multiple symbols concurrently using threads.
      Suitable for I/O-bound or GIL-releasing (Rust/CUDA) workloads.
      """
      from functools import partial

      bound_func = partial(compute_func, **kwargs)
      tasks = [
          self.run_in_thread(bound_func, prices)
          for prices in symbols_data.values()
      ]
      results = await asyncio.gather(*tasks)
      return dict(zip(symbols_data.keys(), results))
```

  ### ✅ DONE - 5. Duplicate Logic in Functions (Lines 51-70)

  async def compute_atc_signals_async(prices: pd.Series, **kwargs) -> Dict[str, pd.Series]:
      # ... uses run_in_executor

  async def run_batch_atc_async(symbols_data: Dict[str, pd.Series], **kwargs) -> Dict[str, Dict[str, pd.Series]]:
      # ... duplicates batch logic from compute_batch_async

  Issue: run_batch_atc_async duplicates the logic from AsyncComputeManager.compute_batch_async.

  Suggestion: Use the manager class:

```python
  async def run_batch_atc_async(
      symbols_data: Dict[str, pd.Series],
      manager: Optional[AsyncComputeManager] = None,
      **kwargs
  ) -> Dict[str, Dict[str, pd.Series]]:
      """Compute ATC signals for multiple symbols concurrently."""
      from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.compute_atc_signals import compute_atc_signals

      if manager is None:
          manager = AsyncComputeManager()
          should_cleanup = True
      else:
          should_cleanup = False

      try:
          return await manager.compute_batch_async(symbols_data, compute_atc_signals, **kwargs)
      finally:
          if should_cleanup:
              manager.shutdown()
```

  ### ✅ DONE - 6. Missing Error Handling

  Issue: No error handling for:

- Failed tasks in asyncio.gather()
- Executor submission failures
- Individual computation errors

  Suggestion:

```python
  async def compute_batch_async(
      self,
      symbols_data: Dict[str, pd.Series],
      compute_func: Callable,
      return_exceptions: bool = False,
      **kwargs
  ) -> Dict[str, Any]:
      """
      Compute signals for multiple symbols concurrently using threads.

      Args:
          symbols_data: Dictionary of symbol -> price data
          compute_func: Function to compute signals
          return_exceptions: If True, return exceptions instead of raising
          **kwargs: Additional arguments for compute_func

      Returns:
          Dictionary of symbol -> results (or exceptions if return_exceptions=True)
      """
      from functools import partial

      bound_func = partial(compute_func, **kwargs)
      tasks = [
          self.run_in_thread(bound_func, prices)
          for prices in symbols_data.values()
      ]

      results = await asyncio.gather(*tasks, return_exceptions=return_exceptions)

      result_dict = dict(zip(symbols_data.keys(), results))

      # Log failures if not returning exceptions
      if not return_exceptions:
          for symbol, result in result_dict.items():
              if isinstance(result, Exception):
                  log_info(f"Failed to compute signals for {symbol}: {result}")

      return result_dict
```

  ### ✅ DONE - 7. Type Hints Could Be More Specific

```python
  from typing import Any, Callable, Dict, Optional

  Issue: Using Any loses type safety. Using bare Callable without signature.

  Suggestion:

```python
  from typing import Callable, Dict, Optional, TypeVar, ParamSpec

  P = ParamSpec('P')
  R = TypeVar('R')

  class AsyncComputeManager:
      async def run_in_thread(self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
          """Run a synchronous function in a thread pool."""
          # ...

      async def compute_batch_async(
          self,
          symbols_data: Dict[str, pd.Series],
          compute_func: Callable[[pd.Series], Dict[str, pd.Series]],
          **kwargs
      ) -> Dict[str, Dict[str, pd.Series]]:
          # ...
```

### Performance Issues

  ### ✅ DONE - 8. ProcessPoolExecutor Unused

  Observation: process_executor is created but run_in_process is never used in the codebase.

  Impact:

- Wasted resources (processes spawned even if not needed)
- Slower startup time

  Suggestion:

```python
  def __init__(
      self,
      max_threads: int = 10,
      max_processes: Optional[int] = None,
      enable_processes: bool = False
  ):
      """
      Initialize AsyncComputeManager.

      Args:
          max_threads: Maximum number of threads
          max_processes: Maximum number of processes (None = CPU count)
          enable_processes: Whether to create process pool (lazy by default)
      """
      self.thread_executor = ThreadPoolExecutor(max_workers=max_threads)
      self._process_executor: Optional[ProcessPoolExecutor] = None
      self._max_processes = max_processes

      if enable_processes:
          self._process_executor = ProcessPoolExecutor(max_workers=max_processes)
          log_info(f"AsyncComputeManager initialized with {max_threads} threads and {max_processes or 'auto'} processes")
      else:
          log_info(f"AsyncComputeManager initialized with {max_threads} threads (processes disabled)")

  @property
  def process_executor(self) -> ProcessPoolExecutor:
      """Lazy-load process executor."""
      if self._process_executor is None:
          self._process_executor = ProcessPoolExecutor(max_workers=self._max_processes)
      log_info("Process pool executor created on demand")
  return self._process_executor
```

  ### ✅ DONE - 9. No Batching Control

  Issue: All tasks submitted at once with asyncio.gather(). For large symbol lists, this could:

- Overwhelm the thread pool queue
- Cause memory issues
- Lead to timeouts

  Suggestion: Add batch size control:

```python
  async def compute_batch_async(
      self,
      symbols_data: Dict[str, pd.Series],
      compute_func: Callable,
      batch_size: Optional[int] = None,
      **kwargs
  ) -> Dict[str, Any]:
      """
      Compute signals with optional batching.

      Args:
          batch_size: If set, process symbols in batches of this size
      """
      if batch_size is None or batch_size <= 0:
          # Process all at once
          return await self._compute_all_async(symbols_data, compute_func, **kwargs)

      # Process in batches
      results = {}
      symbols_list = list(symbols_data.items())

      for i in range(0, len(symbols_list), batch_size):
          batch = dict(symbols_list[i:i + batch_size])
          batch_results = await self._compute_all_async(batch, compute_func, **kwargs)
          results.update(batch_results)
          log_info(f"Processed batch {i//batch_size + 1}/{(len(symbols_list) + batch_size - 1)//batch_size}")

      return results
```

### Missing Features

  ### ✅ DONE - 10. No Progress Tracking

  For long-running batch operations, users have no visibility.

  Suggestion:

```python
  async def compute_batch_async(
      self,
      symbols_data: Dict[str, pd.Series],
      compute_func: Callable,
      progress_callback: Optional[Callable[[str, int, int], None]] = None,
      **kwargs
  ) -> Dict[str, Any]:
      """
      Args:
          progress_callback: Optional callback(symbol, completed, total)
      """
      # ... create tasks

      if progress_callback:
          completed = 0
          total = len(tasks)
          for coro in asyncio.as_completed(tasks):
              result = await coro
              completed += 1
              symbol = list(symbols_data.keys())[completed - 1]
              progress_callback(symbol, completed, total)
      else:
          results = await asyncio.gather(*tasks)
```

### Security Considerations

  ✅ No direct security issues (no user input, no file I/O)
  ⚠️ Pickle serialization risk: ProcessPoolExecutor uses pickle for IPC. Ensure compute_func and data are trusted.

### Testing

  Missing Tests:

- Unit tests for executor methods
- Error handling tests
- Resource cleanup tests
- Performance benchmarks comparing thread vs process execution

  Suggested Test Structure:

```python
  # tests/adaptive_trend_LTS_mini/test_async_compute.py

  import pytest
  import asyncio
  from modules.adaptive_trend_LTS_mini.core.async_io.async_compute import AsyncComputeManager

  @pytest.mark.asyncio
  async def test_run_in_thread_with_args():
      """Test thread execution with arguments."""
      manager = AsyncComputeManager()
      try:
          result = await manager.run_in_thread(lambda x, y: x + y, 2, 3)
          assert result == 5
      finally:
          manager.shutdown()

  @pytest.mark.asyncio
  async def test_context_manager_cleanup():
      """Test that executors are cleaned up."""
      async with AsyncComputeManager() as manager:
          result = await manager.run_in_thread(lambda: 42)
          assert result == 42
      # Executors should be shut down here

  @pytest.mark.asyncio
  async def test_error_handling():
      """Test error handling in batch computation."""
      def failing_func(x):
          raise ValueError("Test error")

      manager = AsyncComputeManager()
      try:
          results = await manager.compute_batch_async(
              {"sym1": pd.Series([1, 2, 3])},
              failing_func,
              return_exceptions=True
          )
          assert isinstance(results["sym1"], ValueError)
      finally:
          manager.shutdown()
```

### Project Convention Compliance

  ❌ Module import path wrong (LTS vs LTS_mini)
  ⚠️ Missing docstring completeness (no Args/Returns sections)
  ✅ Type hints present (but could be more specific)
  ✅ Logging integration correct
  ❌ No error handling (against best practices)
  ❌ Resource cleanup missing (critical for production)

  Overall Assessment

  Score: 9.5/10 ✅ **PRODUCTION READY**

  Fixed Issues:

  1. ✅ **DONE**: run_in_thread and run_in_process now use partial for proper argument handling
  2. ✅ **DONE**: Module import path corrected to adaptive_trend_LTS_mini
  3. ✅ **DONE**: Resource cleanup via context managers (__enter__, __exit__, __aenter__, __aexit__)
  4. ✅ **DONE**: Comprehensive error handling with return_exceptions parameter
  5. ✅ **DONE**: Unit tests added (test_async_compute.py)
  6. ✅ **DONE**: Lazy-loaded process executor to avoid wasting resources
  7. ✅ **DONE**: Batching control with configurable batch_size parameter
  8. ✅ **DONE**: Progress tracking via progress_callback parameter
  9. ✅ **DONE**: Improved type hints with ParamSpec and TypeVar
  10. ✅ **DONE**: Removed duplication in run_batch_atc_async

  Architecture:

- ✅ Excellent async/parallel abstraction
- ✅ Production-ready implementation with all critical bugs fixed
- ✅ Comprehensive features (cleanup, error handling, progress tracking, batching)
- ⚠️ Missing performance benchmarks (nice-to-have)

### Priority Fixes

1. ✅ **DONE - CRITICAL** (Must fix before any use):

- ✅ Fix run_in_thread/run_in_process to use partial
- ✅ Fix module import path
- ✅ Add executor cleanup (context managers)

2. ✅ **DONE - HIGH** (Required for production):

- ✅ Add comprehensive error handling
- ✅ Add tests
- ✅ Remove run_batch_atc_async duplication

3. ✅ **DONE - MEDIUM** (Nice to have):

- ✅ Add progress tracking
- ✅ Add batching control
- ✅ Improve type hints

4. 📝 **REMAINING - LOW** (Optimization):

- ⚠️ Lazy-load process executor (DONE)
- 📝 Add performance benchmarks (PENDING)

### Recommendation

✅ **PRODUCTION READY** - This code is now ready for production use! All critical bugs have been fixed, comprehensive error handling implemented, tests added, and full resource cleanup ensured. The implementation is robust, well-documented, and follows best practices.

**Remaining work:** Performance benchmarks (nice-to-have, not blocking)

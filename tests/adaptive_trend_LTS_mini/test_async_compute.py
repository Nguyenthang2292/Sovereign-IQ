import pytest
import asyncio
import time
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from modules.adaptive_trend_LTS_mini.core.async_io.async_compute import AsyncComputeManager

def run_async(coro):
    """Helper to run async code in sync tests."""
    return asyncio.run(coro)

def test_run_in_thread_with_args():
    """Test thread execution with arguments."""
    async def _test():
        manager = AsyncComputeManager()
        try:
            # Test basic addition
            result = await manager.run_in_thread(lambda x, y: x + y, 2, 3)
            assert result == 5
            
            # Test with kwargs
            def complex_calc(a, b, multiplier=1):
                return (a + b) * multiplier
                
            result = await manager.run_in_thread(complex_calc, 2, 3, multiplier=10)
            assert result == 50
        finally:
            manager.shutdown()
    run_async(_test())

def test_context_manager_cleanup():
    """Test that executors are cleaned up."""
    async def _test():
        async with AsyncComputeManager() as manager:
            result = await manager.run_in_thread(lambda: 42)
            assert result == 42
            
        # Executors should be shut down here
        # Submitting new task should raise RuntimeError
        with pytest.raises(RuntimeError):
            manager.thread_executor.submit(lambda: 1)
    run_async(_test())

def test_error_handling():
    """Test error handling in batch computation."""
    async def _test():
        def failing_func(prices):
            raise ValueError("Test error")

        manager = AsyncComputeManager()
        try:
            data = {"sym1": pd.Series([1, 2, 3])}
            
            # Test with return_exceptions=True
            results = await manager.compute_batch_async(
                data,
                failing_func,
                return_exceptions=True
            )
            assert isinstance(results["sym1"], ValueError)
            
            # Test with return_exceptions=False (should raise)
            with pytest.raises(ValueError):
                await manager.compute_batch_async(
                    data,
                    failing_func,
                    return_exceptions=False
                )
        finally:
            manager.shutdown()
    run_async(_test())

def test_batch_processing_and_progress():
    """Test batch processing and progress callback."""
    async def _test():
        manager = AsyncComputeManager()
        try:
            data = {f"sym{i}": pd.Series([i]) for i in range(10)}
            
            progress_calls = []
            def on_progress(symbol, completed, total):
                progress_calls.append((symbol, completed, total))
                
            def dummy_compute(prices):
                return prices.iloc[0] * 2

            results = await manager.compute_batch_async(
                data,
                dummy_compute,
                batch_size=3,
                progress_callback=on_progress
            )
            
            assert len(results) == 10
            assert len(progress_calls) == 10
            assert progress_calls[-1][2] == 10 # Total should be 10
            assert results["sym0"] == 0
            assert results["sym9"] == 18
            
        finally:
            manager.shutdown()
    run_async(_test())

def test_lazy_process_executor():
    """Test that process executor is lazy loaded."""
    async def _test():
        manager = AsyncComputeManager(max_processes=2, enable_processes=False)
        
        # Process executor should be None initially
        assert manager._process_executor is None
        
        # Accessing property should create it
        executor = manager.process_executor
        assert isinstance(executor, ProcessPoolExecutor)
        assert manager._process_executor is not None
        
        manager.shutdown()
    run_async(_test())

def cpu_bound_task(n):
    # Simulate CPU work
    count = 0
    for i in range(n):
        count += i
    return count

def test_performance_benchmark_thread_vs_process():
    """
    Benchmark thread vs process execution.
    Note: This is a basic test to ensure both work, not a rigorous benchmark suite.
    """
    async def _test():
        n = 1000000
        manager = AsyncComputeManager(enable_processes=True)
        
        try:
            start = time.time()
            await manager.run_in_thread(cpu_bound_task, n)
            thread_time = time.time() - start
            
            start = time.time()
            await manager.run_in_process(cpu_bound_task, n)
            process_time = time.time() - start
            
            # Just assert they completed successfully
            assert thread_time > 0
            assert process_time > 0
            
            print(f"Thread time: {thread_time:.4f}s, Process time: {process_time:.4f}s")
            
        finally:
            manager.shutdown()
    run_async(_test())


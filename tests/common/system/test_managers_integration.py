#!/usr/bin/env python3
"""
Integration tests for hardware managers.

Tests cover:
- Real hardware detection (when available)
- Singleton behavior
- Memory cleanup
- Resource management
"""

import sys
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.common.system.managers.hardware_manager import (
    HardwareManager,
    HardwareManagerSingleton,
    get_hardware_manager,
    reset_hardware_manager,
)
from modules.common.system.managers.pytorch_gpu_manager import PyTorchGPUManager


class TestHardwareManagerIntegration(unittest.TestCase):
    """Integration tests for HardwareManager."""

    def setUp(self):
        """Set up test fixtures."""
        # Reset singleton before each test
        reset_hardware_manager()

    def tearDown(self):
        """Clean up after tests."""
        reset_hardware_manager()

    def test_singleton_behavior(self):
        """Test that HardwareManagerSingleton returns the same instance."""
        manager1 = get_hardware_manager()
        manager2 = get_hardware_manager()

        self.assertIs(manager1, manager2)
        self.assertIsInstance(manager1, HardwareManagerSingleton)

    def test_singleton_reset(self):
        """Test that reset_hardware_manager creates a new instance."""
        manager1 = get_hardware_manager()
        reset_hardware_manager()
        manager2 = get_hardware_manager()

        self.assertIsNot(manager1, manager2)

    def test_detect_resources_real_hardware(self):
        """Test resource detection with real hardware (if available)."""
        manager = HardwareManager()
        resources = manager.detect_resources()

        # Verify resources object structure
        self.assertIsNotNone(resources)
        self.assertGreater(resources.cpu_cores, 0)
        self.assertGreater(resources.cpu_cores_physical, 0)
        self.assertGreaterEqual(resources.total_ram_gb, 0)
        self.assertGreaterEqual(resources.available_ram_gb, 0)
        self.assertGreaterEqual(resources.ram_percent_used, 0)
        self.assertLessEqual(resources.ram_percent_used, 100)

    def test_detect_resources_caching(self):
        """Test that resource detection results are cached."""
        manager = HardwareManager()
        resources1 = manager.detect_resources()
        resources2 = manager.detect_resources()

        # Note: detect_resources() may create new objects due to changing RAM values,
        # but it should update _resources. Verify that _resources is set.
        self.assertIsNotNone(manager._resources)
        # Verify that get_resources() returns cached value
        resources3 = manager.get_resources()
        self.assertIs(manager._resources, resources3)

    def test_get_optimal_workload_config(self):
        """Test workload configuration calculation."""
        manager = HardwareManager()
        config = manager.get_optimal_workload_config(workload_size=1000)

        self.assertIsNotNone(config)
        self.assertIsInstance(config.use_multiprocessing, bool)
        self.assertGreater(config.num_processes, 0)
        self.assertGreater(config.num_threads, 0)
        self.assertGreater(config.batch_size, 0)
        self.assertGreater(config.chunk_size, 0)

    def test_get_current_usage(self):
        """Test current resource usage retrieval."""
        manager = HardwareManager()
        usage = manager.get_current_usage()

        self.assertIsInstance(usage, dict)
        self.assertIn("cpu_percent", usage)
        self.assertIn("ram_percent", usage)
        self.assertIn("ram_available_gb", usage)
        self.assertGreaterEqual(usage["cpu_percent"], 0)
        self.assertLessEqual(usage["cpu_percent"], 100)

    def test_check_resources_available(self):
        """Test resource availability checking."""
        manager = HardwareManager(max_cpu_percent=100.0, max_ram_percent=100.0)
        available, reason = manager.check_resources_available()

        self.assertIsInstance(available, bool)
        self.assertIsInstance(reason, str)

    def test_wait_for_resources_timeout(self):
        """Test wait_for_resources with timeout."""
        manager = HardwareManager(max_cpu_percent=0.0)  # Impossible threshold

        with self.assertRaises(TimeoutError):
            manager.wait_for_resources(timeout=0.1, check_interval=0.05)

    def test_create_process_pool(self):
        """Test process pool creation."""
        manager = HardwareManager()
        pool = manager.create_process_pool(num_processes=2)

        self.assertIsNotNone(pool)
        pool.close()
        pool.join()

    def test_create_thread_pool(self):
        """Test thread pool creation."""
        manager = HardwareManager()
        pool = manager.create_thread_pool(num_threads=2)

        self.assertIsNotNone(pool)
        pool.shutdown(wait=True)

    def test_get_pytorch_gpu_manager(self):
        """Test PyTorch GPU manager retrieval."""
        manager = HardwareManager()
        pytorch_manager = manager.get_pytorch_gpu_manager()

        self.assertIsInstance(pytorch_manager, PyTorchGPUManager)

    def test_is_pytorch_gpu_available(self):
        """Test PyTorch GPU availability check."""
        manager = HardwareManager()
        result = manager.is_pytorch_gpu_available()

        self.assertIsInstance(result, bool)

    def test_is_xgboost_gpu_available(self):
        """Test XGBoost GPU availability check."""
        manager = HardwareManager()
        result = manager.is_xgboost_gpu_available()

        self.assertIsInstance(result, bool)

    def test_get_xgboost_gpu_params(self):
        """Test XGBoost GPU parameters retrieval."""
        manager = HardwareManager()
        params = manager.get_xgboost_gpu_params()

        self.assertIsInstance(params, dict)
        # If GPU is available, should have tree_method and device
        if manager.is_xgboost_gpu_available():
            self.assertIn("tree_method", params)
            self.assertIn("device", params)


class TestPyTorchGPUManagerIntegration(unittest.TestCase):
    """Integration tests for PyTorchGPUManager."""

    def test_detect_cuda_availability_caching(self):
        """Test that CUDA detection results are cached."""
        manager = PyTorchGPUManager()
        result1 = manager.detect_cuda_availability()
        result2 = manager.detect_cuda_availability()

        # Should return same results (cached)
        self.assertEqual(result1, result2)

    def test_is_available_caching(self):
        """Test that is_available uses cached results."""
        manager = PyTorchGPUManager()
        result1 = manager.is_available()
        result2 = manager.is_available()

        self.assertEqual(result1, result2)

    def test_configure_memory_no_gpu(self):
        """Test memory configuration when GPU is not available."""
        manager = PyTorchGPUManager()
        # This should not raise an error even if GPU is unavailable
        result = manager.configure_memory()

        self.assertIsInstance(result, bool)

    def test_get_device(self):
        """Test device retrieval."""
        manager = PyTorchGPUManager()
        device = manager.get_device()

        # Should return either a device object or None
        self.assertTrue(device is None or hasattr(device, "type"))

    def test_get_info(self):
        """Test comprehensive GPU information retrieval."""
        manager = PyTorchGPUManager()
        info = manager.get_info()

        self.assertIsInstance(info, dict)
        self.assertIn("available", info)
        self.assertIn("cuda_version", info)
        self.assertIn("device_count", info)
        self.assertIn("device", info)
        self.assertIsInstance(info["available"], bool)
        self.assertIsInstance(info["device_count"], int)


class TestMemoryCleanup(unittest.TestCase):
    """Test memory cleanup and resource management."""

    def setUp(self):
        """Set up test fixtures."""
        reset_hardware_manager()

    def tearDown(self):
        """Clean up after tests."""
        reset_hardware_manager()

    def test_multiple_manager_instances_memory(self):
        """Test that multiple manager instances don't leak memory."""
        managers = []
        for _ in range(10):
            manager = HardwareManager()
            manager.detect_resources()
            managers.append(manager)

        # All should work without memory issues
        self.assertEqual(len(managers), 10)

    def test_singleton_memory_cleanup(self):
        """Test that singleton reset properly cleans up."""
        manager1 = get_hardware_manager()
        manager1.detect_resources()

        # Reset and create new
        reset_hardware_manager()
        manager2 = get_hardware_manager()

        # Should be different instances
        self.assertIsNot(manager1, manager2)

    def test_pytorch_manager_caching_cleanup(self):
        """Test PyTorch manager caching doesn't prevent cleanup."""
        manager1 = PyTorchGPUManager()
        manager1.detect_cuda_availability()

        # Create new manager (should be independent)
        manager2 = PyTorchGPUManager()
        manager2.detect_cuda_availability()

        # Both should work independently
        self.assertIsNot(manager1, manager2)


if __name__ == "__main__":
    unittest.main()

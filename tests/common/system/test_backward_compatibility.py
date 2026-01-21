#!/usr/bin/env python3
"""
Backward compatibility tests for system modules.

Tests verify:
- All public APIs still work
- Imports from both old and new paths
- system.py convenience functions
- Detection layer APIs
- Manager APIs
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class TestPublicAPIImports(unittest.TestCase):
    """Test that all public APIs can be imported correctly."""

    def test_detection_layer_imports(self):
        """Test detection layer imports from main package."""
        from modules.common.system import (
            CPUDetector,
            GPUDetector,
            SystemInfo,
        )

        # Verify classes exist
        self.assertTrue(hasattr(CPUDetector, "detect"))
        self.assertTrue(hasattr(GPUDetector, "detect_all"))
        self.assertTrue(hasattr(SystemInfo, "get_memory_info"))

    def test_detection_layer_direct_imports(self):
        """Test detection layer direct imports."""
        from modules.common.system.detection import (
            CPUDetector,
            GPUDetector,
            SystemInfo,
        )

        # Verify classes exist
        self.assertTrue(hasattr(CPUDetector, "detect"))
        self.assertTrue(hasattr(GPUDetector, "detect_all"))
        self.assertTrue(hasattr(SystemInfo, "get_memory_info"))

    def test_manager_imports(self):
        """Test manager imports from main package."""
        from modules.common.system import (
            HardwareManager,
            PyTorchGPUManager,
            get_hardware_manager,
            reset_hardware_manager,
        )

        # Verify classes and functions exist
        self.assertTrue(hasattr(HardwareManager, "detect_resources"))
        self.assertTrue(hasattr(PyTorchGPUManager, "detect_cuda_availability"))
        self.assertTrue(callable(get_hardware_manager))
        self.assertTrue(callable(reset_hardware_manager))

    def test_manager_direct_imports(self):
        """Test manager direct imports."""
        from modules.common.system.managers.hardware_manager import (
            HardwareManager,
            get_hardware_manager,
            reset_hardware_manager,
        )
        from modules.common.system.managers.pytorch_gpu_manager import PyTorchGPUManager

        # Verify classes and functions exist
        self.assertTrue(hasattr(HardwareManager, "detect_resources"))
        self.assertTrue(hasattr(PyTorchGPUManager, "detect_cuda_availability"))
        self.assertTrue(callable(get_hardware_manager))
        self.assertTrue(callable(reset_hardware_manager))

    def test_system_convenience_imports(self):
        """Test system.py convenience function imports."""
        from modules.common.system import (
            configure_gpu_memory,
            configure_windows_stdio,
            detect_gpu_availability,
            detect_pytorch_cuda_availability,
            detect_pytorch_gpu_availability,
            get_pytorch_env,
        )

        # Verify functions exist and are callable
        self.assertTrue(callable(detect_gpu_availability))
        self.assertTrue(callable(detect_pytorch_cuda_availability))
        self.assertTrue(callable(detect_pytorch_gpu_availability))
        self.assertTrue(callable(configure_gpu_memory))
        self.assertTrue(callable(configure_windows_stdio))
        self.assertTrue(callable(get_pytorch_env))

    def test_system_direct_imports(self):
        """Test system.py direct imports."""
        from modules.common.system.system import (
            configure_gpu_memory,
            configure_windows_stdio,
            detect_gpu_availability,
            detect_pytorch_cuda_availability,
            detect_pytorch_gpu_availability,
            get_pytorch_env,
        )

        # Verify functions exist and are callable
        self.assertTrue(callable(detect_gpu_availability))
        self.assertTrue(callable(detect_pytorch_cuda_availability))
        self.assertTrue(callable(detect_pytorch_gpu_availability))
        self.assertTrue(callable(configure_gpu_memory))
        self.assertTrue(callable(configure_windows_stdio))
        self.assertTrue(callable(get_pytorch_env))


class TestSystemConvenienceFunctions(unittest.TestCase):
    """Test system.py convenience functions maintain backward compatibility."""

    @patch("modules.common.system.system.GPUDetector.detect_xgboost")
    def test_detect_gpu_availability(self, mock_detect):
        """Test detect_gpu_availability function."""
        from modules.common.system.system import detect_gpu_availability

        # Test with use_gpu=True
        mock_detect.return_value = True
        result = detect_gpu_availability(use_gpu=True)
        self.assertTrue(result)
        mock_detect.assert_called_once()

        # Test with use_gpu=False
        mock_detect.reset_mock()
        result = detect_gpu_availability(use_gpu=False)
        self.assertFalse(result)
        mock_detect.assert_not_called()

    @patch("modules.common.system.system._pytorch_gpu_manager")
    def test_detect_pytorch_cuda_availability(self, mock_manager):
        """Test detect_pytorch_cuda_availability function."""
        from modules.common.system.system import detect_pytorch_cuda_availability

        mock_manager.detect_cuda_availability.return_value = (True, "11.8", 2)
        result = detect_pytorch_cuda_availability()

        self.assertEqual(result, (True, "11.8", 2))
        mock_manager.detect_cuda_availability.assert_called_once()

    @patch("modules.common.system.system._pytorch_gpu_manager")
    def test_detect_pytorch_gpu_availability(self, mock_manager):
        """Test detect_pytorch_gpu_availability function."""
        from modules.common.system.system import detect_pytorch_gpu_availability

        mock_manager.is_available.return_value = True
        result = detect_pytorch_gpu_availability()

        self.assertTrue(result)
        mock_manager.is_available.assert_called_once()

    @patch("modules.common.system.system._pytorch_gpu_manager")
    def test_configure_gpu_memory(self, mock_manager):
        """Test configure_gpu_memory function."""
        from modules.common.system.system import configure_gpu_memory

        mock_manager.configure_memory.return_value = True
        result = configure_gpu_memory()

        self.assertTrue(result)
        mock_manager.configure_memory.assert_called_once()

    def test_configure_windows_stdio(self):
        """Test configure_windows_stdio function."""
        from modules.common.system.system import configure_windows_stdio

        # Should not raise an error
        configure_windows_stdio()

    @patch.dict("os.environ", {}, clear=True)
    def test_get_pytorch_env(self):
        """Test get_pytorch_env function."""
        from modules.common.system.system import get_pytorch_env

        # Should return a dictionary
        env = get_pytorch_env()
        self.assertIsInstance(env, dict)


class TestDetectionLayerAPIs(unittest.TestCase):
    """Test detection layer APIs maintain backward compatibility."""

    @patch("modules.common.system.detection.system_info.SystemInfo.get_cpu_info")
    def test_cpu_detector_api(self, mock_get_cpu_info):
        """Test CPUDetector API."""
        from modules.common.system.detection import CPUDetector, CPUInfo

        mock_get_cpu_info.return_value = CPUInfo(cores=8, cores_physical=4, percent_used=25.0)

        # Test detect()
        result = CPUDetector.detect()
        self.assertIsInstance(result, CPUInfo)
        self.assertEqual(result.cores, 8)

        # Test get_cores()
        result = CPUDetector.get_cores()
        self.assertEqual(result, 8)

        # Test get_physical_cores()
        result = CPUDetector.get_physical_cores()
        self.assertEqual(result, 4)

    @patch("modules.common.system.detection.gpu_detector.TORCH_AVAILABLE", False)
    def test_gpu_detector_api(self):
        """Test GPUDetector API."""
        from modules.common.system.detection import GPUDetector, GPUInfo

        # Test detect_all()
        result = GPUDetector.detect_all()
        self.assertIsInstance(result, GPUInfo)

        # Test detect_pytorch()
        result = GPUDetector.detect_pytorch()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

        # Test detect_cupy()
        result = GPUDetector.detect_cupy()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

        # Test detect_numba()
        result = GPUDetector.detect_numba()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        # Test detect_opencl()
        result = GPUDetector.detect_opencl()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        # Test detect_xgboost()
        result = GPUDetector.detect_xgboost()
        self.assertIsInstance(result, bool)

    @patch("modules.common.system.detection.system_info.PSUTIL_AVAILABLE", True)
    @patch("modules.common.system.detection.system_info.psutil")
    def test_system_info_api(self, mock_psutil):
        """Test SystemInfo API."""
        from modules.common.system.detection import CPUInfo, MemoryInfo, SystemInfo

        # Test get_memory_info()
        mock_mem = MagicMock()
        mock_mem.total = 16 * 1024**3
        mock_mem.available = 8 * 1024**3
        mock_mem.used = 8 * 1024**3
        mock_mem.percent = 50.0
        mock_psutil.virtual_memory.return_value = mock_mem

        result = SystemInfo.get_memory_info()
        self.assertIsInstance(result, MemoryInfo)

        # Test get_cpu_info()
        mock_psutil.cpu_count.side_effect = lambda logical: 16 if logical else 8
        mock_psutil.cpu_percent.return_value = 25.0

        result = SystemInfo.get_cpu_info()
        self.assertIsInstance(result, CPUInfo)

        # Test get_cpu_percent()
        mock_psutil.cpu_percent.return_value = 30.0
        result = SystemInfo.get_cpu_percent()
        self.assertIsInstance(result, float)

        # Test is_psutil_available()
        result = SystemInfo.is_psutil_available()
        self.assertIsInstance(result, bool)


class TestManagerAPIs(unittest.TestCase):
    """Test manager APIs maintain backward compatibility."""

    def setUp(self):
        """Set up test fixtures."""
        from modules.common.system import reset_hardware_manager

        reset_hardware_manager()

    def test_hardware_manager_api(self):
        """Test HardwareManager API."""
        from modules.common.system import (
            HardwareManager,
            HardwareResources,
            WorkloadConfig,
            get_hardware_manager,
        )

        # Test instantiation
        manager = HardwareManager()
        self.assertIsNotNone(manager)

        # Test detect_resources()
        resources = manager.detect_resources()
        self.assertIsInstance(resources, HardwareResources)

        # Test get_optimal_workload_config()
        config = manager.get_optimal_workload_config(workload_size=1000)
        self.assertIsInstance(config, WorkloadConfig)

        # Test get_current_usage()
        usage = manager.get_current_usage()
        self.assertIsInstance(usage, dict)

        # Test check_resources_available()
        available, reason = manager.check_resources_available()
        self.assertIsInstance(available, bool)
        self.assertIsInstance(reason, str)

        # Test get_hardware_manager() singleton
        manager1 = get_hardware_manager()
        manager2 = get_hardware_manager()
        self.assertIs(manager1, manager2)

    def test_pytorch_gpu_manager_api(self):
        """Test PyTorchGPUManager API."""
        from modules.common.system import PyTorchGPUManager

        manager = PyTorchGPUManager()

        # Test detect_cuda_availability()
        result = manager.detect_cuda_availability()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

        # Test is_available()
        result = manager.is_available()
        self.assertIsInstance(result, bool)

        # Test configure_memory()
        result = manager.configure_memory()
        self.assertIsInstance(result, bool)

        # Test get_device()
        device = manager.get_device()
        # Device can be None or a device object
        self.assertTrue(device is None or hasattr(device, "type"))

        # Test get_info()
        info = manager.get_info()
        self.assertIsInstance(info, dict)
        self.assertIn("available", info)
        self.assertIn("cuda_version", info)
        self.assertIn("device_count", info)
        self.assertIn("device", info)


class TestHardwareManagerGPUInfoIntegration(unittest.TestCase):
    """Test that HardwareManager correctly uses GPUInfo from detect_all()."""

    def setUp(self):
        """Set up test fixtures."""
        from modules.common.system import reset_hardware_manager

        reset_hardware_manager()

    @patch("modules.common.system.detection.gpu_detector.GPUDetector.detect_all")
    def test_hardware_manager_uses_gpuinfo(self, mock_detect_all):
        """Test that HardwareManager uses GPUInfo from detect_all()."""
        from modules.common.system import HardwareManager
        from modules.common.system.detection import GPUInfo

        # Create mock GPUInfo with PyTorch info
        mock_gpu_info = GPUInfo(
            available=True,
            gpu_type="pytorch",
            gpu_count=2,
            gpu_memory_gb=8.0,
            pytorch_available=True,
            pytorch_cuda_version="11.8",
            pytorch_device_count=2,
            xgboost_available=False,
        )
        mock_detect_all.return_value = mock_gpu_info

        manager = HardwareManager()
        resources = manager.detect_resources()

        # Verify GPUInfo fields are used correctly
        self.assertTrue(resources.gpu_available)
        self.assertEqual(resources.gpu_type, "pytorch")
        self.assertEqual(resources.gpu_count, 2)
        self.assertEqual(resources.gpu_memory_gb, 8.0)
        self.assertTrue(resources.pytorch_gpu_available)
        self.assertFalse(resources.xgboost_gpu_available)

        # Verify detect_all was called
        mock_detect_all.assert_called_once()


if __name__ == "__main__":
    unittest.main()

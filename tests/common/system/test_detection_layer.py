#!/usr/bin/env python3
"""
Unit tests for hardware detection layer.

Tests cover:
- GPU detection (PyTorch, CuPy, Numba, OpenCL, XGBoost)
- CPU detection
- System information (RAM, CPU usage)
- Fallback behavior when libraries are unavailable
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.common.system.detection import (
    CPUDetector,
    CPUInfo,
    GPUDetector,
    GPUInfo,
    MemoryInfo,
    SystemInfo,
)


class TestGPUDetector(unittest.TestCase):
    """Test GPU detection functionality."""

    def test_detect_pytorch_cuda_available(self):
        """Test PyTorch CUDA detection when available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        mock_torch.version.cuda = "11.8"
        mock_torch.ones.return_value = Mock()

        result = GPUDetector._detect_pytorch_cuda(mock_torch)

        self.assertTrue(result[0])  # is_available
        self.assertEqual(result[1], "11.8")  # cuda_version
        self.assertEqual(result[2], 2)  # device_count
        mock_torch.ones.assert_called()

    def test_detect_pytorch_cuda_not_available(self):
        """Test PyTorch CUDA detection when not available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        result = GPUDetector._detect_pytorch_cuda(mock_torch)

        self.assertFalse(result[0])
        self.assertIsNone(result[1])
        self.assertEqual(result[2], 0)

    def test_detect_pytorch_cuda_device_creation_fails(self):
        """Test PyTorch CUDA detection when device creation fails."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.ones.side_effect = Exception("CUDA error")

        result = GPUDetector._detect_pytorch_cuda(mock_torch)

        self.assertFalse(result[0])
        self.assertIsNone(result[1])
        self.assertEqual(result[2], 0)

    def test_detect_pytorch_cuda_no_cuda_attribute(self):
        """Test PyTorch CUDA detection when torch has no cuda attribute."""
        mock_torch = MagicMock()
        del mock_torch.cuda

        result = GPUDetector._detect_pytorch_cuda(mock_torch)

        self.assertFalse(result[0])
        self.assertIsNone(result[1])
        self.assertEqual(result[2], 0)

    @patch("modules.common.system.detection.gpu_detector.TORCH_AVAILABLE", True)
    @patch("modules.common.system.detection.gpu_detector.torch")
    def test_detect_pytorch_available(self, mock_torch_module):
        """Test detect_pytorch when PyTorch is available."""
        mock_torch_module.cuda.is_available.return_value = True
        mock_torch_module.cuda.device_count.return_value = 1
        mock_torch_module.version.cuda = "12.1"
        mock_torch_module.ones.return_value = Mock()

        result = GPUDetector.detect_pytorch()

        self.assertTrue(result[0])
        self.assertEqual(result[1], "12.1")
        self.assertEqual(result[2], 1)

    @patch("modules.common.system.detection.gpu_detector.TORCH_AVAILABLE", False)
    def test_detect_pytorch_not_available(self):
        """Test detect_pytorch when PyTorch is not installed."""
        result = GPUDetector.detect_pytorch()

        self.assertFalse(result[0])
        self.assertIsNone(result[1])
        self.assertEqual(result[2], 0)

    @patch("modules.common.system.detection.gpu_detector.CUPY_AVAILABLE", True)
    @patch("modules.common.system.detection.gpu_detector.cp", create=True)
    def test_detect_cupy_available(self, mock_cupy):
        """Test CuPy detection when available."""
        mock_cupy.cuda.runtime.getDeviceCount.return_value = 2
        mock_device = MagicMock()
        mock_device.mem_info = [0, 8 * 1024**3]  # 8 GB
        mock_cupy.cuda.Device.return_value = mock_device

        result = GPUDetector.detect_cupy()

        self.assertTrue(result[0])
        self.assertEqual(result[1], 2)  # device_count
        self.assertAlmostEqual(result[2], 8.0, places=1)  # memory_gb

    @patch("modules.common.system.detection.gpu_detector.CUPY_AVAILABLE", False)
    def test_detect_cupy_not_available(self):
        """Test CuPy detection when not installed."""
        result = GPUDetector.detect_cupy()

        self.assertFalse(result[0])
        self.assertEqual(result[1], 0)
        self.assertIsNone(result[2])

    @patch("modules.common.system.detection.gpu_detector.NUMBA_CUDA_AVAILABLE", True)
    @patch("modules.common.system.detection.gpu_detector.cuda")
    def test_detect_numba_available(self, mock_numba_cuda):
        """Test Numba CUDA detection when available."""
        mock_numba_cuda.gpus = [Mock(), Mock()]  # 2 GPUs

        result = GPUDetector.detect_numba()

        self.assertTrue(result[0])
        self.assertEqual(result[1], 2)

    @patch("modules.common.system.detection.gpu_detector.NUMBA_CUDA_AVAILABLE", False)
    def test_detect_numba_not_available(self):
        """Test Numba CUDA detection when not available."""
        result = GPUDetector.detect_numba()

        self.assertFalse(result[0])
        self.assertEqual(result[1], 0)

    @patch("modules.common.system.detection.gpu_detector.PYOPENCL_AVAILABLE", True)
    @patch("modules.common.system.detection.gpu_detector.cl", create=True)
    def test_detect_opencl_available(self, mock_opencl):
        """Test OpenCL detection when available."""
        mock_platform = MagicMock()
        mock_device1 = MagicMock()
        mock_device2 = MagicMock()
        mock_platform.get_devices.return_value = [mock_device1, mock_device2]
        mock_opencl.get_platforms.return_value = [mock_platform]
        mock_opencl.device_type.GPU = "gpu"

        result = GPUDetector.detect_opencl()

        self.assertTrue(result[0])
        self.assertEqual(result[1], 2)

    @patch("modules.common.system.detection.gpu_detector.PYOPENCL_AVAILABLE", False)
    def test_detect_opencl_not_available(self):
        """Test OpenCL detection when not installed."""
        result = GPUDetector.detect_opencl()

        self.assertFalse(result[0])
        self.assertEqual(result[1], 0)

    @patch("subprocess.run")
    def test_detect_xgboost_available(self, mock_subprocess):
        """Test XGBoost GPU detection when nvidia-smi is available."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = b"NVIDIA-SMI 470.00.00"
        mock_subprocess.return_value = mock_result

        result = GPUDetector.detect_xgboost()

        self.assertTrue(result)
        mock_subprocess.assert_called_once()

    @patch("subprocess.run")
    def test_detect_xgboost_not_available(self, mock_subprocess):
        """Test XGBoost GPU detection when nvidia-smi is not available."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_subprocess.return_value = mock_result

        result = GPUDetector.detect_xgboost()

        self.assertFalse(result)

    @patch("modules.common.system.detection.gpu_detector.TORCH_AVAILABLE", True)
    @patch("modules.common.system.detection.gpu_detector.torch")
    def test_detect_all_pytorch_priority(self, mock_torch_module):
        """Test detect_all prioritizes PyTorch detection."""
        mock_torch_module.cuda.is_available.return_value = True
        mock_torch_module.cuda.device_count.return_value = 1
        mock_torch_module.version.cuda = "11.8"
        mock_torch_module.ones.return_value = Mock()
        mock_torch_module.cuda.get_device_properties.return_value.total_memory = 8 * 1024**3

        result = GPUDetector.detect_all()

        self.assertIsInstance(result, GPUInfo)
        self.assertTrue(result.available)
        self.assertEqual(result.gpu_type, "pytorch")
        self.assertTrue(result.pytorch_available)
        self.assertEqual(result.pytorch_cuda_version, "11.8")
        self.assertEqual(result.pytorch_device_count, 1)

    @patch("modules.common.system.detection.gpu_detector.TORCH_AVAILABLE", False)
    @patch("modules.common.system.detection.gpu_detector.CUPY_AVAILABLE", True)
    @patch("modules.common.system.detection.gpu_detector.cp", create=True)
    def test_detect_all_fallback_to_cupy(self, mock_cupy):
        """Test detect_all falls back to CuPy when PyTorch unavailable."""
        mock_cupy.cuda.runtime.getDeviceCount.return_value = 1
        mock_device = MagicMock()
        mock_device.mem_info = [0, 4 * 1024**3]
        mock_cupy.cuda.Device.return_value = mock_device

        result = GPUDetector.detect_all()

        self.assertIsInstance(result, GPUInfo)
        self.assertTrue(result.available)
        self.assertEqual(result.gpu_type, "cuda")


class TestCPUDetector(unittest.TestCase):
    """Test CPU detection functionality."""

    @patch("modules.common.system.detection.system_info.SystemInfo.get_cpu_info")
    def test_detect(self, mock_get_cpu_info):
        """Test CPU detection."""
        mock_cpu_info = CPUInfo(cores=8, cores_physical=4, percent_used=25.5)
        mock_get_cpu_info.return_value = mock_cpu_info

        result = CPUDetector.detect()

        self.assertIsInstance(result, CPUInfo)
        self.assertEqual(result.cores, 8)
        self.assertEqual(result.cores_physical, 4)
        self.assertEqual(result.percent_used, 25.5)

    @patch("modules.common.system.detection.system_info.SystemInfo.get_cpu_info")
    def test_get_cores(self, mock_get_cpu_info):
        """Test get_cores method."""
        mock_cpu_info = CPUInfo(cores=16, cores_physical=8, percent_used=0.0)
        mock_get_cpu_info.return_value = mock_cpu_info

        result = CPUDetector.get_cores()

        self.assertEqual(result, 16)

    @patch("modules.common.system.detection.system_info.SystemInfo.get_cpu_info")
    def test_get_physical_cores(self, mock_get_cpu_info):
        """Test get_physical_cores method."""
        mock_cpu_info = CPUInfo(cores=16, cores_physical=8, percent_used=0.0)
        mock_get_cpu_info.return_value = mock_cpu_info

        result = CPUDetector.get_physical_cores()

        self.assertEqual(result, 8)


class TestSystemInfo(unittest.TestCase):
    """Test system information detection."""

    @patch("modules.common.system.detection.system_info.PSUTIL_AVAILABLE", True)
    @patch("modules.common.system.detection.system_info.psutil")
    def test_get_memory_info_available(self, mock_psutil):
        """Test memory info when psutil is available."""
        mock_mem = MagicMock()
        mock_mem.total = 16 * 1024**3  # 16 GB
        mock_mem.available = 8 * 1024**3  # 8 GB
        mock_mem.used = 8 * 1024**3  # 8 GB
        mock_mem.percent = 50.0
        mock_psutil.virtual_memory.return_value = mock_mem

        result = SystemInfo.get_memory_info()

        self.assertIsInstance(result, MemoryInfo)
        self.assertAlmostEqual(result.total_gb, 16.0, places=1)
        self.assertAlmostEqual(result.available_gb, 8.0, places=1)
        self.assertAlmostEqual(result.used_gb, 8.0, places=1)
        self.assertEqual(result.percent_used, 50.0)

    @patch("modules.common.system.detection.system_info.PSUTIL_AVAILABLE", False)
    def test_get_memory_info_fallback(self):
        """Test memory info fallback when psutil is unavailable."""
        result = SystemInfo.get_memory_info()

        self.assertIsInstance(result, MemoryInfo)
        self.assertEqual(result.total_gb, 0.0)
        self.assertEqual(result.available_gb, 0.0)
        self.assertEqual(result.used_gb, 0.0)
        self.assertEqual(result.percent_used, 0.0)

    @patch("modules.common.system.detection.system_info.PSUTIL_AVAILABLE", True)
    @patch("modules.common.system.detection.system_info.psutil")
    def test_get_cpu_info_available(self, mock_psutil):
        """Test CPU info when psutil is available."""
        mock_psutil.cpu_count.side_effect = lambda logical: 16 if logical else 8
        mock_psutil.cpu_percent.return_value = 45.5

        result = SystemInfo.get_cpu_info()

        self.assertIsInstance(result, CPUInfo)
        self.assertEqual(result.cores, 16)
        self.assertEqual(result.cores_physical, 8)
        self.assertEqual(result.percent_used, 45.5)

    @patch("modules.common.system.detection.system_info.PSUTIL_AVAILABLE", False)
    @patch("multiprocessing.cpu_count")
    def test_get_cpu_info_fallback(self, mock_cpu_count):
        """Test CPU info fallback when psutil is unavailable."""
        mock_cpu_count.return_value = 4

        result = SystemInfo.get_cpu_info()

        self.assertIsInstance(result, CPUInfo)
        self.assertEqual(result.cores, 4)
        self.assertEqual(result.cores_physical, 4)
        self.assertEqual(result.percent_used, 0.0)

    @patch("modules.common.system.detection.system_info.PSUTIL_AVAILABLE", True)
    @patch("modules.common.system.detection.system_info.psutil")
    def test_get_cpu_percent_available(self, mock_psutil):
        """Test CPU percent when psutil is available."""
        mock_psutil.cpu_percent.return_value = 75.5

        result = SystemInfo.get_cpu_percent(interval=0.1)

        self.assertEqual(result, 75.5)
        mock_psutil.cpu_percent.assert_called_once_with(interval=0.1)

    @patch("modules.common.system.detection.system_info.PSUTIL_AVAILABLE", False)
    def test_get_cpu_percent_fallback(self):
        """Test CPU percent fallback when psutil is unavailable."""
        result = SystemInfo.get_cpu_percent()

        self.assertEqual(result, 0.0)

    def test_is_psutil_available(self):
        """Test psutil availability check."""
        # This will test the actual state, but we can verify the method exists
        result = SystemInfo.is_psutil_available()
        self.assertIsInstance(result, bool)


if __name__ == "__main__":
    unittest.main()

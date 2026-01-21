#!/usr/bin/env python3
"""
Test script specifically for testing PyTorch GPU availability detection.
This script can run independently without heavy deep learning imports.
"""

import sys
import warnings
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Suppress warnings
warnings.filterwarnings("ignore")


def test_detect_pytorch_gpu_availability_comprehensive():
    """Comprehensive test for detect_pytorch_gpu_availability function."""
    from modules.common.system import detect_pytorch_gpu_availability
    from modules.common.system.system import _pytorch_gpu_manager

    print("Testing PyTorch GPU availability detection...")

    # Reset cache between tests
    def reset_gpu_cache():
        _pytorch_gpu_manager._gpu_available = None
        _pytorch_gpu_manager._cuda_version = None
        _pytorch_gpu_manager._device_count = 0

    # Test case 1: GPU available and functional
    print("\n1. Testing GPU available and functional")
    reset_gpu_cache()
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=1),
        patch("torch.ones") as mock_ones,
        patch("torch.version.cuda", "11.8"),
        patch("modules.common.utils.system.log_system"),
    ):
        result = detect_pytorch_gpu_availability()
        assert result is True, f"Expected True, got {result}"
        mock_ones.assert_called_with(1, device="cuda:0")
        print("PASS: GPU available test passed")

    # Test case 2: GPU not available
    print("\n2. Testing GPU not available")
    reset_gpu_cache()
    with patch("torch.cuda.is_available", return_value=False), patch("modules.common.utils.system.log_info"):
        result = detect_pytorch_gpu_availability()
        assert result is False, f"Expected False, got {result}"
        print("PASS: GPU not available test passed")

    # Test case 3: GPU reports available but device creation fails
    print("\n3. Testing GPU available but device creation fails")
    reset_gpu_cache()
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=1),
        patch("torch.ones", side_effect=Exception("CUDA error")),
        patch("modules.common.utils.system.log_warn"),
    ):
        result = detect_pytorch_gpu_availability()
        assert result is False, f"Expected False, got {result}"
        print("PASS: GPU creation failure test passed")

    # Test case 4: Multiple GPUs, some functional
    print("\n4. Testing multiple GPUs with mixed functionality")
    reset_gpu_cache()
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=2),
        patch("torch.ones") as mock_ones,
        patch("torch.version.cuda", "12.1"),
        patch("modules.common.utils.system.log_system"),
        patch("modules.common.utils.system.log_warn"),
    ):
        # Mock so device 0 works, device 1 fails
        mock_ones.side_effect = [Mock(), Exception("Device 1 error")]
        result = detect_pytorch_gpu_availability()
        assert result is True, f"Expected True (at least one device works), got {result}"
        print("PASS: Multiple GPUs test passed")

    # Test case 5: All devices fail
    print("\n5. Testing all devices fail")
    reset_gpu_cache()
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=2),
        patch("torch.ones", side_effect=Exception("All devices failed")),
        patch("modules.common.utils.system.log_warn"),
    ):
        result = detect_pytorch_gpu_availability()
        assert result is False, f"Expected False, got {result}"
        print("PASS: All devices fail test passed")

    # Test case 6: PyTorch not available
    print("\n6. Testing PyTorch not available")
    reset_gpu_cache()
    with patch("modules.common.utils.system.PyTorchGPUManager._get_torch_module", return_value=None):
        result = detect_pytorch_gpu_availability()
        assert result is False, f"Expected False, got {result}"
        print("PASS: PyTorch not available test passed")

    print("\nSUCCESS: All GPU detection tests passed!")


def test_pytorch_gpu_manager_caching():
    """Test that GPU detection results are cached."""
    from modules.common.system.system import _pytorch_gpu_manager

    print("\nTesting GPU detection caching...")

    # Reset manager state
    _pytorch_gpu_manager._gpu_available = None
    _pytorch_gpu_manager._cuda_version = None
    _pytorch_gpu_manager._device_count = 0

    # Mock for first call
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=1),
        patch("torch.ones"),
        patch("torch.version.cuda", "11.8"),
        patch("modules.common.utils.system.log_system"),
    ):
        result1 = _pytorch_gpu_manager.is_available()
        assert result1 is True

        # Mock for second call (should use cache)
        with patch("torch.cuda.is_available", return_value=False):
            result2 = _pytorch_gpu_manager.is_available()
            assert result2 is True, "Should use cached result"

    print("PASS: GPU detection caching test passed")


if __name__ == "__main__":
    print("Running PyTorch GPU Detection Tests")
    print("=" * 50)

    try:
        test_detect_pytorch_gpu_availability_comprehensive()
        test_pytorch_gpu_manager_caching()

        print("\n" + "=" * 50)
        print("SUCCESS: ALL TESTS PASSED! GPU detection is working correctly.")
        print("\nTest Coverage:")
        print("  - GPU available and functional")
        print("  - GPU not available")
        print("  - GPU creation failures")
        print("  - Multiple GPUs scenarios")
        print("  - PyTorch not installed")
        print("  - Result caching")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

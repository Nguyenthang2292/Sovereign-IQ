from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from modules.common.system.managers.hardware_manager import HardwareManager


@pytest.fixture
def clean_hardware_manager():
    """Create a temporary HardwareManager and reset it."""
    # Reset singleton if needed, but here we just instantiate a fresh one
    # (since we are testing logic, not singleton state, unless singleton enforces uniqueness)
    # The actual class is HardwareManager, singleton is HardwareManagerSingleton.
    # We test HardwareManager directly to avoid global state issues.
    hm = HardwareManager()

    # Mock resources to simulate environment
    mock_resources = MagicMock()
    mock_resources.cpu_cores = 16
    mock_resources.cpu_cores_physical = 8
    mock_resources.gpu_available = True
    mock_resources.gpu_type = "cuda"

    hm._resources = mock_resources
    return hm


def test_optimal_execution_mode_sequential(clean_hardware_manager):
    """Test selection of sequential mode for small workloads."""
    mode = clean_hardware_manager.get_optimal_execution_mode(workload_size=5)
    assert mode == "sequential"


def test_optimal_execution_mode_threadpool(clean_hardware_manager):
    """Test selection of threadpool mode for medium workloads."""
    mode = clean_hardware_manager.get_optimal_execution_mode(workload_size=20)
    assert mode == "threadpool"


def test_optimal_execution_mode_processpool(clean_hardware_manager):
    """Test selection of processpool for large CPU workloads."""
    # Ensure usage of processpool for 50+ items
    mode = clean_hardware_manager.get_optimal_execution_mode(workload_size=100)
    assert mode == "processpool"


def test_optimal_execution_mode_gpu(clean_hardware_manager):
    """Test selection of gpu_batch for massive workloads when GPU is available."""
    mode = clean_hardware_manager.get_optimal_execution_mode(workload_size=1000)
    assert mode == "gpu_batch"


def test_optimal_execution_mode_gpu_fallback(clean_hardware_manager):
    """Test fallback to processpool if GPU is not available/compatible."""
    clean_hardware_manager._resources.gpu_available = False
    mode = clean_hardware_manager.get_optimal_execution_mode(workload_size=1000)
    assert mode == "processpool"

    # Also check if available but not supported type
    clean_hardware_manager._resources.gpu_available = True
    clean_hardware_manager._resources.gpu_type = "unknown_backend"
    mode = clean_hardware_manager.get_optimal_execution_mode(workload_size=1000)
    assert mode == "processpool"


if __name__ == "__main__":
    # Manually run tests if executed as script
    hm = HardwareManager()
    # Mock
    mock_res = MagicMock()
    mock_res.cpu_cores = 8
    mock_res.cpu_cores_physical = 4
    mock_res.gpu_available = True
    mock_res.gpu_type = "cuda"
    hm._resources = mock_res

    print(f"Size 5 -> {hm.get_optimal_execution_mode(5)}")
    print(f"Size 20 -> {hm.get_optimal_execution_mode(20)}")
    print(f"Size 100 -> {hm.get_optimal_execution_mode(100)}")
    print(f"Size 1000 -> {hm.get_optimal_execution_mode(1000)}")

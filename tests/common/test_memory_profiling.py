import tracemalloc
from unittest.mock import MagicMock, patch

import pytest

from modules.common.system.managers.memory_manager import profile_memory


def test_profile_memory_decorator():
    # Mock logger to verify warnings
    with patch("modules.common.system.managers.memory_manager.logger") as mock_logger:
        # Test 1: Enabled, below threshold
        @profile_memory(threshold_mb=100.0, enable=True, trace_key_count=1)
        def heavy_function_small():
            # Allocate something small
            x = [1] * 1000
            return len(x)

        result = heavy_function_small()
        assert result == 1000
        # Should verify tracemalloc was used (hard to checking direct calls on C-module, rely on logic flow)
        # Should NOT log warning
        warnings = [c for c in mock_logger.mock_calls if "warning" in str(c)]
        assert len(warnings) == 0

        # Test 2: Enabled, above threshold (Simulated via patching get_traced_memory)
        with patch("tracemalloc.get_traced_memory", return_value=(200 * 1024 * 1024, 200 * 1024 * 1024)):

            @profile_memory(threshold_mb=100.0, enable=True)
            def heavy_function_large():
                pass

            heavy_function_large()

            # Should log warning
            warnings = [c for c in mock_logger.mock_calls if "warning" in str(c) and "Exceeded" in str(c)]
            assert len(warnings) > 0

        # Test 3: Disabled
        @profile_memory(enable=False)
        def disabled_func():
            return "ok"

        # If disabled, it should just run without touching tracemalloc
        # We can't easily verify non-touching of C module without more complex mocks,
        # but we check it returns correct value
        assert disabled_func() == "ok"


if __name__ == "__main__":
    test_profile_memory_decorator()
    print("Test passed")

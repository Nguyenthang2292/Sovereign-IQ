"""
Test Script for Phase 6.1: Main Auto-Trade Loop
================================================

Tests main event loop initialization and basic functionality.

Run:python test_main_loop.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from modules.auto_trade.config import get_config, get_testnet_config, load_config, save_config
from modules.auto_trade.main import AutoTradeSystem


def test_config_creation():
    """Test 1: Configuration Creation"""
    print("\n" + "=" * 60)
    print("TEST 1: Configuration Creation")
    print("=" * 60)

    config = get_config()

    print("\n✓ Default configuration created")
    print(config.get_summary())

    assert config is not None, "Config should not be None"
    assert config.risk.leverage == 2, "Default leverage should be 2"
    assert config.martingale.enabled is True, "Martingale should be enabled by default"

    print("\n✅ Configuration creation test PASSED")
    return True


def test_config_validation():
    """Test 2: Configuration Validation"""
    print("\n" + "=" * 60)
    print("TEST 2: Configuration Validation")
    print("=" * 60)

    # Test valid config
    config = get_testnet_config()
    print("✓ Valid testnet config created")

    # Test invalid leverage
    try:
        config2 = get_testnet_config()
        config2.risk.leverage = 200  # Too high
        config2.validate()
        raise AssertionError("Should have raised validation error")
    except ValueError as e:
        print(f"✓ Invalid leverage rejected: {e}")

    # Test invalid scan interval
    try:
        config3 = get_testnet_config()
        config3.scanning.scan_interval = 30  # Too low
        config3.validate()
        raise AssertionError("Should have raised validation error")
    except ValueError as e:
        print(f"✓ Invalid scan interval rejected: {e}")

    print("\n✅ Configuration validation test PASSED")
    return True


def test_config_export_import():
    """Test 3: Configuration Export/Import"""
    print("\n" + "=" * 60)
    print("TEST 3: Configuration Export/Import")
    print("=" * 60)

    # Create config
    config = get_testnet_config()
    config.risk.max_open_positions = 5

    # Export to JSON
    json_path = "data/test_config.json"
    config.to_json(json_path)
    print(f"✓ Config exported to {json_path}")

    # Import from JSON
    loaded_config = load_config(json_path)
    print(f"✓ Config imported from {json_path}")

    # Verify
    assert loaded_config.risk.max_open_positions == 5, "Loaded config mismatch"
    assert loaded_config.binance.testnet is True, "Testnet flag mismatch"
    print("✓ Loaded config matches original")

    print("\n✅ Configuration export/import test PASSED")
    return True


async def test_system_initialization():
    """Test 4: System Initialization"""
    print("\n" + "=" * 60)
    print("TEST 4: System Initialization")
    print("=" * 60)

    # Create system
    config = get_testnet_config()
    system = AutoTradeSystem(config.to_dict())

    print("✓ AutoTradeSystem instance created")

    # Initialize
    await system.initialize()

    print("✓ System initialized successfully")
    print(f"  Database: {config.database.path}")
    print(f"  Dry Run: {config.dry_run}")
    print(f"  Max Positions: {config.risk.max_open_positions}")

    assert system.stats is not None, "Stats should be initialized"
    assert system.stats["loops_completed"] == 0, "Loops should be 0"

    print("\n✅ System initialization test PASSED")
    return True


async def test_main_loop_single_iteration():
    """Test 5: Main Loop Single Iteration"""
    print("\n" + "=" * 60)
    print("TEST 5: Main Loop Single Iteration")
    print("=" * 60)

    # Create system with short scan interval
    config = get_testnet_config()
    config.scanning.scan_interval = 1  # 1 second for testing

    system = AutoTradeSystem(config.to_dict())
    await system.initialize()

    print("✓ System initialized for loop test")

    # Run one iteration
    print("Running single loop iteration...")

    # Start loop in background
    loop_task = asyncio.create_task(system.main_loop())

    # Wait for one iteration
    await asyncio.sleep(2)

    # Stop loop
    system.shutdown_requested = True
    await loop_task

    print(f"✓ Loop completed: {system.stats['loops_completed']} iteration(s)")

    assert system.stats["loops_completed"] >= 1, "At least 1 iteration should complete"

    print("\n✅ Main loop single iteration test PASSED")
    return True


async def test_graceful_shutdown():
    """Test 6: Graceful Shutdown"""
    print("\n" + "=" * 60)
    print("TEST 6: Graceful Shutdown")
    print("=" * 60)

    # Create and initialize system
    system = AutoTradeSystem()
    await system.initialize()

    print("✓ System initialized")

    # Shutdown
    await system.shutdown()

    print("✓ System shutdown completed")

    assert system.running is False, "System should not be running"

    print("\n✅ Graceful shutdown test PASSED")
    return True


async def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  PHASE 6.1: MAIN AUTO-TRADE LOOP - TEST SUITE")
    print("=" * 60)

    tests = [
        ("Configuration Creation", test_config_creation, False),
        ("Configuration Validation", test_config_validation, False),
        ("Configuration Export/Import", test_config_export_import, False),
        ("System Initialization", test_system_initialization, True),
        ("Main Loop Single Iteration", test_main_loop_single_iteration, True),
        ("Graceful Shutdown", test_graceful_shutdown, True),
    ]

    results = []

    for test_name, test_func, is_async in tests:
        try:
            if is_async:
                success = await test_func()
            else:
                success = test_func()
            results.append((test_name, success, None))
        except Exception as e:
            print(f"\n❌ {test_name} FAILED: {e}")
            import traceback

            traceback.print_exc()
            results.append((test_name, False, str(e)))

    # Print summary
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60 + "\n")

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    for test_name, success, error in results:
        if success:
            print(f"✅ {test_name}")
        else:
            print(f"❌ {test_name}")
            if error:
                print(f"   Error: {error}")

    print(f"\n{passed}/{total} tests passed ({(passed / total) * 100:.0f}%)")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Main loop is ready.")
    else:
        print("\n⚠️  Some tests failed. Please review errors above.")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)

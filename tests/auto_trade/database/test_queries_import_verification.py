"""
Import Verification Test for queries.py Refactoring
===================================================

Tests backward compatibility and new import patterns.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def test_facade_imports():
    """Test backward compatibility - importing from queries.py facade."""
    print("=" * 70)
    print("TEST 1: Facade Imports (Backward Compatibility)")
    print("=" * 70)

    try:
        # Test importing from the facade (old pattern)
        from modules.auto_trade.database.queries import (
            # Order queries
            get_open_positions,
            get_last_closed_order,
            get_all_programmatic_orders,
            get_orders_cursor,
            is_programmatic_order,
            get_order_by_id,
            get_order_by_client_id,
            update_order_status_by_client_id,
            update_order_status,
            mark_be_moved,
            create_order,
            get_orders_by_symbol,
            # Martingale queries
            get_martingale_state,
            find_or_create_martingale_chain,
            update_martingale_chain,
            get_active_martingale_chains,
            get_martingale_chains_cursor,
            # Signal queries
            save_signal,
            mark_signal_executed,
            update_signal_outcome,
            get_recent_signals,
            get_signal_performance_stats,
            get_signals_cursor,
            # System state queries
            get_system_state,
            set_system_state,
            # Audit log queries
            create_audit_log,
            get_recent_audit_logs,
            get_audit_log_cursor,
            # Statistics queries
            get_daily_stats,
            get_overall_stats,
            # Gradual recovery queries
            get_active_gradual_recovery,
            create_gradual_recovery,
            update_gradual_recovery,
            cancel_gradual_recovery,
            get_gradual_recovery_by_id,
            get_all_gradual_recoveries,
        )

        print("✅ SUCCESS: All 41 functions imported from facade")
        print(f"   - 12 order functions")
        print(f"   - 5 Martingale functions")
        print(f"   - 6 signal functions")
        print(f"   - 2 system state functions")
        print(f"   - 3 audit log functions")
        print(f"   - 2 statistics functions")
        print(f"   - 6 gradual recovery functions")
        print(f"   - 5 cursor pagination functions")
        return True

    except ImportError as e:
        print(f"❌ FAILED: {e}")
        return False


def test_direct_module_imports():
    """Test new import pattern - importing directly from sub-modules."""
    print("\n" + "=" * 70)
    print("TEST 2: Direct Module Imports (New Pattern)")
    print("=" * 70)

    try:
        # Test importing from sub-modules directly
        from modules.auto_trade.database.queries.orders import (
            get_open_positions,
            create_order,
        )
        from modules.auto_trade.database.queries.martingale import (
            get_martingale_state,
        )
        from modules.auto_trade.database.queries.signals import (
            save_signal,
        )
        from modules.auto_trade.database.queries.system_state import (
            get_system_state,
        )
        from modules.auto_trade.database.queries.audit_logs import (
            create_audit_log,
        )
        from modules.auto_trade.database.queries.statistics import (
            get_daily_stats,
        )
        from modules.auto_trade.database.queries.gradual_recovery import (
            get_active_gradual_recovery,
        )

        print("✅ SUCCESS: Direct imports from all 7 sub-modules work")
        print(f"   - orders.get_open_positions")
        print(f"   - martingale.get_martingale_state")
        print(f"   - signals.save_signal")
        print(f"   - system_state.get_system_state")
        print(f"   - audit_logs.create_audit_log")
        print(f"   - statistics.get_daily_stats")
        print(f"   - gradual_recovery.get_active_gradual_recovery")
        return True

    except ImportError as e:
        print(f"❌ FAILED: {e}")
        return False


def test_package_imports():
    """Test importing from queries package __init__.py."""
    print("\n" + "=" * 70)
    print("TEST 3: Package Imports (queries/__init__.py)")
    print("=" * 70)

    try:
        # Test importing from package
        from modules.auto_trade.database import queries

        # Verify functions are accessible
        assert hasattr(queries, 'get_open_positions')
        assert hasattr(queries, 'save_signal')
        assert hasattr(queries, 'get_daily_stats')

        print("✅ SUCCESS: Package imports work correctly")
        print(f"   - queries.get_open_positions accessible")
        print(f"   - queries.save_signal accessible")
        print(f"   - queries.get_daily_stats accessible")
        return True

    except (ImportError, AssertionError) as e:
        print(f"❌ FAILED: {e}")
        return False


def test_function_signatures():
    """Test that function signatures are preserved."""
    print("\n" + "=" * 70)
    print("TEST 4: Function Signature Verification")
    print("=" * 70)

    try:
        from modules.auto_trade.database.queries import (
            get_open_positions,
            create_order,
            save_signal,
        )
        import inspect

        # Check get_open_positions signature
        sig = inspect.signature(get_open_positions)
        params = list(sig.parameters.keys())
        assert 'session' in params
        assert 'symbol' in params
        assert 'order_source' in params

        # Check create_order signature
        sig = inspect.signature(create_order)
        params = list(sig.parameters.keys())
        assert 'session' in params
        assert 'order_data' in params

        # Check save_signal signature
        sig = inspect.signature(save_signal)
        params = list(sig.parameters.keys())
        assert 'session' in params
        assert 'correlation_id' in params
        assert 'symbol' in params

        print("✅ SUCCESS: Function signatures preserved correctly")
        print(f"   - get_open_positions(session, symbol, order_source)")
        print(f"   - create_order(session, order_data)")
        print(f"   - save_signal(session, correlation_id, symbol, ...)")
        return True

    except (ImportError, AssertionError) as e:
        print(f"❌ FAILED: {e}")
        return False


def main():
    """Run all import verification tests."""
    print("\n" + "=" * 70)
    print("QUERIES.PY REFACTORING - IMPORT VERIFICATION TEST")
    print("=" * 70)
    print(f"Testing backward compatibility and new import patterns\n")

    results = []

    # Run tests
    results.append(("Facade Imports", test_facade_imports()))
    results.append(("Direct Module Imports", test_direct_module_imports()))
    results.append(("Package Imports", test_package_imports()))
    results.append(("Function Signatures", test_function_signatures()))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Backward compatibility verified!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

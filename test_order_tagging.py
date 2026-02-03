"""
Test Script for Order Tagging System
======================================

Tests order tagging functionality and client order ID generation.

Run: python test_order_tagging.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import time

from modules.auto_trade.execution.order_tagging import (
    CLIENT_ORDER_ID_PREFIX,
    EXECUTION_MODE_AUTO,
    ORDER_SOURCE_PROGRAMMATIC,
    OrderTagger,
    extract_order_info,
    generate_order_id,
    get_order_tag_stats,
    is_auto_trade_order,
    tag_multiple_orders,
    tag_programmatic_order,
    validate_order_metadata,
)


def print_section(title: str):
    """Print section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def test_client_order_id_generation():
    """Test 1: Client Order ID Generation"""
    print_section("TEST 1: Client Order ID Generation")

    # Generate client order ID
    symbol = "BTCUSDT"
    client_order_id = OrderTagger.generate_client_order_id(symbol)

    print(f"Generated Client Order ID: {client_order_id}")
    print(f"✓ Format verified: starts with '{CLIENT_ORDER_ID_PREFIX}'")

    # Verify format
    assert client_order_id.startswith(CLIENT_ORDER_ID_PREFIX), "Missing AT_ prefix"
    assert symbol in client_order_id, "Symbol not in client_order_id"

    # Verify uniqueness
    ids = [OrderTagger.generate_client_order_id(symbol) for _ in range(100)]
    unique_ids = set(ids)

    print(f"✓ Uniqueness test: Generated 100 IDs, {len(unique_ids)} unique")
    assert len(unique_ids) == 100, "Duplicate IDs detected!"

    # Test with milliseconds
    id_ms = OrderTagger.generate_client_order_id(symbol, use_milliseconds=True)
    print(f"✓ With milliseconds: {id_ms}")

    print("\n✅ Client Order ID generation test PASSED")
    return True


def test_order_id_parsing():
    """Test 2: Parse Client Order ID"""
    print_section("TEST 2: Client Order ID Parsing")

    # Generate and parse
    symbol = "ETHUSDT"
    client_order_id = OrderTagger.generate_client_order_id(symbol)

    print(f"Original ID: {client_order_id}")

    info = OrderTagger.parse_client_order_id(client_order_id)

    if info:
        print(f"✓ Parsed successfully:")
        print(f"  - Symbol: {info['symbol']}")
        print(f"  - Timestamp: {info['timestamp']}")
        print(f"  - DateTime: {info['datetime']}")
        print(f"  - Random Suffix: {info['random_suffix']}")
        print(f"  - Is Programmatic: {info['is_programmatic']}")

        assert info["symbol"] == symbol, "Symbol mismatch"
        assert info["is_programmatic"] is True, "Should be programmatic"
    else:
        raise AssertionError("Failed to parse client_order_id")

    # Test invalid ID
    invalid_info = OrderTagger.parse_client_order_id("MANUAL_ORDER_123")
    assert invalid_info is None, "Should reject non-programmatic IDs"
    print("✓ Correctly rejected non-programmatic ID")

    print("\n✅ Client Order ID parsing test PASSED")
    return True


def test_order_identification():
    """Test 3: Order Identification"""
    print_section("TEST 3: Order Identification")

    # Programmatic order
    prog_id = generate_order_id("BTCUSDT")
    assert is_auto_trade_order(prog_id) is True, "Should identify as auto_trade"
    print(f"✓ Programmatic ID identified: {prog_id}")

    # Manual order
    manual_id = "MANUAL_ORDER_12345"
    assert is_auto_trade_order(manual_id) is False, "Should reject manual order"
    print(f"✓ Manual ID rejected: {manual_id}")

    # Empty/None
    assert is_auto_trade_order("") is False, "Should reject empty string"
    assert is_auto_trade_order(None) is False, "Should reject None"
    print("✓ Edge cases handled correctly")

    print("\n✅ Order identification test PASSED")
    return True


def test_order_metadata_creation():
    """Test 4: Order Metadata Creation"""
    print_section("TEST 4: Order Metadata Creation")

    symbol = "BTCUSDT"
    signal_id = "SIGNAL_BTCUSDT_LONG_1707043200_abc123"

    # Tag order
    metadata = tag_programmatic_order(symbol=symbol, signal_id=signal_id)

    print("✓ Order Metadata Created:")
    print(f"  - Client Order ID: {metadata['client_order_id']}")
    print(f"  - Order Source: {metadata['order_source']}")
    print(f"  - Execution Mode: {metadata['execution_mode']}")
    print(f"  - Signal Correlation ID: {metadata.get('signal_correlation_id')}")
    print(f"  - Is Programmatic: {metadata['is_programmatic']}")
    print(f"  - Created At: {metadata['created_at']}")

    # Verify
    assert metadata["order_source"] == ORDER_SOURCE_PROGRAMMATIC, "Wrong order source"
    assert metadata["execution_mode"] == EXECUTION_MODE_AUTO, "Wrong execution mode"
    assert metadata["is_programmatic"] is True, "Should be programmatic"
    assert metadata["signal_correlation_id"] == signal_id, "Signal ID mismatch"

    print("\n✅ Order metadata creation test PASSED")
    return True


def test_metadata_validation():
    """Test 5: Metadata Validation"""
    print_section("TEST 5: Metadata Validation")

    # Valid metadata
    valid_metadata = tag_programmatic_order("BTCUSDT")
    is_valid, error = validate_order_metadata(valid_metadata)

    assert is_valid is True, f"Should be valid: {error}"
    print("✓ Valid metadata accepted")

    # Invalid metadata - missing field
    invalid_metadata = {"client_order_id": "AT_123_BTCUSDT_abc"}
    is_valid, error = validate_order_metadata(invalid_metadata)

    assert is_valid is False, "Should reject incomplete metadata"
    print(f"✓ Invalid metadata rejected: {error}")

    # Invalid metadata - wrong prefix
    invalid_prefix = {
        "client_order_id": "WRONG_PREFIX_123",
        "order_source": ORDER_SOURCE_PROGRAMMATIC,
        "execution_mode": EXECUTION_MODE_AUTO,
    }
    is_valid, error = validate_order_metadata(invalid_prefix)

    assert is_valid is False, "Should reject wrong prefix"
    print(f"✓ Wrong prefix rejected: {error}")

    print("\n✅ Metadata validation test PASSED")
    return True


def test_batch_operations():
    """Test 6: Batch Operations"""
    print_section("TEST 6: Batch Operations")

    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"]

    # Tag multiple orders
    batch_metadata = tag_multiple_orders(symbols)

    print(f"✓ Tagged {len(batch_metadata)} orders:")
    for meta in batch_metadata[:3]:  # Show first 3
        print(f"  - {meta['client_order_id']}")

    assert len(batch_metadata) == len(symbols), "Count mismatch"

    # Verify all unique
    client_ids = [m["client_order_id"] for m in batch_metadata]
    assert len(set(client_ids)) == len(client_ids), "Duplicate IDs in batch!"
    print("✓ All IDs unique in batch")

    print("\n✅ Batch operations test PASSED")
    return True


def test_martingale_chain_id():
    """Test 7: Martingale Chain ID Generation"""
    print_section("TEST 7: Martingale Chain ID")

    symbol = "BTCUSDT"
    initial_order_id = "ORDER_12345678"

    chain_id = OrderTagger.generate_martingale_chain_id(symbol, initial_order_id)

    print(f"Generated Chain ID: {chain_id}")
    print(f"✓ Format: CHAIN_{symbol}_{{timestamp}}_{{suffix}}")

    assert chain_id.startswith("CHAIN_"), "Wrong prefix"
    assert symbol in chain_id, "Symbol missing"

    print("\n✅ Martingale chain ID test PASSED")
    return True


def test_signal_correlation_id():
    """Test 8: Signal Correlation ID Generation"""
    print_section("TEST 8: Signal Correlation ID")

    symbol = "ETHUSDT"
    signal_type = "LONG"

    correlation_id = OrderTagger.generate_signal_correlation_id(symbol, signal_type)

    print(f"Generated Correlation ID: {correlation_id}")
    print(f"✓ Format: SIGNAL_{symbol}_{signal_type}_{{timestamp}}_{{random}}")

    assert correlation_id.startswith("SIGNAL_"), "Wrong prefix"
    assert symbol in correlation_id, "Symbol missing"
    assert signal_type in correlation_id, "Signal type missing"

    print("\n✅ Signal correlation ID test PASSED")
    return True


def test_statistics():
    """Test 9: Order Tag Statistics"""
    print_section("TEST 9: Order Tag Statistics")

    # Generate mixed order IDs
    programmatic_ids = [generate_order_id("BTCUSDT") for _ in range(7)]
    manual_ids = [f"MANUAL_{i}" for i in range(3)]

    all_ids = programmatic_ids + manual_ids

    stats = get_order_tag_stats(all_ids)

    print("✓ Order Tag Statistics:")
    print(f"  - Total Orders: {stats['total_orders']}")
    print(f"  - Programmatic: {stats['programmatic_orders']}")
    print(f"  - Manual: {stats['manual_orders']}")
    print(f"  - Programmatic %: {stats['programmatic_percentage']:.1f}%")
    print(f"  - Unique Symbols: {stats['unique_symbols']}")

    assert stats["total_orders"] == 10, "Count mismatch"
    assert stats["programmatic_orders"] == 7, "Programmatic count wrong"
    assert stats["manual_orders"] == 3, "Manual count wrong"

    print("\n✅ Statistics test PASSED")
    return True


def test_integration_example():
    """Test 10: Full Integration Example"""
    print_section("TEST 10: Integration Example")

    print("Simulating order creation workflow:\n")

    # Step 1: Generate signal correlation ID
    signal_id = OrderTagger.generate_signal_correlation_id("BTCUSDT", "LONG")
    print(f"1. Signal Generated: {signal_id}")

    # Step 2: Tag order with signal
    order_metadata = tag_programmatic_order("BTCUSDT", signal_id=signal_id)
    print(f"2. Order Tagged: {order_metadata['client_order_id']}")

    # Step 3: Validate metadata
    is_valid, error = validate_order_metadata(order_metadata)
    print(f"3. Metadata Valid: {is_valid}")

    # Step 4: Verify it's programmatic
    is_programmatic = is_auto_trade_order(order_metadata["client_order_id"])
    print(f"4. Is Programmatic: {is_programmatic}")

    # Step 5: Extract info
    info = extract_order_info(order_metadata["client_order_id"])
    print(f"5. Extracted Symbol: {info['symbol']}")
    print(f"   Extracted DateTime: {info['datetime']}")

    print("\n✓ Full integration workflow completed successfully")

    print("\n✅ Integration example test PASSED")
    return True


def run_all_tests():
    """Run all order tagging tests."""
    print("\n" + "=" * 60)
    print("  ORDER TAGGING SYSTEM - COMPREHENSIVE TEST")
    print("=" * 60)

    tests = [
        ("Client Order ID Generation", test_client_order_id_generation),
        ("Client Order ID Parsing", test_order_id_parsing),
        ("Order Identification", test_order_identification),
        ("Order Metadata Creation", test_order_metadata_creation),
        ("Metadata Validation", test_metadata_validation),
        ("Batch Operations", test_batch_operations),
        ("Martingale Chain ID", test_martingale_chain_id),
        ("Signal Correlation ID", test_signal_correlation_id),
        ("Statistics", test_statistics),
        ("Integration Example", test_integration_example),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success, None))
        except Exception as e:
            print(f"\n❌ {test_name} FAILED: {e}")
            import traceback

            traceback.print_exc()
            results.append((test_name, False, str(e)))

    # Print summary
    print_section("TEST SUMMARY")

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
        print("\n🎉 ALL TESTS PASSED! Order tagging system is ready.")
    else:
        print("\n⚠️  Some tests failed. Please review errors above.")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

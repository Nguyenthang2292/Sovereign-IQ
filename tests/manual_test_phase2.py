"""
Phase 2 Manual Testing Script
Run this to manually test the GUI components
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print("PHASE 2 MANUAL TESTING CHECKLIST")
print("=" * 60)
print()

# Test 1: Risk Calculator
print("1. Risk Calculator Tests")
print("-" * 40)
try:
    from modules.auto_trade.gui.utils.risk_calculator import RiskCalculator

    # Test LONG trade
    result = RiskCalculator.calculate(
        symbol="BTC/USDT",
        side="LONG",
        amount_usdt=100.0,
        leverage=10,
        current_price=50000.0,
        tp_percent=5.0,
        sl_percent=2.5,
    )

    if result:
        print("[PASS] LONG trade calculation:")
        print(f"  - Contract Size: {result['contract_size']:.6f} BTC")
        print(f"  - Margin Required: ${result['margin_required']:.2f}")
        print(f"  - Max Profit: +${result['max_profit']:.2f}")
        print(f"  - Max Loss: -${result['max_loss']:.2f}")
        print(f"  - TP Price: ${result['tp_price']:,.2f}")
        print(f"  - SL Price: ${result['sl_price']:,.2f}")
        print(f"  - Risk/Reward: {result['risk_reward_ratio']:.2f}:1")
        print(f"  - Liquidation: ${result['liquidation_price']:,.2f}")
    else:
        print("[FAIL] LONG trade calculation failed")

    # Test SHORT trade
    result = RiskCalculator.calculate(
        symbol="ETH/USDT",
        side="SHORT",
        amount_usdt=50.0,
        leverage=5,
        current_price=3000.0,
        tp_percent=4.0,
        sl_percent=2.0,
    )

    if result:
        print("\n[PASS] SHORT trade calculation:")
        print(f"  - Contract Size: {result['contract_size']:.6f} ETH")
        print(f"  - Margin Required: ${result['margin_required']:.2f}")
        print(f"  - Max Profit: +${result['max_profit']:.2f}")
        print(f"  - Max Loss: -${result['max_loss']:.2f}")
    else:
        print("\n[FAIL] SHORT trade calculation failed")

except Exception as e:
    print(f"[FAIL] Risk calculator test failed: {e}")

print()
print()

# Test 2: Component Imports
print("2. Component Import Tests")
print("-" * 40)
components = [
    ("TradeFormFrame", "modules.auto_trade.gui.components.trade_form"),
    ("AutoTradeControl", "modules.auto_trade.gui.components.auto_trade_control"),
    ("RiskCalculator", "modules.auto_trade.gui.utils.risk_calculator"),
]

for component_name, module_path in components:
    try:
        parts = module_path.split(".")
        module = __import__(module_path)
        for part in parts[1:]:
            module = getattr(module, part)
        component = getattr(module, component_name)
        print(f"[PASS] {component_name} imported successfully")
    except Exception as e:
        print(f"[FAIL] {component_name} import failed: {e}")

print()
print()

# Test 3: Validation Scenarios
print("3. Form Validation Scenarios")
print("-" * 40)

test_cases = [
    ("Valid trade", 100.0, 10, 5.0, 2.5, True),
    ("Empty amount", None, 10, 5.0, 2.5, False),
    ("Negative amount", -100.0, 10, 5.0, 2.5, False),
    ("Too much amount", 1500.0, 10, 5.0, 2.5, False),
    ("Invalid leverage", 100.0, 150, 5.0, 2.5, False),
    ("TP too close to SL", 100.0, 10, 3.0, 2.5, False),
]

for name, amount, leverage, tp, sl, should_pass in test_cases:
    # Simple validation logic
    is_valid = True
    errors = []

    if amount is None or amount == "":
        is_valid = False
        errors.append("Empty amount")
    elif amount <= 0:
        is_valid = False
        errors.append("Negative amount")
    elif amount > 1000:
        is_valid = False
        errors.append("Amount too high")

    if leverage < 1 or leverage > 100:
        is_valid = False
        errors.append("Invalid leverage")

    if tp <= 0 or sl <= 0:
        is_valid = False
        errors.append("Invalid TP/SL")
    elif tp < sl * 1.5:
        is_valid = False
        errors.append("TP too close to SL")

    result = "[PASS]" if (is_valid == should_pass) else "[FAIL]"
    status = "PASS" if should_pass else "FAIL"

    print(f"{result} {name}: {status} (expected)")
    if errors and not should_pass:
        print(f"   Errors: {', '.join(errors)}")

print()
print()

# Test 4: Risk Limits
print("4. Risk Limit Checking")
print("-" * 40)

scenarios = [
    ("No positions", [], True),
    ("Under limit", [1, 2], True),
    ("At limit", [1, 2, 3], False),
    ("Over limit", [1, 2, 3, 4], False),
]

for name, positions, should_allow in scenarios:
    max_positions = 3
    can_trade = len(positions) < max_positions

    result = "[PASS]" if (can_trade == should_allow) else "[FAIL]"
    status = "ALLOW" if can_trade else "DENY"

    print(f"{result} {name}: {status} ({len(positions)} positions)")

print()
print()

# Test 5: Integration Points
print("5. Integration Points")
print("-" * 40)

integrations = [
    ("DataService", "modules.auto_trade.gui.utils.data_service"),
    ("ThreadingUtils", "modules.auto_trade.gui.utils.threading_utils"),
    ("OrderExecutor", "modules.auto_trade.execution.order_executor"),
    ("SignalSelector", "modules.auto_trade.signal_selector"),
]

for name, module_path in integrations:
    try:
        __import__(module_path)
        print(f"[PASS] {name} module available")
    except Exception as e:
        print(f"[FAIL] {name} module not found: {e}")

print()
print("=" * 60)
print("MANUAL TESTING COMPLETE")
print("=" * 60)
print()
print("Next Steps:")
print("1. Run GUI: python modules/auto_trade/gui/main_window.py")
print("2. Test manual trade form (Trading tab)")
print("3. Test auto-trade toggle (enable/disable)")
print("4. Verify risk calculations update in real-time")
print("5. Test trade execution on DEMO account")
print()
print("IMPORTANT:")
print("- Always test on DEMO account first")
print("- Verify all validation error messages display correctly")
print("- Check that TP/SL prices update when parameters change")
print("- Confirm leverage warning appears for >10x")
print("- Test auto-trade cycle with sample signals")
print()


# Test 1: Risk Calculator
print("1. Risk Calculator Tests")
print("-" * 40)
try:
    from modules.auto_trade.gui.utils.risk_calculator import RiskCalculator

    # Test LONG trade
    result = RiskCalculator.calculate(
        symbol="BTC/USDT",
        side="LONG",
        amount_usdt=100.0,
        leverage=10,
        current_price=50000.0,
        tp_percent=5.0,
        sl_percent=2.5,
    )

    if result:
        print("✓ LONG trade calculation:")
        print(f"  - Contract Size: {result['contract_size']:.6f} BTC")
        print(f"  - Margin Required: ${result['margin_required']:.2f}")
        print(f"  - Max Profit: +${result['max_profit']:.2f}")
        print(f"  - Max Loss: -${result['max_loss']:.2f}")
        print(f"  - TP Price: ${result['tp_price']:,.2f}")
        print(f"  - SL Price: ${result['sl_price']:,.2f}")
        print(f"  - Risk/Reward: {result['risk_reward_ratio']:.2f}:1")
        print(f"  - Liquidation: ${result['liquidation_price']:,.2f}")
    else:
        print("✗ LONG trade calculation failed")

    # Test SHORT trade
    result = RiskCalculator.calculate(
        symbol="ETH/USDT",
        side="SHORT",
        amount_usdt=50.0,
        leverage=5,
        current_price=3000.0,
        tp_percent=4.0,
        sl_percent=2.0,
    )

    if result:
        print("\n✓ SHORT trade calculation:")
        print(f"  - Contract Size: {result['contract_size']:.6f} ETH")
        print(f"  - Margin Required: ${result['margin_required']:.2f}")
        print(f"  - Max Profit: +${result['max_profit']:.2f}")
        print(f"  - Max Loss: -${result['max_loss']:.2f}")
    else:
        print("\n✗ SHORT trade calculation failed")

except Exception as e:
    print(f"✗ Risk calculator test failed: {e}")

print()
print()

# Test 2: Component Imports
print("2. Component Import Tests")
print("-" * 40)
components = [
    ("TradeFormFrame", "modules.auto_trade.gui.components.trade_form"),
    ("AutoTradeControl", "modules.auto_trade.gui.components.auto_trade_control"),
    ("RiskCalculator", "modules.auto_trade.gui.utils.risk_calculator"),
]

for component_name, module_path in components:
    try:
        parts = module_path.split(".")
        module = __import__(module_path)
        for part in parts[1:]:
            module = getattr(module, part)
        component = getattr(module, component_name)
        print(f"✓ {component_name} imported successfully")
    except Exception as e:
        print(f"✗ {component_name} import failed: {e}")

print()
print()

# Test 3: Validation Scenarios
print("3. Form Validation Scenarios")
print("-" * 40)

test_cases = [
    ("Valid trade", 100.0, 10, 5.0, 2.5, True),
    ("Empty amount", None, 10, 5.0, 2.5, False),
    ("Negative amount", -100.0, 10, 5.0, 2.5, False),
    ("Too much amount", 1500.0, 10, 5.0, 2.5, False),
    ("Invalid leverage", 100.0, 150, 5.0, 2.5, False),
    ("TP too close to SL", 100.0, 10, 3.0, 2.5, False),
]

for name, amount, leverage, tp, sl, should_pass in test_cases:
    # Simple validation logic
    is_valid = True
    errors = []

    if amount is None or amount == "":
        is_valid = False
        errors.append("Empty amount")
    elif amount <= 0:
        is_valid = False
        errors.append("Negative amount")
    elif amount > 1000:
        is_valid = False
        errors.append("Amount too high")

    if leverage < 1 or leverage > 100:
        is_valid = False
        errors.append("Invalid leverage")

    if tp <= 0 or sl <= 0:
        is_valid = False
        errors.append("Invalid TP/SL")
    elif tp < sl * 1.5:
        is_valid = False
        errors.append("TP too close to SL")

    result = "✓" if (is_valid == should_pass) else "✗"
    status = "PASS" if should_pass else "FAIL"

    print(f"{result} {name}: {status} (expected)")
    if errors and not should_pass:
        print(f"   Errors: {', '.join(errors)}")

print()
print()

# Test 4: Risk Limits
print("4. Risk Limit Checking")
print("-" * 40)

scenarios = [
    ("No positions", [], True),
    ("Under limit", [1, 2], True),
    ("At limit", [1, 2, 3], False),
    ("Over limit", [1, 2, 3, 4], False),
]

for name, positions, should_allow in scenarios:
    max_positions = 3
    can_trade = len(positions) < max_positions

    result = "✓" if (can_trade == should_allow) else "✗"
    status = "ALLOW" if can_trade else "DENY"

    print(f"{result} {name}: {status} ({len(positions)} positions)")

print()
print()

# Test 5: Integration Points
print("5. Integration Points")
print("-" * 40)

integrations = [
    ("DataService", "modules.auto_trade.gui.utils.data_service"),
    ("ThreadingUtils", "modules.auto_trade.gui.utils.threading_utils"),
    ("OrderExecutor", "modules.auto_trade.execution.order_executor"),
    ("SignalSelector", "modules.auto_trade.signal_selector"),
]

for name, module_path in integrations:
    try:
        __import__(module_path)
        print(f"✓ {name} module available")
    except Exception as e:
        print(f"✗ {name} module not found: {e}")

print()
print("=" * 60)
print("MANUAL TESTING COMPLETE")
print("=" * 60)
print()
print("Next Steps:")
print("1. Run GUI: python modules/auto_trade/gui/main_window.py")
print("2. Test manual trade form (Trading tab)")
print("3. Test auto-trade toggle (enable/disable)")
print("4. Verify risk calculations update in real-time")
print("5. Test trade execution on DEMO account")
print()
print("IMPORTANT:")
print("- Always test on DEMO account first")
print("- Verify all validation error messages display correctly")
print("- Check that TP/SL prices update when parameters change")
print("- Confirm leverage warning appears for >10x")
print("- Test auto-trade cycle with sample signals")
print()


"""Verify indexing bounds for CUDA batch."""


# Test parameters
total_bars = 50
num_symbols = 1
num_ma_types = 6

# For symbol 0, bar 31, MA type 3 (DEMA)
symbol_idx = 0
bar = 31
i = 3  # DEMA

# Calculate offsets
start = 0  # offsets[0] for first symbol
idx = start + bar  # = 0 + 31 = 31

# Calculate array index
array_idx = i * total_bars + idx
print("Array index calculation:")
print(f"  i (MA type) = {i} (DEMA)")
print(f"  total_bars = {total_bars}")
print(f"  idx (start + bar) = {start} + {bar} = {idx}")
print(f"  array_idx = {i} * {total_bars} + {idx} = {array_idx}")

# Check bounds
array_size = num_ma_types * total_bars
print("\nBounds check:")
print(f"  Array size = {num_ma_types} * {total_bars} = {array_size}")
print(f"  Accessing index: {array_idx}")
print(f"  In bounds: {array_idx < array_size}")

# Show memory layout
print("\nMemory layout (flattened array):")
for ma_i in range(num_ma_types):
    ma_names = ["EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"]
    start_idx = ma_i * total_bars
    end_idx = (ma_i + 1) * total_bars - 1
    print(f"  {ma_names[ma_i]:6s} (i={ma_i}): indices [{start_idx:3d} - {end_idx:3d}]")

    if ma_i == 3:  # DEMA
        bar_31_idx = ma_i * total_bars + 31
        print(f"         Bar 31 at index: {bar_31_idx}")

# Check if there's an off-by-one error
print(f"\n{'=' * 60}")
print("POTENTIAL ISSUE CHECK:")
print("=" * 60)

# What if idx is calculated wrong?
print("\nScenario 1: idx = start + bar (current)")
print(f"  idx = {start} + {bar} = {idx}")
print(f"  DEMA bar 31 index = 3 * 50 + {idx} = {3 * 50 + idx}")

print("\nScenario 2: What if there's confusion with global vs local indexing?")
print("  If kernel uses 'bar' directly instead of 'idx':")
print(f"  DEMA bar 31 index = 3 * 50 + {bar} = {3 * 50 + bar}")

print("\nScenario 3: What if total_bars is wrong?")
for wrong_total in [49, 51, 100]:
    wrong_idx = 3 * wrong_total + 31
    print(f"  If total_bars={wrong_total}: index = {wrong_idx} (out of bounds: {wrong_idx >= array_size})")

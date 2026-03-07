# 🕵️ CUDA Investigation To-Do List (Reviewing D-G)

## 🎯 Objective
Identify the root cause of the specific numerical drift in CUDA signals (0% match rate vs CPU) by investigating algorithmic logic, bindings, and compilation settings.

## 📋 Tasks

### 🔍 Option E: Compare HMA/EMA Implementation (Algorithm Logic)
- [x] **Locate Rust HMA Logic**: Found in `src/batch_processing.rs`.
- [x] **Simulate CUDA HMA in Python**: Created `simulate_cuda_hma.py`.
- [x] **Compare with pandas_ta**: ✅ RESULT: Perfect match (e-14 diff). **HMA algorithm is CORRECT.**
- [x] **Analyze Differences**: Error starts at bar 28, but HMA logic is fine. Pointed to ROC accumulation bug.

### 🔍 Option G: Deep-dive Rust PyO3 Bindings
- [x] **Inspect `lib.rs` / `batch.rs`**: Reviewed `batch_processing.rs`.
- [x] **Check Data Types**: Python casts to `float64` explicitly, Rust expects `f64`. **Correct.**
- [x] **Check Memory Layout**: `as_slice()` handles contiguous memory correctly.

### 🔍 Option F: Strict Compiler Flags inspection
- [ ] **Deprioritized**: Found logic bug, flags unlikely to be primary cause.

### 🔍 Option D: Debug Logs Strategy
- [ ] **Deprioritized**: Logic bug identified by code review.

### 🚨 Option H: Fix Strided Accumulation Bug in ROC Kernel (NEW)
- [ ] **Identify Bug**: `batch_roc_with_growth_kernel` incorrectly accumulates `growth` in strided parallel loop.
- [ ] **Fix Code**: Modify `.cu` file to calculate growth staticaly: `exp(La * i)`.
- [ ] **Verify**: Rebuild and run diagnostic script.

---
**Status Tracking:**
- **Start Date**: 2026-01-31
- **Current Status**: **ROOT CAUSE FOUND** - Fixing now.

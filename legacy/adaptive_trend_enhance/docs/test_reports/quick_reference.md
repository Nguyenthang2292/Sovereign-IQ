# ⚡ Performance Test - Quick Reference

## 🚀 Quick Commands

### Development (Fast - Recommended)
```bash
# Run all performance tests (both files)
pytest tests/adaptive_trend_enhance/test_performance*.py -n 0 -m "not slow"

# Or use script
.\tests\adaptive_trend_enhance\run_perf_tests.ps1 fast all

# Only regression tests
.\tests\adaptive_trend_enhance\run_perf_tests.ps1 fast regression

# Only comparison tests (base vs enhanced)
.\tests\adaptive_trend_enhance\run_perf_tests.ps1 fast comparison
```
⏱️ **Time**: ~10-20s | 💾 **Memory**: ~200MB

---

### Full Test Suite
```bash
# All tests with 5 iterations
PERF_ITERATIONS=5 pytest tests/adaptive_trend_enhance/test_performance*.py -n 0 -m performance

# Or use script
.\tests\adaptive_trend_enhance\run_perf_tests.ps1 full all
```
⏱️ **Time**: ~25-35s | 💾 **Memory**: ~300MB

---

### CI/Production
```bash
# Full suite with coverage
PERF_ITERATIONS=10 pytest tests/adaptive_trend_enhance/test_performance*.py -n 0 -m performance --cov

# Or use script
.\tests\adaptive_trend_enhance\run_perf_tests.ps1 ci all
```
⏱️ **Time**: ~40-50s | 💾 **Memory**: ~400MB

---

## 📂 Test Files

### test_performance_regression.py
- Performance baselines & targets
- Regression detection
- CI metrics export
- **Tests**: 8 tests total

### test_performance.py
- Base vs Enhanced comparison
- Memory leak detection
- Individual version benchmarking
- **Tests**: 4 tests total (includes parametrized)

---

## 🎯 Common Use Cases

### Run specific test file
```bash
# Only regression tests
pytest tests/adaptive_trend_enhance/test_performance_regression.py -n 0 -m "not slow"

# Only comparison tests
pytest tests/adaptive_trend_enhance/test_performance.py -n 0 -m "not slow"
```

### Run specific test class
```bash
pytest tests/adaptive_trend_enhance/test_performance_regression.py::TestPerformanceBaseline -n 0
```

### Run specific test
```bash
pytest tests/adaptive_trend_enhance/test_performance.py::test_performance_comparison -n 0
```

### Using script with suite selector
```bash
# PowerShell
.\tests\adaptive_trend_enhance\run_perf_tests.ps1 fast all         # Both files
.\tests\adaptive_trend_enhance\run_perf_tests.ps1 fast regression  # Regression only
.\tests\adaptive_trend_enhance\run_perf_tests.ps1 fast comparison  # Comparison only

# Windows CMD
.\tests\adaptive_trend_enhance\run_perf_tests.bat fast all
.\tests\adaptive_trend_enhance\run_perf_tests.bat fast regression
.\tests\adaptive_trend_enhance\run_perf_tests.bat fast comparison
```

### Run without slow tests
```bash
pytest -m "performance and not slow" -n 0
```

### Run only slow tests (thorough benchmarking)
```bash
pytest -m "slow" -n 0
```

---

## 🔧 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PERF_ITERATIONS` | 3 | Number of benchmark iterations |

### Examples
```bash
# Super fast (1 iteration - for quick checks only)
PERF_ITERATIONS=1 pytest ... -n 0

# Standard (3 iterations - default)
pytest ... -n 0

# Thorough (10 iterations - CI/Production)
PERF_ITERATIONS=10 pytest ... -n 0
```

---

## 📊 Performance Comparison

| Mode | Time | Memory | Iterations | Use Case |
|------|------|--------|------------|----------|
| **Fast** | 10-15s | 200MB | 3 | Development, quick checks |
| **Full** | 20-25s | 300MB | 5 | Pre-commit, thorough testing |
| **CI** | 30-40s | 400MB | 10 | CI/CD, production validation |

---

## 🏷️ Pytest Markers

### Available Markers
- `@pytest.mark.performance` - All performance tests
- `@pytest.mark.slow` - Slow tests (can be skipped)

### Filter Examples
```bash
# All performance tests
pytest -m performance -n 0

# Fast performance tests only
pytest -m "performance and not slow" -n 0

# Only slow tests
pytest -m slow -n 0

# Exclude slow tests
pytest -m "not slow" -n 0
```

---

## 📁 Important Files

| File | Purpose |
|------|---------|
| `test_performance_regression.py` | Main test file (optimized) |
| `complete_summary.md` | Full documentation with best practices |
| `implementation_summary.md` | Implementation details |
| `run_perf_tests.ps1` | PowerShell quick runner |
| `run_perf_tests.bat` | Windows CMD quick runner |

---

## 💡 Tips & Tricks

### 1. Fast Iteration During Development
```bash
# Run only equity_series (fastest test)
pytest tests/adaptive_trend_enhance/test_performance_regression.py::TestPerformanceBaseline::test_benchmark_equity_series -n 0
```

### 2. Check What Will Run
```bash
# Collect tests without running
pytest tests/adaptive_trend_enhance/test_performance_regression.py -n 0 --collect-only
```

### 3. Verbose Output
```bash
# See detailed timing
pytest ... -n 0 -v
```

### 4. Very Verbose Output
```bash
# See setup/teardown details
pytest ... -n 0 -vv --setup-show
```

### 5. Monitor Memory
```bash
# With memory profiling
pytest ... -n 0 --memory-profile
```

---

## 🐛 Troubleshooting

### Tests running slow?
1. ✅ Check using `-n 0` (single thread)
2. ✅ Check `PERF_ITERATIONS` value
3. ✅ Use `-m "not slow"` to skip slow tests

### Out of memory?
1. ✅ Use fast mode with 3 iterations
2. ✅ Run one test at a time
3. ✅ Check memory usage: `pytest ... --memory-profile`

### Inconsistent results?
1. ✅ Close other applications
2. ✅ Use more iterations: `PERF_ITERATIONS=10`
3. ✅ Check garbage collection is working

---

## 📈 Expected Results

### Development Mode (Fast)
```
======================== test session starts =========================
collected 6 items (2 skipped)

test_performance_regression.py::TestPerformanceBaseline::test_benchmark_equity_series PASSED
test_performance_regression.py::TestPerformanceTargets::test_set_target_metrics PASSED
...

===================== 4 passed, 2 skipped in 10.23s =================
```

### CI Mode (Full)
```
======================== test session starts =========================
collected 8 items

test_performance_regression.py::TestPerformanceBaseline::test_benchmark_compute_atc_signals PASSED
test_performance_regression.py::TestPerformanceBaseline::test_benchmark_equity_series PASSED
...

===================== 8 passed in 35.47s ============================
```

---

## 🎯 Common Workflows

### Before Commit
```bash
# Quick sanity check (10-15s)
pytest tests/adaptive_trend_enhance/test_performance_regression.py -n 0 -m "not slow"

# If passed, run full suite (20-25s)
PERF_ITERATIONS=5 pytest tests/adaptive_trend_enhance/test_performance_regression.py -n 0 -m performance
```

### Daily Development
```bash
# Fast iteration
pytest tests/adaptive_trend_enhance/test_performance_regression.py -n 0 -m "not slow" -v
```

### Weekly/CI Testing
```bash
# Full comprehensive test with coverage
PERF_ITERATIONS=10 pytest tests/adaptive_trend_enhance/test_performance_regression.py -n 0 -m performance --cov=modules.adaptive_trend_enhance --cov-report=html
```

---

## 📞 Need Help?

1. 📖 Read [complete_summary.md](complete_summary.md)
2. 📝 Check [implementation_summary.md](implementation_summary.md)
3. 🔍 Run verification tests (see guide)

---

**Last Updated**: 2026-01-22
**Version**: 1.0 (Optimized)

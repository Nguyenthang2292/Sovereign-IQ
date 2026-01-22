# 🧪 Testing Guide - Sovereign IQ

## 📋 Mục lục

1. [Thiết lập môi trường test](#thiết-lập-môi-trường-test)
2. [Chạy tests](#chạy-tests)
3. [Các loại tests](#các-loại-tests)
4. [Best Practices](#best-practices)

---

## 🔧 Thiết lập môi trường test

### 1. Kích hoạt Virtual Environment (venv)

**PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Command Prompt:**

```cmd
.venv\Scripts\activate.bat
```

### 2. Cài đặt dependencies cho testing

```bash
pip install -r requirements-dev.txt
```

### 3. Kiểm tra pytest đã được cài đặt

```bash
python -m pytest --version
```

---

## 🚀 Chạy tests

### Cách 1: Sử dụng script tự động (Khuyến nghị)

**PowerShell:**

```powershell
.\run_tests.ps1
```

**Command Prompt:**

```cmd
run_tests.bat
```

Script này sẽ:

- ✅ Tự động kích hoạt venv
- ✅ Kiểm tra pytest đã cài đặt
- ✅ Chạy tests với cấu hình tối ưu
- ✅ Hiển thị kết quả với màu sắc

### Cách 2: Chạy trực tiếp với pytest

**Sau khi đã activate venv:**

```bash
# Chạy tất cả tests
pytest

# Chạy tests trong một thư mục cụ thể
pytest tests/adaptive_trend_enhance/

# Chạy một file test cụ thể
pytest tests/adaptive_trend_enhance/test_gpu_logic.py

# Chạy một test function cụ thể
pytest tests/adaptive_trend_enhance/test_gpu_logic.py::test_specific_function

# Chạy với verbose output
pytest -v

# Chạy với coverage report
pytest --cov=modules --cov-report=html

# Chạy parallel với nhiều workers
pytest -n 4

# Chạy chỉ tests đã fail trước đó
pytest --lf

# Chạy tests theo marker
pytest -m "not slow"  # Bỏ qua slow tests
pytest -m "unit"      # Chỉ chạy unit tests
pytest -m "gpu"       # Chỉ chạy GPU tests
```

### Cách 3: Sử dụng VS Code Testing UI

1. Mở VS Code
2. Click vào icon Testing ở sidebar (🧪)
3. VS Code sẽ tự động discover tests
4. Click vào ▶️ để chạy tests

**Lưu ý:** VS Code sẽ tự động sử dụng venv nhờ cấu hình trong `.vscode/settings.json`

---

## 📊 Các loại tests

### Unit Tests

```bash
pytest -m unit
```

- Tests các function/class riêng lẻ
- Nhanh, không phụ thuộc external services
- Nên chiếm 70-80% tổng số tests

### Integration Tests

```bash
pytest -m integration
```

- Tests tích hợp giữa các modules
- Có thể chậm hơn unit tests
- Test workflow hoàn chỉnh

### Performance Tests

```bash
pytest -m performance
```

- Đo lường hiệu suất
- Benchmark các operations
- Kiểm tra memory usage

### GPU Tests

```bash
pytest -m gpu
```

- Tests yêu cầu CUDA/GPU
- Tự động skip nếu không có GPU

### Memory Intensive Tests

```bash
pytest -m memory_intensive
```

- Tests sử dụng nhiều RAM
- Có thể chạy riêng để tránh OOM

### Slow Tests

```bash
# Chạy tất cả tests bao gồm slow tests
pytest

# Bỏ qua slow tests
pytest -m "not slow"
```

---

## 📝 Best Practices

### 1. Cấu trúc Test File

```python
"""
Test module for [component name]
"""
import pytest
from modules.your_module import YourClass


class TestYourClass:
    """Test suite for YourClass"""

    @pytest.fixture
    def sample_data(self):
        """Fixture providing sample test data"""
        return {"key": "value"}

    def test_basic_functionality(self, sample_data):
        """Test basic functionality"""
        result = YourClass().process(sample_data)
        assert result is not None

    @pytest.mark.slow
    def test_slow_operation(self):
        """Test that takes a long time"""
        # ... slow test code
        pass

    @pytest.mark.gpu
    def test_gpu_operation(self):
        """Test requiring GPU"""
        # ... GPU test code
        pass
```

### 2. Sử dụng Fixtures

```python
@pytest.fixture(scope="session")
def shared_resource():
    """Fixture shared across all tests in session"""
    resource = expensive_setup()
    yield resource
    resource.cleanup()

@pytest.fixture(scope="function")
def fresh_data():
    """Fixture created for each test function"""
    return create_test_data()
```

### 3. Parametrize Tests

```python
@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_double(input, expected):
    assert double(input) == expected
```

### 4. Skip và XFail

```python
@pytest.mark.skip(reason="Not implemented yet")
def test_future_feature():
    pass

@pytest.mark.skipif(not has_gpu(), reason="Requires GPU")
def test_gpu_feature():
    pass

@pytest.mark.xfail(reason="Known bug #123")
def test_known_issue():
    pass
```

### 5. Memory Profiling

```bash
# Chạy với memory profiling
pytest --memory-profile --memory-threshold=0.5

# Xem chi tiết memory usage
pytest --memory-profile --memory-threshold=0.1 -v
```

---

## 🎯 Markers Reference

| Marker                          | Mô tả            | Cách sử dụng                 |
| ------------------------------- | ---------------- | ---------------------------- |
| `@pytest.mark.unit`             | Unit test        | `pytest -m unit`             |
| `@pytest.mark.integration`      | Integration test | `pytest -m integration`      |
| `@pytest.mark.slow`             | Test chậm        | `pytest -m "not slow"`       |
| `@pytest.mark.gpu`              | Cần GPU          | `pytest -m gpu`              |
| `@pytest.mark.memory_intensive` | Dùng nhiều RAM   | `pytest -m memory_intensive` |
| `@pytest.mark.performance`      | Performance test | `pytest -m performance`      |

---

## 🔍 Debugging Tests

### 1. Chạy với pdb

```bash
pytest --pdb  # Drop vào debugger khi fail
pytest -x --pdb  # Stop at first failure và debug
```

### 2. Print output

```bash
pytest -s  # Hiển thị print statements
pytest -v -s  # Verbose + print output
```

### 3. Chỉ chạy failed tests

```bash
pytest --lf  # Last failed
pytest --ff  # Failed first, then others
```

---

## 📈 Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=modules --cov-report=html

# Open report
start htmlcov/index.html  # Windows
```

---

## ⚙️ Configuration Files

- **`pytest.ini`**: Cấu hình chính của pytest
- **`conftest.py`**: Shared fixtures và hooks
- **`.vscode/settings.json`**: VS Code testing configuration
- **`pyproject.toml`**: Project metadata và tool configs

---

## 🆘 Troubleshooting

### Vấn đề: "Module not found"

**Giải pháp:**

```bash
# Đảm bảo PYTHONPATH được set
set PYTHONPATH=.
pytest
```

### Vấn đề: "pytest not found"

**Giải pháp:**

```bash
# Kích hoạt venv trước
.\.venv\Scripts\Activate.ps1
python -m pip install pytest
```

### Vấn đề: Tests chạy chậm

**Giải pháp:**

```bash
# Chạy parallel
pytest -n auto

# Bỏ qua slow tests
pytest -m "not slow"

# Chỉ chạy failed tests
pytest --lf
```

### Vấn đề: Out of memory

**Giải pháp:**

```bash
# Chạy sequential thay vì parallel
pytest -n 0

# Bỏ qua memory intensive tests
pytest -m "not memory_intensive"
```

---

## 📚 Tài liệu tham khảo

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)
- [Testing Python Applications](https://realpython.com/pytest-python-testing/)

---

**Cập nhật lần cuối:** 2026-01-22

# Auto Trade System - Test Suite

Comprehensive test suite for the auto trading system.

## 📋 Test Structure

```
tests/auto_trade/
├── __init__.py
├── test_database.py        # Database operations tests
├── test_order_tagging.py   # Order tagging system tests
├── test_config.py          # Configuration management tests
└── README.md               # This file
```

## 🚀 Running Tests

### Run All Tests
```bash
pytest tests/auto_trade/ -v
```

### Run Specific Test File
```bash
pytest tests/auto_trade/test_database.py -v
pytest tests/auto_trade/test_order_tagging.py -v
pytest tests/auto_trade/test_config.py -v
```

### Run Specific Test Class
```bash
pytest tests/auto_trade/test_database.py::TestOrderOperations -v
```

### Run Specific Test Method
```bash
pytest tests/auto_trade/test_database.py::TestOrderOperations::test_create_order -v
```

### Run with Coverage
```bash
pytest tests/auto_trade/ --cov=modules.auto_trade --cov-report=html
```

Coverage report will be in `htmlcov/index.html`

## 📊 Test Categories

### Database Tests (`test_database.py`)
- ✅ Database initialization
- ✅ Order CRUD operations
- ✅ Signal operations
- ✅ Statistics calculations
- ✅ Programmatic order filtering
- ✅ Data validation

**Coverage**: Order creation, updates, queries, statistics

### Order Tagging Tests (`test_order_tagging.py`)
- ✅ Client order ID generation
- ✅ ID parsing and validation
- ✅ Order identification
- ✅ Metadata creation
- ✅ Batch operations
- ✅ Statistics

**Coverage**: Unique ID generation, metadata tagging, validation

### Configuration Tests (`test_config.py`)
- ✅ Config creation and validation
- ✅ Export/Import (JSON)
- ✅ Preset configurations
- ✅ Parameter modification
- ✅ Sub-config classes

**Coverage**: Configuration management, validation rules

## 🎯 Test Markers

Tests are organized with pytest markers:

```bash
# Run only unit tests
pytest -m unit

# Run only database tests
pytest -m database

# Run only configuration tests  
pytest -m config

# Run only quick tests (exclude slow)
pytest -m "not slow"
```

## ✅ Expected Results

All tests should pass with **100% success rate** when:
- Database is properly initialized
- All required modules are available
- Test environment is clean (temp databases)

## 📝 Writing New Tests

### Test Structure Template

```python
import pytest
from modules.auto_trade.your_module import YourClass

class TestYourFeature:
    """Test your feature description."""
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        result = YourClass.method()
        assert result is not None
    
    def test_error_handling(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            YourClass.invalid_method()
```

### Using Fixtures

```python
@pytest.fixture
def sample_data():
    """Provide sample test data."""
    return {'key': 'value'}

def test_with_fixture(sample_data):
    """Test using fixture data."""
    assert sample_data['key'] == 'value'
```

## 🔧 Test Utilities

### Temporary Database Fixture

```python
@pytest.fixture
def test_db(tmp_path):
    """Create temporary test database."""
    db_path = tmp_path / "test.db"
    initialize_database(str(db_path))
    yield str(db_path)
```

### Sample Order Data

```python
@pytest.fixture
def sample_order():
    """Provide sample order data."""
    return {
        'order_id': 'TEST_001',
        'symbol': 'BTCUSDT',
        'side': 'LONG',
        ...
    }
```

## 📈 Code Coverage Goals

| Module | Target Coverage |
|--------|----------------|
| Database | > 90% |
| Order Tagging | > 95% |
| Configuration | > 85% |
| Main Loop | > 70% |

## 🐛 Debugging Failed Tests

### View Test Output
```bash
pytest tests/auto_trade/test_database.py -v -s
```

### Run Single Failing Test
```bash
pytest tests/auto_trade/test_database.py::TestOrderOperations::test_create_order -v -s
```

### Show Traceback
```bash
pytest tests/auto_trade/ -v --tb=long
```

### Drop into Debugger on Failure
```bash
pytest tests/auto_trade/ --pdb
```

## 🔄 Continuous Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# GitHub Actions example
- name: Run Tests
  run: pytest tests/auto_trade/ -v --cov=modules.auto_trade
```

## ⚠️ Important Notes

1. **Isolation**: Tests use temporary databases to avoid conflicts
2. **Cleanup**: Fixtures handle cleanup automatically
3. **Independence**: Each test should be independent
4. **Speed**: Unit tests should complete in < 1 second each

## 📚 Next Steps

After unit tests pass:
1. Integration tests (Phase 6.4)
2. End-to-end tests (Phase 6.4)
3. Performance benchmarks (Phase 6.6)
4. Load testing (Phase 6.6)

---

**Created**: 2026-02-03  
**Status**: ✅ Active  
**Coverage Target**: > 85%

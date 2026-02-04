# GUI Utils Tests

This directory contains comprehensive unit tests for the `modules/auto_trade/gui/utils` package.

## Test Coverage

All utility modules are fully covered with unit tests:

### Core Utilities
- **test_credential_manager.py**: Secure credential storage, environment management, connection testing
- **test_data_service.py**: Unified data layer, mode-based operations, exchange integration
- **test_dry_run_db.py**: SQLite database operations, position management, data persistence
- **test_dry_run_executor.py**: Simulated order execution, TP/SL management, PnL calculations
- **test_settings_manager.py**: Settings persistence, validation, import/export, migration

### Supporting Utilities
- **test_utils_comprehensive.py**: All remaining utilities in one file
  - MockPriceFeed: Price simulation and updates
  - Formatters: Price, PnL, percent, timestamp formatting
  - RetryUtils: Exponential backoff and retry logic
  - ThreadingUtils: Periodic updates and background tasks
  - Toast: Notification display system
  - Modes: Trading mode constants
  - RiskCalculator: Risk metrics and calculations
  - Colors: Theme-aware color system

## Running Tests

### Run all GUI utils tests:
```bash
pytest tests/auto_trade/gui/utils/ -v
```

### Run specific test file:
```bash
pytest tests/auto_trade/gui/utils/test_credential_manager.py -v
```

### Run with coverage:
```bash
pytest tests/auto_trade/gui/utils/ --cov=modules.auto_trade.gui.utils --cov-report=html
```

### Run specific test class:
```bash
pytest tests/auto_trade/gui/utils/test_data_service.py::TestDataService -v
```

### Run specific test method:
```bash
pytest tests/auto_trade/gui/utils/test_data_service.py::TestDataService::test_init_dry_run_mode -v
```

## Test Fixtures

The `conftest.py` file provides shared fixtures:

- `temp_env_file`: Temporary .env file for credential tests
- `temp_db_file`: Temporary database file for dry run tests
- `temp_settings_file`: Temporary settings file for settings tests
- `mock_env_vars`: Mocked environment variables
- `mock_exchange_manager`: Mocked ExchangeManager
- `mock_data_fetcher`: Mocked DataFetcher
- `mock_database_manager`: Mocked DatabaseManager
- `sample_position_data`: Sample position data
- `sample_settings`: Sample settings configuration

## Test Structure

Each test file follows this structure:

1. **Import statements**: All necessary imports at the top
2. **Test class**: One class per module being tested
3. **Test methods**: Individual tests for each function/method
4. **Fixtures**: Used from conftest.py or defined locally
5. **Assertions**: Clear, specific assertions for each test case

## Coverage Goals

- **Line Coverage**: >95% for all utils modules
- **Branch Coverage**: >90% for conditional logic
- **Edge Cases**: All error paths and edge cases tested
- **Integration**: Tests verify integration between components

## Best Practices

1. **Isolation**: Each test is independent and can run in any order
2. **Mocking**: External dependencies are mocked (databases, APIs, etc.)
3. **Descriptive Names**: Test names clearly describe what is being tested
4. **AAA Pattern**: Arrange, Act, Assert structure in each test
5. **Error Testing**: Both success and failure cases are tested

## Adding New Tests

When adding new utility modules:

1. Create a new test file: `test_<module_name>.py`
2. Add necessary fixtures to `conftest.py` if shared
3. Write comprehensive tests covering:
   - Happy path scenarios
   - Error conditions
   - Edge cases
   - Integration with other modules
4. Update this README with the new test coverage

## Known Issues

None currently. All tests passing.

## Dependencies

Test dependencies (from requirements-dev.txt):
- pytest
- pytest-cov
- pytest-mock
- unittest.mock (built-in)

## Continuous Integration

These tests are run automatically on:
- Every commit (pre-commit hook)
- Pull requests
- Daily CI/CD pipeline

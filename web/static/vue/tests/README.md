# Vue Frontend Tests

Test suite cho Vue frontend của Gemini Chart Analyzer.

## Cấu trúc Test

```
tests/
├── setup.js                    # Test setup và mocks
├── services/
│   ├── api.test.js             # Tests cho API service
│   └── logPoller.test.js       # Tests cho LogPoller service
├── components/
│   ├── App.test.js             # Tests cho App component
│   ├── BatchScanner.test.js    # Tests cho BatchScanner component
│   ├── ChartAnalyzer.test.js   # Tests cho ChartAnalyzer component
│   ├── LoadingSpinner.test.js  # Tests cho LoadingSpinner component
│   └── ResultsTable.test.js     # Tests cho ResultsTable component
└── router/
    └── index.test.js           # Tests cho Vue Router
```

## Chạy Tests

```bash
# Chạy tất cả tests
npm test

# Chạy tests với UI
npm run test:ui

# Chạy tests với coverage
npm run test:coverage

# Chạy tests trong watch mode
npm test -- --watch
```

## Test Coverage

Tests cover các phần sau:

### Services
- ✅ API service (chartAnalyzerAPI, batchScannerAPI, logsAPI, chartAnalyzerStatusAPI)
- ✅ Error handling trong API calls
- ✅ LogPoller service (polling, status updates, completion handling)

### Components
- ✅ BatchScanner (form validation, API calls, log polling, error handling)
- ✅ ChartAnalyzer (single/multi timeframe, form validation, API calls, results display)
- ✅ ResultsTable (filtering, sorting, pagination, symbol click events)
- ✅ LoadingSpinner (rendering với/không có message)
- ✅ App (navigation, routing, symbol click handling)

### Router
- ✅ Route definitions
- ✅ Navigation
- ✅ Query parameters

## Dependencies

- `vitest`: Test framework
- `@vue/test-utils`: Vue component testing utilities
- `jsdom`: DOM environment cho tests


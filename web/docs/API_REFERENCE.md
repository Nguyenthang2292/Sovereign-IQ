# API Reference

Complete API documentation for all web applications.

## Table of Contents

- [Gemini Analyzer API](#gemini-analyzer-api)
  - [Chart Analyzer Endpoints](#chart-analyzer-endpoints)
  - [Batch Scanner Endpoints](#batch-scanner-endpoints)
  - [Logs Endpoints](#logs-endpoints)
- [ATC Visualizer API](#atc-visualizer-api)
- [Shared Utilities](#shared-utilities)
- [Common Patterns](#common-patterns)
- [Error Handling](#error-handling)

---

## Gemini Analyzer API

**Base URL:** `http://localhost:8001`

**API Prefix:** `/api`

**Documentation:** http://localhost:8001/docs

### Chart Analyzer Endpoints

#### POST /api/analyze/single

Analyze a single symbol on a single timeframe.

**Request:**
```json
{
  "symbol": "BTC/USDT",
  "timeframe": "4h",
  "indicators": {
    "ma_periods": [20, 50, 200],
    "rsi_period": 14,
    "enable_macd": true,
    "enable_bb": false,
    "bb_period": 20
  },
  "prompt_type": "detailed",
  "custom_prompt": null,
  "limit": 500,
  "chart_figsize": [16, 10],
  "chart_dpi": 150,
  "no_cleanup": false
}
```

**Parameters:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| symbol | string | ✅ | Trading symbol (e.g., BTC/USDT) |
| timeframe | string | ✅ | Timeframe (e.g., 1m, 5m, 15m, 1h, 4h, 1d) |
| indicators | object | ❌ | Indicators configuration |
| indicators.ma_periods | array[int] | ❌ | Moving Average periods (default: [20, 50, 200]) |
| indicators.rsi_period | int | ❌ | RSI period (default: 14) |
| indicators.enable_macd | bool | ❌ | Enable MACD (default: true) |
| indicators.enable_bb | bool | ❌ | Enable Bollinger Bands (default: false) |
| indicators.bb_period | int | ❌ | Bollinger Bands period (default: 20) |
| prompt_type | string | ❌ | Prompt type: detailed, simple, or custom (default: detailed) |
| custom_prompt | string | ❌ | Custom prompt (if prompt_type is custom) |
| limit | int | ❌ | Number of candles (default: 500) |
| chart_figsize | array[int] | ❌ | Chart figure size [width, height] (default: [16, 10]) |
| chart_dpi | int | ❌ | Chart DPI (default: 150) |
| no_cleanup | bool | ❌ | Don't cleanup old charts (default: false) |

**Response:**
```json
{
  "success": true,
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "message": "Analysis started. Use GET /api/analyze/{session_id}/status to check progress."
}
```

**Status Flow:**
1. `running` - Analysis in progress
2. `completed` - Analysis finished successfully
3. `error` - Analysis failed
4. `cancelled` - Analysis cancelled by user

---

#### POST /api/analyze/multi

Analyze a single symbol across multiple timeframes.

**Request:**
```json
{
  "symbol": "BTC/USDT",
  "timeframes": ["15m", "1h", "4h", "1d"],
  "indicators": {
    "ma_periods": [20, 50, 200],
    "rsi_period": 14,
    "enable_macd": true
  },
  "prompt_type": "detailed",
  "limit": 500,
  "chart_figsize": [16, 10],
  "chart_dpi": 150,
  "no_cleanup": false
}
```

**Parameters:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| symbol | string | ✅ | Trading symbol |
| timeframes | array[string] | ✅ | List of timeframes (e.g., ['15m', '1h', '4h', '1d']) |
| indicators | object | ❌ | Same as single analysis |
| prompt_type | string | ❌ | Same as single analysis |
| custom_prompt | string | ❌ | Same as single analysis |
| limit | int | ❌ | Same as single analysis |
| chart_figsize | array[int] | ❌ | Same as single analysis |
| chart_dpi | int | ❌ | Same as single analysis |
| no_cleanup | bool | ❌ | Same as single analysis |

**Response:**
```json
{
  "success": true,
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "message": "Multi-timeframe analysis started. Use GET /api/analyze/{session_id}/status to check progress."
}
```

---

#### GET /api/analyze/{session_id}/status

Get status of an analysis task.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| session_id | string | ✅ | Session ID from analyze_single or analyze_multi |

**Response (running):**
```json
{
  "success": true,
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "started_at": "2026-01-16T00:00:00Z",
  "completed_at": null
}
```

**Response (completed - single):**
```json
{
  "success": true,
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "started_at": "2026-01-16T00:00:00Z",
  "completed_at": "2026-01-16T00:00:30Z",
  "result": {
    "success": true,
    "symbol": "BTC/USDT",
    "timeframe": "4h",
    "analysis": "...",
    "chart_path": "/path/to/chart.png",
    "chart_url": "/static/charts/BTC_USDT_4h.png",
    "signal": "LONG",
    "confidence": 0.7,
    "exchange": "binance"
  }
}
```

**Response (completed - multi):**
```json
{
  "success": true,
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "started_at": "2026-01-16T00:00:00Z",
  "completed_at": "2026-01-16T00:02:00Z",
  "result": {
    "success": true,
    "symbol": "BTC/USDT",
    "timeframes": ["15m", "1h", "4h", "1d"],
    "timeframes_results": {
      "15m": {
        "chart_path": "/path/to/chart_15m.png",
        "chart_url": "/static/charts/BTC_USDT_15m.png",
        "signal": "LONG",
        "confidence": 0.65,
        "analysis": "..."
      },
      "1h": { ... },
      "4h": { ... },
      "1d": { ... }
    },
    "aggregated": {
      "signal": "LONG",
      "confidence": 0.72,
      "weights_used": {
        "15m": 0.1,
        "1h": 0.2,
        "4h": 0.3,
        "1d": 0.4
      }
    }
  }
}
```

**Response (error):**
```json
{
  "success": true,
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "error",
  "started_at": "2026-01-16T00:00:00Z",
  "completed_at": "2026-01-16T00:00:05Z",
  "error": "Error message here"
}
```

---

#### POST /api/analyze/{session_id}/cancel

Cancel a running analysis task.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| session_id | string | ✅ | Session ID from analyze_single or analyze_multi |

**Response:**
```json
{
  "success": true,
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "cancelled",
  "message": "Analysis cancelled successfully. The task will stop processing after current operation."
}
```

**Errors:**
- `404` - Session not found
- `400` - Cannot cancel task with current status (only running tasks can be cancelled)

---

### Batch Scanner Endpoints

#### POST /api/batch/scan

Scan entire market and return LONG/SHORT signals.

**Request (single timeframe mode):**
```json
{
  "timeframe": "4h",
  "timeframes": null,
  "max_symbols": 50,
  "limit": 500,
  "cooldown": 2.5,
  "charts_per_batch": 100,
  "quote_currency": "USDT",
  "exchange_name": "binance"
}
```

**Request (multi-timeframe mode):**
```json
{
  "timeframe": null,
  "timeframes": ["15m", "1h", "4h"],
  "max_symbols": 50,
  "limit": 500,
  "cooldown": 2.5,
  "charts_per_batch": 100,
  "quote_currency": "USDT",
  "exchange_name": "binance"
}
```

**Parameters:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| timeframe | string | ❌* | Single timeframe (e.g., 1h, 4h, 1d) |
| timeframes | array[string] | ❌* | List of timeframes for multi-timeframe mode |
| max_symbols | int | ❌ | Maximum symbols to scan (null = all, min: 1) |
| limit | int | ❌ | Candles per symbol (default: 500, range: 1-5000) |
| cooldown | float | ❌ | Cooldown between batches in seconds (default: 2.5, range: 0-60) |
| charts_per_batch | int | ❌ | Charts per batch (default: 100, range: 1-1000) |
| quote_currency | string | ❌ | Quote currency filter (default: "USDT") |
| exchange_name | string | ❌ | Exchange name (default: "binance") |

*At least one of `timeframe` or `timeframes` must be provided.

**Response:**
```json
{
  "success": true,
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "message": "Scan started. Use GET /api/batch/scan/{session_id}/status to check progress."
}
```

---

#### GET /api/batch/scan/{session_id}/status

Get status of a batch scan task.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| session_id | string | ✅ | Session ID from batch_scan |

**Response (running):**
```json
{
  "success": true,
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "started_at": "2026-01-16T00:00:00Z",
  "completed_at": null
}
```

**Response (completed):**
```json
{
  "success": true,
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "started_at": "2026-01-16T00:00:00Z",
  "completed_at": "2026-01-16T00:10:00Z",
  "result": {
    "success": true,
    "mode": "single-timeframe",
    "timeframe": "4h",
    "timeframes": null,
    "summary": {
      "total_symbols": 50,
      "long_count": 15,
      "short_count": 8,
      "none_count": 27
    },
    "long_symbols": ["BTC/USDT", "ETH/USDT", ...],
    "short_symbols": ["XRP/USDT", "ADA/USDT", ...],
    "long_symbols_with_confidence": [
      {"symbol": "BTC/USDT", "confidence": 0.75},
      {"symbol": "ETH/USDT", "confidence": 0.70},
      ...
    ],
    "short_symbols_with_confidence": [
      {"symbol": "XRP/USDT", "confidence": 0.68},
      {"symbol": "ADA/USDT", "confidence": 0.65},
      ...
    ],
    "all_results": { ... },
    "results_file": "/path/to/results.json",
    "results_url": "/static/results/batch_scan/results_20260116_000000.json"
  }
}
```

**Response (cancelled):**
```json
{
  "success": true,
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "cancelled",
  "started_at": "2026-01-16T00:00:00Z",
  "completed_at": "2026-01-16T00:05:00Z",
  "message": "Scan was cancelled by user"
}
```

---

#### POST /api/batch/scan/{session_id}/cancel

Cancel a running batch scan task.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| session_id | string | ✅ | Session ID from batch_scan |

**Response:**
```json
{
  "success": true,
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "cancelled",
  "message": "Scan cancelled successfully. The task will stop processing after current batch."
}
```

---

#### GET /api/batch/results/{filename}

Get saved batch scan results by filename.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| filename | string | ✅ | Name of the results JSON file |

**Example URL:** `/api/batch/results/results_20260116_000000.json`

**Response:**
```json
{
  "success": true,
  "filename": "results_20260116_000000.json",
  "results": {
    "mode": "single-timeframe",
    "timeframe": "4h",
    "summary": { ... },
    "long_symbols": [ ... ],
    "short_symbols": [ ... ],
    "all_results": { ... }
  }
}
```

**Security:**
- Directory traversal is prevented
- Only JSON files are allowed
- File must exist in batch_scan directory

**Errors:**
- `400` - Directory traversal detected or invalid file type
- `404` - File not found
- `500` - Server error

---

#### GET /api/batch/list

List all available batch scan results.

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| skip | int | ❌ | 0 | Number of results to skip (pagination) |
| limit | int | ❌ | 50 | Max results to return (range: 1-200) |
| metadata_only | bool | ❌ | false | Skip reading JSON files for better performance |

**Example URL:** `/api/batch/list?skip=0&limit=20&metadata_only=false`

**Response:**
```json
{
  "success": true,
  "count": 20,
  "results": [
    {
      "filename": "results_20260116_000000.json",
      "size": 524288,
      "modified": "2026-01-16T00:00:00Z",
      "summary": {
        "total_symbols": 50,
        "long_count": 15,
        "short_count": 8,
        "none_count": 27
      },
      "url": "/api/batch/results/results_20260116_000000.json"
    },
    ...
  ]
}
```

**Notes:**
- Results are sorted by modified time (newest first)
- Use `metadata_only=true` for faster listing without reading file contents

---

### Logs Endpoints

#### GET /api/logs/list

List available log files.

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| session_id | string | ❌ | null | Filter by session ID |
| log_type | string | ❌ | null | Filter by log type (analyze, scan) |
| limit | int | ❌ | 50 | Max results to return (range: 1-200) |

**Example URL:** `/api/logs/list?session_id=xxx&log_type=analyze&limit=20`

**Response:**
```json
{
  "success": true,
  "count": 10,
  "logs": [
    {
      "session_id": "550e8400-e29b-41d4-a716-446655440000",
      "log_type": "analyze",
      "filename": "analyze_550e8400-e29b-41d4-a716-446655440000.log",
      "created_at": "2026-01-16T00:00:00Z",
      "size": 1024
    },
    ...
  ]
}
```

---

#### GET /api/logs/stream/{session_id}

Stream logs for a specific session in real-time.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| session_id | string | ✅ | Session ID |

**Response (Server-Sent Events):**
```
data: [INFO] Starting analysis...
data: [INFO] Fetching OHLCV data...
data: [INFO] Generating chart...
data: [INFO] Analyzing with Gemini...
data: [INFO] Analysis complete!
```

**Usage (Frontend):**
```javascript
const eventSource = new EventSource(`/api/logs/stream/${sessionId}`);

eventSource.onmessage = (event) => {
  const logLine = event.data;
  appendToLog(logLine);
};

eventSource.onerror = () => {
  eventSource.close();
};
```

---

#### GET /api/logs/download/{session_id}

Download log file for a session.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| session_id | string | ✅ | Session ID |

**Response:**
- `Content-Type: text/plain`
- `Content-Disposition: attachment; filename="analyze_{session_id}.log"`

---

---

## ATC Visualizer API

**Base URL:** `http://localhost:8002`

**API Prefix:** `/api`

**Documentation:** http://localhost:8002/docs

### Chart Data Endpoints

#### GET /api/health

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "service": "ATC Visualizer API"
}
```

---

#### GET /api/symbols

List available trading symbols from Binance.

**Response:**
```json
{
  "success": true,
  "symbols": [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    ...
  ]
}
```

**Note:** Returns top 50 USDT-M futures symbols by volume.

---

#### GET /api/ohlcv

Fetch OHLCV data for a symbol.

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| symbol | string | ❌ | BTC/USDT | Trading symbol |
| timeframe | string | ❌ | 15m | Timeframe (e.g., 15m, 1h, 1d) |
| limit | int | ❌ | 1500 | Number of candles (range: 100-5000) |

**Example URL:** `/api/ohlcv?symbol=BTC/USDT&timeframe=15m&limit=1500`

**Response:**
```json
{
  "success": true,
  "data": {
    "symbol": "BTC/USDT",
    "exchange": "binance",
    "timeframe": "15m",
    "data": [
      {
        "x": 1640995200000,
        "y": [47000.0, 47500.0, 46800.0, 47200.0]
      },
      ...
    ]
  }
}
```

**Note:** Each data point is `[open, high, low, close]` in ApexCharts candlestick format.

---

#### GET /api/atc-signals

Compute ATC signals for a symbol.

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| symbol | string | ❌ | BTC/USDT | Trading symbol |
| timeframe | string | ❌ | 15m | Timeframe |
| limit | int | ❌ | 1500 | Number of candles (range: 100-5000) |
| ema_len | int | ❌ | 28 | EMA length (min: 1) |
| hma_len | int | ❌ | 28 | HMA length (min: 1) |
| wma_len | int | ❌ | 28 | WMA length (min: 1) |
| dema_len | int | ❌ | 28 | DEMA length (min: 1) |
| lsma_len | int | ❌ | 28 | LSMA length (min: 1) |
| kama_len | int | ❌ | 28 | KAMA length (min: 1) |
| robustness | string | ❌ | Medium | Robustness: Narrow, Medium, Wide |
| lambda_param | float | ❌ | 0.02 | Lambda parameter (range: 0.0-1.0) |
| decay | float | ❌ | 0.03 | Decay parameter (range: 0.0-1.0) |
| cutout | int | ❌ | 0 | Cutout parameter (min: 0) |

**Example URL:** `/api/atc-signals?symbol=BTC/USDT&timeframe=15m&limit=1500&ema_len=28&robustness=Medium`

**Response:**
```json
{
  "success": true,
  "data": {
    "symbol": "BTC/USDT",
    "exchange": "binance",
    "timeframe": "15m",
    "current_price": 47200.0,
    "ohlcv": [
      {
        "x": 1640995200000,
        "y": [47000.0, 47500.0, 46800.0, 47200.0]
      },
      ...
    ],
    "moving_averages": { ... },
    "signals": {
      "Average_Signal": [
        {"x": 1640995200000, "y": 1.0},
        {"x": 1640996100000, "y": 0.0},
        {"x": 1640997000000, "y": -1.0}
      ],
      "EMA_Signal": [ ... ],
      "HMA_Signal": [ ... ],
      ...
    }
  }
}
```

---

#### GET /api/moving-averages

Get all Moving Averages for a symbol.

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| symbol | string | ❌ | BTC/USDT | Trading symbol |
| timeframe | string | ❌ | 15m | Timeframe |
| limit | int | ❌ | 1500 | Number of candles (range: 100-5000) |
| ema_len | int | ❌ | 28 | EMA length (min: 1) |
| hma_len | int | ❌ | 28 | HMA length (min: 1) |
| wma_len | int | ❌ | 28 | WMA length (min: 1) |
| dema_len | int | ❌ | 28 | DEMA length (min: 1) |
| lsma_len | int | ❌ | 28 | LSMA length (min: 1) |
| kama_len | int | ❌ | 28 | KAMA length (min: 1) |
| robustness | string | ❌ | Medium | Robustness: Narrow, Medium, Wide |

**Response:**
```json
{
  "success": true,
  "data": {
    "EMA_MA": [
      {"x": 1640995200000, "y": 47000.0},
      ...
    ],
    "EMA_MA1": [...],
    "EMA_MA2": [...],
    ...
    "HMA_MA": [...],
    ...
    "KAMA_MA_4": [...]
  }
}
```

**Note:** Returns 9 MAs for each MA type (MA, MA1-MA4, MA_1-MA_4).

---

#### GET /api/timeframes

Get available timeframes.

**Response:**
```json
{
  "success": true,
  "timeframes": [
    {"value": "1m", "label": "1 Minute"},
    {"value": "5m", "label": "5 Minutes"},
    {"value": "15m", "label": "15 Minutes"},
    {"value": "30m", "label": "30 Minutes"},
    {"value": "1h", "label": "1 Hour"},
    {"value": "4h", "label": "4 Hours"},
    {"value": "1d", "label": "1 Day"}
  ]
}
```

---

## Shared Utilities

### Task Manager

The TaskManager provides background task execution with status tracking.

**Location:** `web/shared/utils/task_manager.py`

**Methods:**
```python
# Start a task
task_manager.start_task(session_id, task_func, task_type)

# Get task status
status = task_manager.get_status(session_id)

# Cancel a task
cancelled = task_manager.cancel_task(session_id)

# Check if cancelled
is_cancelled = task_manager.is_cancelled(session_id)

# Set result
task_manager.set_result(session_id, result)

# Set error
task_manager.set_error(session_id, error_msg)
```

**Status Types:**
- `pending` - Task queued but not started
- `running` - Task in progress
- `completed` - Task finished successfully
- `error` - Task failed with error
- `cancelled` - Task cancelled by user

---

### Log Manager

The LogManager handles log file creation and management.

**Location:** `web/shared/utils/log_manager.py`

**Methods:**
```python
# Create log file for session
log_manager.create_log_file(session_id, log_type)

# List available logs
logs = log_manager.list_logs(session_id=None, log_type=None, limit=50)

# Get log file path
log_path = log_manager.get_log_path(session_id)

# Cleanup old logs
log_manager.cleanup_logs(max_age_hours=24)
```

---

### CORS Middleware

Shared CORS configuration for all apps.

**Location:** `web/shared/middleware/cors.py`

**Usage:**
```python
from web.shared.middleware.cors import setup_cors

setup_cors(app, allowed_origins=CORS_ORIGINS)
```

---

## Common Patterns

### Background Task Pattern

For long-running operations, use the background task pattern:

**Backend:**
```python
@router.post("/api/long-operation")
async def start_long_operation(request: Request):
    session_id = str(uuid.uuid4())

    def run_operation():
        # Long-running code here
        result = perform_task()
        task_manager.set_result(session_id, result)

    task_manager.start_task(session_id, run_operation, "operation")

    return {"session_id": session_id, "status": "running"}
```

**Frontend:**
```javascript
// Start operation
const { data } = await axios.post('/api/long-operation');
const sessionId = data.session_id;

// Poll for status
const pollInterval = setInterval(async () => {
  const { data } = await axios.get(`/api/operation/${sessionId}/status`);
  
  if (data.status === 'completed') {
    clearInterval(pollInterval);
    handleResult(data.result);
  } else if (data.status === 'error') {
    clearInterval(pollInterval);
    handleError(data.error);
  }
}, 1000);
```

---

### Error Handling Pattern

Use HTTPException for API errors:

```python
from fastapi import HTTPException

# Validation error
raise HTTPException(status_code=400, detail="Invalid input")

# Not found
raise HTTPException(status_code=404, detail="Resource not found")

# Server error
raise HTTPException(status_code=500, detail="Internal server error")
```

---

### Pagination Pattern

For listing endpoints, use pagination:

```python
from fastapi import Query

@router.get("/api/items")
async def list_items(
    skip: int = Query(0, ge=0, description="Items to skip"),
    limit: int = Query(50, ge=1, le=200, description="Items per page")
):
    items = get_items()[skip:skip+limit]
    return {"success": True, "items": items}
```

---

## Error Handling

### Standard Error Response Format

```json
{
  "detail": "Error message here"
}
```

### Common HTTP Status Codes

| Code | Meaning | Example |
|------|---------|---------|
| 200 | Success | Operation completed successfully |
| 400 | Bad Request | Invalid input or parameters |
| 404 | Not Found | Resource or session not found |
| 500 | Internal Error | Server-side error occurred |

### Error Response Examples

**400 - Bad Request:**
```json
{
  "detail": "Either 'timeframe' (single) or 'timeframes' (multi) must be provided"
}
```

**404 - Not Found:**
```json
{
  "detail": "Session 550e8400-e29b-41d4-a716-446655440000 not found"
}
```

**500 - Internal Error:**
```json
{
  "detail": "Lỗi khi khởi động phân tích: Unable to fetch data"
}
```

---

## Port Reference

| App | Backend Port | Frontend Port | API Docs |
|-----|-------------|---------------|-----------|
| Gemini Analyzer | 8001 | 5173 | http://localhost:8001/docs |
| ATC Visualizer | 8002 | 5174 | http://localhost:8002/docs |

---

## Testing APIs

### Using curl

```bash
# Health check
curl http://localhost:8001/health

# Get symbols
curl http://localhost:8002/api/symbols

# Start analysis
curl -X POST http://localhost:8001/api/analyze/single \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTC/USDT","timeframe":"4h"}'

# Get status
curl http://localhost:8001/api/analyze/{session_id}/status
```

### Using Python requests

```python
import requests

# Health check
response = requests.get('http://localhost:8001/health')
print(response.json())

# Start analysis
response = requests.post(
    'http://localhost:8001/api/analyze/single',
    json={'symbol': 'BTC/USDT', 'timeframe': '4h'}
)
session_id = response.json()['session_id']

# Poll status
while True:
    response = requests.get(f'http://localhost:8001/api/analyze/{session_id}/status')
    status = response.json()['status']
    if status in ['completed', 'error']:
        break
    time.sleep(1)
```

---

## Rate Limiting

Currently, there are no enforced rate limits. Future implementations may include:

- Per-IP rate limiting
- Per-endpoint rate limiting
- API key-based throttling

---

## Authentication

Currently, all endpoints are public. Future implementations may include:

- JWT token authentication
- API key authentication
- OAuth2 integration

---

## WebSocket Support (Planned)

Future versions will support WebSocket connections for:

- Real-time log streaming
- Live chart updates
- Status change notifications

---

## Changelog

### v1.0.0 (2026-01-16)
- Initial API documentation
- Gemini Analyzer API (chart analysis, batch scanner)
- ATC Visualizer API (OHLCV, MA, signals)
- Shared utilities documentation

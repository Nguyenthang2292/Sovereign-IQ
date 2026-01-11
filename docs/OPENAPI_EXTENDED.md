## OpenAPI (Machine-readable)

OpenAPI YAML
```
openapi: 3.0.3
info:
  title: Gemini Chart Analyzer API
  version: 1.0.0
  description: REST API for Gemini Chart Analyzer and Batch Scanner
servers:
  - url: http://localhost:8000/api
paths:
  "/health":
    get:
      responses:
        "200": {"description": "OK"}
  "/":
    get:
      responses:
        "200": {"description": "OK"}
  "/logs/{session_id}":
    get:
      parameters:
        - in: path; name: session_id; required: true; schema: {"type": "string"}
      responses:
        "200": {"description": "OK"}
  "/analyze/single":
    post:
      requestBody:
        required: true
        content:
          application/json:
            schema:
              "$ref": "#/components/schemas/SingleAnalysisRequest"
      responses:
        "200": {"description": "OK"}
  "/analyze/multi":
    post:
      requestBody:
        required: true
        content:
          application/json:
            schema:
              "$ref": "#/components/schemas/MultiAnalysisRequest"
      responses:
        "200": {"description": "OK"}
  "/analyze/{session_id}/status":
    get:
      parameters:
        - in: path; name: session_id; required: true; schema: {"type": "string"}
      responses:
        "200": {"description": "OK"}
  "/batch/scan":
    post:
      requestBody:
        required: true
        content:
          application/json:
            schema:
              "$ref": "#/components/schemas/BatchScanRequest"
      responses:
        "200": {"description": "OK"}
  "/batch/scan/{session_id}/status":
    get:
      parameters:
        - in: path; name: session_id; required: true; schema: {"type": "string"}
      responses:
        "200": {"description": "OK"}
  "/batch/scan/{session_id}/cancel":
    post:
      parameters:
        - in: path; name: session_id; required: true; schema: {"type": "string"}
      responses:
        "200": {"description": "OK"}
  "/batch/results/{filename}":
    get:
      parameters:
        - in: path; name: filename; required: true; schema: {"type": "string"}
      responses:
        "200": {"description": "OK"}
  "/batch/list":
    get:
      responses:
        "200": {"description": "OK"}

components:
  schemas:
    SingleAnalysisRequest:
      type: object
      properties:
        symbol: {type: string}
        timeframe: {type: string}
        indicators: {"$ref": "#/components/schemas/IndicatorsConfig"}
        prompt_type: {type: string}
        custom_prompt: {type: string}
        limit: {type: integer}
        chart_figsize: {type: array, items: {type: integer}, minItems: 2, maxItems: 2}
        chart_dpi: {type: integer}
        no_cleanup: {type: boolean}
      required: [symbol, timeframe]
    MultiAnalysisRequest:
      type: object
      properties:
        symbol: {type: string}
        timeframes: {type: array, items: {type: string}}
        indicators: {"$ref": "#/components/schemas/IndicatorsConfig"}
        prompt_type: {type: string}
        custom_prompt: {type: string}
        limit: {type: integer}
        chart_figsize: {type: array, items: {type: integer}, minItems: 2, maxItems: 2}
        chart_dpi: {type: integer}
        no_cleanup: {type: boolean}
      required: [symbol, timeframes]
    BatchScanRequest:
      type: object
      properties:
        timeframe: {type: string}
        timeframes: {type: array, items: {type: string}}
        max_symbols: {type: integer}
        limit: {type: integer}
        cooldown: {type: number}
        charts_per_batch: {type: integer}
        quote_currency: {type: string}
        exchange_name: {type: string}
    IndicatorsConfig:
      type: object
      properties:
        ma_periods: {type: array, items: {type: integer}}
        rsi_period: {type: integer}
        enable_macd: {type: boolean}
        enable_bb: {type: boolean}
        bb_period: {type: integer}
```

OpenAPI JSON
```
{ /* compact JSON snippet omitted for readability in doc; see openapi.json file in repo for full content */ }
```

## Schemas

- IndicatorsConfig: fields for moving averages, RSI, MACD, Bollinger Bands.
- SingleAnalysisRequest: symbol, timeframe, indicators, prompts, chart options.
- MultiAnalysisRequest: symbol, timeframes, indicators, prompts, chart options.
- BatchScanRequest: timeframe/timeframes, scan limits, cooldowns, and exchange options.

## Examples

- Analyze Single (request)
  - curl:
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"symbol":"BTC/USDT","timeframe":"1h"}' http://localhost:8000/api/analyze/single
    ```
  - response (example):
    ```json
    {"success": true, "session_id": "abc-123", "status": "running", "message": "Analysis started."}
    ```

- Analyze Multi (request)
  - curl:
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"symbol":"BTC/USDT","timeframes":["1h","4h"]}' http://localhost:8000/api/analyze/multi
    ```
  - response (example):
    ```json
    {"success": true, "session_id": "def-456", "status": "running", "message": "Analysis started."}
    ```

- Batch Scan (request)
  - curl:
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"timeframe":"1h"}' http://localhost:8000/api/batch/scan
    ```
  - response (example):
    ```json
    {"success": true, "session_id": "ghi-789", "status": "running", "message": "Scan started."}
    ```

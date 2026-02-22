# Future Design Considerations

As the `gemini_chart_analyzer` module evolves, several structural and performance improvements should be considered. These designs focus on asynchronous execution and data caching to improve throughput and response times.

## 1. Asynchronous Processing (`asyncio`)

Currently, the analysis pipeline operates synchronously. Each step—fetching data, generating a chart, sending it to Gemini, and saving the HTML report—blocks the execution thread. By migrating to an asynchronous architecture, we can achieve significant speedups, especially when processing batches of symbols.

### Proposed Architecture

- **`core/protocols.py` Updates:**
  Change interface methods to be asynchronous:

  ```python
  class AsyncChartAnalyzerProtocol(Protocol):
      async def analyze_chart(self, image_path: str, symbol: str, timeframe: str, ...) -> str: ...

  class AsyncBatchChartAnalyzerProtocol(Protocol):
      async def analyze_batch_chart(self, image_path: str, symbols: List[str], timeframes: List[str]) -> Dict[str, Any]: ...
  ```

- **Async HTTP Clients for External APIs:**
  Replace synchronous API calls in Gemini (`google-generativeai` or raw REST) with asynchronous requests (e.g., using `aiohttp` or the official async Gemini SDKs) to prevent I/O blocking during model inference.

- **Non-blocking Chart Generation:**
  Matplotlib is inherently synchronous and not thread-safe. To avoid blocking the event loop:
  - Run chart generation tasks in a `ProcessPoolExecutor`.
  - Example: `chart_path = await asyncio.get_event_loop().run_in_executor(process_pool, generate_chart_func, df, symbol, timeframe)`

- **Concurrent Batch Processing:**
  In `BatchProcessor`, `asyncio.gather` can be used to prepare multiple sub-batches concurrently (fetch data and generate charts) before sequentially sending them to Gemini (to manage rate limits carefully), or to parallelize API requests if rate limits allow.

---

## 2. Result Caching Strategy

The most expensive operation in the pipeline is the Gemini API call. For the same symbol and timeframe (given a specific time window), the chart and its corresponding analysis shouldn't change. Implementing a caching layer will drastically reduce API load and costs.

### Proposed Architecture

- **Cache Keys:**
  Keys must uniquely identify the analysis input to prevent stale or incorrect retrievals. A suggested structure:

  ```
  Hash(Symbol + Timeframe + End_Timestamp + Indicators_Config + Prompt)
  ```

  - `End_Timestamp`: Can be rounded to the nearest timeframe interval (e.g., nearest hour) to ensure recent data triggers a new analysis, but identical intervals hit the cache.

- **Storage Layer:**
  - **Memory/Redis:** Ideal for fast retrieval across distributed worker nodes.
  - **Local SQLite/JSON:** Acceptable for local single-node execution. Store the Gemini response string and a reference to the chart image path.

- **Integration Point (`ChartAnalysisService`):**

  ```python
  async def run_chart_analysis(config, data_fetcher, cache):
      cache_key = generate_cache_key(config, last_candle_timestamp)
      
      cached_result = await cache.get(cache_key)
      if cached_result:
          return cached_result
          
      # Run full pipeline...
      result = await full_pipeline(...)
      
      await cache.set(cache_key, result, ttl=appropriate_ttl)
      return result
  ```

- **Cache Invalidation:**
  - Standard TTL (Time-To-Live): Set TTL to the timeframe duration (e.g., 1 hour cache TTL for a 1h timeframe chart).
  - Event-driven: Invalidate cache if fresh data processing detects a major price movement or indicator divergence outside the normal range.

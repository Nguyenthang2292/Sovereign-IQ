# Cache Monitoring Guide

## Overview

The Adaptive Trend LTS Mini cache system includes comprehensive monitoring capabilities to track cache performance, identify bottlenecks, and optimize caching strategies.

## Features

- **Real-time Metrics**: Hit/miss rates, cache utilization, eviction counts
- **Historical Tracking**: Trend analysis and performance over time
- **Automatic Logging**: Periodic metrics logging at configurable intervals
- **Performance Insights**: Automatic detection of performance issues
- **Multiple Interfaces**: CLI dashboard, Python API, and HTTP endpoints

## Quick Start

### View Current Cache Status

```python
from modules.adaptive_trend_LTS_mini.utils.cache_manager import get_cache_manager

cache = get_cache_manager()
cache.log_stats()
```

### Display Interactive Dashboard

```bash
# Single snapshot
python modules/adaptive_trend_LTS_mini/utils/cache_monitor.py

# Continuous monitoring (updates every 10 seconds)
python modules/adaptive_trend_LTS_mini/utils/cache_monitor.py --continuous

# Custom update interval
python modules/adaptive_trend_LTS_mini/utils/cache_monitor.py --continuous --interval 5

# Monitor for specific duration
python modules/adaptive_trend_LTS_mini/utils/cache_monitor.py --continuous --duration 60
```

### Start Metrics API Server

```bash
# Start HTTP API server
python modules/adaptive_trend_LTS_mini/utils/cache_metrics_api.py --port 8080

# Access metrics
curl http://localhost:8080/cache/metrics
curl http://localhost:8080/cache/health
curl http://localhost:8080/cache/summary
```

## Metrics Reference

### Core Metrics

| Metric | Description | Type |
|--------|-------------|------|
| `hit_rate_percent` | Overall cache hit rate | Percentage |
| `hit_rate_l1_percent` | L1 cache hit rate | Percentage |
| `hit_rate_l2_percent` | L2 cache hit rate | Percentage |
| `recent_hit_rate_percent` | Hit rate in last 60 seconds | Percentage |
| `total_requests` | Total cache requests | Counter |
| `hits` | Total cache hits (L1 + L2) | Counter |
| `misses` | Total cache misses | Counter |
| `evictions` | Number of cache evictions | Counter |
| `promotions` | L2 to L1 promotions | Counter |

### Cache Size Metrics

| Metric | Description |
|--------|-------------|
| `entries_l1` | Current L1 cache entries |
| `entries_l2` | Current L2 cache entries |
| `size_l2_mb` | L2 cache size in MB |
| `max_entries_l1` | Maximum L1 entries |
| `max_entries_l2` | Maximum L2 entries |
| `max_size_bytes_l2` | Maximum L2 size in bytes |

## Usage Examples

### Python API

#### Basic Monitoring

```python
from modules.adaptive_trend_LTS_mini.utils.cache_manager import get_cache_manager

cache = get_cache_manager()

# Get current statistics
stats = cache.get_stats()
print(f"Hit Rate: {stats['hit_rate_percent']:.2f}%")
print(f"Total Requests: {stats['total_requests']}")

# Get detailed metrics with insights
detailed = cache.get_detailed_metrics()
print(f"Insights: {detailed['insights']}")
```

#### Configure Automatic Logging

```python
from modules.adaptive_trend_LTS_mini.utils.cache_manager import get_cache_manager

cache = get_cache_manager()

# Log metrics every 30 seconds (default: 60)
cache.set_metrics_log_interval(30)

# Disable automatic logging
cache.set_metrics_log_interval(0)
```

#### Advanced Monitoring

```python
from modules.adaptive_trend_LTS_mini.utils.cache_monitor import CacheMonitor

monitor = CacheMonitor(history_size=100)

# Take periodic snapshots
import time
for _ in range(10):
    snapshot = monitor.take_snapshot()
    print(f"Hit Rate: {snapshot['hit_rate_percent']:.2f}%")
    time.sleep(10)

# Analyze trends
trends = monitor.get_trends()
print(f"Hit rate trend: {trends['hit_rate']['trend']}")
print(f"Average hit rate: {trends['hit_rate']['avg']:.2f}%")

# Display dashboard
monitor.display_dashboard(show_trends=True)

# Export metrics
monitor.export_metrics("cache_metrics.json")
monitor.export_history("cache_history.json")

# Generate report
report = monitor.get_summary_report()
print(report)
```

### Command Line Interface

#### Export Metrics

```bash
# Export current metrics to JSON
python modules/adaptive_trend_LTS_mini/utils/cache_monitor.py --export metrics.json

# Export historical data
python modules/adaptive_trend_LTS_mini/utils/cache_monitor.py --export-history history.json
```

#### Continuous Monitoring

```bash
# Monitor with updates every 5 seconds
python modules/adaptive_trend_LTS_mini/utils/cache_monitor.py --continuous --interval 5

# Monitor for 2 minutes
python modules/adaptive_trend_LTS_mini/utils/cache_monitor.py --continuous --duration 120
```

### HTTP API

#### Start Server

```python
from fastapi import FastAPI
from modules.adaptive_trend_LTS_mini.utils.cache_metrics_api import create_cache_metrics_router

app = FastAPI()
app.include_router(create_cache_metrics_router())

# Or use standalone server
# python modules/adaptive_trend_LTS_mini/utils/cache_metrics_api.py --port 8080
```

#### API Endpoints

```bash
# Get current metrics
curl http://localhost:8080/cache/metrics

# Get health status
curl http://localhost:8080/cache/health

# Get performance summary
curl http://localhost:8080/cache/summary

# Get historical snapshots (last 50)
curl http://localhost:8080/cache/snapshots?limit=50

# Record new snapshot
curl -X POST http://localhost:8080/cache/snapshots
```

#### Response Examples

**GET /cache/metrics**
```json
{
  "timestamp": 1707234567.89,
  "metrics": {
    "hit_rate_percent": 87.5,
    "hit_rate_l1_percent": 45.2,
    "hit_rate_l2_percent": 42.3,
    "recent_hit_rate_percent": 89.1,
    "total_requests": 10000,
    "hits": 8750,
    "misses": 1250,
    "evictions": 123,
    "promotions": 456,
    "insights": []
  }
}
```

**GET /cache/health**
```json
{
  "status": "healthy",
  "warnings": [],
  "timestamp": 1707234567.89
}
```

## Performance Insights

The monitoring system automatically detects common performance issues:

### LOW_HIT_RATE
- **Trigger**: Hit rate < 50%
- **Recommendation**: Consider warming cache or increasing cache size

### L2_HEAVY
- **Trigger**: L1 hit rate < 20% while L2 hit rate > 30%
- **Recommendation**: Increase L1 size for better performance

### HIGH_EVICTION
- **Trigger**: Evictions > total entries
- **Recommendation**: Cache thrashing detected, increase cache size

### DEGRADING_PERFORMANCE
- **Trigger**: Recent hit rate < overall hit rate - 10%
- **Recommendation**: Performance is declining, investigate workload changes

## Dashboard Interpretation

### Sample Dashboard Output

```
================================================================================
CACHE PERFORMANCE DASHBOARD
================================================================================
Timestamp: 2024-02-06T14:30:45.123456

CACHE SIZE:
  L1 Entries: 98 / 128
  L2 Entries: 856 / 1,024
  L2 Size: 245.67 MB / 1000.00 MB

HIT/MISS STATISTICS:
  Total Requests: 15,432
  Total Hits: 13,245 (85.83%)
    - L1 Hits: 6,234 (40.40%)
    - L2 Hits: 7,011 (45.43%)
  Total Misses: 2,187 (14.17%)

RECENT PERFORMANCE (Last 60s):
  Recent Hit Rate: 87.45%
  Recent Hits: 523
  Recent Misses: 75

CACHE OPERATIONS:
  Evictions: 234
  Promotions (L2→L1): 789

PERFORMANCE INSIGHTS:
  ⚠ L2_HEAVY: Consider increasing L1 size for better performance

TRENDS:
  Hit Rate: 85.83% (avg: 84.21%, min: 78.50%, max: 88.92%)
  Trend: INCREASING
  Eviction Rate: 2.34 evictions/snapshot
  Monitoring Duration: 145.3s (15 samples)
================================================================================
```

### Key Indicators

1. **Hit Rate > 80%**: Excellent performance
2. **Hit Rate 60-80%**: Good performance
3. **Hit Rate < 60%**: Needs optimization
4. **Recent Hit Rate > Overall**: Performance improving
5. **Recent Hit Rate < Overall**: Performance degrading

## Integration Examples

### Integration with Web Dashboard

```python
# In your FastAPI app
from fastapi import FastAPI
from modules.adaptive_trend_LTS_mini.utils.cache_metrics_api import create_cache_metrics_router

app = FastAPI()
app.include_router(create_cache_metrics_router())

# Frontend can fetch metrics via:
# GET /cache/metrics - Real-time metrics
# GET /cache/snapshots?limit=100 - Historical data for charts
```

### Integration with Monitoring Tools

```python
# Prometheus-style metrics export
from modules.adaptive_trend_LTS_mini.utils.cache_manager import get_cache_manager

def export_prometheus_metrics():
    cache = get_cache_manager()
    stats = cache.get_stats()

    metrics = []
    metrics.append(f'cache_hit_rate {stats["hit_rate_percent"]}')
    metrics.append(f'cache_requests_total {stats["total_requests"]}')
    metrics.append(f'cache_hits_total {stats["hits"]}')
    metrics.append(f'cache_misses_total {stats["misses"]}')
    metrics.append(f'cache_evictions_total {stats["evictions"]}')

    return '\n'.join(metrics)
```

### Periodic Snapshot Collection

```python
import threading
import time
from modules.adaptive_trend_LTS_mini.utils.cache_metrics_api import get_metrics_service

def snapshot_collector(interval_seconds=60):
    """Collect snapshots periodically for historical analysis."""
    service = get_metrics_service()

    while True:
        try:
            service.record_snapshot()
            time.sleep(interval_seconds)
        except Exception as e:
            print(f"Error collecting snapshot: {e}")

# Start background thread
thread = threading.Thread(target=snapshot_collector, args=(60,), daemon=True)
thread.start()
```

## Troubleshooting

### Low Hit Rate

**Symptoms**: Hit rate < 50%

**Possible Causes**:
- Cache size too small
- High data variability (random access patterns)
- Cache not warmed up

**Solutions**:
```python
# Increase cache size
cache = CacheManager(
    max_entries_l1=256,  # Increase from 128
    max_entries_l2=2048,  # Increase from 1024
    max_size_mb_l2=2000.0  # Increase from 1000.0
)

# Warm cache before operations
cache.warm_cache(symbols_data, configs)
```

### High Eviction Rate

**Symptoms**: Evictions > total entries, frequent cache turnover

**Possible Causes**:
- Cache size too small for workload
- Poor key locality (working set too large)

**Solutions**:
```python
# Increase L2 size
cache = CacheManager(
    max_entries_l2=4096,  # Double the size
    max_size_mb_l2=5000.0  # Increase memory limit
)
```

### L2-Heavy Performance

**Symptoms**: L2 hit rate much higher than L1 hit rate

**Possible Causes**:
- L1 too small to hold working set
- Frequent cache promotions

**Solutions**:
```python
# Increase L1 size
cache = CacheManager(
    max_entries_l1=256  # Double from default 128
)
```

## Best Practices

1. **Monitor Regularly**: Check cache metrics during development and production
2. **Set Appropriate Intervals**: Configure automatic logging based on workload
3. **Export Historical Data**: Keep snapshots for trend analysis
4. **Adjust Cache Sizes**: Tune based on observed hit rates and workload
5. **Warm Cache**: Pre-populate cache for production workloads
6. **Track Insights**: Act on performance warnings immediately

## API Reference

### CacheManager Methods

```python
# Get statistics
stats = cache.get_stats()
detailed = cache.get_detailed_metrics()

# Configure logging
cache.set_metrics_log_interval(60)  # seconds

# Manual logging
cache.log_stats()
cache.log_cache_effectiveness()

# Clear cache (resets some metrics)
cache.clear()
```

### CacheMonitor Class

```python
monitor = CacheMonitor(history_size=100)

# Take snapshot
snapshot = monitor.take_snapshot()

# Get trends
trends = monitor.get_trends()

# Display dashboard
monitor.display_dashboard(show_trends=True)

# Export data
monitor.export_metrics("metrics.json")
monitor.export_history("history.json")

# Generate report
report = monitor.get_summary_report()
```

### CacheMetricsService Class

```python
from modules.adaptive_trend_LTS_mini.utils.cache_metrics_api import get_metrics_service

service = get_metrics_service()

# Get metrics
metrics = service.get_current_metrics()
health = service.get_health_status()
summary = service.get_performance_summary()

# Record/retrieve snapshots
service.record_snapshot()
snapshots = service.get_snapshots(limit=50)
```

## See Also

- [Cache Manager Documentation](../README.md#cache-system)
- [Performance Tuning Guide](./PERFORMANCE_TUNING.md)
- [API Reference](./API_REFERENCE.md)

# CloudWatch Monitoring and Alerts Setup

**Module**: ATC Serverless Lambda  
**Date**: February 15, 2026  
**Purpose**: Memory monitoring, performance metrics, and automated alerting

---

## Overview

This document describes the CloudWatch monitoring setup for the ATC Serverless Lambda function, including custom metrics, log insights queries, and alarm configurations.

---

## Custom Metrics

The Lambda handler emits the following custom metrics via structured logging:

### 1. Memory Usage Metrics

#### MemoryUsageMB

- **Description**: Peak memory usage during batch processing
- **Unit**: Megabytes
- **Dimensions**: `batch_id`
- **Typical Range**: 50-500 MB (depending on batch size)
- **Alert Threshold**:
  - Warning: 512 MB
  - Critical: 768 MB (for 1GB Lambda)

#### MemoryDeltaMB

- **Description**: Memory increase from start to peak
- **Unit**: Megabytes
- **Dimensions**: `batch_id`
- **Typical Range**: 20-300 MB
- **Use Case**: Detect memory leaks or unexpected growth

### 2. Performance Metrics

#### SymbolsPerSecond

- **Description**: Processing throughput
- **Unit**: Count/Second
- **Dimensions**: `batch_id`
- **Typical Range**: 3,000-10,000 symbols/second
- **Alert Threshold**: < 1,000 symbols/second (performance degradation)

---

## CloudWatch Logs Insights Queries

### Query 1: Memory Usage Over Time

```sql
fields @timestamp, batch_id, peak_memory_mb, memory_delta_mb, symbol_count
| filter metric_name = "MemoryUsageMB"
| sort @timestamp desc
| limit 100
```

**Purpose**: Track memory usage trends across batches

### Query 2: High Memory Usage Alerts

```sql
fields @timestamp, batch_id, peak_memory_mb, symbol_count
| filter metric_name = "MemoryUsageMB" and metric_value >= 512
| sort metric_value desc
| limit 50
```

**Purpose**: Identify batches approaching memory limits

### Query 3: Performance Analysis

```sql
fields @timestamp, batch_id, processing_duration_ms, symbols_per_second, success_count, error_count
| filter metric_name = "SymbolsPerSecond"
| stats avg(metric_value) as avg_throughput, 
        max(metric_value) as max_throughput, 
        min(metric_value) as min_throughput 
  by bin(5m)
```

**Purpose**: Monitor processing performance over time

### Query 4: Error Rate Monitoring

```sql
fields @timestamp, batch_id, error_count, total_symbols, error_rate
| filter ispresent(error_rate)
| filter error_rate > 0.1
| sort error_rate desc
| limit 50
```

**Purpose**: Detect batches with high error rates (>10%)

### Query 5: Memory vs Batch Size Correlation

```sql
fields symbol_count, peak_memory_mb, memory_delta_mb
| filter metric_name = "MemoryUsageMB"
| stats avg(peak_memory_mb) as avg_memory, 
        max(peak_memory_mb) as max_memory 
  by symbol_count
| sort symbol_count asc
```

**Purpose**: Understand memory scaling with batch size

---

## CloudWatch Alarms Configuration

### Alarm 1: High Memory Usage Warning

```json
{
  "AlarmName": "ATC-Lambda-MemoryWarning",
  "AlarmDescription": "Memory usage exceeds 512MB threshold",
  "MetricName": "MemoryUsageMB",
  "Namespace": "ATC/Serverless",
  "Statistic": "Maximum",
  "Period": 300,
  "EvaluationPeriods": 1,
  "Threshold": 512,
  "ComparisonOperator": "GreaterThanThreshold",
  "TreatMissingData": "notBreaching",
  "ActionsEnabled": true,
  "AlarmActions": [
    "arn:aws:sns:REGION:ACCOUNT_ID:atc-lambda-alerts"
  ]
}
```

**AWS CLI Command**:

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name ATC-Lambda-MemoryWarning \
  --alarm-description "Memory usage exceeds 512MB threshold" \
  --metric-name MemoryUsageMB \
  --namespace ATC/Serverless \
  --statistic Maximum \
  --period 300 \
  --evaluation-periods 1 \
  --threshold 512 \
  --comparison-operator GreaterThanThreshold \
  --treat-missing-data notBreaching \
  --alarm-actions arn:aws:sns:REGION:ACCOUNT_ID:atc-lambda-alerts
```

### Alarm 2: Critical Memory Usage

```json
{
  "AlarmName": "ATC-Lambda-MemoryCritical",
  "AlarmDescription": "CRITICAL: Memory usage exceeds 768MB (approaching 1GB limit)",
  "MetricName": "MemoryUsageMB",
  "Namespace": "ATC/Serverless",
  "Statistic": "Maximum",
  "Period": 60,
  "EvaluationPeriods": 1,
  "Threshold": 768,
  "ComparisonOperator": "GreaterThanThreshold",
  "TreatMissingData": "notBreaching",
  "ActionsEnabled": true,
  "AlarmActions": [
    "arn:aws:sns:REGION:ACCOUNT_ID:atc-lambda-critical-alerts"
  ]
}
```

**AWS CLI Command**:

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name ATC-Lambda-MemoryCritical \
  --alarm-description "CRITICAL: Memory usage exceeds 768MB" \
  --metric-name MemoryUsageMB \
  --namespace ATC/Serverless \
  --statistic Maximum \
  --period 60 \
  --evaluation-periods 1 \
  --threshold 768 \
  --comparison-operator GreaterThanThreshold \
  --treat-missing-data notBreaching \
  --alarm-actions arn:aws:sns:REGION:ACCOUNT_ID:atc-lambda-critical-alerts
```

### Alarm 3: Low Throughput Performance

```json
{
  "AlarmName": "ATC-Lambda-LowThroughput",
  "AlarmDescription": "Processing throughput below 1000 symbols/second",
  "MetricName": "SymbolsPerSecond",
  "Namespace": "ATC/Serverless",
  "Statistic": "Average",
  "Period": 300,
  "EvaluationPeriods": 2,
  "Threshold": 1000,
  "ComparisonOperator": "LessThanThreshold",
  "TreatMissingData": "notBreaching",
  "ActionsEnabled": true,
  "AlarmActions": [
    "arn:aws:sns:REGION:ACCOUNT_ID:atc-lambda-performance-alerts"
  ]
}
```

**AWS CLI Command**:

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name ATC-Lambda-LowThroughput \
  --alarm-description "Processing throughput below 1000 symbols/second" \
  --metric-name SymbolsPerSecond \
  --namespace ATC/Serverless \
  --statistic Average \
  --period 300 \
  --evaluation-periods 2 \
  --threshold 1000 \
  --comparison-operator LessThanThreshold \
  --treat-missing-data notBreaching \
  --alarm-actions arn:aws:sns:REGION:ACCOUNT_ID:atc-lambda-performance-alerts
```

### Alarm 4: High Error Rate

```json
{
  "AlarmName": "ATC-Lambda-HighErrorRate",
  "AlarmDescription": "Error rate exceeds 10% of processed symbols",
  "MetricName": "ErrorRate",
  "Namespace": "ATC/Serverless",
  "Statistic": "Average",
  "Period": 300,
  "EvaluationPeriods": 2,
  "Threshold": 0.1,
  "ComparisonOperator": "GreaterThanThreshold",
  "TreatMissingData": "notBreaching",
  "ActionsEnabled": true,
  "AlarmActions": [
    "arn:aws:sns:REGION:ACCOUNT_ID:atc-lambda-error-alerts"
  ]
}
```

---

## SNS Topic Setup

### Create SNS Topics for Alerts

```bash
# Create warning alerts topic
aws sns create-topic --name atc-lambda-alerts

# Create critical alerts topic
aws sns create-topic --name atc-lambda-critical-alerts

# Create performance alerts topic
aws sns create-topic --name atc-lambda-performance-alerts

# Create error alerts topic
aws sns create-topic --name atc-lambda-error-alerts
```

### Subscribe Email to Topics

```bash
# Subscribe to warning alerts
aws sns subscribe \
  --topic-arn arn:aws:sns:REGION:ACCOUNT_ID:atc-lambda-alerts \
  --protocol email \
  --notification-endpoint your-email@example.com

# Subscribe to critical alerts (use SMS for urgent notifications)
aws sns subscribe \
  --topic-arn arn:aws:sns:REGION:ACCOUNT_ID:atc-lambda-critical-alerts \
  --protocol sms \
  --notification-endpoint +1234567890

# Subscribe to performance alerts
aws sns subscribe \
  --topic-arn arn:aws:sns:REGION:ACCOUNT_ID:atc-lambda-performance-alerts \
  --protocol email \
  --notification-endpoint devops-team@example.com
```

---

## Dashboard Configuration

### CloudWatch Dashboard JSON

```json
{
  "widgets": [
    {
      "type": "metric",
      "properties": {
        "metrics": [
          [ "ATC/Serverless", "MemoryUsageMB", { "stat": "Maximum" } ],
          [ ".", "MemoryDeltaMB", { "stat": "Average" } ]
        ],
        "period": 300,
        "stat": "Average",
        "region": "us-east-1",
        "title": "Memory Usage",
        "yAxis": {
          "left": {
            "label": "MB"
          }
        }
      }
    },
    {
      "type": "metric",
      "properties": {
        "metrics": [
          [ "ATC/Serverless", "SymbolsPerSecond", { "stat": "Average" } ]
        ],
        "period": 300,
        "stat": "Average",
        "region": "us-east-1",
        "title": "Processing Throughput",
        "yAxis": {
          "left": {
            "label": "Symbols/Second"
          }
        }
      }
    },
    {
      "type": "log",
      "properties": {
        "query": "SOURCE '/aws/lambda/atc-serverless'\n| fields @timestamp, batch_id, peak_memory_mb, symbols_per_second\n| filter metric_name = \"MemoryUsageMB\"\n| sort @timestamp desc\n| limit 20",
        "region": "us-east-1",
        "title": "Recent Batch Processing"
      }
    }
  ]
}
```

**Create Dashboard**:

```bash
aws cloudwatch put-dashboard \
  --dashboard-name ATC-Serverless-Monitoring \
  --dashboard-body file://dashboard.json
```

---

## Monitoring Best Practices

### 1. Regular Review

- Review CloudWatch Insights queries weekly
- Analyze memory usage trends monthly
- Adjust thresholds based on actual usage patterns

### 2. Proactive Alerts

- Set up PagerDuty integration for critical alerts
- Configure Slack notifications for warning alerts
- Create runbooks for common alert scenarios

### 3. Cost Optimization

- Use metric filters to reduce log ingestion costs
- Set log retention to 7 days for debug logs, 30 days for metrics
- Archive historical data to S3 for long-term analysis

### 4. Performance Baselines

- Establish baseline metrics after initial deployment
- Track performance degradation over time
- Correlate memory usage with batch size and complexity

---

## Troubleshooting Guide

### High Memory Usage

**Symptoms**: MemoryUsageMB > 512 MB

**Possible Causes**:

1. Large batch size (>500 symbols)
2. Memory leak in processing logic
3. Inefficient data structures

**Actions**:

1. Check batch size in logs
2. Review recent code changes
3. Analyze memory delta trends
4. Consider increasing Lambda memory allocation

### Low Throughput

**Symptoms**: SymbolsPerSecond < 1000

**Possible Causes**:

1. CPU throttling
2. Network latency
3. Inefficient algorithms

**Actions**:

1. Check Lambda CPU utilization
2. Review processing duration logs
3. Profile hot paths with flamegraph
4. Consider SIMD optimizations

### High Error Rate

**Symptoms**: ErrorRate > 10%

**Possible Causes**:

1. Invalid input data
2. API rate limiting
3. Timeout issues

**Actions**:

1. Review error logs for patterns
2. Check symbol-specific failures
3. Validate input data quality
4. Adjust timeout configurations

---

## Related Files

- **Lambda Handler**: `lambda/src/handler.rs`
- **Aggregation Module**: `src/aggregation.rs`
- **Performance Profile**: `PERFORMANCE_PROFILE.md`
- **TODO Tracking**: `archive/phase_1_2_issues_todo.md`

---

**Last Updated**: February 15, 2026  
**Status**: ✅ Production Ready  
**Maintainer**: DevOps Team

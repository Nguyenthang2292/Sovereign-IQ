# Error Recovery and Failure Handling Guide

**Module**: ATC Serverless Lambda  
**Version**: 0.1.0  
**Last Updated**: February 16, 2026  
**Status**: Production Ready

---

## Table of Contents

- [Overview](#overview)
- [Failure Scenarios](#failure-scenarios)
- [SQS Error Handling](#sqs-error-handling)
- [Dead Letter Queue (DLQ)](#dead-letter-queue-dlq)
- [Recovery Procedures](#recovery-procedures)
- [Monitoring and Alerts](#monitoring-and-alerts)
- [Troubleshooting](#troubleshooting)
- [Runbooks](#runbooks)

---

## Overview

This guide documents all failure modes in the ATC Serverless Lambda system and provides detailed recovery procedures. The system implements multiple layers of error handling to ensure reliability and data integrity.

### Error Handling Layers

1. **Input Validation** - Reject invalid requests before processing
2. **Processing Error Recovery** - Continue processing valid symbols even if some fail
3. **SQS Retry Logic** - Automatic retry with exponential backoff
4. **Dead Letter Queue** - Capture failed messages for manual recovery
5. **CloudWatch Alarms** - Alert on critical failures

---

## Failure Scenarios

### 1. Input Validation Failures

**What happens**: Lambda rejects the request immediately before processing.

**Causes**:
- Invalid batch size (>1500 symbols)
- Missing required fields
- Invalid OHLCV data format
- Schema version mismatch

**Impact**: Request is rejected, no processing occurs.

**Recovery**: Fix the input data and retry the request.

**Example Error**:
```json
{
  "error": "Input validation failed: batch size 2000 exceeds maximum of 1500"
}
```

**Prevention**:
- Validate requests on the client side before sending
- Use the latest schema version
- Follow the Python integration guide

---

### 2. Symbol Processing Failures

**What happens**: Individual symbols fail processing, but the batch continues.

**Causes**:
- Insufficient OHLCV data (< diflen requirements)
- NaN values in price data
- Invalid MA configuration
- Numerical overflow/underflow

**Impact**: Failed symbols are logged in the `errors` array of ScanResult.

**Recovery**: Automatic - batch continues processing other symbols.

**Error Tracking**:
```json
{
  "batch_id": "batch_123",
  "results": [...],
  "errors": [
    {
      "symbol": "AAPL",
      "error": "Insufficient data: have 50 bars, need 120"
    }
  ],
  "success_count": 499,
  "error_count": 1
}
```

**Best Practices**:
- Monitor error rate per batch
- Alert if error_rate > 10%
- Investigate symbols with repeated failures

---

### 3. SQS Send Failures

**What happens**: Lambda cannot send results to the SQS queue.

**Causes**:
- Network connectivity issues
- SQS service throttling
- IAM permission errors
- Queue does not exist

**Impact**: Results are lost unless retry succeeds or DLQ is configured.

**Recovery**: Automatic retry with exponential backoff (see [SQS Error Handling](#sqs-error-handling)).

---

### 4. Memory Exhaustion

**What happens**: Lambda runs out of memory during processing.

**Causes**:
- Batch size too large for Lambda memory allocation
- Memory leak (very rare in Rust)
- Unexpected data volume

**Impact**: Lambda terminates, entire batch fails.

**Recovery**: Reduce batch size or increase Lambda memory.

**Detection**:
```
CRITICAL: Memory usage exceeds critical threshold
peak_memory_mb=780, threshold_mb=768
```

**Prevention**:
- Monitor memory metrics
- Use CloudWatch alarms (see CLOUDWATCH_MONITORING.md)
- Follow batch size recommendations (see PERFORMANCE_TUNING.md)

---

### 5. Lambda Timeout

**What happens**: Processing takes longer than Lambda timeout (default 300s).

**Causes**:
- Batch size too large
- Complex MA configurations
- Cold start delay

**Impact**: Partial results may be lost.

**Recovery**: Reduce batch size or increase timeout.

**Detection**:
```
Task timed out after 300.00 seconds
```

**Prevention**:
- Test with production data volumes
- Monitor processing duration
- Set appropriate timeout values

---

## SQS Error Handling

### Retry Strategy

The Lambda implements **automatic retry with exponential backoff** for SQS send failures:

```
Attempt 1: Immediate send
  ↓ (fails)
Attempt 2: Wait 100ms, retry
  ↓ (fails)
Attempt 3: Wait 200ms, retry
  ↓ (fails)
Attempt 4: Wait 400ms, retry (FINAL)
  ↓ (fails)
Dead Letter Queue (if configured)
  OR
Return error to Lambda runtime
```

### Retry Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MAX_RETRY_ATTEMPTS` | 3 | Maximum number of retry attempts |
| `INITIAL_BACKOFF_MS` | 100 | Initial backoff delay in milliseconds |
| `MAX_BACKOFF_MS` | 5000 | Maximum backoff delay (capped) |

### Backoff Calculation

Exponential backoff formula:
```
backoff_ms = min(INITIAL_BACKOFF * 2^(attempt-1), MAX_BACKOFF)
```

Example progression:
- Attempt 1: 100ms
- Attempt 2: 200ms
- Attempt 3: 400ms
- Attempt 4+: 800ms, 1600ms, 3200ms, 5000ms (capped)

### Error Classification

#### Retryable Errors
- Network timeouts
- Service throttling (503 errors)
- Temporary service unavailability

#### Non-Retryable Errors
- Serialization failures (JSON conversion)
- IAM permission denied
- Queue does not exist

### Logging

All retry attempts are logged with structured data:

```json
{
  "level": "WARN",
  "batch_id": "batch_123",
  "attempt": 2,
  "max_attempts": 3,
  "backoff_ms": 200,
  "error": "RequestTimeout: request timed out",
  "message": "SQS send failed, retrying after backoff"
}
```

---

## Dead Letter Queue (DLQ)

### Overview

The Dead Letter Queue captures messages that fail after all retry attempts, enabling manual recovery and investigation.

### Configuration

**CloudFormation/SAM Template**:
```yaml
# Dead Letter Queue
ATCDeadLetterQueue:
  Type: AWS::SQS::Queue
  Properties:
    QueueName: atc-sqs-dlq
    MessageRetentionPeriod: 1209600  # 14 days

# Main Queue with Redrive Policy
ATCSQSPollerQueue:
  Type: AWS::SQS::Queue
  Properties:
    QueueName: atc-sqs-poller-queue
    RedrivePolicy:
      deadLetterTargetArn: !GetAtt ATCDeadLetterQueue.Arn
      maxReceiveCount: 3  # Consumer-side failures

# Lambda Environment
Environment:
  Variables:
    DLQ_URL: !Ref ATCDeadLetterQueue
```

### Message Attributes

Messages in DLQ include metadata for debugging:

| Attribute | Type | Description |
|-----------|------|-------------|
| `FailureReason` | String | Why the message went to DLQ |
| `OriginalBatchId` | String | Batch ID from the original request |

### Two-Level DLQ Protection

The system implements **two levels** of Dead Letter Queue protection:

#### Level 1: Consumer-Side DLQ (SQS RedrivePolicy)
- Handles **processing failures** (Lambda errors, timeouts)
- Configured via `maxReceiveCount: 3`
- If Lambda fails 3 times, message moves to DLQ

#### Level 2: Producer-Side DLQ (Lambda SqsClient)
- Handles **SQS send failures** (network, throttling)
- Configured in `SqsClient::send_scan_result()`
- If all retries fail, message sent to DLQ programmatically

### DLQ Processing Flow

```
Lambda Processing
  ↓
SQS Send with Retry
  ↓ (3 attempts fail)
Send to DLQ with Metadata
  ↓
CloudWatch Alarm Triggered
  ↓
Manual Investigation Required
```

### Monitoring DLQ

**CloudWatch Alarm**:
```yaml
DLQAlarm:
  Type: AWS::CloudWatch::Alarm
  Properties:
    AlarmName: ATC-DLQ-Messages-Detected
    MetricName: ApproximateNumberOfMessagesVisible
    Namespace: AWS/SQS
    Threshold: 1  # Alert on ANY message in DLQ
    Period: 60  # Check every minute
```

**Query DLQ Messages** (AWS CLI):
```bash
# Check DLQ depth
aws sqs get-queue-attributes \
  --queue-url https://sqs.us-east-1.amazonaws.com/123456789012/atc-sqs-dlq \
  --attribute-names ApproximateNumberOfMessages

# Receive message from DLQ
aws sqs receive-message \
  --queue-url https://sqs.us-east-1.amazonaws.com/123456789012/atc-sqs-dlq \
  --max-number-of-messages 10 \
  --attribute-names All \
  --message-attribute-names All
```

---

## Recovery Procedures

### Procedure 1: Retry Failed Batch

**When**: SQS send failed, no DLQ configured.

**Steps**:
1. Check Lambda CloudWatch logs for batch_id
2. Retrieve original request payload
3. Re-invoke Lambda with same payload
4. Verify results sent to SQS successfully

**Example**:
```bash
# Find failed batch in logs
aws logs filter-log-events \
  --log-group-name /aws/lambda/atc-serverless \
  --filter-pattern "\"Failed to send results to SQS\""

# Re-invoke Lambda
aws lambda invoke \
  --function-name atc-serverless \
  --payload file://batch_123_retry.json \
  --cli-binary-format raw-in-base64-out \
  response.json
```

---

### Procedure 2: Process DLQ Messages

**When**: Messages accumulate in Dead Letter Queue.

**Steps**:

#### Step 1: Investigate Root Cause
```bash
# Check DLQ messages
aws sqs receive-message \
  --queue-url https://sqs.us-east-1.amazonaws.com/123456789012/atc-sqs-dlq \
  --max-number-of-messages 10 \
  --message-attribute-names All
```

#### Step 2: Identify Failure Pattern
- Check `FailureReason` attribute
- Review CloudWatch logs for corresponding batch_id
- Determine if issue is systemic or isolated

#### Step 3: Fix Root Cause
Common fixes:
- **IAM Permissions**: Update Lambda execution role
- **Queue Issues**: Verify queue exists and is accessible
- **Network**: Check VPC/security group configuration

#### Step 4: Replay Messages
```bash
# Option A: Manual replay (small volume)
aws sqs receive-message \
  --queue-url $DLQ_URL | \
  jq -r '.Messages[].Body' | \
  while read body; do
    aws sqs send-message \
      --queue-url $PRIMARY_QUEUE_URL \
      --message-body "$body"
  done

# Option B: Redrive (large volume)
aws sqs start-message-move-task \
  --source-arn arn:aws:sqs:us-east-1:123456789012:atc-sqs-dlq \
  --destination-arn arn:aws:sqs:us-east-1:123456789012:atc-sqs-poller-queue
```

#### Step 5: Verify Success
- Monitor primary queue processing
- Confirm results appear in downstream systems
- Clear DLQ once confirmed

---

### Procedure 3: Recover from Memory Exhaustion

**When**: Lambda terminates due to out-of-memory error.

**Steps**:

#### Step 1: Identify Problematic Batch
```bash
# Find OOM errors in logs
aws logs filter-log-events \
  --log-group-name /aws/lambda/atc-serverless \
  --filter-pattern "Runtime.ExitError"
```

#### Step 2: Analyze Memory Usage
```bash
# Get memory metrics for batch
aws logs filter-log-events \
  --log-group-name /aws/lambda/atc-serverless \
  --filter-pattern "{ $.peak_memory_mb > 768 }"
```

#### Step 3: Adjust Configuration
```bash
# Option A: Reduce batch size
# Split 1500 symbols into 2 batches of 750

# Option B: Increase Lambda memory
aws lambda update-function-configuration \
  --function-name atc-serverless \
  --memory-size 2048  # Increase from 1024MB
```

#### Step 4: Reprocess Batch
```bash
# Retry with adjusted configuration
aws lambda invoke \
  --function-name atc-serverless \
  --payload file://batch_reduced.json \
  response.json
```

---

### Procedure 4: Handle Partial Failures

**When**: Batch completes but has high error rate.

**Steps**:

#### Step 1: Extract Failed Symbols
```python
# Parse ScanResult from SQS
import json

scan_result = json.loads(sqs_message_body)
failed_symbols = [err['symbol'] for err in scan_result['errors']]
error_messages = {err['symbol']: err['error'] for err in scan_result['errors']}

print(f"Failed symbols: {failed_symbols}")
print(f"Error summary: {error_messages}")
```

#### Step 2: Categorize Errors
```python
# Group errors by type
from collections import defaultdict

error_categories = defaultdict(list)
for err in scan_result['errors']:
    error_type = err['error'].split(':')[0]
    error_categories[error_type].append(err['symbol'])

for error_type, symbols in error_categories.items():
    print(f"{error_type}: {len(symbols)} symbols")
    print(f"  Examples: {symbols[:5]}")
```

#### Step 3: Retry Recoverable Errors
```python
# Retry symbols with insufficient data after fetching more history
symbols_to_retry = error_categories.get('Insufficient data', [])

if symbols_to_retry:
    # Fetch more historical data
    # Rebuild batch request
    retry_batch = {
        "batch_id": f"{original_batch_id}_retry",
        "symbols": symbols_to_retry,
        "config": original_config
    }
    # Invoke Lambda with retry batch
```

---

## Monitoring and Alerts

### Key Metrics to Monitor

| Metric | Threshold | Alert Level | Action |
|--------|-----------|-------------|--------|
| DLQ Message Count | > 0 | **CRITICAL** | Investigate immediately |
| Error Rate per Batch | > 10% | **WARNING** | Review error patterns |
| Memory Usage | > 768MB | **CRITICAL** | Adjust batch size |
| Processing Duration | > 180s | **WARNING** | Consider optimization |
| SQS Send Failures | > 5/hour | **WARNING** | Check SQS health |

### CloudWatch Log Insights Queries

#### Query 1: Find All Failed Batches
```
fields @timestamp, batch_id, error
| filter message like /Failed to send results to SQS/
| sort @timestamp desc
| limit 100
```

#### Query 2: Analyze Error Rates
```
fields batch_id, success_count, error_count, 
       (error_count / (success_count + error_count)) as error_rate
| filter error_count > 0
| sort error_rate desc
| limit 50
```

#### Query 3: Memory Usage Trends
```
fields @timestamp, batch_id, peak_memory_mb, symbol_count
| filter peak_memory_mb > 512
| stats avg(peak_memory_mb), max(peak_memory_mb) by bin(5m)
```

#### Query 4: SQS Retry Attempts
```
fields @timestamp, batch_id, attempt, backoff_ms, error
| filter message like /retrying after backoff/
| sort @timestamp desc
| limit 100
```

### SNS Alert Configuration

```yaml
# Create SNS topic for alerts
AlertTopic:
  Type: AWS::SNS::Topic
  Properties:
    TopicName: atc-serverless-alerts
    Subscription:
      - Endpoint: ops-team@example.com
        Protocol: email

# Link CloudWatch alarms to SNS
DLQAlarm:
  Properties:
    AlarmActions:
      - !Ref AlertTopic
```

---

## Troubleshooting

### Issue: High SQS Send Failure Rate

**Symptoms**:
- Multiple retry attempts logged
- CloudWatch shows frequent SQS errors
- DLQ accumulating messages

**Possible Causes**:
1. SQS throttling (TPS limit exceeded)
2. Network connectivity issues
3. IAM permission problems

**Diagnosis**:
```bash
# Check SQS metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/SQS \
  --metric-name NumberOfMessagesSent \
  --dimensions Name=QueueName,Value=atc-sqs-poller-queue \
  --start-time 2026-02-16T00:00:00Z \
  --end-time 2026-02-16T23:59:59Z \
  --period 300 \
  --statistics Sum

# Check for throttling
aws logs filter-log-events \
  --log-group-name /aws/lambda/atc-serverless \
  --filter-pattern "Throttling"
```

**Resolution**:
1. **If throttled**: Request SQS limit increase
2. **If network**: Check VPC/NAT Gateway configuration
3. **If IAM**: Update Lambda execution role with `sqs:SendMessage`

---

### Issue: Messages Stuck in DLQ

**Symptoms**:
- DLQ message count not decreasing
- Messages older than expected

**Diagnosis**:
```bash
# Check message age
aws sqs receive-message \
  --queue-url $DLQ_URL \
  --attribute-names SentTimestamp \
  --max-number-of-messages 10
```

**Resolution**:
1. Investigate root cause (see [Procedure 2](#procedure-2-process-dlq-messages))
2. Fix underlying issue
3. Replay messages to primary queue
4. Monitor for recurrence

---

### Issue: Inconsistent Error Rates

**Symptoms**:
- Some batches have 0% error rate
- Others have 20%+ error rate
- No obvious pattern

**Diagnosis**:
```bash
# Correlate error rates with batch characteristics
aws logs filter-log-events \
  --log-group-name /aws/lambda/atc-serverless \
  --filter-pattern "{ $.error_count > 0 }" | \
  jq '.events[] | .message | fromjson | 
      {batch_id, symbol_count, error_count, error_rate}'
```

**Possible Causes**:
- Data quality varies by symbol
- Insufficient historical data for some symbols
- MA configuration incompatible with certain symbols

**Resolution**:
1. Analyze failed symbols for commonalities
2. Improve data validation on client side
3. Document symbol requirements clearly
4. Consider filtering symbols before sending batch

---

## Runbooks

### Runbook 1: DLQ Alert Response

**Trigger**: CloudWatch Alarm "ATC-DLQ-Messages-Detected"

**Priority**: P1 (High)

**Steps**:

1. **Acknowledge Alert** (< 5 minutes)
   ```bash
   # Check DLQ depth
   aws sqs get-queue-attributes \
     --queue-url $DLQ_URL \
     --attribute-names ApproximateNumberOfMessages
   ```

2. **Assess Impact** (< 10 minutes)
   ```bash
   # How many messages?
   # How old are they?
   # What's the failure reason?
   aws sqs receive-message --queue-url $DLQ_URL \
     --message-attribute-names All
   ```

3. **Identify Root Cause** (< 20 minutes)
   - Check Lambda logs for corresponding batch_ids
   - Review recent deployments or config changes
   - Check AWS Service Health Dashboard

4. **Implement Fix** (< 30 minutes)
   - Apply appropriate recovery procedure
   - Test fix with a single message first

5. **Replay Messages** (< 60 minutes)
   - Use redrive or manual replay
   - Monitor processing

6. **Document and Close** (< 90 minutes)
   - Document root cause and fix
   - Update runbook if new scenario
   - Create ticket to prevent recurrence

---

### Runbook 2: High Error Rate Alert

**Trigger**: Error rate > 10% in multiple consecutive batches

**Priority**: P2 (Medium)

**Steps**:

1. **Identify Pattern** (< 10 minutes)
   ```bash
   # Get recent error patterns
   aws logs filter-log-events \
     --log-group-name /aws/lambda/atc-serverless \
     --filter-pattern "{ $.error_count > 0 }" \
     --start-time $(date -u -d '1 hour ago' +%s)000
   ```

2. **Analyze Failed Symbols** (< 20 minutes)
   - Extract error messages
   - Group by error type
   - Identify common characteristics

3. **Determine Action** (< 30 minutes)
   - **Data Quality Issue**: Contact data provider
   - **Configuration Issue**: Review MA configs
   - **System Issue**: Escalate to engineering

4. **Communicate** (< 45 minutes)
   - Notify stakeholders
   - Provide error summary and ETA for fix

5. **Implement Fix** (< 2 hours)
   - Apply appropriate resolution
   - Reprocess failed symbols if needed

6. **Monitor and Close** (< 4 hours)
   - Verify error rate returns to normal
   - Document resolution
   - Update monitoring thresholds if needed

---

### Runbook 3: Lambda Timeout Investigation

**Trigger**: Lambda timeout errors in CloudWatch

**Priority**: P2 (Medium)

**Steps**:

1. **Identify Timeout Events** (< 5 minutes)
   ```bash
   aws logs filter-log-events \
     --log-group-name /aws/lambda/atc-serverless \
     --filter-pattern "Task timed out"
   ```

2. **Analyze Batch Characteristics** (< 15 minutes)
   - Batch size
   - MA configurations
   - Symbol complexity

3. **Check Resource Utilization** (< 20 minutes)
   ```bash
   # Review processing metrics before timeout
   aws logs filter-log-events \
     --log-group-name /aws/lambda/atc-serverless \
     --filter-pattern "{ $.batch_id = \"timeout_batch_id\" }"
   ```

4. **Determine Solution** (< 30 minutes)
   - Reduce batch size
   - Increase timeout
   - Optimize processing (if recurring)

5. **Implement and Verify** (< 60 minutes)
   - Apply configuration change
   - Reprocess timed-out batch
   - Monitor for recurrence

---

## Best Practices

### Error Handling Best Practices

1. **Always Use DLQ**: Configure DLQ for production deployments
2. **Monitor Error Rates**: Set up alerts for abnormal patterns
3. **Log Structured Data**: Use JSON logging for easy querying
4. **Preserve Context**: Include batch_id in all log messages
5. **Test Failure Scenarios**: Regularly test error recovery procedures

### Operational Best Practices

1. **Regular DLQ Checks**: Review DLQ weekly, even if no alerts
2. **Trend Analysis**: Monitor error rates over time for patterns
3. **Documentation**: Keep runbooks updated with new scenarios
4. **Automation**: Automate common recovery procedures
5. **Communication**: Establish escalation paths and notification channels

### Development Best Practices

1. **Fail Fast**: Validate inputs early to avoid wasted processing
2. **Fail Gracefully**: Continue processing even if some symbols fail
3. **Retry Smartly**: Use exponential backoff, not constant retries
4. **Log Comprehensively**: Log all decision points and errors
5. **Test Edge Cases**: Include failure scenarios in test suite

---

## Related Documents

- **CloudWatch Monitoring**: `CLOUDWATCH_MONITORING.md`
- **Performance Tuning**: `PERFORMANCE_TUNING.md` (Phase 7.4)
- **Python Integration**: `PYTHON_INTEGRATION.md`
- **Testing Guide**: `TESTING.md`
- **Deployment Guide**: `../README.md`

---

## Appendix A: Error Codes Reference

| Error Code | Description | Retryable | Recovery |
|------------|-------------|-----------|----------|
| `INPUT_VALIDATION_FAILED` | Invalid request format | No | Fix input data |
| `INSUFFICIENT_DATA` | Not enough OHLCV bars | No | Provide more history |
| `SERIALIZATION_ERROR` | JSON conversion failed | No | Check data format |
| `SQS_TIMEOUT` | SQS request timed out | Yes | Automatic retry |
| `SQS_THROTTLING` | SQS rate limit exceeded | Yes | Automatic retry |
| `IAM_PERMISSION_DENIED` | Missing SQS permissions | No | Update IAM role |
| `MEMORY_EXHAUSTED` | Lambda out of memory | No | Reduce batch size |
| `LAMBDA_TIMEOUT` | Processing took too long | No | Optimize or split |

---

## Appendix B: Sample Error Messages

### Serialization Error
```json
{
  "level": "ERROR",
  "batch_id": "batch_123",
  "error": "Failed to serialize ScanResult - this is a non-retryable error",
  "details": "invalid type: floating point `NaN`, expected f64 at line 1 column 42"
}
```

### SQS Retry Log
```json
{
  "level": "WARN",
  "batch_id": "batch_456",
  "attempt": 2,
  "max_attempts": 3,
  "backoff_ms": 200,
  "error": "RequestTimeout: request timed out",
  "message": "SQS send failed, retrying after backoff"
}
```

### DLQ Send Success
```json
{
  "level": "WARN",
  "batch_id": "batch_789",
  "dlq_url": "https://sqs.us-east-1.amazonaws.com/123456789012/atc-sqs-dlq",
  "message": "Message sent to DLQ after primary queue failures"
}
```

### Memory Warning
```json
{
  "level": "WARN",
  "batch_id": "batch_101",
  "peak_memory_mb": 650,
  "threshold_mb": 512,
  "message": "WARNING: Memory usage exceeds warning threshold"
}
```

---

**Document Version**: 1.0  
**Maintenance**: Update after any changes to error handling logic  
**Owner**: ATC Serverless Team  
**Review Cycle**: Quarterly

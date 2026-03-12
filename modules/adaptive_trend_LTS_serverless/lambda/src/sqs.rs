use atc_serverless::ScanResult;
use aws_sdk_sqs::error::SdkError;
use aws_sdk_sqs::operation::send_message::SendMessageError;
use aws_sdk_sqs::types::MessageAttributeValue;
use aws_sdk_sqs::Client;
use std::collections::HashMap;
use std::time::Duration;
use tracing::{error, info, warn};

/// Maximum number of retry attempts for SQS send operations
const MAX_RETRY_ATTEMPTS: u32 = 3;

/// Initial backoff delay in milliseconds
const INITIAL_BACKOFF_MS: u64 = 100;

/// Maximum backoff delay in milliseconds (5 seconds)
const MAX_BACKOFF_MS: u64 = 5000;

/// Hard limit for an SQS message body in bytes (256 KB)
const MAX_SQS_MESSAGE_BYTES: usize = 256 * 1024;

/// SQS Client with retry logic and error handling
///
/// This client implements exponential backoff retry logic for transient failures
/// and provides comprehensive error logging for debugging.
pub struct SqsClient {
    client: Client,
    queue_url: String,
    /// DLQ URL for messages that fail after all retries (optional)
    dlq_url: Option<String>,
}

fn invalid_input_error(message: impl Into<String>) -> SdkError<SendMessageError> {
    SdkError::construction_failure(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        message.into(),
    ))
}

fn validate_queue_url(queue_url: &str) -> Result<(), SdkError<SendMessageError>> {
    if queue_url.trim().is_empty() {
        return Err(invalid_input_error(
            "SQS queue URL is empty. Set SQS_QUEUE_URL to a valid queue URL.",
        ));
    }
    Ok(())
}

fn validate_message_size(body: &str, batch_id: &str) -> Result<(), SdkError<SendMessageError>> {
    let body_size = body.as_bytes().len();
    if body_size > MAX_SQS_MESSAGE_BYTES {
        error!(
            batch_id = %batch_id,
            body_size_bytes = body_size,
            max_size_bytes = MAX_SQS_MESSAGE_BYTES,
            "Serialized ScanResult exceeds SQS message size limit"
        );
        return Err(invalid_input_error(format!(
            "Serialized ScanResult for batch '{}' is {} bytes, exceeds SQS max {} bytes. Reduce batch size.",
            batch_id, body_size, MAX_SQS_MESSAGE_BYTES
        )));
    }
    Ok(())
}

impl SqsClient {
    /// Create a new SqsClient
    ///
    /// # Arguments
    /// * `client` - AWS SQS SDK client
    /// * `queue_url` - URL of the primary SQS queue
    pub fn new(client: Client, queue_url: String) -> Self {
        Self {
            client,
            queue_url,
            dlq_url: None,
        }
    }

    /// Create a new SqsClient with Dead Letter Queue support
    ///
    /// # Arguments
    /// * `client` - AWS SQS SDK client
    /// * `queue_url` - URL of the primary SQS queue
    /// * `dlq_url` - URL of the Dead Letter Queue
    pub fn new_with_dlq(client: Client, queue_url: String, dlq_url: String) -> Self {
        Self {
            client,
            queue_url,
            dlq_url: Some(dlq_url),
        }
    }

    /// Send scan result to SQS with retry logic
    ///
    /// Implements exponential backoff retry strategy:
    /// - Attempt 1: immediate
    /// - Attempt 2: wait 100ms
    /// - Attempt 3: wait 200ms
    /// - Attempt 4: wait 400ms (final attempt)
    ///
    /// If all retries fail and DLQ is configured, the message is sent to DLQ.
    ///
    /// # Arguments
    /// * `result` - The scan result to send
    ///
    /// # Returns
    /// * `Ok(())` - Message sent successfully (to primary queue or DLQ)
    /// * `Err(SdkError<SendMessageError>)` - All retries failed and no DLQ configured
    ///
    /// # Error Scenarios
    /// 1. **Serialization Failure**: Unable to convert ScanResult to JSON
    ///    - This is a non-retryable error (returns immediately)
    /// 2. **Network/Throttling**: Transient AWS service errors
    ///    - These are retried with exponential backoff
    /// 3. **Permission Errors**: IAM permissions insufficient
    ///    - These are retried but typically fail all attempts
    pub async fn send_scan_result(
        &self,
        result: &ScanResult,
    ) -> Result<(), SdkError<SendMessageError>> {
        validate_queue_url(&self.queue_url)?;

        // Serialize the result (non-retryable error)
        let body = match serde_json::to_string(result) {
            Ok(b) => b,
            Err(e) => {
                error!(
                    batch_id = %result.batch_id,
                    error = %e,
                    "Failed to serialize ScanResult - this is a non-retryable error"
                );
                return Err(SdkError::construction_failure(e));
            }
        };
        validate_message_size(&body, &result.batch_id)?;

        info!(
            batch_id = %result.batch_id,
            queue_url = %self.queue_url,
            "Sending ScanResult to SQS"
        );

        // Try sending with retry logic
        let mut last_error = None;

        for attempt in 1..=MAX_RETRY_ATTEMPTS {
            match self
                .try_send_message(&body, &result.batch_id, attempt)
                .await
            {
                Ok(_) => {
                    if attempt > 1 {
                        info!(
                            batch_id = %result.batch_id,
                            attempt = attempt,
                            "SQS send succeeded after retry"
                        );
                    }
                    return Ok(());
                }
                Err(e) => {
                    last_error = Some(e);

                    if attempt < MAX_RETRY_ATTEMPTS {
                        let backoff_ms = Self::calculate_backoff(attempt);
                        warn!(
                            batch_id = %result.batch_id,
                            attempt = attempt,
                            max_attempts = MAX_RETRY_ATTEMPTS,
                            backoff_ms = backoff_ms,
                            error = %last_error.as_ref().unwrap(),
                            "SQS send failed, retrying after backoff"
                        );
                        tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                    }
                }
            }
        }

        // All retries exhausted - try DLQ if configured
        if let Some(dlq_url) = &self.dlq_url {
            error!(
                batch_id = %result.batch_id,
                attempts = MAX_RETRY_ATTEMPTS,
                error = %last_error.as_ref().unwrap(),
                "All SQS send attempts failed, sending to Dead Letter Queue"
            );

            match self.send_to_dlq(&body, &result.batch_id, dlq_url).await {
                Ok(_) => {
                    warn!(
                        batch_id = %result.batch_id,
                        dlq_url = %dlq_url,
                        "Message sent to DLQ after primary queue failures"
                    );
                    return Ok(()); // Consider DLQ success as overall success
                }
                Err(dlq_error) => {
                    error!(
                        batch_id = %result.batch_id,
                        dlq_error = %dlq_error,
                        "Failed to send to DLQ after primary queue failures"
                    );
                    // Return the original error, not the DLQ error
                    return Err(last_error.unwrap());
                }
            }
        }

        // No DLQ configured - return the last error
        error!(
            batch_id = %result.batch_id,
            attempts = MAX_RETRY_ATTEMPTS,
            error = %last_error.as_ref().unwrap(),
            "All SQS send attempts failed and no DLQ configured"
        );

        Err(last_error.unwrap())
    }

    /// Attempt to send a message to SQS (single attempt, no retry)
    async fn try_send_message(
        &self,
        body: &str,
        batch_id: &str,
        attempt: u32,
    ) -> Result<(), SdkError<SendMessageError>> {
        self.client
            .send_message()
            .queue_url(&self.queue_url)
            .message_body(body)
            .send()
            .await
            .map(|_| ())
            .map_err(|e| {
                // Log detailed error information
                match &e {
                    SdkError::ServiceError(service_err) => {
                        warn!(
                            batch_id = %batch_id,
                            attempt = attempt,
                            error_code = ?service_err.err().meta().code(),
                            error_message = ?service_err.err().meta().message(),
                            "SQS service error"
                        );
                    }
                    SdkError::TimeoutError(_) => {
                        warn!(
                            batch_id = %batch_id,
                            attempt = attempt,
                            "SQS request timeout"
                        );
                    }
                    _ => {
                        warn!(
                            batch_id = %batch_id,
                            attempt = attempt,
                            error = %e,
                            "SQS send error"
                        );
                    }
                }
                e
            })
    }

    /// Send a message to the Dead Letter Queue
    async fn send_to_dlq(
        &self,
        body: &str,
        batch_id: &str,
        dlq_url: &str,
    ) -> Result<(), SdkError<SendMessageError>> {
        info!(
            batch_id = %batch_id,
            dlq_url = %dlq_url,
            "Attempting to send message to DLQ"
        );

        // Create message attributes for DLQ metadata
        let mut attributes = HashMap::new();

        attributes.insert(
            "FailureReason".to_string(),
            MessageAttributeValue::builder()
                .data_type("String")
                .string_value("PrimaryQueueRetriesExhausted")
                .build()
                .map_err(|e| SdkError::construction_failure(e))?,
        );

        attributes.insert(
            "OriginalBatchId".to_string(),
            MessageAttributeValue::builder()
                .data_type("String")
                .string_value(batch_id)
                .build()
                .map_err(|e| SdkError::construction_failure(e))?,
        );

        self.client
            .send_message()
            .queue_url(dlq_url)
            .message_body(body)
            .set_message_attributes(Some(attributes))
            .send()
            .await
            .map(|_| ())
    }

    /// Calculate exponential backoff delay with jitter in milliseconds
    ///
    /// Formula: min(INITIAL_BACKOFF * 2^(attempt-1), MAX_BACKOFF) + random jitter
    /// Jitter prevents thundering-herd when multiple Lambda invocations retry simultaneously.
    fn calculate_backoff(attempt: u32) -> u64 {
        let base_backoff = INITIAL_BACKOFF_MS * 2_u64.pow(attempt - 1);
        let capped = base_backoff.min(MAX_BACKOFF_MS);
        // Add ±25% jitter using a simple deterministic-enough source (no rand crate needed)
        let jitter_range = capped / 4;
        let jitter = if jitter_range > 0 {
            // Use current time nanos as cheap entropy for jitter
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos() as u64;
            nanos % (jitter_range * 2)
        } else {
            0
        };
        capped.saturating_sub(jitter_range).saturating_add(jitter)
    }
}

#[cfg(test)]
mod tests {
    use super::{validate_message_size, validate_queue_url, MAX_SQS_MESSAGE_BYTES};

    #[test]
    fn test_validate_queue_url_rejects_empty() {
        assert!(validate_queue_url("").is_err());
        assert!(validate_queue_url("   ").is_err());
    }

    #[test]
    fn test_validate_queue_url_accepts_non_empty() {
        assert!(validate_queue_url("https://sqs.us-east-1.amazonaws.com/123/queue").is_ok());
    }

    #[test]
    fn test_validate_message_size_accepts_limit_boundary() {
        let body = "a".repeat(MAX_SQS_MESSAGE_BYTES);
        assert!(validate_message_size(&body, "batch-test").is_ok());
    }

    #[test]
    fn test_validate_message_size_rejects_oversized_payload() {
        let body = "a".repeat(MAX_SQS_MESSAGE_BYTES + 1);
        assert!(validate_message_size(&body, "batch-test").is_err());
    }
}

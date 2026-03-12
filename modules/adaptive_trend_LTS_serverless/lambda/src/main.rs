mod handler;
mod sqs;

use crate::handler::handle_request;
use crate::sqs::SqsClient;
use aws_config::BehaviorVersion;
use aws_sdk_sqs::Client;
use lambda_runtime::{run, service_fn, Error};
use std::env;
use std::sync::Arc;

fn require_non_empty_env(var_name: &str, value: Option<String>) -> Result<String, Error> {
    match value {
        Some(v) if !v.trim().is_empty() => Ok(v),
        _ => Err(format!(
            "Missing or empty required environment variable: {}",
            var_name
        )
        .into()),
    }
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .json()
        .init();

    // Initialize AWS clients once
    let config = aws_config::load_defaults(BehaviorVersion::latest()).await;
    let sqs_sdk_client = Client::new(&config);

    // Get queue URLs from environment
    let queue_url = require_non_empty_env("SQS_QUEUE_URL", env::var("SQS_QUEUE_URL").ok())?;
    let dlq_url = env::var("DLQ_URL").ok(); // Optional DLQ

    // Create SQS client with or without DLQ support
    let sqs_client = Arc::new(match dlq_url {
        Some(dlq) if !dlq.is_empty() => {
            tracing::info!(
                queue_url = %queue_url,
                dlq_url = %dlq,
                "Initializing SQS client with Dead Letter Queue support"
            );
            SqsClient::new_with_dlq(sqs_sdk_client, queue_url, dlq)
        }
        _ => {
            tracing::info!(
                queue_url = %queue_url,
                "Initializing SQS client without Dead Letter Queue"
            );
            SqsClient::new(sqs_sdk_client, queue_url)
        }
    });

    let func = service_fn(|event| {
        let client = Arc::clone(&sqs_client);
        async move { handle_request(event, client.as_ref()).await }
    });

    run(func).await
}

#[cfg(test)]
mod tests {
    use super::require_non_empty_env;

    #[test]
    fn test_require_non_empty_env_rejects_missing() {
        let result = require_non_empty_env("SQS_QUEUE_URL", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_require_non_empty_env_rejects_blank() {
        let result = require_non_empty_env("SQS_QUEUE_URL", Some("   ".to_string()));
        assert!(result.is_err());
    }

    #[test]
    fn test_require_non_empty_env_accepts_value() {
        let result = require_non_empty_env(
            "SQS_QUEUE_URL",
            Some("https://sqs.us-east-1.amazonaws.com/123/queue".to_string()),
        );
        assert!(result.is_ok());
    }
}

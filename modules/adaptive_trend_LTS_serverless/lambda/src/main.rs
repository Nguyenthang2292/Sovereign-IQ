mod handler;
mod sqs;

use lambda_runtime::{run, service_fn, Error};
use aws_sdk_sqs::Client;
use aws_config::BehaviorVersion;
use std::env;
use std::sync::Arc;
use crate::handler::handle_request;
use crate::sqs::SqsClient;

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
    let queue_url = env::var("SQS_QUEUE_URL").unwrap_or_default();
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
        async move {
            handle_request(event, client.as_ref()).await
        }
    });
    
    run(func).await
}

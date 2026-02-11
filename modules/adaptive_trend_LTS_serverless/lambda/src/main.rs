mod handler;
mod sqs;

use lambda_runtime::{run, service_fn, Error};
use aws_sdk_sqs::Client;
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
    let config = aws_config::load_from_env().await;
    let sqs_sdk_client = Client::new(&config);
    let queue_url = env::var("SQS_QUEUE_URL").unwrap_or_default();
    
    let sqs_client = Arc::new(SqsClient::new(sqs_sdk_client, queue_url));

    let func = service_fn(|event| {
        let client = Arc::clone(&sqs_client);
        async move {
            handle_request(event, &client).await
        }
    });
    
    run(func).await
}

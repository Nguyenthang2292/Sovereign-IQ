use aws_sdk_sqs::Client;
use aws_sdk_sqs::error::SdkError;
use aws_sdk_sqs::operation::send_message::SendMessageError;
use atc_serverless::ScanResult;
use tracing::{info, error};

pub struct SqsClient {
    client: Client,
    queue_url: String,
}

impl SqsClient {
    pub fn new(client: Client, queue_url: String) -> Self {
        Self { client, queue_url }
    }

    pub async fn send_scan_result(&self, result: &ScanResult) -> Result<(), SdkError<SendMessageError>> {
        if self.queue_url.is_empty() {
            info!("Skipping SQS send: queue_url is empty");
            return Ok(());
        }

        let body = match serde_json::to_string(result) {
            Ok(b) => b,
            Err(e) => {
                error!("Failed to serialize ScanResult: {}", e);
                return Err(SdkError::ConstructionFailure(Box::new(e)));
            }
        };

        info!("Sending ScanResult for batch {} to SQS", result.batch_id);

        self.client
            .send_message()
            .queue_url(&self.queue_url)
            .message_body(body)
            .send()
            .await?;

        Ok(())
    }
}

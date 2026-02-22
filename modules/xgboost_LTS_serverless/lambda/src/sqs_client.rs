use anyhow::Result;
use aws_sdk_sqs::Client;
use serde::Serialize;

pub struct SqsClient {
    client: Client,
}

impl SqsClient {
    pub fn new(config: &aws_config::SdkConfig) -> Self {
        let client = Client::new(config);
        Self { client }
    }

    pub async fn send_json<T: Serialize>(&self, queue_url: &str, payload: &T) -> Result<()> {
        let body = serde_json::to_string(payload)?;
        self.client
            .send_message()
            .queue_url(queue_url)
            .message_body(body)
            .send()
            .await?;
        Ok(())
    }
}

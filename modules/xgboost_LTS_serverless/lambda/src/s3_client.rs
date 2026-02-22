use anyhow::Result;
use aws_sdk_s3::Client;

pub struct S3Client {
    client: Client,
    bucket_name: String,
}

impl S3Client {
    pub fn new(config: &aws_config::SdkConfig, bucket_name: String) -> Self {
        let client = Client::new(config);
        Self {
            client,
            bucket_name,
        }
    }

    pub async fn download_model(&self, key: &str) -> Result<Vec<u8>> {
        let response = self
            .client
            .get_object()
            .bucket(&self.bucket_name)
            .key(key)
            .send()
            .await?;

        let body = response.body;
        let bytes = body.collect().await?;
        Ok(bytes.to_vec())
    }
}

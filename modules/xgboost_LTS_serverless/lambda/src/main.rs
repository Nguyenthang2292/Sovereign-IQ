use lambda_runtime::{run, service_fn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod handler;
mod s3_client;
mod sqs_client;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), lambda_runtime::Error> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    run(service_fn(handler::handle_request)).await
}

mod handler;

use crate::handler::handle_request;
use lambda_runtime::{run, service_fn, Error};

#[tokio::main]
async fn main() -> Result<(), Error> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .json()
        .init();

    run(service_fn(handle_request)).await
}

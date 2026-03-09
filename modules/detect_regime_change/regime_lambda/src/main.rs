mod models;
mod handler;

use lambda_http::{run, service_fn, Body, Error, Request, RequestPayloadExt, Response};
use crate::models::RegimeAnalysisRequest;

async fn function_handler(event: Request) -> Result<Response<Body>, Error> {
    let payload = match event.payload::<RegimeAnalysisRequest>() {
        Ok(Some(req)) => req,
        Ok(None) => return Ok(Response::builder()
            .status(400)
            .body("Empty payload".into())?),
        Err(e) => return Ok(Response::builder()
            .status(400)
            .body(format!("Invalid request: {}", e).into())?),
    };

    let res = handler::process_request(payload);

    let js = serde_json::to_string(&res)?;
    
    Ok(Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .body(js.into())?)
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    run(service_fn(function_handler)).await
}

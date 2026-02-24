#!/usr/bin/env python3
"""
Local OpenAI-compatible proxy server that forwards requests to OpenRouter.
Runs on http://localhost:9000
"""

import logging

import requests
from flask import Flask, Response, jsonify, request

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

# OpenRouter configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = "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"


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """Proxy endpoint for chat completions"""
    try:
        # Get the request data
        data = request.get_json()
        logger.info(f"Received request for model: {data.get('model', 'unknown')}")

        # Prepare headers for OpenRouter
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": request.headers.get("Referer", "http://localhost:9000"),
            "X-Title": "Local Proxy",
        }

        # Check if reasoning is enabled in the request
        if data.get("reasoning") or (data.get("extra_body") and data.get("extra_body", {}).get("reasoning")):
            # Move reasoning config to extra_body if needed
            if "reasoning" in data and "extra_body" not in data:
                data["extra_body"] = {"reasoning": data.pop("reasoning")}

        # Forward the request to OpenRouter
        response = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions", headers=headers, json=data, stream=data.get("stream", False)
        )

        # Handle streaming responses
        if data.get("stream", False):

            def generate():
                for chunk in response.iter_content(chunk_size=None):
                    if chunk:
                        yield chunk

            return Response(generate(), content_type="text/event-stream")

        # Handle non-streaming responses
        response_data = response.json()
        logger.info(f"Response status: {response.status_code}")

        return jsonify(response_data), response.status_code

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": {"message": str(e), "type": "proxy_error", "code": "internal_error"}}), 500


@app.route("/v1/models", methods=["GET"])
def list_models():
    """Proxy endpoint for listing models"""
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        }

        response = requests.get(f"{OPENROUTER_BASE_URL}/models", headers=headers)

        return jsonify(response.json()), response.status_code

    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return jsonify({"error": {"message": str(e), "type": "proxy_error", "code": "internal_error"}}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "openrouter-proxy"}), 200


@app.route("/", methods=["GET"])
def index():
    """Root endpoint with usage information"""
    return jsonify(
        {
            "service": "OpenRouter Local Proxy",
            "version": "1.0.0",
            "endpoints": {"chat_completions": "/v1/chat/completions", "models": "/v1/models", "health": "/health"},
            "usage": {
                "base_url": "http://localhost:9000",
                "example": "Use this as your OpenAI base_url in your client",
            },
        }
    ), 200


if __name__ == "__main__":
    logger.info("Starting OpenRouter proxy server on http://localhost:9000")
    logger.info("Use http://localhost:9000 as your OpenAI API base_url")
    app.run(host="0.0.0.0", port=9000, debug=True)

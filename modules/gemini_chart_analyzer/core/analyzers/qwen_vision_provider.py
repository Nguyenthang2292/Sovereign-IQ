"""Qwen Vision Provider for analyzing chart images."""

import base64
import os
import time
from pathlib import Path
from typing import List, Optional

from modules.common.ui.logging import log_error, log_info, log_warn


class QwenAPIError(Exception):
    """Base error for Qwen API failures."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class QwenVisionProvider:
    """Analyze chart images using Alibaba Qwen Vision API (DashScope)."""

    provider_name: str = "qwen"
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 1  # seconds

    def __init__(self, api_key: Optional[str] = None, models: Optional[List[str]] = None):
        """
        Initialize QwenVisionProvider.

        Args:
            api_key: DashScope API key (if None, will load from config)
            models: Optional explicit list of models to use. If None, fetches from API.
        """
        self._api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self._client = None
        self._models: List[str] = []

        if self._api_key:
            try:
                from openai import OpenAI

                self._client = OpenAI(
                    api_key=self._api_key,
                    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                )
                self._models = models if models else self._load_qwen_models()
            except ImportError as e:
                log_error(f"Failed to initialize OpenAI client for Qwen: {e}")

    def is_available(self) -> bool:
        """Check if provider is configured and available."""
        return bool(self._api_key and self._client)

    def _load_qwen_models(self) -> List[str]:
        """Fetch available Qwen vision models from DashScope API."""
        if not self._api_key:
            return []

        try:
            import requests

            url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models"
            headers = {"Authorization": f"Bearer {self._api_key}"}
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                all_models = response.json().get("data", [])
                vision_models = [
                    m["id"]
                    for m in all_models
                    if "-vl" in m["id"].lower() and "edit" not in m["id"].lower() and "wan" not in m["id"].lower()
                ]
                vision_models.sort()

                # Ensure qwen-vl-max is preferred if available
                if "qwen-vl-max" in vision_models:
                    vision_models.remove("qwen-vl-max")
                    vision_models.insert(0, "qwen-vl-max")

                return vision_models
        except Exception as e:
            log_warn(f"[{self.provider_name}] Failed to load model list: {e}")

        # Static fallback list if API fails
        return ["qwen-vl-max", "qwen-vl-plus"]

    def _encode_image(self, image_path: str) -> str:
        """Read and base64-encode an image to a Data URL format."""
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            ext = os.path.splitext(image_path)[1].lower().strip(".")
            if ext == "jpg":
                ext = "jpeg"
            return f"data:image/{ext};base64,{encoded_string}"

    def _get_prompt(self, symbol: str, timeframe: str, prompt_type: str, custom_prompt: Optional[str]) -> str:
        """Generate prompt for Qwen (reuses prompts from gemini directory)."""
        if prompt_type == "custom" and custom_prompt:
            return custom_prompt

        prompts_dir = Path(__file__).parent.parent / "prompts"

        if prompt_type == "detailed":
            prompt_file = prompts_dir / "detailed.txt"
        elif prompt_type == "simple":
            prompt_file = prompts_dir / "simple.txt"
        elif prompt_type == "batch":
            prompt_file = prompts_dir / "batch.txt"
        else:
            prompt_file = prompts_dir / "default.txt"

        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt_template = f.read()
        except FileNotFoundError:
            log_error(f"Prompt file not found: {prompt_file}")
            return (
                f"Phân tích biểu đồ {symbol} {timeframe} và đưa ra tín hiệu giao dịch "
                "với Entry, Stop Loss, Take Profit."
            )

        return prompt_template.replace("{symbol}", symbol).replace("{timeframe}", timeframe)

    def analyze_chart(
        self,
        image_path: str,
        symbol: str,
        timeframe: str,
        prompt_type: str = "detailed",
        custom_prompt: Optional[str] = None,
    ) -> str:
        """Analyze the chart using Qwen Vision models."""
        if not self.is_available() or self._client is None:
            raise QwenAPIError(f"[{self.provider_name}] Provider is not configured or available.")

        prompt = self._get_prompt(symbol, timeframe, prompt_type, custom_prompt)
        base64_image = self._encode_image(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": base64_image}},
                ],
            }
        ]

        last_error = None
        models_to_try = self._models or ["qwen-vl-max", "qwen-vl-plus"]

        for model in models_to_try:
            log_info(f"[{self.provider_name}] Trying model from downloaded list: {model}")

            for attempt in range(self.MAX_RETRIES):
                try:
                    response = self._client.chat.completions.create(
                        model=model,
                        messages=messages,  # type: ignore
                    )

                    if not response.choices or not response.choices[0].message.content:
                        raise ValueError("Empty or invalid response from Qwen API")

                    log_info(f"[{self.provider_name}] Request succeeded with model: {model}")
                    return response.choices[0].message.content

                except Exception as e:
                    error_message = str(e)
                    error_code = getattr(e, "status_code", getattr(e, "code", None))

                    # Parse common OpenAI error objects if present
                    if error_code is None:
                        if (
                            "429" in error_message
                            or "quota" in error_message.lower()
                            or "rate limit" in error_message.lower()
                        ):
                            error_code = 429
                        elif "503" in error_message or "unavailable" in error_message.lower():
                            error_code = 503

                    last_error = QwenAPIError(f"API error: {error_message}", status_code=error_code)

                    is_rate_limited = error_code == 429 or "quota" in error_message.lower()
                    is_retryable = (
                        error_code in [503]
                        or "overloaded" in error_message.lower()
                        or "unavailable" in error_message.lower()
                    )

                    if is_rate_limited:
                        log_warn(
                            f"[{self.provider_name}] Rate limit / quota exceeded for model {model}: "
                            f"{error_message}. Skipping to next model."
                        )
                        break  # Skip remaining retries for this model
                    elif is_retryable and attempt < self.MAX_RETRIES - 1:
                        wait_time = self.RETRY_DELAY * (2**attempt)
                        log_warn(
                            f"[{self.provider_name}] Retryable error with model {model}, "
                            f"attempt {attempt + 1}/{self.MAX_RETRIES}: {error_message}. "
                            f"Waiting {wait_time}s"
                        )
                        time.sleep(wait_time)
                        continue
                    elif not is_retryable:
                        log_error(f"[{self.provider_name}] Non-retryable error with model {model}: {error_message}.")
                        break
                    else:
                        break  # Max retries reached

        # If we exit the loop and no model succeeded
        if last_error:
            raise last_error

        raise QwenAPIError(f"[{self.provider_name}] Failed to analyze chart using any model in list.")

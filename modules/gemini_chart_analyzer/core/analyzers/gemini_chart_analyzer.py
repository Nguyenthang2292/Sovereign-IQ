"""
Gemini Analyzer for analyzing chart images using Google Gemini API.

Send a chart image to Google Gemini for analysis and receive LONG/SHORT signals with TP/SL.
"""

import os
import time
from pathlib import Path
from typing import List, Optional

import PIL.Image
from PIL.Image import Image as PILImage

# Try importing with the newest syntax first: from google import genai
# Note: There may be a conflict with other google packages, so try direct import first
try:
    # Try importing google.genai directly first (to avoid namespace collisions)
    import google.genai as genai  # type: ignore[import-untyped]  # pyright: ignore[reportMissingTypeStubs]
except ImportError as e1:
    try:
        # Fallback: from google import genai
        from google import genai  # type: ignore[import-untyped]  # pyright: ignore[reportMissingTypeStubs]
    except ImportError as e2:
        try:
            # Fallback: google-generativeai (legacy package)
            import google.generativeai as genai  # type: ignore[import-untyped]  # pyright: ignore[reportMissingTypeStubs]
        except ImportError as e3:
            raise ImportError(
                f"Unable to import Google Generative AI. "
                f"import google.genai error: {e1}. "
                f"from google import genai error: {e2}. "
                f"import google.generativeai error: {e3}. "
                f"Please install: pip install google-genai"
            )

from modules.common.ui.logging import log_error, log_info, log_success, log_warn

from ..exceptions import GeminiAnalysisError
from .components.analyzer_config import GeminiModelType, ImageValidationConfig
from .components.exceptions import (
    GeminiAPIError,
    GeminiAuthenticationError,
    GeminiImageValidationError,
    GeminiInvalidRequestError,
    GeminiModelNotFoundError,
    GeminiQuotaExceededError,
    GeminiRateLimitError,
    GeminiResponseParseError,
)
from .components.token_limit import (
    MAX_TOKENS_PER_REQUEST,
    PROMPT_TOKEN_WARNING_THRESHOLD,
    estimate_token_count,
)


def select_best_model(available_models: Optional[List[str]] = None) -> str:
    """Choose highest priority Gemini model from available list."""
    if available_models is None:
        return GeminiModelType.PRO_31_PREVIEW.name

    if len(available_models) == 0:
        return GeminiModelType.PRO_31_PREVIEW_CUSTOMTOOLS.name

    available_model_types: List[GeminiModelType] = []
    for model_name in available_models:
        model_type = GeminiModelType.from_name(model_name)
        if model_type:
            available_model_types.append(model_type)

    if available_model_types:
        available_model_types.sort(key=lambda m: m.priority)
        return available_model_types[0].name

    return available_models[0]


def validate_image(image_path: str, config: Optional[ImageValidationConfig] = None) -> tuple[bool, Optional[str]]:
    """Validate an image file against the configured limits."""
    if config is None:
        config = ImageValidationConfig()

    if not os.path.exists(image_path):
        return False, f"Image file not found: {image_path}"

    file_ext = os.path.splitext(image_path)[1].upper().lstrip(".")
    if file_ext not in config.supported_formats:
        return False, f"Unsupported image format: {file_ext}. Supported: {config.supported_formats}"

    file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
    if file_size_mb > config.max_file_size_mb:
        return False, f"Image file too large: {file_size_mb:.2f}MB (max: {config.max_file_size_mb}MB)"

    try:
        with PIL.Image.open(image_path) as img:
            width, height = img.size
            if width < config.min_width or height < config.min_height:
                return False, f"Image too small: {width}x{height} (min: {config.min_width}x{config.min_height})"
            if width > config.max_width or height > config.max_height:
                return False, f"Image too large: {width}x{height} (max: {config.max_width}x{config.max_height})"
    except Exception as exc:
        return False, f"Failed to validate image: {exc}"

    return True, None


class GeminiChartAnalyzer:
    """Analyze chart images using Google Gemini AI."""

    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 1  # seconds
    MODEL_LIST_CACHE_TTL_SECONDS: int = 3600
    _MODEL_LIST_CACHE: Optional[List[str]] = None
    _MODEL_LIST_CACHE_AT: float = 0.0

    def __init__(self, api_key: Optional[str] = None, image_config: Optional[ImageValidationConfig] = None):
        """
        Initialize GeminiChartAnalyzer.

        Args:
            api_key: Google Gemini API key (if None, will load from config)
            image_config: Image validation configuration (optional)
        """
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")

        if api_key is None:
            try:
                from config.config_api import get_gemini_api_key

                # Use thread-safe getter function for API key
                api_key = get_gemini_api_key()
            except ImportError:
                raise ValueError(
                    "GEMINI_API_KEY was not found in config.config_api. "
                    "Please add GEMINI_API_KEY to config/config_api.py"
                )

        if not api_key:
            raise ValueError("GEMINI_API_KEY not provided")

        self.image_config = image_config or ImageValidationConfig()

        # Try new API first (google-genai with Client), fallback to old API
        if hasattr(genai, "Client"):
            # Use new Client API of google-genai
            self.client = genai.Client(api_key=api_key)

            # Try to list available models to find the right one (with cache)
            available_models = self._get_cached_available_models()

            # Store available models for fallback filtering
            self._available_models = available_models

            # Use helper function for consistent model selection
            self.model_name = select_best_model(available_models)

            self.use_new_api = True
        elif hasattr(genai, "configure"):
            # Fallback to old API (google-generativeai)
            genai.configure(api_key=api_key)  # type: ignore

            # Use the same selection logic (no available_models for legacy)
            # Helper will return default model
            selected_model = select_best_model(None)

            # Legacy API may need the format without 'models/' prefix
            if selected_model.startswith("models/"):
                model_for_legacy = selected_model.replace("models/", "")
            else:
                model_for_legacy = selected_model

            # If the model doesn't exist in legacy, fallback to gemini-2.5-pro
            # (legacy likely doesn't support flash models)
            if "flash" in model_for_legacy.lower():
                # Try flash first; if not possible, fallback
                try:
                    self.model = genai.GenerativeModel(model_for_legacy)  # type: ignore
                    # Store the model name for fallback logic
                    self.model_name = selected_model  # Keep the format with 'models/' prefix
                except Exception as e:
                    # Log exception and context before falling back
                    log_error(
                        f"Failed to initialize flash model '{model_for_legacy}' for legacy API: {e}. "
                        f"Falling back to 'gemini-2.5-pro'"
                    )
                    # Fall back to pro if flash is not available
                    self.model = genai.GenerativeModel("gemini-2.5-pro")  # type: ignore
                    self.model_name = "models/gemini-2.5-pro"
            else:
                self.model = genai.GenerativeModel(model_for_legacy)  # type: ignore
                # Store the model name for fallback logic
                self.model_name = selected_model  # Keep the format with 'models/' prefix

            self.use_new_api = False
        else:
            raise AttributeError("genai module is missing Client, configure, or GenerativeModel")

        log_success("Successfully connected to the Google Gemini API")

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        name = model_name.strip()
        if not name:
            return ""
        return name if name.startswith("models/") else f"models/{name}"

    @staticmethod
    def _is_vision_candidate(model_name: str) -> bool:
        """Heuristic filter for Gemini multimodal chart-analysis models."""
        normalized = model_name.lower()
        if "embedding" in normalized or "imagen" in normalized or "tts" in normalized:
            return False
        return "gemini" in normalized and ("flash" in normalized or "pro" in normalized)

    @staticmethod
    def _supports_generate_content(model_obj: object) -> bool:
        """Best-effort capability check across SDK/REST field naming variants."""
        for field_name in ("supported_actions", "supportedGenerationMethods", "supported_generation_methods"):
            actions = getattr(model_obj, field_name, None)
            if not actions:
                continue
            lowered = [str(action).lower() for action in actions]
            if any("generatecontent" in action for action in lowered):
                return True
        # If capability fields are missing, keep backward compatibility by allowing name-based filtering.
        return True

    def _fetch_available_models_from_api(self) -> Optional[List[str]]:
        """Fetch Gemini model list from Google API and keep only vision-capable candidates."""
        models = self.client.models.list()
        filtered_models: List[str] = []
        seen: set[str] = set()

        for model_obj in models:
            model_name_raw = getattr(model_obj, "name", None)
            if not isinstance(model_name_raw, str):
                continue
            model_name = self._normalize_model_name(model_name_raw)
            if not model_name:
                continue
            if not self._supports_generate_content(model_obj):
                continue
            if not self._is_vision_candidate(model_name):
                continue
            if model_name in seen:
                continue
            filtered_models.append(model_name)
            seen.add(model_name)

        return filtered_models or None

    def _get_cached_available_models(self) -> Optional[List[str]]:
        """Return cached model list if fresh; otherwise refresh from API."""
        now = time.time()
        cache_age = now - self.__class__._MODEL_LIST_CACHE_AT
        if self.__class__._MODEL_LIST_CACHE and cache_age < self.MODEL_LIST_CACHE_TTL_SECONDS:
            return list(self.__class__._MODEL_LIST_CACHE)

        try:
            models = self._fetch_available_models_from_api()
            if models:
                self.__class__._MODEL_LIST_CACHE = list(models)
                self.__class__._MODEL_LIST_CACHE_AT = now
                log_info(f"Loaded {len(models)} Gemini models from API list (cached)")
            return models
        except Exception as e:
            log_error(f"Failed to list available Gemini models: {e}, falling back to defaults")
            return list(self.__class__._MODEL_LIST_CACHE) if self.__class__._MODEL_LIST_CACHE else None

    def _build_model_chain(self, current_model: Optional[str]) -> List[str]:
        """Build model try-order using dynamic API list first, then safe static fallbacks."""
        fallback_models: List[str] = []
        available = getattr(self, "_available_models", None)

        if current_model:
            model_type = GeminiModelType.from_name(current_model)
            if model_type:
                fallback_models = [
                    m.name
                    for m in GeminiModelType.get_fallback_models(model_type)
                    if available is None or m.name in available
                ]

        # Prefer dynamic model list order from API.
        if available:
            for dynamic_model in available:
                if dynamic_model != current_model and dynamic_model not in fallback_models:
                    fallback_models.append(dynamic_model)

        # Keep static emergency fallbacks to preserve reliability if API list is incomplete.
        stable_fallbacks = GeminiModelType.get_rate_limit_fallbacks()
        for stable in stable_fallbacks:
            if stable != current_model and stable not in fallback_models:
                fallback_models.append(stable)

        if not current_model:
            return []

        models_to_try: List[str] = [current_model]
        models_to_try.extend(fallback_models)
        return models_to_try

    def validate_image(self, image_path: str) -> tuple[bool, Optional[str]]:
        """Validate image file using analyzer's configured validation rules."""
        return validate_image(image_path, self.image_config)

    def _resolve_current_model(self) -> Optional[str]:
        """
        Resolve the current model name from various possible attributes.

        This method handles:
        - Inspecting self.model_name, self.model, self.model_id
        - Handling GenerativeModel objects via model_name attribute or str()
        - Guarding against empty/placeholder strings
        - Applying fallback to getattr(self,'model_name',None) or 'models/gemini-2.5-pro'

        Returns:
            Normalized model name as string, or None if no valid model found
        """
        # Check both new-API and legacy attributes (model_name, model, model_id)
        current_model = (
            getattr(self, "model_name", None) or getattr(self, "model", None) or getattr(self, "model_id", None)
        )

        # If current_model is a GenerativeModel object (legacy API), try to get its name
        if current_model and not isinstance(current_model, str):
            # Try to extract model name from GenerativeModel object
            if hasattr(current_model, "model_name"):
                current_model = current_model.model_name
            else:
                # Fallback to string conversion, then to model_name attribute
                str_model = str(current_model)
                # If string conversion is empty or not valid, use fallback
                if str_model and str_model.strip() and not str_model.startswith("<"):
                    current_model = str_model
                else:
                    current_model = getattr(self, "model_name", None)

        # Ensure current_model is a string when finished (and not an empty string)
        if current_model:
            if not isinstance(current_model, str):
                current_model = str(current_model)
            # If it's an empty string after converting, use fallback
            if not current_model.strip():
                current_model = getattr(self, "model_name", None) or "models/gemini-2.5-pro"

        return current_model

    def _call_model_with_retries(self, prompt: str, image: PILImage) -> str:
        """
        Protected helper method to call Gemini API with retry logic and fallback models.

        This method handles:
        - Retry logic with exponential backoff
        - Fallback model selection
        - Response extraction from different API formats

        Args:
            prompt: The prompt text to send to the model
            image: PIL Image object to analyze

        Returns:
            Response text from Gemini

        Raises:
            Exception: If all retries and fallback models are exhausted
        """
        max_retries = self.MAX_RETRIES
        retry_delay = self.RETRY_DELAY  # seconds
        last_error: Optional[GeminiAPIError] = None
        response = None

        # Get current model and fallback models
        current_model = self._resolve_current_model()
        built_model_chain = self._build_model_chain(current_model)
        models_to_try: list[str | None] = []
        models_to_try.extend(built_model_chain)
        if not models_to_try:
            models_to_try.append(None)

        for model_idx, model_to_use in enumerate(models_to_try):
            if model_to_use is None and getattr(self, "use_new_api", False):
                continue  # Skip None models for new API

            # Log which model is being attempted before the inner retry loop starts
            model_identifier = model_to_use if model_to_use else (getattr(self, "model_name", None) or "primary")
            log_info(f"Trying model: {model_identifier}")

            for attempt in range(max_retries):
                try:
                    if getattr(self, "use_new_api", False):
                        # New API (google-genai with Client)
                        # Try with PIL Image directly (simpler and works according to logs)
                        model_name = model_to_use if model_to_use else self.model_name
                        response = self.client.models.generate_content(model=model_name, contents=[prompt, image])
                    else:
                        # Legacy API (google-generativeai with GenerativeModel)
                        # For legacy, we need to recreate the model if using fallback
                        if model_idx > 0 and model_to_use:
                            # Try fallback model for legacy API
                            if model_to_use.startswith("models/"):
                                model_for_legacy = model_to_use.replace("models/", "")
                            else:
                                model_for_legacy = model_to_use
                            try:
                                fallback_model = genai.GenerativeModel(model_for_legacy)  # type: ignore
                                response = fallback_model.generate_content([prompt, image])
                            except Exception as e:
                                # Log exception with stack trace before continuing to next model
                                log_error(
                                    f"Failed to create fallback model '{model_for_legacy}': {e}. "
                                    f"Switching to next model"
                                )
                                # If fallback model creation fails, continue to next model
                                break
                        else:
                            response = self.model.generate_content([prompt, image])

                    # Log success when a request returns
                    log_info(f"Request succeeded with model: {model_identifier}")
                    break  # Success, exit retry loop

                except Exception as e:
                    # Extract error code and message
                    error_code = getattr(e, "status_code", getattr(e, "code", None))
                    error_message = str(e)

                    if error_code is None and ("503" in error_message or "UNAVAILABLE" in error_message):
                        error_code = 503

                    # Map to custom exceptions
                    if error_code in [401, 403] or "permission" in error_message.lower():
                        last_error = GeminiAuthenticationError(f"Authentication failed: {error_message}")
                    elif error_code == 404 or "not found" in error_message.lower():
                        last_error = GeminiModelNotFoundError(f"Model not found: {error_message}")
                    elif error_code == 429:
                        if "quota" in error_message.lower():
                            last_error = GeminiQuotaExceededError(f"Quota exceeded: {error_message}")
                        else:
                            last_error = GeminiRateLimitError(f"Rate limit exceeded: {error_message}")
                    elif error_code == 400:
                        last_error = GeminiInvalidRequestError(f"Invalid request: {error_message}")
                    else:
                        last_error = GeminiAPIError(f"API error ({error_code}): {error_message}")

                    # Check if error is retryable (503, or network/overloaded errors only)
                    # 429 is handled separately via is_rate_limited above.
                    is_rate_limited = error_code == 429 or "quota" in error_message.lower()
                    is_retryable = (
                        error_code in [503]
                        or "overloaded" in error_message.lower()
                        or "unavailable" in error_message.lower()
                    )

                    if is_rate_limited:
                        # 429 / quota: do NOT retry same model — jump to next immediately.
                        log_warn(
                            f"Rate limit / quota exceeded for model {model_identifier} "
                            f"(attempt {attempt + 1}/{max_retries}): {error_message}. "
                            f"Switching to next model immediately."
                        )
                        break  # Skip remaining retries for this model
                    elif is_retryable and attempt < max_retries - 1:
                        # Exponential backoff for transient errors (503, overloaded)
                        wait_time = retry_delay * (2**attempt)
                        log_warn(
                            f"Retryable error with model {model_identifier}, "
                            f"attempt {attempt + 1}/{max_retries}: {error_message}. "
                            f"Waiting {wait_time}s before retrying"
                        )
                        time.sleep(wait_time)
                        continue
                    elif not is_retryable:
                        # Non-retryable error, try next model
                        log_error(
                            f"Non-retryable error with model {model_identifier}: {error_message}. "
                            f"Switching to next model"
                        )
                        break  # Exit retry loop, try next model
                    else:
                        # Max retries reached for this model, try next model
                        if model_idx < len(models_to_try) - 1:
                            log_warn(
                                f"Max retries ({max_retries}) reached with model {model_identifier}: {error_message}. "
                                f"Falling back to the next model"
                            )
                            break  # Try next model
                        else:
                            # All models exhausted
                            log_error(
                                f"Max retries ({max_retries}) reached with model {model_identifier}: {error_message}. "
                                f"All models have been tried"
                            )
                            if last_error is not None:
                                raise last_error
                            raise GeminiAPIError("Failed after retries with no captured exception")

            if response is not None:
                break  # Success, exit model loop

        if response is None:
            # All models and retries exhausted
            if last_error:
                raise last_error
            raise GeminiAPIError("Failed to get response from any model after all retries.")

        try:
            # Extract text from response
            if hasattr(response, "text"):
                res_text = response.text
                return str(res_text) if res_text is not None else ""

            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and candidate.content:
                    content = candidate.content
                    if hasattr(content, "parts") and content.parts:
                        # Extract text from parts, ensuring they are strings
                        parts = []
                        for part in content.parts:
                            part_text = getattr(part, "text", None)
                            if part_text:
                                parts.append(str(part_text))
                        if parts:
                            return "".join(parts)

                    content_text = getattr(content, "text", None)
                    if content_text:
                        return str(content_text)

                    return str(content)

            return str(response)
        except Exception as e:
            log_error(f"Error parsing Gemini response: {e}")
            raise GeminiResponseParseError(f"Failed to parse Gemini response: {e}")

    def analyze_chart(
        self,
        image_path: str,
        symbol: str,
        timeframe: str,
        prompt_type: str = "detailed",
        custom_prompt: Optional[str] = None,
    ) -> str:
        """
        Analyze the chart and return the result from Gemini.

        Args:
            image_path: Path to the chart image file
            symbol: Symbol name (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h', '4h')
            prompt_type: Type of prompt ('detailed', 'simple', 'custom')
            custom_prompt: Custom prompt (if prompt_type='custom')

        Returns:
            Analysis result from Gemini
        """
        # Validate image first
        is_valid, validation_error = validate_image(image_path, self.image_config)
        if not is_valid:
            raise GeminiImageValidationError(validation_error)

        # Select prompt
        prompt = self._get_prompt(symbol, timeframe, prompt_type, custom_prompt)

        # Check token count
        token_count = estimate_token_count(prompt)
        if token_count > MAX_TOKENS_PER_REQUEST:
            raise GeminiInvalidRequestError(
                f"Prompt is too long (estimated {token_count} tokens). Max allowed is {MAX_TOKENS_PER_REQUEST} tokens."
            )
        elif token_count > (MAX_TOKENS_PER_REQUEST * PROMPT_TOKEN_WARNING_THRESHOLD):
            log_warn(f"Prompt is relatively long (estimated {token_count} tokens).")

        log_info("Sending image to Google Gemini for analysis...")
        log_info(f"Symbol: {symbol}, Timeframe: {timeframe}")

        try:
            # Open the image with a context manager to ensure the file is closed automatically
            with PIL.Image.open(image_path) as img:
                # Copy the image for use after the file is closed
                img_copy = img.copy()

            # Use helper method to call API with retry logic and fallback models
            result = self._call_model_with_retries(prompt, img_copy)

            log_success("Successfully received analysis result from Gemini")

            return result

        except Exception as e:
            log_error(f"Error while analyzing chart {symbol}")
            raise GeminiAnalysisError(f"Failed to analyze chart {symbol} on {timeframe}") from e

    def _get_prompt(self, symbol: str, timeframe: str, prompt_type: str, custom_prompt: Optional[str]) -> str:
        """Generate prompt for Gemini."""

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
            # Default prompt
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

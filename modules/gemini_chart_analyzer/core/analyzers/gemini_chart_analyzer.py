
from typing import Optional
import time

from PIL.Image import Image as PILImage
import PIL.Image
import PIL.Image

"""
Gemini Analyzer for analyzing chart images using Google Gemini API.

Send a chart image to Google Gemini for analysis and receive LONG/SHORT signals with TP/SL.
"""


# Try importing with the newest syntax first: from google import genai
# Note: There may be a conflict with other google packages, so try direct import first
try:
    # Try importing google.genai directly first (to avoid namespace collisions)
    import google.genai as genai
except ImportError as e1:
    try:
        # Fallback: from google import genai
        from google import genai
    except ImportError as e2:
        try:
            # Fallback: google-generativeai (legacy package)
            import google.generativeai as genai  # pyright: ignore[reportMissingImports]
        except ImportError as e3:
            raise ImportError(
                f"Unable to import Google Generative AI. "
                f"import google.genai error: {e1}. "
                f"from google import genai error: {e2}. "
                f"import google.generativeai error: {e3}. "
                f"Please install: pip install google-genai"
            )

from modules.common.ui.logging import log_info, log_error, log_success, log_warn
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
from .components.helpers import select_best_model, validate_image
from .components.image_config import ImageValidationConfig
from .components.model_config import GeminiModelType
from .components.token_limit import (
    MAX_TOKENS_PER_REQUEST,
    PROMPT_TOKEN_WARNING_THRESHOLD,
    estimate_token_count,
)

class GeminiChartAnalyzer:
    """Analyze chart images using Google Gemini AI."""
    def __init__(self, api_key: Optional[str] = None, image_config: Optional[ImageValidationConfig] = None):
        """
        Initialize GeminiChartAnalyzer.

        Args:
            api_key: Google Gemini API key (if None, will load from config)
            image_config: Image validation configuration (optional)
        """
        if api_key is None:
            try:
                from config.config_api import GEMINI_API_KEY
                api_key = GEMINI_API_KEY
            except ImportError:
                raise ValueError(
                    "GEMINI_API_KEY was not found in config.config_api. "
                    "Please add GEMINI_API_KEY to config/config_api.py"
                )

        if not api_key:
            raise ValueError("GEMINI_API_KEY not provided")

        self.image_config = image_config or ImageValidationConfig()


        # Try new API first (google-genai with Client), fallback to old API
        if hasattr(genai, 'Client'):
            # Use new Client API of google-genai
            self.client = genai.Client(api_key=api_key)

            # Try to list available models to find the right one
            available_models = None
            try:
                models = self.client.models.list()
                available_models = [
                    str(m.name) for m in models
                    if m.name and ('flash' in m.name.lower() or 'pro' in m.name.lower())
                ]
                # Sort to prioritize models using the same logic as select_best_model
                # This ensures consistency between sorting and selection
            except Exception:
                # If unable to list models, will use default via helper
                log_error("Failed to list available Gemini models, falling back to default model")

            # Use helper function for consistent model selection
            self.model_name = select_best_model(available_models)

            self.use_new_api = True
        elif hasattr(genai, 'configure'):
            # Fallback to old API (google-generativeai)
            genai.configure(api_key=api_key)  # type: ignore

            # Use the same selection logic (no available_models for legacy)
            # Helper will return default model
            selected_model = select_best_model(None)

            # Legacy API may need the format without 'models/' prefix
            model_for_legacy = selected_model.replace('models/', '') if selected_model.startswith('models/') else selected_model

            # If the model doesn't exist in legacy, fallback to gemini-1.5-pro
            # (legacy likely doesn't support flash models)
            if 'flash' in model_for_legacy.lower():
                # Try flash first; if not possible, fallback
                try:
                    self.model = genai.GenerativeModel(model_for_legacy)  # type: ignore
                    # Store the model name for fallback logic
                    self.model_name = selected_model  # Keep the format with 'models/' prefix
                except Exception as e:
                    # Log exception and context before falling back
                    log_error(
                        f"Failed to initialize flash model '{model_for_legacy}' for legacy API. "
                        f"Falling back to 'gemini-1.5-pro'"
                    )
                    # Fall back to pro if flash is not available
                    self.model = genai.GenerativeModel('gemini-1.5-pro')  # type: ignore
                    self.model_name = 'models/gemini-1.5-pro'
            else:
                self.model = genai.GenerativeModel(model_for_legacy)  # type: ignore
                # Store the model name for fallback logic
                self.model_name = selected_model  # Keep the format with 'models/' prefix

            self.use_new_api = False
        else:
            raise AttributeError("genai module is missing Client, configure, or GenerativeModel")

        log_success("Successfully connected to the Google Gemini API")

    def _resolve_current_model(self) -> Optional[str]:
        """
        Resolve the current model name from various possible attributes.

        This method handles:
        - Inspecting self.model_name, self.model, self.model_id
        - Handling GenerativeModel objects via model_name attribute or str()
        - Guarding against empty/placeholder strings
        - Applying fallback to getattr(self,'model_name',None) or 'models/gemini-1.5-pro'

        Returns:
            Normalized model name as string, or None if no valid model found
        """
        # Check both new-API and legacy attributes (model_name, model, model_id)
        current_model = (
            getattr(self, 'model_name', None) or
            getattr(self, 'model', None) or
            getattr(self, 'model_id', None)
        )

        # If current_model is a GenerativeModel object (legacy API), try to get its name
        if current_model and not isinstance(current_model, str):
            # Try to extract model name from GenerativeModel object
            if hasattr(current_model, 'model_name'):
                current_model = current_model.model_name
            else:
                # Fallback to string conversion, then to model_name attribute
                str_model = str(current_model)
                # If string conversion is empty or not valid, use fallback
                if str_model and str_model.strip() and not str_model.startswith('<'):
                    current_model = str_model
                else:
                    current_model = getattr(self, 'model_name', None)

        # Ensure current_model is a string when finished (and not an empty string)
        if current_model:
            if not isinstance(current_model, str):
                current_model = str(current_model)
            # If it's an empty string after converting, use fallback
            if not current_model.strip():
                current_model = getattr(self, 'model_name', None) or 'models/gemini-1.5-pro'

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
        max_retries = 3
        retry_delay = 1  # seconds
        last_error = None
        response = None

        # Get current model and fallback models
        current_model = self._resolve_current_model()
        fallback_models = []
        if current_model:
            model_type = GeminiModelType.from_name(current_model)
            if model_type:
                fallback_models = [m.name for m in GeminiModelType.get_fallback_models(model_type)]

        models_to_try = [current_model] + fallback_models if current_model else [None]

        for model_idx, model_to_use in enumerate(models_to_try):
            if model_to_use is None and getattr(self, 'use_new_api', False):
                continue  # Skip None models for new API

            # Log which model is being attempted before the inner retry loop starts
            model_identifier = model_to_use if model_to_use else (getattr(self, 'model_name', None) or "primary")
            log_info(f"Trying model: {model_identifier}")

            for attempt in range(max_retries):
                try:
                    if getattr(self, 'use_new_api', False):
                        # New API (google-genai with Client)
                        # Try with PIL Image directly (simpler and works according to logs)
                        model_name = model_to_use if model_to_use else self.model_name
                        response = self.client.models.generate_content(
                            model=model_name,
                            contents=[prompt, image]
                        )
                    else:
                        # Legacy API (google-generativeai with GenerativeModel)
                        # For legacy, we need to recreate the model if using fallback
                        if model_idx > 0 and model_to_use:
                            # Try fallback model for legacy API
                            model_for_legacy = model_to_use.replace('models/', '') if model_to_use.startswith('models/') else model_to_use
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
                    error_code = getattr(e, 'status_code', getattr(e, 'code', None))
                    error_message = str(e)

                    if error_code is None and ('503' in error_message or 'UNAVAILABLE' in error_message):
                        error_code = 503

                    # Map to custom exceptions
                    if error_code in [401, 403] or 'permission' in error_message.lower():
                        last_error = GeminiAuthenticationError(f"Authentication failed: {error_message}")
                    elif error_code == 404 or 'not found' in error_message.lower():
                        last_error = GeminiModelNotFoundError(f"Model not found: {error_message}")
                    elif error_code == 429:
                        if 'quota' in error_message.lower():
                            last_error = GeminiQuotaExceededError(f"Quota exceeded: {error_message}")
                        else:
                            last_error = GeminiRateLimitError(f"Rate limit exceeded: {error_message}")
                    elif error_code == 400:
                        last_error = GeminiInvalidRequestError(f"Invalid request: {error_message}")
                    else:
                        last_error = GeminiAPIError(f"API error ({error_code}): {error_message}")

                    # Check if error is retryable (503, 429, or network errors)
                    is_retryable = (
                        error_code in [503, 429] or
                        'overloaded' in error_message.lower() or
                        'rate limit' in error_message.lower() or
                        'unavailable' in error_message.lower()
                    )

                    if is_retryable and attempt < max_retries - 1:
                        # Exponential backoff
                        wait_time = retry_delay * (2 ** attempt)
                        # Log retryable errors with attempt number and computed wait_time before sleeping
                        log_warn(
                            f"Retryable error with model {model_identifier}, "
                            f"attempt {attempt + 1}/{max_retries}: {error_message}. "
                            f"Waiting {wait_time}s before retrying"
                        )
                        time.sleep(wait_time)
                        continue
                    elif not is_retryable:
                        # Non-retryable error, try next model
                        # Log non-retryable errors with the model name when breaking to the next model
                        log_error(
                            f"Non-retryable error with model {model_identifier}: {error_message}. "
                            f"Switching to next model"
                        )
                        break  # Exit retry loop, try next model
                    else:
                        # Max retries reached for this model, try next model
                        # Log when max retries are reached and the code is falling back to the next model
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
                            raise last_error

            if response is not None:
                break  # Success, exit model loop

        if response is None:
            # All models and retries exhausted
            if last_error:
                raise last_error
            raise GeminiAPIError("Failed to get response from any model after all retries.")

        try:
            # Extract text from response
            if hasattr(response, 'text'):
                res_text = response.text
                return str(res_text) if res_text is not None else ""

            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    content = candidate.content
                    if hasattr(content, 'parts') and content.parts:
                        # Extract text from parts, ensuring they are strings
                        parts = []
                        for part in content.parts:
                            part_text = getattr(part, 'text', None)
                            if part_text:
                                parts.append(str(part_text))
                        if parts:
                            return "".join(parts)

                    content_text = getattr(content, 'text', None)
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
        custom_prompt: Optional[str] = None
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
                f"Prompt is too long (estimated {token_count} tokens). "
                f"Max allowed is {MAX_TOKENS_PER_REQUEST} tokens."
            )
        elif token_count > (MAX_TOKENS_PER_REQUEST * PROMPT_TOKEN_WARNING_THRESHOLD):
            log_warn(f"Prompt is relatively long (estimated {token_count} tokens).")

        log_info(f"Sending image to Google Gemini for analysis...")
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
            log_error(f"Error while analyzing chart: {e}")
            raise

    def _get_prompt(
        self,
        symbol: str,
        timeframe: str,
        prompt_type: str,
        custom_prompt: Optional[str]
    ) -> str:
        """Generate prompt for Gemini."""

        if prompt_type == "custom" and custom_prompt:
            return custom_prompt

        if prompt_type == "detailed":
            return f"""Bạn là một chuyên gia phân tích kỹ thuật tài chính cao cấp. Nhiệm vụ của bạn là nhận diện các mẫu hình giá, các vùng hỗ trợ/kháng cự và các chỉ báo kỹ thuật trên hình ảnh biểu đồ được cung cấp.

Biểu đồ này là của {symbol} trên khung thời gian {timeframe}.

Hãy phân tích biểu đồ và phản hồi theo cấu trúc sau:

1. **Xu hướng chính**: Mô tả xu hướng hiện tại (tăng, giảm, hoặc đi ngang) và độ mạnh của xu hướng.

2. **Các mẫu hình nhận diện được**:
   - Các mẫu hình nến (candlestick patterns) quan trọng
   - Các mẫu hình giá (chart patterns) như triangle, head and shoulders, double top/bottom, etc.
   - Các vùng hỗ trợ và kháng cự quan trọng

3. **Phân tích Indicators**:
   - RSI: Xác định vùng overbought (>70) hoặc oversold (<30), tìm phân kỳ nếu có
   - MACD: Phân tích tín hiệu giao cắt và histogram
   - Moving Averages: Xác định vị trí giá so với các MA (20, 50, 200)
   - Bollinger Bands: Xác định vùng giá có thể bật lại hoặc breakout
   - Volume: Phân tích volume để xác nhận xu hướng

4. **Điểm vào lệnh/cắt lỗ tham khảo**:
   - **LONG**: Nếu có tín hiệu mua, đề xuất:
     * Entry: Giá vào lệnh
     * Stop Loss: Giá cắt lỗ
     * Take Profit 1: Mục tiêu lợi nhuận đầu tiên
     * Take Profit 2: Mục tiêu lợi nhuận thứ hai (nếu có)
   - **SHORT**: Nếu có tín hiệu bán, đề xuất:
     * Entry: Giá vào lệnh
     * Stop Loss: Giá cắt lỗ
     * Take Profit 1: Mục tiêu lợi nhuận đầu tiên
     * Take Profit 2: Mục tiêu lợi nhuận thứ hai (nếu có)
   - **KHÔNG GIAO DỊCH**: Nếu không có tín hiệu rõ ràng, hãy nêu lý do

5. **Cảnh báo rủi ro**:
   - Các yếu tố rủi ro cần lưu ý
   - Độ tin cậy của tín hiệu (cao/trung bình/thấp)
   - Điều kiện để tín hiệu bị vô hiệu hóa

6. **Khu vực thanh khoản (Liquidity)**:
   - Xác định các vùng thanh khoản quan trọng (liquidity zones)
   - Các vùng có thể có stop loss hunting

Hãy phân tích chi tiết và đưa ra các con số cụ thể về giá khi có thể."""

        elif prompt_type == "simple":
            return f"""Hãy phân tích biểu đồ {symbol} khung {timeframe} này.

Cho tôi biết:
1. Giá đang nằm trong mô hình gì?
2. Có dấu hiệu phân kỳ RSI nào không?
3. Hãy khoanh vùng các khu vực thanh khoản (Liquidity) quan trọng.
4. Đưa ra tín hiệu LONG hoặc SHORT với Entry, Stop Loss và Take Profit nếu có cơ hội giao dịch."""

        else:
            # Default prompt
            return f"""Phân tích biểu đồ {symbol} {timeframe} và đưa ra tín hiệu giao dịch với Entry, Stop Loss, Take Profit."""
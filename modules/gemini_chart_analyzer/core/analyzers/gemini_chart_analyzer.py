"""
Gemini Analyzer for analyzing chart images using Google Gemini API.

Send a chart image to Google Gemini for analysis and receive LONG/SHORT signals with TP/SL.
"""

import os
import time
import logging
import re
from typing import Optional
import PIL.Image
from PIL.Image import Image as PILImage

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

from modules.common.ui.logging import log_info, log_error, log_success

logger = logging.getLogger(__name__)


def _get_fallback_models(primary_model: str) -> list:
    """
    Get list of fallback models in order of preference.
    
    Args:
        primary_model: The primary model that failed
        
    Returns:
        List of fallback model names (excluding the primary_model)
    """
    fallbacks = []
    
    # If primary is flash, try other flash models then pro
    if 'flash' in primary_model.lower():
        if '3' in primary_model:
            # If primary is gemini-3-flash, try 2.5-flash then 1.5-flash then pro
            fallbacks.extend([
                'models/gemini-2.5-flash',
                'models/gemini-1.5-flash',
                'models/gemini-1.5-pro'
            ])
        elif '2.5' in primary_model:
            # If primary is gemini-2.5-flash, try 1.5-flash then pro
            fallbacks.extend([
                'models/gemini-1.5-flash',
                'models/gemini-1.5-pro'
            ])
        else:
            # Try newer flash then pro
            fallbacks.extend([
                'models/gemini-2.5-flash',
                'models/gemini-1.5-pro'
            ])
    else:
        # If primary is pro, try flash models (newer first)
        fallbacks.extend([
            'models/gemini-3-flash',
            'models/gemini-2.5-flash',
            'models/gemini-1.5-flash'
        ])
    
    # Filter out any entries that equal the primary model (case-insensitive, trimmed)
    primary_normalized = primary_model.strip().lower()
    filtered_fallbacks = []
    seen = set()
    
    for fallback in fallbacks:
        fallback_normalized = fallback.strip().lower()
        # Skip if it matches primary model or if we've already seen it (preserve order)
        if fallback_normalized != primary_normalized and fallback_normalized not in seen:
            filtered_fallbacks.append(fallback)
            seen.add(fallback_normalized)
    
    return filtered_fallbacks


def _normalize_and_tokenize(model_name: str) -> list[str]:
    """
    Normalize model name to lowercase and split into tokens on non-alphanumeric characters.
    
    Args:
        model_name: Model name string (e.g., "models/gemini-3-flash")
        
    Returns:
        List of tokens (e.g., ["models", "gemini", "3", "flash"])
    """
    normalized = model_name.lower()
    # Split on non-alphanumeric characters
    tokens = re.split(r'[^a-z0-9]+', normalized)
    # Filter out empty strings
    return [token for token in tokens if token]


def _is_flash_model(tokens: list[str]) -> bool:
    """
    Check if tokens represent a flash model (gemini-3-flash or gemini-2.5-flash).
    
    Args:
        tokens: List of normalized tokens from model name
        
    Returns:
        True if tokens match gemini-3-flash or gemini-2.5-flash pattern
    """
    if 'gemini' not in tokens or 'flash' not in tokens:
        return False
    
    try:
        gemini_idx = tokens.index('gemini')
        flash_idx = tokens.index('flash')
        
        # Extract tokens between gemini and flash
        between_tokens = tokens[gemini_idx + 1:flash_idx]
        
        # Check for gemini-3-flash: must have exactly "3" between gemini and flash
        if between_tokens == ['3']:
            return True
        
        # Check for gemini-2.5-flash: must have exactly "2", "5" between gemini and flash
        if between_tokens == ['2', '5']:
            return True
    except (ValueError, IndexError):
        pass
    
    return False


def _is_pro_model(tokens: list[str]) -> bool:
    """
    Check if tokens represent a pro model (gemini-3 or gemini-2.5) without flash.
    
    Args:
        tokens: List of normalized tokens from model name
        
    Returns:
        True if tokens match gemini-3 or gemini-2.5 pattern (without flash)
    """
    if 'gemini' not in tokens or 'flash' in tokens:
        return False
    
    try:
        gemini_idx = tokens.index('gemini')
        
        # Check for gemini-3: must have "gemini" followed by "3"
        # Ensure it's exactly "3" (not part of "3.5" which would have tokens ["3", "5"])
        if gemini_idx + 1 < len(tokens):
            next_token = tokens[gemini_idx + 1]
            # If next token is "3", check that it's not followed by "5" (which would be "3.5")
            if next_token == '3':
                # Make sure the token after "3" is not "5" (to avoid matching "gemini-3.5")
                if gemini_idx + 2 >= len(tokens) or tokens[gemini_idx + 2] != '5':
                    return True
        
        # Check for gemini-2.5: must have "gemini" followed by "2" then "5"
        if (gemini_idx + 2 < len(tokens) and 
            tokens[gemini_idx + 1] == '2' and 
            tokens[gemini_idx + 2] == '5'):
            return True
    except (ValueError, IndexError):
        pass
    
    return False


def _select_best_model(available_models: Optional[list] = None) -> str:
    """
    Select the best model from the list of available models or return a default model.
    
    Priority logic:
    1. If available_models is provided:
       - Prefer flash variants (gemini-3-flash or gemini-2.5-flash); return the first matching flash model
       - If no flash model is found, fall back to non-flash pro variants (gemini-3 or gemini-2.5)
       - If neither is found, select the first model in the list
    2. If available_models is not provided: Return 'models/gemini-3-flash' as the default
    
    Note: Legacy API may not support flash models, so legacy branches will
    handle fallback to gemini-1.5-pro if needed.
    
    Args:
        available_models: List of available model names (may be None)
        
    Returns:
        The selected model name as a string (could have a 'models/' prefix),
        keeping the format from available_models.
    """
    if available_models:
        # Step 1: Prefer flash variants (gemini-3-flash or gemini-2.5-flash)
        flash_match = next(
            (m for m in available_models 
             if _is_flash_model(_normalize_and_tokenize(m))),
            None
        )
        if flash_match:
            return flash_match
        
        # Step 2: If no flash, fall back to non-flash pro variants
        # Look for gemini-3 or gemini-2.5 that are not flash
        pro_match = next(
            (m for m in available_models 
             if _is_pro_model(_normalize_and_tokenize(m))),
            None
        )
        if pro_match:
            return pro_match
        
        # Step 3: If neither found, select the first model in the list
        return available_models[0]
    
    # Fallback: return default model (prefer a newer model)
    # Prefer gemini-3-flash, fallback to gemini-2.5-flash or gemini-1.5-flash if needed
    # Note: Legacy API might not support flash, so legacy branch will handle fallback separately
    return 'models/gemini-3-flash'


class GeminiChartAnalyzer:
    """Analyze chart images using Google Gemini AI."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize GeminiChartAnalyzer.
        
        Args:
            api_key: Google Gemini API key (if None, will load from config)
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
        
        # Try new API first (google-genai with Client), fallback to old API
        if hasattr(genai, 'Client'):
            # Use new Client API of google-genai
            self.client = genai.Client(api_key=api_key)
            
            # Try to list available models to find the right one
            available_models = None
            try:
                models = self.client.models.list()
                available_models = [
                    m.name for m in models 
                    if 'flash' in m.name.lower() or 'pro' in m.name.lower()
                ]
            except Exception:
                # If unable to list models, will use default via helper
                logger.exception("Failed to list available Gemini models, falling back to default model")
            
            # Use helper function for consistent model selection
            self.model_name = _select_best_model(available_models)
            
            self.use_new_api = True
        elif hasattr(genai, 'configure'):
            # Fallback to old API (google-generativeai)
            genai.configure(api_key=api_key)
            
            # Use the same selection logic (no available_models for legacy)
            # Helper will return default model
            selected_model = _select_best_model(None)
            
            # Legacy API may need the format without 'models/' prefix
            model_for_legacy = selected_model.replace('models/', '') if selected_model.startswith('models/') else selected_model
            
            # If the model doesn't exist in legacy, fallback to gemini-1.5-pro
            # (legacy likely doesn't support flash models)
            if 'flash' in model_for_legacy.lower():
                # Try flash first; if not possible, fallback
                try:
                    self.model = genai.GenerativeModel(model_for_legacy)
                    # Store the model name for fallback logic
                    self.model_name = selected_model  # Keep the format with 'models/' prefix
                except Exception as e:
                    # Log exception and context before falling back
                    logger.exception(
                        f"Failed to initialize flash model '{model_for_legacy}' for legacy API. "
                        f"Falling back to 'gemini-1.5-pro'"
                    )
                    # Fall back to pro if flash is not available
                    self.model = genai.GenerativeModel('gemini-1.5-pro')
                    self.model_name = 'models/gemini-1.5-pro'
            else:
                self.model = genai.GenerativeModel(model_for_legacy)
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
        fallback_models = _get_fallback_models(current_model) if current_model else []
        models_to_try = [current_model] + fallback_models if current_model else [None]
        
        for model_idx, model_to_use in enumerate(models_to_try):
            if model_to_use is None and getattr(self, 'use_new_api', False):
                continue  # Skip None models for new API
            
            # Log which model is being attempted before the inner retry loop starts
            model_identifier = model_to_use if model_to_use else (getattr(self, 'model_name', None) or "primary")
            logger.info(f"Trying model: {model_identifier}")
                
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
                                fallback_model = genai.GenerativeModel(model_for_legacy)
                                response = fallback_model.generate_content([prompt, image])
                            except Exception as e:
                                # Log exception with stack trace before continuing to next model
                                logger.exception(
                                    f"Failed to create fallback model '{model_for_legacy}': {e}. "
                                    f"Switching to next model"
                                )
                                # If fallback model creation fails, continue to next model
                                break
                        else:
                            response = self.model.generate_content([prompt, image])
                    
                    # Log success when a request returns
                    logger.info(f"Request succeeded with model: {model_identifier}")
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    last_error = e
                    error_code = None
                    error_message = str(e)
                    
                    # Extract error code from exception
                    if hasattr(e, 'status_code'):
                        error_code = e.status_code
                    elif hasattr(e, 'code'):
                        error_code = e.code
                    elif '503' in error_message or 'UNAVAILABLE' in error_message:
                        error_code = 503
                    
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
                        logger.warning(
                            f"Retryable error with model {model_identifier}, "
                            f"attempt {attempt + 1}/{max_retries}: {error_message}. "
                            f"Waiting {wait_time}s before retrying"
                        )
                        time.sleep(wait_time)
                        continue
                    elif not is_retryable:
                        # Non-retryable error, try next model
                        # Log non-retryable errors with the model name when breaking to the next model
                        logger.error(
                            f"Non-retryable error with model {model_identifier}: {error_message}. "
                            f"Switching to next model"
                        )
                        break  # Exit retry loop, try next model
                    else:
                        # Max retries reached for this model, try next model
                        # Log when max retries are reached and the code is falling back to the next model
                        if model_idx < len(models_to_try) - 1:
                            logger.warning(
                                f"Max retries ({max_retries}) reached with model {model_identifier}: {error_message}. "
                                f"Falling back to the next model"
                            )
                            break  # Try next model
                        else:
                            # All models exhausted
                            logger.error(
                                f"Max retries ({max_retries}) reached with model {model_identifier}: {error_message}. "
                                f"All models have been tried"
                            )
                            raise
            
            if response is not None:
                break  # Success, exit model loop
        
        if response is None:
            # All models and retries exhausted
            raise last_error if last_error else Exception("Failed to get response from any model")
        
        # Extract text from response
        # New API has a different response structure
        if hasattr(response, 'text'):
            return response.text
        elif hasattr(response, 'candidates') and response.candidates is not None and isinstance(response.candidates, (list, tuple)) and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content'):
                if hasattr(candidate.content, 'parts'):
                    text_parts = ''.join([part.text for part in candidate.content.parts if hasattr(part, 'text')])
                    # Fall back to str(candidate.content) if no text parts found
                    return text_parts if text_parts else str(candidate.content)
                elif hasattr(candidate.content, 'text'):
                    return candidate.content.text
                else:
                    return str(candidate.content)
            else:
                return str(response)
        else:
            return str(response)
    
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
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Chart image file not found: {image_path}")

        # Select prompt
        prompt = self._get_prompt(symbol, timeframe, prompt_type, custom_prompt)

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



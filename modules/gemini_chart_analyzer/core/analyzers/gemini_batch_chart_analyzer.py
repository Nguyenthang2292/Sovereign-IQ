
from typing import Any, Dict, List, Optional
import json
import logging
import os
import re
import threading
import time

from modules.common.ui.logging import log_error, log_info, log_success, log_warn
from modules.gemini_chart_analyzer.core.analyzers.gemini_chart_analyzer import GeminiChartAnalyzer
import PIL.Image
import PIL.Image

"""
Batch Gemini Analyzer for analyzing batch chart images.

Extends GeminiChartAnalyzer to handle batch images with JSON response parsing.
"""




logger = logging.getLogger(__name__)


class GeminiBatchChartAnalyzer(GeminiChartAnalyzer):
    """Analyze batch chart images using Gemini API with JSON response."""

    def __init__(self, api_key: Optional[str] = None, cooldown_seconds: float = 2.5):
        """
        Initialize GeminiBatchChartAnalyzer.

        Args:
            api_key: Google Gemini API key (if None, gets from config)
            cooldown_seconds: Cooldown time between batch requests (default: 2.5s)
        """
        super().__init__(api_key)
        self.cooldown_seconds = cooldown_seconds
        self.last_request_time = 0.0
        self._request_lock = threading.Lock()

    def _is_actionable_signal(self, v: Any) -> bool:
        """Check if a value represents an actionable signal (LONG or SHORT)."""
        return isinstance(v, dict) and v.get("signal") in ["LONG", "SHORT"]

    def _clamp_confidence(self, confidence: float) -> float:
        """Clamp confidence value to [0.0, 1.0]."""
        return max(0.0, min(1.0, confidence))

    def analyze_batch_chart(
        self, image_path: str, batch_id: int, total_batches: int, symbols: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze a batch chart image and return JSON with LONG/SHORT signals and confidence.

        Args:
            image_path: Path to batch chart image
            batch_id: Current batch ID (for logging)
            total_batches: Total number of batches (for logging)
            symbols: List of symbols in this batch (for validation)

        Returns:
            Dictionary mapping symbol to signal and confidence:
            {symbol: {"signal": "LONG"|"SHORT"|"NONE", "confidence": float}}
        """
        # Apply cooldown
        self._apply_cooldown()

        log_info(f"Analyzing batch {batch_id}/{total_batches} with {len(symbols)} symbols...")

        # Create prompt for batch analysis
        prompt = self._create_batch_prompt(symbols)

        try:
            # Use parent's analyze_chart method but with custom prompt
            response_text = self._analyze_with_custom_prompt(image_path, prompt)

            # Update request timestamp after successful API call (thread-safe)
            with self._request_lock:
                self.last_request_time = time.time()

            # Parse JSON from response
            result = self._parse_json_response(response_text, symbols)

            signals_count = sum(1 for v in result.values() if self._is_actionable_signal(v))
            log_success(f"Batch {batch_id}/{total_batches} completed: {signals_count} signals found")

            return result

        except Exception as e:
            log_error(f"Error analyzing batch {batch_id}: {e}")
            # Return empty dict with NONE for all symbols
            return {symbol: {"signal": "NONE", "confidence": 0.0} for symbol in symbols}

    def analyze_multi_tf_batch_chart(
        self, batch_chart_path: str, symbols: List[str], normalized_timeframes: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze a multi-timeframe batch chart image and return parsed results.

        This method encapsulates the full workflow:
        1. Creates the multi-timeframe batch prompt
        2. Invokes the LLM with the chart image
        3. Parses the JSON response

        Args:
            batch_chart_path: Path to the multi-timeframe batch chart image
            symbols: List of symbols in this batch
            normalized_timeframes: List of normalized timeframe strings (e.g., ['15m', '1h', '4h'])

        Returns:
            Dictionary mapping symbol to timeframe results:
            {
                symbol: {
                    'timeframes': {
                        tf: {'signal': 'LONG'|'SHORT'|'NONE', 'confidence': float},
                        ...
                    },
                    'aggregated': {'signal': '...', 'confidence': ...}  # Optional, may be None
                },
                ...
            }
        """
        # Apply cooldown
        self._apply_cooldown()

        log_info(
            f"Analyzing multi-TF batch chart with {len(symbols)} symbols across {len(normalized_timeframes)} timeframes..."
        )

        try:
            # Step 1: Create multi-timeframe batch prompt
            prompt = self._create_multi_tf_batch_prompt(symbols, normalized_timeframes)

            # Step 2: Analyze with custom prompt (invokes LLM)
            response_text = self._analyze_with_custom_prompt(batch_chart_path, prompt)

            # Update request timestamp after successful API call (thread-safe)
            with self._request_lock:
                self.last_request_time = time.time()

            # Step 3: Parse multi-TF JSON response
            parsed_results = self._parse_multi_tf_json_response(
                response_text=response_text, expected_symbols=symbols, expected_timeframes=normalized_timeframes
            )

            log_success(f"Multi-TF batch analysis completed for {len(symbols)} symbols")

            return parsed_results

        except Exception as e:
            log_error(f"Error analyzing multi-TF batch chart: {e}")
            # Return empty result structure for all symbols
            return self._create_empty_multi_tf_result(symbols, normalized_timeframes)

    def _apply_cooldown(self):
        """
        Apply cooldown between requests (wait if needed, but don't update timestamp).

        Thread-safe: Uses lock to ensure only one thread calculates and waits at a time.
        """
        current_time = time.time()
        wait_time = 0.0

        # Calculate wait time atomically within lock
        with self._request_lock:
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.cooldown_seconds:
                wait_time = self.cooldown_seconds - time_since_last

        # Sleep outside lock to avoid blocking other threads unnecessarily
        # But wait_time is already calculated, so no race condition
        if wait_time > 0:
            log_info(f"Cooldown: waiting {wait_time:.2f}s...")
            time.sleep(wait_time)

    def _format_symbols_list(self, symbols: List[str], max_display: int) -> tuple[str, str]:
        """
        Format symbols list with truncation if needed.

        Args:
            symbols: List of symbols to format
            max_display: Maximum number of symbols to display before truncation

        Returns:
            Tuple of (symbols_list_str, instruction_str)
        """
        all_symbols_list = ", ".join(symbols)

        if len(symbols) > max_display:
            symbols_list = ", ".join(symbols[:max_display])
            symbols_list += f", ... (và {len(symbols) - max_display} symbols khác)"
            instruction = f"\n\n⚠️ QUAN TRỌNG: Danh sách trên chỉ hiển thị {max_display} symbols đầu tiên. Bạn PHẢI phân tích TẤT CẢ {len(symbols)} symbols có trong ảnh."
        else:
            symbols_list = all_symbols_list
            instruction = ""

        return (symbols_list, instruction)

    def _create_batch_prompt(self, symbols: List[str]) -> str:
        """
        Create prompt for batch analysis.

        Args:
            symbols: List of symbols in the batch

        Returns:
            Prompt string
        """
        # Format symbols list with truncation if needed
        max_display_symbols = 50
        symbols_list, base_instruction = self._format_symbols_list(symbols, max_display_symbols)

        # Customize instruction for batch prompt (more detailed)
        if base_instruction:
            all_symbols_instruction = f"\n\n⚠️ QUAN TRỌNG: Danh sách trên chỉ hiển thị {max_display_symbols} symbols đầu tiên làm ví dụ. Bạn PHẢI phân tích TẤT CẢ {len(symbols)} biểu đồ/symbols có trong ảnh, không chỉ những symbols được liệt kê ở trên. Hãy xem kỹ ảnh và tìm tất cả các symbols có nhãn trên biểu đồ."
        else:
            all_symbols_instruction = ""

        prompt = f"""Bạn là một chuyên gia phân tích kỹ thuật tài chính.

Trong ảnh này có {len(symbols)} biểu đồ nến (candlestick charts) của các cặp tiền mã hóa. Mỗi biểu đồ có nhãn symbol ở góc trên trái.

Danh sách tất cả symbols trong batch này: {symbols_list}{all_symbols_instruction}

Nhiệm vụ của bạn:
1. Phân tích TẤT CẢ biểu đồ trong ảnh (tất cả {len(symbols)} symbols)
2. Đánh giá xu hướng và tín hiệu kỹ thuật cho mỗi symbol
3. Đánh dấu mỗi symbol theo một trong các tín hiệu sau:
   - "LONG": Nếu có tín hiệu mua rõ ràng (xu hướng tăng, breakout, reversal tăng, etc.)
   - "SHORT": Nếu có tín hiệu bán rõ ràng (xu hướng giảm, breakdown, reversal giảm, etc.)
   - "NONE": Nếu không có tín hiệu rõ ràng hoặc thị trường đi ngang

QUAN TRỌNG: Bạn PHẢI trả về kết quả dưới dạng JSON hợp lệ với format sau:
{{
  "BTC/USDT": {{"signal": "LONG", "confidence": 0.85}},
  "ETH/USDT": {{"signal": "SHORT", "confidence": 0.70}},
  "BNB/USDT": {{"signal": "NONE", "confidence": 0.50}},
  ...
}}

Lưu ý:
- Chỉ trả về JSON, không có text thêm trước hoặc sau
- Đảm bảo TẤT CẢ {len(symbols)} symbols trong danh sách đều có trong JSON response
- Bạn phải phân tích tất cả biểu đồ có trong ảnh, không bỏ sót symbol nào
- Tên symbol phải chính xác (khớp với nhãn trên biểu đồ)
- confidence là số từ 0.0 đến 1.0:
  * 0.8-1.0: Tín hiệu rất mạnh và rõ ràng
  * 0.6-0.8: Tín hiệu khá mạnh
  * 0.4-0.6: Tín hiệu yếu hoặc không rõ ràng (thường là NONE)
  * 0.0-0.4: Rất yếu, không có tín hiệu (NONE)
- Nếu không thấy rõ symbol nào, đánh dấu signal là "NONE" với confidence thấp
"""
        return prompt

    def _create_multi_tf_batch_prompt(self, symbols: List[str], timeframes: List[str]) -> str:
        """
        Create prompt for multi-timeframe batch analysis.

        Args:
            symbols: List of symbols in the batch
            timeframes: List of timeframes to analyze

        Returns:
            Prompt string
        """
        all_timeframes_list = ", ".join(timeframes)

        # Format symbols list with truncation if needed
        max_display_symbols = 30  # Reduced because prompt is longer
        symbols_list, all_symbols_instruction = self._format_symbols_list(symbols, max_display_symbols)

        prompt = f"""Bạn là một chuyên gia phân tích kỹ thuật tài chính.

Trong ảnh này có {len(symbols)} symbols, mỗi symbol có {len(timeframes)} biểu đồ nến cho các timeframes khác nhau: {all_timeframes_list}.

Mỗi symbol được hiển thị với các biểu đồ timeframes được sắp xếp trong một grid. Mỗi biểu đồ có nhãn symbol và timeframe ở góc trên trái.

Danh sách tất cả symbols trong batch này: {symbols_list}{all_symbols_instruction}

Nhiệm vụ của bạn:
1. Phân tích TẤT CẢ biểu đồ trong ảnh (tất cả {len(symbols)} symbols × {len(timeframes)} timeframes)
2. Đánh giá xu hướng và tín hiệu kỹ thuật cho mỗi symbol trên mỗi timeframe
3. Đánh dấu mỗi symbol+timeframe theo một trong các tín hiệu sau:
   - "LONG": Nếu có tín hiệu mua rõ ràng
   - "SHORT": Nếu có tín hiệu bán rõ ràng
   - "NONE": Nếu không có tín hiệu rõ ràng

QUAN TRỌNG: Bạn PHẢI trả về kết quả dưới dạng JSON hợp lệ với format sau:
{{
  "BTC/USDT": {{
    "15m": {{"signal": "LONG", "confidence": 0.70}},
    "1h": {{"signal": "LONG", "confidence": 0.80}},
    "4h": {{"signal": "SHORT", "confidence": 0.60}},
    "1d": {{"signal": "LONG", "confidence": 0.75}},
    "aggregated": {{"signal": "LONG", "confidence": 0.71}}
  }},
  "ETH/USDT": {{
    "15m": {{"signal": "SHORT", "confidence": 0.65}},
    ...
  }},
  ...
}}

Lưu ý:
- Chỉ trả về JSON, không có text thêm trước hoặc sau
- Đảm bảo TẤT CẢ {len(symbols)} symbols đều có trong JSON response
- Mỗi symbol PHẢI có tất cả {len(timeframes)} timeframes: {all_timeframes_list}
- Mỗi symbol PHẢI có field "aggregated" với signal và confidence tổng hợp từ các timeframes
- confidence là số từ 0.0 đến 1.0
- aggregated confidence nên là weighted average của các timeframes (timeframe lớn hơn = weight cao hơn)
"""
        return prompt

    def _analyze_with_custom_prompt(self, image_path: str, prompt: str) -> str:
        """
        Analyze image with custom prompt using parent's infrastructure.

        Args:
            image_path: Path to image
            prompt: Custom prompt text

        Returns:
            Response text from Gemini

        Raises:
            FileNotFoundError: If image file does not exist
            Exception: If API call fails (raised by parent's helper method)
        """
        # Load image - only handle I/O errors here
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        with PIL.Image.open(image_path) as img:
            img_copy = img.copy()

        # Delegate API call + retry/response-extraction to parent's helper method
        # This method will handle all retry logic, fallback models, and response parsing
        return self._call_model_with_retries(prompt, img_copy)

    def _extract_json_from_text(self, response_text: str) -> Optional[str]:
        """
        Extract JSON string from response text using multiple strategies.

        Tries in order:
        1. Fenced code block regex (```json ... ``` or ``` ... ```)
        2. Try parsing JSON candidates by iterating start positions and validating with json.loads()
        3. Greedy regex fallback

        Args:
            response_text: Raw response text that may contain JSON

        Returns:
            Extracted JSON string, or None if no JSON found
        """
        # Strategy 1: Try fenced code block regex
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if json_match:
            candidate = json_match.group(1)
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass  # Fall through to next strategy

        # Strategy 2: Try parsing JSON candidates by iterating start positions
        # Find all positions of '{' that could be the start of a JSON object
        start_positions = []
        for i, char in enumerate(response_text):
            if char == "{":
                start_positions.append(i)

        # Try parsing from each start position
        # Start from the first '{' and try progressively longer substrings
        for start_idx in start_positions:
            # Try parsing from start_idx to end of text first
            candidate = response_text[start_idx:]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

            # If that fails, try finding the matching closing brace
            # by looking for '}' positions after start_idx
            end_positions = []
            for i in range(start_idx + 1, len(response_text)):
                if response_text[i] == "}":
                    end_positions.append(i)

            # Try parsing from start_idx to each potential end position
            # Start from the longest candidate (last '}') and work backwards
            for end_idx in reversed(end_positions):
                candidate = response_text[start_idx : end_idx + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    continue

        # Strategy 3: Fallback to greedy regex
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            candidate = json_match.group(0)
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

        # No JSON found
        return None

    def _parse_json_response(self, response_text: str, expected_symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Parse JSON response from Gemini, handling various formats.

        Args:
            response_text: Raw response text from Gemini
            expected_symbols: List of expected symbols (for validation)

        Returns:
            Dictionary mapping symbol to signal and confidence:
            {symbol: {"signal": "LONG"|"SHORT"|"NONE", "confidence": float}}
        """
        # Extract JSON from response text using helper method
        json_str = self._extract_json_from_text(response_text)
        if json_str is None:
            log_error("No JSON found in response")
            return {symbol: {"signal": "NONE", "confidence": 0.0} for symbol in expected_symbols}

        try:
            # Parse JSON
            result = json.loads(json_str)

            # Validate result is a dict
            if not isinstance(result, dict):
                log_error(f"JSON response is not a dictionary, got {type(result).__name__}")
                return {symbol: {"signal": "NONE", "confidence": 0.0} for symbol in expected_symbols}

            # Validate and normalize
            normalized_result = {}
            missing_symbols = []
            for symbol in expected_symbols:
                # Try exact match first
                if symbol in result:
                    value = result[symbol]

                    # Handle different formats
                    if isinstance(value, dict):
                        # New format: {"signal": "LONG", "confidence": 0.85}
                        signal = str(value.get("signal", "NONE")).upper().strip()
                        confidence = float(value.get("confidence", 0.5))
                        # Clamp confidence to [0.0, 1.0]
                        confidence = self._clamp_confidence(confidence)

                        if signal in ["LONG", "SHORT", "NONE"]:
                            normalized_result[symbol] = {"signal": signal, "confidence": confidence}
                        else:
                            normalized_result[symbol] = {"signal": "NONE", "confidence": 0.0}
                    elif isinstance(value, str):
                        # Old format: "LONG" or "SHORT" or "NONE"
                        signal = value.upper().strip()
                        if signal in ["LONG", "SHORT", "NONE"]:
                            # Default confidence based on signal
                            confidence = 0.7 if signal in ["LONG", "SHORT"] else 0.5
                            normalized_result[symbol] = {"signal": signal, "confidence": confidence}
                        else:
                            normalized_result[symbol] = {"signal": "NONE", "confidence": 0.0}
                    else:
                        normalized_result[symbol] = {"signal": "NONE", "confidence": 0.0}
                else:
                    # Symbol not found in response
                    missing_symbols.append(symbol)
                    normalized_result[symbol] = {"signal": "NONE", "confidence": 0.0}

            # Log missing symbols if any
            if missing_symbols:
                log_warn(
                    f"{len(missing_symbols)} symbols not found in JSON response: {missing_symbols[:5]}{'...' if len(missing_symbols) > 5 else ''}"
                )

            return normalized_result

        except json.JSONDecodeError as e:
            log_error(f"JSON decode error: {e}")
            log_error(f"Response text: {response_text[:500]}")
            return {symbol: {"signal": "NONE", "confidence": 0.0} for symbol in expected_symbols}
        except (ValueError, TypeError) as e:
            log_error(f"Error parsing confidence: {e}")
            return {symbol: {"signal": "NONE", "confidence": 0.0} for symbol in expected_symbols}

    def _parse_multi_tf_json_response(
        self, response_text: str, expected_symbols: List[str], expected_timeframes: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Parse multi-timeframe JSON response from Gemini.

        Expected format:
        {
            "BTC/USDT": {
                "15m": {"signal": "LONG", "confidence": 0.70},
                "1h": {"signal": "LONG", "confidence": 0.80},
                "aggregated": {"signal": "LONG", "confidence": 0.71}
            },
            ...
        }

        Args:
            response_text: Raw response text from Gemini
            expected_symbols: List of expected symbols (for validation)
            expected_timeframes: List of expected timeframes (for validation)

        Returns:
            Dictionary mapping symbol to timeframe results:
            {
                symbol: {
                    'timeframes': {
                        tf: {'signal': '...', 'confidence': ...},
                        ...
                    },
                    'aggregated': {'signal': '...', 'confidence': ...}  # Optional, may be missing
                },
                ...
            }
        """
        # Extract JSON from response text using helper method
        json_str = self._extract_json_from_text(response_text)
        if json_str is None:
            log_error("No JSON found in multi-TF response")
            return self._create_empty_multi_tf_result(expected_symbols, expected_timeframes)

        try:
            # Parse JSON
            batch_result_raw = json.loads(json_str)

            # Validate result is a dict
            if not isinstance(batch_result_raw, dict):
                log_error(f"Multi-TF JSON response is not a dictionary, got {type(batch_result_raw).__name__}")
                return self._create_empty_multi_tf_result(expected_symbols, expected_timeframes)

            # Parse and validate each symbol
            result = {}
            missing_symbols = []

            for symbol in expected_symbols:
                if symbol not in batch_result_raw:
                    missing_symbols.append(symbol)
                    result[symbol] = self._create_empty_symbol_multi_tf_result(expected_timeframes)
                    continue

                symbol_data = batch_result_raw[symbol]

                # Validate symbol_data is a dict
                if not isinstance(symbol_data, dict):
                    log_error(f"Symbol {symbol} data is not a dictionary, got {type(symbol_data).__name__}")
                    result[symbol] = self._create_empty_symbol_multi_tf_result(expected_timeframes)
                    continue

                # Extract timeframe signals
                tf_signals = {}
                for tf in expected_timeframes:
                    if tf in symbol_data:
                        tf_data = symbol_data[tf]

                        # Validate and normalize timeframe signal
                        if isinstance(tf_data, dict):
                            signal = str(tf_data.get("signal", "NONE")).upper().strip()
                            confidence = float(tf_data.get("confidence", 0.5))
                            confidence = self._clamp_confidence(confidence)

                            if signal in ["LONG", "SHORT", "NONE"]:
                                tf_signals[tf] = {"signal": signal, "confidence": confidence}
                            else:
                                log_warn(f"Invalid signal '{signal}' for {symbol} {tf}, using NONE")
                                tf_signals[tf] = {"signal": "NONE", "confidence": 0.0}
                        else:
                            log_warn(f"Invalid format for {symbol} {tf}, expected dict, got {type(tf_data).__name__}")
                            tf_signals[tf] = {"signal": "NONE", "confidence": 0.0}
                    else:
                        # Timeframe missing from response
                        tf_signals[tf] = {"signal": "NONE", "confidence": 0.0}

                # Extract aggregated signal if present (optional)
                aggregated = None
                if "aggregated" in symbol_data:
                    agg_data = symbol_data["aggregated"]
                    if isinstance(agg_data, dict):
                        signal = str(agg_data.get("signal", "NONE")).upper().strip()
                        confidence = float(agg_data.get("confidence", 0.5))
                        confidence = self._clamp_confidence(confidence)

                        if signal in ["LONG", "SHORT", "NONE"]:
                            aggregated = {"signal": signal, "confidence": confidence}
                        else:
                            log_warn(f"Invalid aggregated signal '{signal}' for {symbol}")
                    else:
                        log_warn(f"Invalid aggregated format for {symbol}, expected dict")

                result[symbol] = {
                    "timeframes": tf_signals,
                    "aggregated": aggregated,  # May be None if not provided
                }

            # Log missing symbols if any
            if missing_symbols:
                log_warn(
                    f"{len(missing_symbols)} symbols not found in multi-TF JSON response: {missing_symbols[:5]}{'...' if len(missing_symbols) > 5 else ''}"
                )

            return result

        except json.JSONDecodeError as e:
            log_error(f"JSON decode error in multi-TF response: {e}")
            log_error(f"Response text (first 500 chars): {response_text[:500]}")
            return self._create_empty_multi_tf_result(expected_symbols, expected_timeframes)
        except (ValueError, TypeError) as e:
            log_error(f"Error parsing multi-TF response: {e}")
            return self._create_empty_multi_tf_result(expected_symbols, expected_timeframes)

    def _create_empty_multi_tf_result(
        self, expected_symbols: List[str], expected_timeframes: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Create empty result structure for multi-TF response."""
        return {symbol: self._create_empty_symbol_multi_tf_result(expected_timeframes) for symbol in expected_symbols}

    def _create_empty_symbol_multi_tf_result(self, expected_timeframes: List[str]) -> Dict[str, Any]:
        """Create empty result structure for a single symbol in multi-TF response."""
        return {
            "timeframes": {tf: {"signal": "NONE", "confidence": 0.0} for tf in expected_timeframes},
            "aggregated": None,
        }

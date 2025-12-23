"""
Batch Gemini Analyzer for analyzing batch chart images.

Extends GeminiAnalyzer to handle batch images with JSON response parsing.
"""

import os
import json
import re
import time
import threading
from typing import Dict, Optional, List, Any
import logging
import PIL.Image

from modules.gemini_chart_analyzer.core.gemini_analyzer import GeminiAnalyzer
from modules.common.ui.logging import log_info, log_error, log_success, log_warn

logger = logging.getLogger(__name__)


class BatchGeminiAnalyzer(GeminiAnalyzer):
    """Analyze batch chart images using Gemini API with JSON response."""
    
    def __init__(self, api_key: Optional[str] = None, cooldown_seconds: float = 2.5):
        """
        Initialize BatchGeminiAnalyzer.
        
        Args:
            api_key: Google Gemini API key (if None, gets from config)
            cooldown_seconds: Cooldown time between batch requests (default: 2.5s)
        """
        super().__init__(api_key)
        self.cooldown_seconds = cooldown_seconds
        self.last_request_time = 0.0
        self._request_lock = threading.Lock()
    
    def analyze_batch_chart(
        self,
        image_path: str,
        batch_id: int,
        total_batches: int,
        symbols: List[str]
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
            
            signals_count = len([v for v in result.values() if isinstance(v, dict) and v.get('signal') in ['LONG', 'SHORT']])
            log_success(f"Batch {batch_id}/{total_batches} completed: {signals_count} signals found")
            
            return result
            
        except Exception as e:
            log_error(f"Error analyzing batch {batch_id}: {e}")
            # Return empty dict with NONE for all symbols
            return {symbol: {"signal": "NONE", "confidence": 0.0} for symbol in symbols}
    
    def _apply_cooldown(self):
        """Apply cooldown between requests (wait if needed, but don't update timestamp)."""
        current_time = time.time()
        # Thread-safe read of last_request_time
        with self._request_lock:
            time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.cooldown_seconds:
            wait_time = self.cooldown_seconds - time_since_last
            log_info(f"Cooldown: waiting {wait_time:.2f}s...")
            time.sleep(wait_time)
    
    def _create_batch_prompt(self, symbols: List[str]) -> str:
        """
        Create prompt for batch analysis.
        
        Args:
            symbols: List of symbols in the batch
            
        Returns:
            Prompt string
        """
        # Include all symbols in the list for clarity
        all_symbols_list = ", ".join(symbols)
        
        # If list is very long, truncate but add explicit instruction
        max_display_symbols = 50
        if len(symbols) > max_display_symbols:
            symbols_list = ", ".join(symbols[:max_display_symbols])
            symbols_list += f", ... (và {len(symbols) - max_display_symbols} symbols khác)"
            all_symbols_instruction = f"\n\n⚠️ QUAN TRỌNG: Danh sách trên chỉ hiển thị {max_display_symbols} symbols đầu tiên làm ví dụ. Bạn PHẢI phân tích TẤT CẢ {len(symbols)} biểu đồ/symbols có trong ảnh, không chỉ những symbols được liệt kê ở trên. Hãy xem kỹ ảnh và tìm tất cả các symbols có nhãn trên biểu đồ."
        else:
            symbols_list = all_symbols_list
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
        # Try to extract JSON from response text
        # Gemini might wrap JSON in markdown code blocks or add text
        
        # Remove markdown code blocks if present
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object directly
            # Use non-greedy matching to avoid matching multiple JSON objects
            # Also try to find the largest valid JSON object by counting braces
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
            if not json_match:
                # Fallback: try greedy but validate it's valid JSON
                json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
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
                        signal = str(value.get('signal', 'NONE')).upper().strip()
                        confidence = float(value.get('confidence', 0.5))
                        # Clamp confidence to [0.0, 1.0]
                        confidence = max(0.0, min(1.0, confidence))
                        
                        if signal in ['LONG', 'SHORT', 'NONE']:
                            normalized_result[symbol] = {
                                "signal": signal,
                                "confidence": confidence
                            }
                        else:
                            normalized_result[symbol] = {"signal": "NONE", "confidence": 0.0}
                    elif isinstance(value, str):
                        # Old format: "LONG" or "SHORT" or "NONE"
                        signal = value.upper().strip()
                        if signal in ['LONG', 'SHORT', 'NONE']:
                            # Default confidence based on signal
                            confidence = 0.7 if signal in ['LONG', 'SHORT'] else 0.5
                            normalized_result[symbol] = {
                                "signal": signal,
                                "confidence": confidence
                            }
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
                log_warn(f"{len(missing_symbols)} symbols not found in JSON response: {missing_symbols[:5]}{'...' if len(missing_symbols) > 5 else ''}")
            
            return normalized_result
            
        except json.JSONDecodeError as e:
            log_error(f"JSON decode error: {e}")
            log_error(f"Response text: {response_text[:500]}")
            return {symbol: {"signal": "NONE", "confidence": 0.0} for symbol in expected_symbols}
        except (ValueError, TypeError) as e:
            log_error(f"Error parsing confidence: {e}")
            return {symbol: {"signal": "NONE", "confidence": 0.0} for symbol in expected_symbols}


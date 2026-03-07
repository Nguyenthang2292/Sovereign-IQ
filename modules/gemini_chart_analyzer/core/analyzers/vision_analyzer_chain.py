"""Vision Analyzer Chain orchestrator."""

from typing import List, Optional

from modules.common.ui.logging import log_warn

from .gemini_chart_analyzer import GeminiChartAnalyzer
from .qwen_vision_provider import QwenVisionProvider
from .vision_provider_protocol import VisionProvider


class VisionChainExhaustedError(Exception):
    """Raised when all configured vision providers in the chain fail."""

    pass


class GeminiVisionProvider:
    """Wrapper around GeminiChartAnalyzer to implement VisionProvider protocol."""

    provider_name: str = "gemini"

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key
        self._analyzer = None
        try:
            self._analyzer = GeminiChartAnalyzer(api_key=api_key)
        except Exception as e:
            log_warn(f"[{self.provider_name}] Initialization failed (key missing or invalid): {e}")

    def is_available(self) -> bool:
        """Return True if Gemini API SDK is successfully configured."""
        return self._analyzer is not None

    def analyze_chart(
        self,
        image_path: str,
        symbol: str,
        timeframe: str,
        prompt_type: str = "detailed",
        custom_prompt: Optional[str] = None,
    ) -> str:
        """Call Gemini vision API."""
        if not self._analyzer:
            raise RuntimeError("Gemini analyzer is not available.")

        return self._analyzer.analyze_chart(
            image_path=image_path,
            symbol=symbol,
            timeframe=timeframe,
            prompt_type=prompt_type,
            custom_prompt=custom_prompt,
        )


class VisionAnalyzerChain:
    """Orchestrates multiple vision providers with automatic fallback."""

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        qwen_api_key: Optional[str] = None,
        qwen_models: Optional[List[str]] = None,
        skip_unavailable: bool = True,
    ):
        """
        Initialize the provider chain.
        """
        self._providers: List[VisionProvider] = []

        # 1. Primary: Gemini
        gemini = GeminiVisionProvider(api_key=gemini_api_key)
        if not skip_unavailable or gemini.is_available():
            self._providers.append(gemini)

        # 2. Secondary/Fallback: Qwen (Dashscope)
        qwen = QwenVisionProvider(api_key=qwen_api_key, models=qwen_models)
        if not skip_unavailable or qwen.is_available():
            self._providers.append(qwen)

        if not self._providers:
            raise VisionChainExhaustedError("No vision providers configured or available.")

    def is_available(self) -> bool:
        """Return True if at least one provider is available."""
        return any(p.is_available() for p in self._providers)

    def analyze_chart(
        self,
        image_path: str,
        symbol: str,
        timeframe: str,
        prompt_type: str = "detailed",
        custom_prompt: Optional[str] = None,
    ) -> str:
        """Iterate through providers and return the first successful result."""
        last_error = None

        for provider in self._providers:
            if not provider.is_available():
                continue

            try:
                result = provider.analyze_chart(
                    image_path=image_path,
                    symbol=symbol,
                    timeframe=timeframe,
                    prompt_type=prompt_type,
                    custom_prompt=custom_prompt,
                )
                return result
            except Exception as e:
                provider_name = getattr(provider, "provider_name", type(provider).__name__)
                log_warn(f"[{provider_name}] failed: {e}. Trying next provider...")
                last_error = e

        error_msg = (
            f"All vision providers exhausted. Last error: {last_error}"
            if last_error
            else "All configured vision providers failed or were unavailable."
        )
        raise VisionChainExhaustedError(error_msg)

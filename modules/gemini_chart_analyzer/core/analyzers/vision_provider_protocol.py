"""Vision provider protocol for chart analysis."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class VisionProvider(Protocol):
    """Protocol for vision chart analysis providers."""

    def analyze_chart(
        self,
        image_path: str,
        symbol: str,
        timeframe: str,
        prompt_type: str = "detailed",
        custom_prompt: str | None = None,
    ) -> str:
        """
        Analyze a chart image.

        Args:
            image_path: Path to the chart image
            symbol: Symbol name (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h', '4h')
            prompt_type: Type of prompt ('detailed', 'simple', 'custom')
            custom_prompt: Custom prompt (if prompt_type='custom')

        Returns:
            str: Analysis result text
        """
        ...

    def is_available(self) -> bool:
        """
        Check if the provider is available (has valid API key).

        Returns:
            bool: True if provider can be used
        """
        ...

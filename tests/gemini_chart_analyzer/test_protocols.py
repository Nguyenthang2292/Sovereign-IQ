import pytest
from typing import Any, Dict, List, Optional
from modules.gemini_chart_analyzer.core.protocols import ChartAnalyzerProtocol


class MockAnalyzer:
    def analyze_chart(
        self,
        image_path: str,
        symbol: str,
        timeframe: str,
        prompt_type: str = "detailed",
        custom_prompt: Optional[str] = None,
    ) -> str:
        return "MOCK LONG"


def test_protocol_implementation():
    analyzer: ChartAnalyzerProtocol = MockAnalyzer()
    result = analyzer.analyze_chart(image_path="dummy.png", symbol="BTC/USDT", timeframe="1h")
    assert result == "MOCK LONG"

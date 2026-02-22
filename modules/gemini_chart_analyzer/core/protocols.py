from typing import Any, Dict, List, Optional, Protocol


class ChartAnalyzerProtocol(Protocol):
    def analyze_chart(
        self,
        image_path: str,
        symbol: str,
        timeframe: str,
        prompt_type: str = "detailed",
        custom_prompt: Optional[str] = None,
    ) -> str: ...


class BatchChartAnalyzerProtocol(Protocol):
    def analyze_batch_chart(
        self, image_path: str, batch_id: int, total_batches: Optional[int], symbols: List[str]
    ) -> Dict[str, Dict[str, Any]]: ...

    def analyze_multi_tf_batch_chart(
        self, batch_chart_path: str, symbols: List[str], normalized_timeframes: List[str]
    ) -> Any: ...


class ChartGeneratorProtocol(Protocol):
    def create_chart(
        self,
        df: Any,
        symbol: str,
        timeframe: str,
        indicators: Optional[Dict] = None,
        output_path: Optional[str] = None,
        show_volume: bool = True,
    ) -> str: ...

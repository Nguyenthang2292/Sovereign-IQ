"""
HTML report generators package.

This package provides modular HTML report generation for chart analysis:
- Single timeframe reports
- Multi-timeframe reports
- Batch scan reports

Public API:
    - generate_single_report: Generate single timeframe analysis report
    - generate_multi_tf_report: Generate multi-timeframe analysis report
    - generate_batch_report: Generate batch scan results report
"""

from modules.gemini_chart_analyzer.core.reporting.generators.batch_report import (
    generate_batch_report,
)
from modules.gemini_chart_analyzer.core.reporting.generators.multi_tf_report import (
    generate_multi_tf_report,
)
from modules.gemini_chart_analyzer.core.reporting.generators.single_report import (
    generate_single_report,
)

__all__ = [
    "generate_single_report",
    "generate_multi_tf_report",
    "generate_batch_report",
]

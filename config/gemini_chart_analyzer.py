"""
Configuration for Gemini Chart Analyzer module.

Default settings for multi-timeframe analysis, weights, and chart layouts.
"""

# Default timeframes for multi-timeframe analysis
DEFAULT_TIMEFRAMES = ['15m', '30m', '1h']

# Timeframe weights for signal aggregation
# Larger timeframes have higher weights (more important)
# Weights sum to 1.0 for proper signal aggregation
TIMEFRAME_WEIGHTS = {
    '15m': 0.2,   # 20% weight
    '30m': 0.3,   # 30% weight  
    '1h': 0.5     # 50% weight (largest timeframe gets highest weight)
}

# Multi-timeframe batch chart layout configuration
MULTI_TF_BATCH_CONFIG = {
    'charts_per_batch': 25,  # Reduced because each symbol has multiple TFs
    'timeframes_per_symbol': 4,  # Default: 4 timeframes per symbol
    'chart_size': (3.0, 2.0),  # Size of each individual chart in inches (width, height)
    'dpi': 100  # DPI for batch charts
}

# Single timeframe batch chart configuration
SINGLE_TF_BATCH_CONFIG = {
    'charts_per_batch': 100,
    'grid_rows': 10,
    'grid_cols': 10,
    'chart_size': (2.0, 1.5),
    'dpi': 100
}

# Deep analysis chart configuration
DEEP_ANALYSIS_CHART_CONFIG = {
    'figsize': (16, 10),
    'style': 'dark_background',
    'dpi': 150
}


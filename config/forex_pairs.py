"""
Forex pairs configuration.

Contains definitions for major and minor forex pairs used in forex market scanning.
"""

# 7 Major Forex Pairs
FOREX_MAJOR_PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD"]

# Common Minor Forex Pairs (Cross pairs without USD)
# These are the most commonly traded minor pairs
FOREX_MINOR_PAIRS = [
    # EUR crosses
    "EUR/GBP",
    "EUR/JPY",
    "EUR/CHF",
    "EUR/AUD",
    "EUR/CAD",
    "EUR/NZD",
    # GBP crosses
    "GBP/JPY",
    "GBP/CHF",
    "GBP/AUD",
    "GBP/CAD",
    "GBP/NZD",
    # JPY crosses
    "AUD/JPY",
    "CAD/JPY",
    "CHF/JPY",
    "NZD/JPY",
    # AUD crosses
    "AUD/CAD",
    "AUD/CHF",
    "AUD/NZD",
    # CAD crosses
    "CAD/CHF",
    "CAD/NZD",
    # CHF crosses
    "CHF/NZD",
    # NZD crosses
    "NZD/CAD",
    "NZD/CHF",
]

"""Centralized monospace font definitions for Matrix GUI theme."""


class Fonts:
    """Matrix theme monospace font system."""

    FAMILY = "Consolas"
    FALLBACKS = ("Consolas", "Courier New", "monospace")

    H1 = (FAMILY, 16, "bold")
    H2 = (FAMILY, 14, "bold")
    H3 = (FAMILY, 12, "bold")
    BODY = (FAMILY, 11)
    SMALL = (FAMILY, 10)
    TINY = (FAMILY, 9)
    DATA = (FAMILY, 18, "bold")
    INPUT = (FAMILY, 12)

    # Button font hierarchy — 2 standardized sizes
    BUTTON = (FAMILY, 13, "bold")  # Primary CTA (large): START SCAN, PLACE ORDER, ENABLE AUTO-TRADE
    BUTTON_SM = (FAMILY, 11, "bold")  # Secondary/utility buttons: REFRESH, CANCEL, CREDENTIALS, etc.

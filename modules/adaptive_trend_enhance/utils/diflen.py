from typing import Tuple

from modules.common.utils import log_error, log_warn

"""Calculate length offsets for Moving Averages based on robustness setting."""


def diflen(length: int, robustness: str = "Medium") -> Tuple[int, int, int, int, int, int, int, int]:
    """Calculate length offsets for Moving Averages based on robustness setting.

    Port of Pine Script `diflen(length)` function. Returns 8 length values
    (4 positive offsets and 4 negative offsets) based on the robustness parameter.

    Args:
        length: Base length for Moving Average (must be > 0).
        robustness: Robustness setting determining offset spread:
            - "Narrow": Small offsets (±1, ±2, ±3, ±4)
            - "Medium": Medium offsets (±1, ±2, ±4, ±6)
            - "Wide": Large offsets (±1, ±3, ±5, ±7)

    Returns:
        Tuple of 8 integers: (L1, L2, L3, L4, L_1, L_2, L_3, L_4)
        where L1-L4 are positive offsets and L_1-L_4 are negative offsets.
        All returned values are guaranteed to be > 0.

    Raises:
        ValueError: If length is invalid or robustness is invalid.
        TypeError: If length is not an integer.
    """
    if not isinstance(length, int):
        raise TypeError(f"length must be an integer, got {type(length)}")

    if length <= 0:
        raise ValueError(f"length must be > 0, got {length}")

    robustness = robustness or "Medium"

    if not isinstance(robustness, str):
        raise TypeError(f"robustness must be a string, got {type(robustness)}")

    VALID_ROBUSTNESS = {"Narrow", "Medium", "Wide"}
    robustness_normalized = robustness.strip() if isinstance(robustness, str) else str(robustness)

    if robustness_normalized not in VALID_ROBUSTNESS:
        log_warn(
            f"Invalid robustness '{robustness}'. Valid values: {', '.join(VALID_ROBUSTNESS)}. Using default 'Medium'."
        )
        robustness_normalized = "Medium"

    try:
        if robustness_normalized == "Narrow":
            L1, L_1 = length + 1, length - 1
            L2, L_2 = length + 2, length - 2
            L3, L_3 = length + 3, length - 3
            L4, L_4 = length + 4, length - 4
        elif robustness_normalized == "Medium":
            L1, L_1 = length + 1, length - 1
            L2, L_2 = length + 2, length - 2
            L3, L_3 = length + 4, length - 4
            L4, L_4 = length + 6, length - 6
        else:  # "Wide" or any other value (fallback to Wide)
            L1, L_1 = length + 1, length - 1
            L2, L_2 = length + 3, length - 3
            L3, L_3 = length + 5, length - 5
            L4, L_4 = length + 7, length - 7

        # Ensure all lengths are positive (negative offsets should still be > 0)
        lengths = [L1, L2, L3, L4, L_1, L_2, L_3, L_4]
        min_length = min(lengths)

        if min_length <= 0:
            # Raise error for invalid lengths (matching Pine Script behavior)
            # Pine Script: length - offset < 0 would cause calculation errors
            invalid_lengths = [len_val for len_val in lengths if len_val <= 0]
            # Calculate minimum length needed to make all offsets positive
            max_negative_offset = max(0 - len_val for len_val in invalid_lengths)
            min_required_length = max_negative_offset + length
            raise ValueError(
                f"Calculated length offsets contain values <= 0: {invalid_lengths}. "
                f"Base length={length}, robustness={robustness_normalized}. "
                f"Please increase base length to at least {min_required_length}."
            )

        return L1, L2, L3, L4, L_1, L_2, L_3, L_4

    except Exception as e:
        log_error(f"Error calculating diflen: {e}")
        raise

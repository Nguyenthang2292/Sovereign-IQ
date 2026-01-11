
from functools import wraps
from typing import Any, List
import threading

import numpy as np
import numpy as np

"""
HMM-Swings Utility Functions.

This module contains utility functions like timeout decorator, safe_forward_backward, and swing distance calculation.
"""




def timeout(seconds):
    """Timeout decorator for function execution."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result: List[Any] = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)

            if thread.is_alive():
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")

            if isinstance(result[0], Exception):
                raise result[0]
            return result[0]

        return wrapper

    return decorator


@timeout(30)
def safe_forward_backward(model, observations):
    """Safely execute forward_backward algorithm with timeout."""
    return model.forward_backward(observations)


def average_swing_distance(swing_highs_info, swing_lows_info):
    """
    Calculate the average time interval (in seconds) between consecutive swing highs and swing lows,
    and return the overall average.

    Args:
        swing_highs_info (pd.DataFrame): DataFrame containing swing high information with datetime index.
        swing_lows_info (pd.DataFrame): DataFrame containing swing low information with datetime index.

    Returns:
        float: The average time distance between swing points in seconds.
    """
    # Calculate high intervals
    swing_high_times = swing_highs_info.index
    intervals_seconds_high = [
        (swing_high_times[i] - swing_high_times[i - 1]).total_seconds() for i in range(1, len(swing_high_times))
    ]
    avg_distance_high = np.mean(intervals_seconds_high) if intervals_seconds_high else 0

    # Calculate low intervals
    swing_low_times = swing_lows_info.index
    intervals_seconds_low = [
        (swing_low_times[i] - swing_low_times[i - 1]).total_seconds() for i in range(1, len(swing_low_times))
    ]
    avg_distance_low = np.mean(intervals_seconds_low) if intervals_seconds_low else 0

    # Return average
    if avg_distance_high and avg_distance_low:
        return (avg_distance_high + avg_distance_low) / 2
    return avg_distance_high or avg_distance_low

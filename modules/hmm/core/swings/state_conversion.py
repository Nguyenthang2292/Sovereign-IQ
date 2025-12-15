"""
HMM-Swings State Conversion.

This module handles conversion from swing points to market states.
"""

from typing import List
import pandas as pd

from modules.common.utils import log_warn


def convert_swing_to_state(swing_highs_info: pd.DataFrame, swing_lows_info: pd.DataFrame, strict_mode: bool = False) -> List[float]:
    """
    Convert swing high and low points to market state sequence.
    
    States:
    - 0: Downtrend
    - 1: Sideways/Consolidation
    - 2: Uptrend
    
    Methods:
    - strict_mode=True: Compares consecutive swing values requiring equal counts
    - strict_mode=False: Uses chronological transitions between highs and lows
    
    Parameters
    ----------
    swing_highs_info : DataFrame
        Swing highs with 'high' column and datetime index
    swing_lows_info : DataFrame
        Swing lows with 'low' column and datetime index
    strict_mode : bool, default=False
        Whether to use strict comparison mode
        
    Returns
    -------
    List[float]
        Market state values (0, 1, or 2)
    """
    if swing_highs_info.empty or swing_lows_info.empty:
        log_warn("One of the swing DataFrames is empty. Returning empty list.")
        return []
    
    if strict_mode:
        """
        Strict mode: Match swing highs and lows by timestamp proximity,
        then compare consecutive matched pairs to determine state transitions.
        
        This ensures we compare swings that actually occurred close in time,
        rather than just using array indices which can be misaligned.
        """
        states = []
        
        # Ensure DataFrames are sorted by timestamp
        swing_highs_info = swing_highs_info.sort_index()
        swing_lows_info = swing_lows_info.sort_index()
        
        # Create lists of swing points with timestamps
        high_points = []
        for idx, row in swing_highs_info.iterrows():
            high_points.append({'time': idx, 'type': 'high', 'value': row['high']})
        
        low_points = []
        for idx, row in swing_lows_info.iterrows():
            low_points.append({'time': idx, 'type': 'low', 'value': row['low']})
        
        # Linear O(N) algorithm: Maintain state of nearest peak/trough while traversing
        # Strategy: Duyệt tuyến tính, duy trì peak/trough gần nhất, tạo pair khi gặp opposite
        
        # Merge and sort all swings by time
        all_swings = sorted(high_points + low_points, key=lambda x: x['time'])
        
        if len(all_swings) < 2:
            log_warn("Not enough swing points for strict mode comparison")
            return []
        
        matched_pairs = []
        last_high = None  # Track last unmatched high (highest so far)
        last_low = None   # Track last unmatched low (lowest so far)
        
        # Linear traversal: O(N)
        for swing in all_swings:
            if swing['type'] == 'high':
                # If new high is higher than last high (and no low intervened), update
                if last_high is None or swing['value'] > last_high['value']:
                    last_high = swing
                # If we have a low waiting, create pair with current high
                if last_low is not None:
                    pair_time = max(last_high['time'], last_low['time'])
                    matched_pairs.append({
                        'time': pair_time,
                        'high': last_high['value'],
                        'low': last_low['value'],
                    })
                    # Reset low after pairing, keep high for potential next pair
                    last_low = None
            else:  # swing['type'] == 'low'
                # If new low is lower than last low (and no high intervened), update
                if last_low is None or swing['value'] < last_low['value']:
                    last_low = swing
                # If we have a high waiting, create pair with current low
                if last_high is not None:
                    pair_time = max(last_high['time'], last_low['time'])
                    matched_pairs.append({
                        'time': pair_time,
                        'high': last_high['value'],
                        'low': last_low['value'],
                    })
                    # Reset high after pairing, keep low for potential next pair
                    last_high = None
        
        # Sort pairs by timestamp - O(N log N) where N = number of pairs
        unique_pairs = sorted(matched_pairs, key=lambda x: x['time'])
        
        # Compare consecutive pairs to determine states
        for i in range(1, len(unique_pairs)):
            current_pair = unique_pairs[i]
            previous_pair = unique_pairs[i - 1]
            
            current_high = current_pair['high']
            previous_high = previous_pair['high']
            current_low = current_pair['low']
            previous_low = previous_pair['low']
            
            if current_high < previous_high and current_low < previous_low:
                state = 0  # Downtrend
            elif current_high > previous_high and current_low > previous_low:
                state = 2  # Uptrend
            else:
                state = 1  # Sideways/Consolidation
            
            states.append(state)
        
        if len(states) == 0:
            log_warn("No valid state transitions found in strict mode")
        
        return states
    else:
        # Remove rows with NaN values
        swing_highs_info = swing_highs_info.dropna(subset=['high'])
        swing_lows_info = swing_lows_info.dropna(subset=['low'])
        
        # Combine high and low swing points
        swings = []
        for idx in swing_highs_info.index:
            swings.append({'time': idx, 'type': 'high', 'value': swing_highs_info.loc[idx, 'high']})
        for idx in swing_lows_info.index:
            swings.append({'time': idx, 'type': 'low', 'value': swing_lows_info.loc[idx, 'low']})
        
        # Sort and remove duplicates
        swings.sort(key=lambda x: x['time'])
        unique_swings, prev_time = [], None
        for swing in swings:
            if swing['time'] != prev_time:
                unique_swings.append(swing)
                prev_time = swing['time']
        
        # Determine states
        states, prev_swing = [], None
        for swing in unique_swings:
            if prev_swing is None:
                prev_swing = swing
                continue
            
            if prev_swing['type'] == 'low' and swing['type'] == 'high':
                state = 2  # price increase
            elif prev_swing['type'] == 'high' and swing['type'] == 'low':
                state = 0  # price decrease
            else:
                state = 1  # unchanged or mixed
            
            states.append(state)
            prev_swing = swing
        
        return states


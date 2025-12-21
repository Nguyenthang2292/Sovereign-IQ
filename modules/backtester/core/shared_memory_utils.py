"""
Shared memory utilities for parallel processing.

This module provides utilities for setting up shared memory for DataFrames
to enable efficient data sharing between processes without pickling overhead.
"""

from typing import Dict, Optional, Any
import pandas as pd
import numpy as np
import pickle
import uuid

try:
    from multiprocessing import shared_memory
    SHARED_MEMORY_AVAILABLE = True
except ImportError:
    # Python < 3.8 doesn't have shared_memory
    SHARED_MEMORY_AVAILABLE = False


def setup_shared_memory_for_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create shared memory for DataFrame arrays.
    
    This function converts a DataFrame into shared memory arrays that can be
    accessed by multiple processes without pickling overhead.
    
    Args:
        df: DataFrame to share
        
    Returns:
        Dictionary containing:
        - 'shm_objects': Dict mapping column name to shared memory info
        - 'index_info': Pickled index information
        - 'columns': List of column names
        - 'dtypes': Dict mapping column name to dtype string
    """
    if not SHARED_MEMORY_AVAILABLE:
        raise RuntimeError("Shared memory is not available (requires Python 3.8+)")
    
    shm_objects = {}
    shm_refs = {}  # Keep references to SharedMemory objects to prevent garbage collection
    dtypes = {}
    unique_id = str(uuid.uuid4())[:8]  # Short unique ID for naming
    
    # Convert index to pickle bytes (index is typically small)
    index_info = pickle.dumps(df.index, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Process each numeric column
    for col in df.columns:
        if col not in ['open', 'high', 'low', 'close', 'volume']:
            continue  # Skip non-OHLCV columns
        
        # Convert to numpy array
        arr = df[col].values
        
        # Skip if empty
        if arr.size == 0:
            continue
        
        # Ensure contiguous array
        if not arr.flags['C_CONTIGUOUS']:
            arr = np.ascontiguousarray(arr)
        
        # Create shared memory
        shm_name = f'shm_{col}_{unique_id}'
        try:
            shm = shared_memory.SharedMemory(
                create=True,
                size=arr.nbytes,
                name=shm_name
            )
        except FileExistsError:
            # Handle case where shared memory with same name exists
            # Try with different name
            shm_name = f'shm_{col}_{unique_id}_{uuid.uuid4().hex[:8]}'
            shm = shared_memory.SharedMemory(
                create=True,
                size=arr.nbytes,
                name=shm_name
            )
        
        # Copy data to shared memory
        np_array = np.ndarray(
            arr.shape,
            dtype=arr.dtype,
            buffer=shm.buf
        )
        np_array[:] = arr[:]
        
        # Store SharedMemory object reference to prevent garbage collection (Windows requirement)
        shm_refs[col] = shm
        
        shm_objects[col] = {
            'shm_name': shm_name,
            'shape': arr.shape,
            'dtype': arr.dtype,
        }
        dtypes[col] = str(arr.dtype)
        
        # IMPORTANT: Don't close the shared memory here - it needs to remain available
        # We keep the reference in shm_refs to prevent garbage collection on Windows
    
    return {
        'shm_objects': shm_objects,
        'shm_refs': shm_refs,  # Keep references to prevent garbage collection on Windows
        'index_info': index_info,
        'columns': list(df.columns),
        'dtypes': dtypes,
    }


def reconstruct_dataframe_from_shared_memory(shm_info: Dict[str, Any]) -> pd.DataFrame:
    """
    Reconstruct DataFrame from shared memory.
    
    This function reconstructs a DataFrame from shared memory arrays created
    by setup_shared_memory_for_dataframe().
    
    Args:
        shm_info: Dictionary returned by setup_shared_memory_for_dataframe()
        
    Returns:
        Reconstructed DataFrame
    """
    if not SHARED_MEMORY_AVAILABLE:
        raise RuntimeError("Shared memory is not available (requires Python 3.8+)")
    
    # Unpickle index
    index = pickle.loads(shm_info['index_info'])
    
    # Reconstruct data dictionary
    data = {}
    shm_objects = shm_info['shm_objects']
    
    for col, info in shm_objects.items():
        try:
            # Attach to existing shared memory
            shm = shared_memory.SharedMemory(name=info['shm_name'])
            
            # Create numpy array from shared memory buffer
            arr = np.ndarray(
                info['shape'],
                dtype=info['dtype'],
                buffer=shm.buf
            )
            
            # Copy from shared memory to local (to avoid issues when shared memory is closed)
            data[col] = arr.copy()
            
            # Close shared memory connection (doesn't delete the memory)
            shm.close()
        except FileNotFoundError:
            # Shared memory was already deleted/unlinked
            # Skip this column or use NaN
            data[col] = np.full(info['shape'], np.nan, dtype=info['dtype'])
    
    # Create DataFrame
    df = pd.DataFrame(data, index=index)
    
    return df


def cleanup_shared_memory(shm_info: Dict[str, Any]) -> None:
    """
    Clean up shared memory objects.
    
    This function unlinks (deletes) all shared memory objects created by
    setup_shared_memory_for_dataframe(). Call this after all processes
    have finished using the shared memory.
    
    Args:
        shm_info: Dictionary returned by setup_shared_memory_for_dataframe()
    """
    if not SHARED_MEMORY_AVAILABLE:
        return
    
    shm_objects = shm_info.get('shm_objects', {})
    shm_refs = shm_info.get('shm_refs', {})
    
    # First, close all SharedMemory references if they exist
    for col, shm in shm_refs.items():
        try:
            shm.close()
        except (FileNotFoundError, ValueError):
            pass
    
    # Then unlink all shared memory objects
    for col, info in shm_objects.items():
        try:
            # Try to attach to unlink (in case reference is not available)
            shm = shared_memory.SharedMemory(name=info['shm_name'])
            shm.close()
            shm.unlink()  # Delete the shared memory
        except FileNotFoundError:
            # Shared memory already deleted/unlinked (expected on Windows when worker processes finish)
            # This is normal behavior - worker processes may have already cleaned up
            # Silently ignore this expected case
            pass
        except ValueError:
            # Unexpected ValueError - ignore it
            pass
        except Exception:
            # Unexpected error - ignore it
            pass


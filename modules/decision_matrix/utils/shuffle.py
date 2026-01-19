from typing import List, Optional, Union

import numpy as np


class ShuffleMechanism:
    """
    Shuffle mechanism using NumPy for performance.

    Provides methods to:
    1. Generate shuffled indices (vectorized)
    2. Shuffle matrix rows (vectorized)

    Args:
        seed: Optional seed for reproducible random generation.
               If None, uses unpredictable random state.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Initialize shuffle mechanism with optional seed.

        Args:
            seed: Optional random seed for reproducible results.
                   Use None for non-deterministic random generation.

        Raises:
            TypeError: If seed is not int or None.
        """
        # Type validation for seed
        if seed is not None and not isinstance(seed, int):
            raise TypeError(f"seed must be int or None, got {type(seed)}")
        
        self.seed: Optional[int] = seed
        self.rng: np.random.Generator = np.random.default_rng(seed)

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return f"ShuffleMechanism(seed={self.seed})"

    def shuffle_indices(self, n: int) -> np.ndarray:
        """
        Shuffle indices from 0 to n-1 using NumPy.

        Args:
            n: Number of indices to generate

        Returns:
            Shuffled numpy array of indices
        """
        return self.rng.permutation(n)

    def shuffle_matrix(self, matrix: Union[List[list], np.ndarray], rows: int) -> np.ndarray:
        """
        Shuffle the rows of a matrix using NumPy.

        Args:
            matrix: Input matrix (list of lists or numpy array)
            rows: Number of rows to shuffle

        Returns:
            New shuffled numpy array
        """
        # Convert to numpy array if it's a list
        if not isinstance(matrix, np.ndarray):
            matrix_arr = np.array(matrix)
        else:
            matrix_arr = matrix

        if len(matrix_arr) == 0:
            # Return empty array with correct shape if possible, or just empty
            return np.array([])

        # Ensure we don't try to access more rows than available
        rows = min(rows, len(matrix_arr))

        # In Pine Script implementation, it specifically shuffles 0..rows-1 indices
        # and creates a new matrix.
        # Here we can just shuffle the first 'rows' of the input data

        # Take the slice we care about
        subset = matrix_arr[:rows]

        # Shuffle it
        # self.rng.shuffle shuffles in-place along the first axis (rows)
        # To avoid modifying original if it was passed as array, make copy
        shuffled_subset = subset.copy()
        self.rng.shuffle(shuffled_subset)

        return shuffled_subset


__all__ = ["ShuffleMechanism"]

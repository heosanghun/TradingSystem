"""Training rows whose one-bar return is realized within the same split."""
import numpy as np


def within_split_next_bar(indices):
    rows = np.asarray(indices)
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
        raise ValueError("Expected one-dimensional integer bar positions")
    if np.any(rows < 0) or np.any(rows[1:] <= rows[:-1]):
        raise ValueError("Training bar positions must be nonnegative, sorted and unique")
    return rows[:-1][np.diff(rows) == 1]

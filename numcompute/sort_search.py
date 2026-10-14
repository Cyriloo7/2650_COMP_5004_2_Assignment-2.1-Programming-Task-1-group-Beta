"""
sort_search.py
Sorting, top-k selection, quickselect and binary search.

Provides
--------
- stable_sort(arr, axis=-1)
- multi_key_sort(data, keys, ascending=None)
- topk(values, k, largest=True, return_indices=True)
- quickselect(arr, k)
- binary_search(sorted_array, x)

NaN policy (one rule per function, all documented in the docstrings)
--------------------------------------------------------------------
- stable_sort / multi_key_sort : NaNs are sorted to the END (NumPy's
  behaviour), so they never disturb the order of real values.
- topk                         : NaNs are IGNORED. They can never be in
  the result, and they do not count towards `len(values)` when k is
  clamped.
- quickselect                  : NaNs are NOT supported (comparisons with
  NaN are always False, which would break partitioning), so a ValueError
  is raised.
- binary_search                : a NaN search value `x` raises ValueError.
"""

import numpy as np


# ----------------------------------------------------------------------
# Sorting
# ----------------------------------------------------------------------

def stable_sort(arr, axis=-1):
    """Return a sorted copy of `arr` using a stable sorting algorithm.

    "Stable" means that equal elements keep their original relative
    order. This is a thin wrapper over ``np.sort(kind='stable')``.

    Parameters
    ----------
    arr : array_like
        Input array of any shape (including empty and non-contiguous
        views such as ``a[::2]``).
    axis : int, default -1
        Axis along which to sort.

    Returns
    -------
    sorted_arr : ndarray, same shape as `arr`
        Sorted copy. NaNs are placed at the end.

    Raises
    ------
    numpy.exceptions.AxisError
        If `axis` is out of bounds for `arr`.

    Complexity
    ----------
    Time: O(n log n). Space: O(n) for the copy.
    """
    return np.sort(np.asarray(arr), axis=axis, kind="stable")

def multi_key_sort(data, keys, ascending=None):
    """Sort the rows of a 2D array by several columns at once.

    Uses ``np.lexsort`` under the hood. IMPORTANT: following lexsort's
    convention, the LAST key in `keys` is the PRIMARY key (sorted by
    first), and the first key is the last tie-breaker.

    Example: ``keys=[0, 1]`` sorts by column 1 first; rows with equal
    column-1 values are then ordered by column 0.

    Parameters
    ----------
    data : array_like, shape (n_rows, n_cols)
        2D array; each row is one record.
    keys : sequence of int
        Column indices to sort by. The last one is the primary key.
    ascending : sequence of bool, optional
        One flag per key (same order as `keys`). ``True`` = ascending.
        Defaults to all ascending.

    Returns
    -------
    sorted_data : ndarray, shape (n_rows, n_cols)
        A sorted copy of `data` (rows reordered, values unchanged).

    Raises
    ------
    ValueError
        If `data` is not 2D, `keys` is empty, a key is out of range, or
        `ascending` has a different length from `keys`.

    Notes
    -----
    The sort is stable, so rows that tie on every key keep their
    original order. Descending order is done by negating the dense rank
    of a column (not the raw values), so it works for any dtype and
    never overflows. NaNs are placed at the end of ascending keys.

    Complexity
    ----------
    Time: O(k * n log n) for k keys and n rows. Space: O(k * n).
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"multi_key_sort() expects a 2D array, got {data.ndim}D")

    keys = list(keys)
    if len(keys) == 0:
        raise ValueError("keys must contain at least one column index")

    n_cols = data.shape[1]
    for key in keys:
        if not -n_cols <= key < n_cols:
            raise ValueError(
                f"key {key} is out of range for data with {n_cols} columns"
            )

    if ascending is None:
        ascending = [True] * len(keys)
    else:
        ascending = list(ascending)
        if len(ascending) != len(keys):
            raise ValueError(
                f"ascending has length {len(ascending)} but keys has length {len(keys)}"
            )

    sort_columns = []
    for key, asc in zip(keys, ascending):
        column = data[:, key]
        if not asc:
            # Dense rank (0..u-1), negated: reverses the order, keeps ties tied.
            _, ranks = np.unique(column, return_inverse=True)
            column = -ranks.reshape(-1)
        sort_columns.append(column)

    order = np.lexsort(tuple(sort_columns))
    return data[order]
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
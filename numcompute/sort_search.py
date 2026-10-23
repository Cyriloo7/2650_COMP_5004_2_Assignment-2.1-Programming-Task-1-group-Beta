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
- stable_sort                  : NaNs are sorted to the END (NumPy's
  behaviour), so they never disturb the order of real values.
- multi_key_sort               : NaNs are sorted to the END of an
  ASCENDING key and to the START of a DESCENDING key. A descending sort
  is the exact reverse of an ascending one, and NaN counts as the
  largest value. Example for one column: ``[1, nan, 3]`` gives
  ``[1, 3, nan]`` ascending and ``[nan, 3, 1]`` descending.
- topk                         : NaNs are IGNORED. They can never be in
  the result, and they do not count towards `len(values)` when k is
  clamped.
- quickselect                  : NaNs are NOT supported (comparisons with
  NaN are always False, which would break partitioning), so a ValueError
  is raised.
- binary_search                : a NaN search value `x` raises ValueError.

Input-type policy
-----------------
- Integer parameters (`k`, the entries of `keys`) must be real integers.
  Floats such as ``1.5`` or ``2.0``, strings and booleans raise
  TypeError. Integers that are out of range raise ValueError.
- topk / quickselect / binary_search need numeric (or boolean) arrays.
  Text or object arrays raise TypeError.
"""

import numbers

import numpy as np


# ----------------------------------------------------------------------
# Private validation helpers
# ----------------------------------------------------------------------

def _check_integer(value, name):
    """Return `value` as a Python int, or raise TypeError if it is not an integer.

    Booleans are rejected on purpose (``True`` is technically an int in
    Python, but ``k=True`` is almost certainly a mistake). NumPy integer
    scalars such as ``np.int64(3)`` are accepted.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(
            f"{name} must be an integer, got {type(value).__name__} ({value!r})"
        )
    return int(value)


def _check_numeric_dtype(arr, func_name):
    """Raise TypeError unless `arr` holds booleans, integers or floats.

    This turns an obscure crash deep inside NumPy (for example
    ``np.isnan`` on a string array) into a clear message up front.
    """
    if arr.dtype.kind not in "biuf":
        raise TypeError(
            f"{func_name}() expects a numeric array, got dtype {arr.dtype}"
        )


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
        Negative indices count from the last column.
    ascending : sequence of bool, optional
        One flag per key (same order as `keys`). ``True`` = ascending.
        Defaults to all ascending.

    Returns
    -------
    sorted_data : ndarray, shape (n_rows, n_cols)
        A sorted copy of `data` (rows reordered, values unchanged).

    Raises
    ------
    TypeError
        If `keys` is not a sequence, or any key is not an integer
        (for example ``1.5`` or ``"a"``).
    ValueError
        If `data` is not 2D, `keys` is empty, a key is out of range, or
        `ascending` has a different length from `keys`.

    Notes
    -----
    The sort is stable, so rows that tie on every key keep their
    original order. Descending order is done by negating the dense rank
    of a column (not the raw values), so it works for any dtype and
    never overflows.

    NaN position: NaNs go to the END of an ascending key but to the
    START of a descending key (NaN counts as the largest value, and a
    descending sort reverses the order). For one column,
    ``[1, nan, 3]`` becomes ``[1, 3, nan]`` ascending and
    ``[nan, 3, 1]`` descending.

    Complexity
    ----------
    Time: O(k * n log n) for k keys and n rows. Space: O(k * n).
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"multi_key_sort() expects a 2D array, got {data.ndim}D")

    try:
        keys = list(keys)
    except TypeError:
        raise TypeError(
            f"keys must be a sequence of column indices, got {type(keys).__name__}"
        ) from None
    if len(keys) == 0:
        raise ValueError("keys must contain at least one column index")

    # Type check first, range check second: comparing a string or a float
    # key against n_cols would otherwise give a confusing message.
    keys = [_check_integer(key, "key") for key in keys]

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


# ----------------------------------------------------------------------
# Top-k
# ----------------------------------------------------------------------

def topk(values, k, largest=True, return_indices=True):
    """Return the k largest (or smallest) values of a 1D array, in order.

    Strategy: ``np.argpartition`` finds the k best elements in O(n)
    without sorting everything, then ``np.argsort`` orders just those
    k elements.

    Parameters
    ----------
    values : array_like, shape (n,)
        1D numeric input. NaNs are ignored (see Notes).
    k : int
        How many elements to return. Must be an integer >= 1.
    largest : bool, default True
        ``True`` -> k largest, sorted from largest to smallest.
        ``False`` -> k smallest, sorted from smallest to largest.
    return_indices : bool, default True
        If True, also return the positions of the chosen elements in the
        ORIGINAL array.

    Returns
    -------
    top_values : ndarray, shape (min(k, m),)
        The selected values, best first (m = number of non-NaN values).
    top_indices : ndarray of int, shape (min(k, m),)
        Only returned when ``return_indices=True``. Indices into `values`.

    Raises
    ------
    TypeError
        If `k` is not an integer (for example ``2.7``), or `values` is
        not a numeric array (for example a string).
    ValueError
        If `k <= 0` or `values` is not 1D.

    Notes
    -----
    - k larger than the number of usable values is CLAMPED (not an
      error): you simply get everything, sorted.
    - NaN policy: NaNs are ignored and never returned. If all values are
      NaN (or the array is empty) the result is empty.
    - Among equal values the order is not guaranteed to follow the
      original positions when ``largest=True``.

    Complexity
    ----------
    Time: O(n + k log k) expected. Space: O(n).
    """
    values = np.asarray(values)
    _check_numeric_dtype(values, "topk")
    if values.ndim != 1:
        raise ValueError(f"topk() expects a 1D array, got {values.ndim}D")
    k = _check_integer(k, "k")
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")

    # NaN policy: drop NaNs but remember where the real values were.
    valid_positions = np.flatnonzero(~np.isnan(values))
    valid = values[valid_positions]
    n = valid.size
    k_eff = min(k, n)  # clamp

    if k_eff == 0:
        empty_values = values[:0]
        empty_idx = np.empty(0, dtype=np.intp)
        return (empty_values, empty_idx) if return_indices else empty_values

    if largest:
        # The k largest sit in the last k slots once position n-k is partitioned.
        part = np.argpartition(valid, n - k_eff)[n - k_eff:]
    else:
        part = np.argpartition(valid, k_eff - 1)[:k_eff]

    # Order only the k chosen elements (cheap: k log k, not n log n).
    ordered = part[np.argsort(valid[part], kind="stable")]
    if largest:
        ordered = ordered[::-1]

    top_indices = valid_positions[ordered]
    top_values = values[top_indices]
    return (top_values, top_indices) if return_indices else top_values


# ----------------------------------------------------------------------
# Quickselect
# ----------------------------------------------------------------------

def quickselect(arr, k):
    """Return the k-th smallest element of `arr` (0-based) without fully sorting.

    Hand-written, iterative Lomuto-partition quickselect (the
    "educational" version). The loop is intentional: each pass
    partitions the array around a pivot and then keeps only the side
    that contains position k.

    Parameters
    ----------
    arr : array_like, shape (n,)
        1D numeric input. It is NOT modified (a copy is partitioned).
    k : int
        0-based rank to find: ``k=0`` is the minimum,
        ``k=n-1`` is the maximum, ``k=n//2`` is roughly the median.
        Must be an integer.

    Returns
    -------
    value : scalar
        The element that would be at index k if `arr` were sorted.

    Raises
    ------
    TypeError
        If `arr` is not a numeric array, or `k` is not an integer
        (for example ``1.5``).
    ValueError
        If `arr` is not 1D, is empty, contains NaN, or `k` is outside
        ``[0, n-1]``.

    Notes
    -----
    The middle element is used as the pivot (moved to the end first),
    which avoids the classic worst case on already-sorted input. Lomuto
    partitioning still degrades on arrays where almost all values are
    equal; that is the O(n^2) case below.

    Complexity
    ----------
    Time: O(n) expected, O(n^2) worst case. Space: O(n) for the copy
    (the partitioning itself is in place and iterative, no recursion).
    """
    a = np.array(arr, copy=True)
    _check_numeric_dtype(a, "quickselect")
    if a.ndim != 1:
        raise ValueError(f"quickselect() expects a 1D array, got {a.ndim}D")
    n = a.size
    if n == 0:
        raise ValueError("quickselect() cannot select from an empty array")
    if np.any(np.isnan(a)):
        raise ValueError("quickselect() does not support NaN values")
    k = _check_integer(k, "k")
    if not 0 <= k < n:
        raise ValueError(f"k={k} is out of range for an array of length {n}")

    lo, hi = 0, n - 1
    while True:
        if lo == hi:
            return a[lo]

        # Move the middle element to the end to use it as the pivot.
        mid = (lo + hi) // 2
        a[mid], a[hi] = a[hi], a[mid]
        pivot = a[hi]

        # Lomuto partition: everything < pivot gets packed to the left.
        store = lo
        for j in range(lo, hi):
            if a[j] < pivot:
                a[store], a[j] = a[j], a[store]
                store += 1
        a[store], a[hi] = a[hi], a[store]  # pivot lands in its final spot

        if k == store:
            return a[store]
        elif k < store:
            hi = store - 1
        else:
            lo = store + 1


# ----------------------------------------------------------------------
# Binary search
# ----------------------------------------------------------------------

def binary_search(sorted_array, x):
    """Binary search in an ascending sorted 1D array.

    Behaves like ``bisect.bisect_left`` / ``np.searchsorted(side='left')``:
    the returned index is the FIRST position where `x` could be inserted
    while keeping the array sorted.

    Parameters
    ----------
    sorted_array : array_like, shape (n,)
        1D numeric array sorted in ascending order. This is assumed, not
        checked (checking would cost O(n) and defeat the purpose).
    x : scalar
        Real number to look for.

    Returns
    -------
    index : int
        Insertion index in ``[0, n]``. If `x` is present it is the
        position of its first occurrence.
    exists : bool
        True if `x` is actually in the array.

    Raises
    ------
    TypeError
        If `sorted_array` is not a numeric array, or `x` is not a real
        number (for example a string).
    ValueError
        If `sorted_array` is not 1D, or `x` is NaN.

    Complexity
    ----------
    Time: O(log n). Space: O(1).
    """
    a = np.asarray(sorted_array)
    _check_numeric_dtype(a, "binary_search")
    if a.ndim != 1:
        raise ValueError(f"binary_search() expects a 1D array, got {a.ndim}D")
    if not isinstance(x, numbers.Real):
        raise TypeError(
            f"x must be a real number, got {type(x).__name__} ({x!r})"
        )
    if np.isnan(x):
        raise ValueError("binary_search() cannot search for NaN")

    lo, hi = 0, a.size
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid

    exists = bool(lo < a.size and a[lo] == x)
    return lo, exists
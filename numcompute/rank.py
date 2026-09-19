"""
rank.py
Tie-aware ranking and percentile utilities.
 
Provides
--------
- rank(data, method='average'|'dense'|'ordinal', axis=-1)
- percentile(data, q, interpolation='linear'|'lower'|'higher'|'midpoint')
 
Design notes
------------
All ranking methods share one base pass:
    1. np.argsort the data to get order statistics
    2. np.unique(..., return_inverse=True, return_counts=True) to find
       groups of tied values and their sizes
 
From that shared base we derive 'ordinal', 'dense', and 'average' ranks
without duplicating the sorting logic, per the assignment spec.
"""
 
import numpy as np
 
 
def _rank_1d(x, method):
    """Rank a single 1D array, handling ties according to `method`.
 
    NaN policy
    ----------
    NaN values are always ranked last (highest ranks), regardless of
    `method`, and are treated as mutually tied with each other (so with
    method='average' multiple NaNs share the average of the top ranks).
    This mirrors pandas' default `na_option='keep'`-adjacent behaviour
    of pushing missing data to the end rather than silently propagating
    NaN into every rank.
 
    Parameters
    ----------
    x : array_like, shape (n,)
        1D array of values to rank. May contain NaN.
    method : {'average', 'dense', 'ordinal'}
        Tie-breaking method.
 
    Returns
    -------
    ranks : ndarray of float, shape (n,)
        1-based ranks of each value in `x`.
 
    Raises
    ------
    ValueError
        If `method` is not one of the supported options.
 
    Complexity
    ----------
    Time: O(n log n) for the sort. Space: O(n).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
 
    if n == 0:
        return np.array([], dtype=float)
 
    # Order statistics: stable sort so ties keep their original relative order.
    order = np.argsort(x, kind="stable")
 
    # 'ordinal': every element gets a unique rank based on sort position.
    ordinal_ranks = np.empty(n, dtype=float)
    ordinal_ranks[order] = np.arange(1, n + 1)
 
    if method == "ordinal":
        return ordinal_ranks
 
    # Shared base for 'dense' and 'average': group identical values together.
    sorted_x = x[order]
    unique_vals, inverse, counts = np.unique(
        sorted_x, return_inverse=True, return_counts=True
    )
 
    if method == "dense":
        # Dense rank = 1-based index of the value's group among unique values.
        dense_sorted = inverse + 1
        dense_ranks = np.empty(n, dtype=float)
        dense_ranks[order] = dense_sorted
        return dense_ranks
 
    if method == "average":
        # Average rank = mean of the ordinal ranks within each tied group.
        # cumsum(counts) gives the LAST ordinal rank of each group;
        # subtracting (count-1)/2 gives that group's average ordinal rank.
        group_last_rank = np.cumsum(counts)
        group_avg_rank = group_last_rank - (counts - 1) / 2.0
        avg_sorted = group_avg_rank[inverse]
        avg_ranks = np.empty(n, dtype=float)
        avg_ranks[order] = avg_sorted
        return avg_ranks
 
    raise ValueError(
        f"Unknown method {method!r}; expected 'average', 'dense', or 'ordinal'"
    )
 
 
def rank(data, method="average", axis=-1):
    """Assign ranks to data, handling ties according to `method`.
 
    Parameters
    ----------
    data : array_like, shape (n,) or (n, m)
        Input data. 1D or 2D. For 2D input, ranking is applied
        independently to each column.
    method : {'average', 'dense', 'ordinal'}, default 'average'
        - 'average': tied values get the mean of the ranks they would
          occupy (matches scipy.stats.rankdata(method='average')).
        - 'dense': like 'ordinal' but ties share a rank and the next rank
          is not skipped (like SQL DENSE_RANK).
        - 'ordinal': every value gets a distinct rank based on stable sort
          order, ties broken by original position.
    axis : int, default -1
        Present for API compatibility. 2D input is always ranked per
        column (a Python loop over columns is acceptable per the spec).
 
    Returns
    -------
    ranks : ndarray of float, same shape as `data`
        1-based ranks.
 
    Raises
    ------
    ValueError
        If `method` is invalid or `data` has more than 2 dimensions.
 
    Complexity
    ----------
    1D: O(n log n) time, O(n) space.
    2D with m columns: O(m * n log n) time (looping over columns).
    """
    data = np.asarray(data, dtype=float)
 
    if data.ndim == 1:
        return _rank_1d(data, method)
 
    if data.ndim == 2:
        n_rows, n_cols = data.shape
        out = np.empty_like(data, dtype=float)
        for col in range(n_cols):
            out[:, col] = _rank_1d(data[:, col], method)
        return out
 
    raise ValueError(f"rank() supports 1D or 2D input, got {data.ndim}D")
 
 
def percentile(data, q, interpolation="linear"):
    """Compute the q-th percentile(s) of data, ignoring NaNs.
 
    Thin wrapper around numpy.nanpercentile. This is the single place
    that owns percentile semantics for the package -- stats.quantile()
    should call this instead of duplicating the logic.
 
    Parameters
    ----------
    data : array_like
        Input data; NaNs are ignored.
    q : float or array_like of float
        Percentile(s) to compute, in [0, 100].
    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}, default 'linear'
        Interpolation method used when the desired percentile lies
        between two data points. NumPy renamed this argument to
        `method=`; this wrapper keeps the assignment's original name so
        call sites don't need to know about the rename.
 
    Returns
    -------
    result : float or ndarray
        The requested percentile(s).
 
    Raises
    ------
    ValueError
        If `q` is outside [0, 100], or `interpolation` is not a value
        NumPy accepts for `method=`.
 
    Complexity
    ----------
    Time: O(n log n) (NumPy sorts internally). Space: O(n).
    """
    data = np.asarray(data, dtype=float)
    q_arr = np.asarray(q, dtype=float)
 
    if np.any((q_arr < 0) | (q_arr > 100)):
        raise ValueError("q must be between 0 and 100")
 
    try:
        return np.nanpercentile(data, q, method=interpolation)
    except ValueError as e:
        raise ValueError(f"Invalid interpolation {interpolation!r}: {e}") from e
 
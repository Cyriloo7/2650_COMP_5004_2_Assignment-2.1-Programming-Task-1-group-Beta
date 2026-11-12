"""
rank.py
Tie-aware ranking and percentile utilities.

Provides
--------
- rank(data, method='average'|'dense'|'ordinal', axis=0)
- percentile(data, q, interpolation='linear'|'lower'|'higher'|'midpoint'|'nearest', axis=None)

Design notes
------------
All ranking methods share one base pass:
    1. np.argsort the data to get order statistics
    2. np.unique(..., return_inverse=True, return_counts=True) to find
       groups of tied values and their sizes

From that shared base we derive 'ordinal', 'dense', and 'average' ranks
without duplicating the sorting logic, per the assignment spec.

Axis convention (same as NumPy)
-------------------------------
`axis` is the axis that gets ranked / reduced. For a 2-D array,
``axis=0`` works down each column and ``axis=1`` works along each row.
Negative axes count from the end, so ``axis=-1`` is the last axis.
"""

import numpy as np

_RANK_METHODS = ("average", "dense", "ordinal")
_PERCENTILE_METHODS = ("linear", "lower", "higher", "midpoint", "nearest")


def _as_float_array(data, owner):
    """Convert `data` to a float ndarray, or raise a clear TypeError.

    Parameters
    ----------
    data : array_like
        Anything NumPy can turn into a numeric array.
    owner : str
        Name of the calling function, used in the error message.

    Returns
    -------
    arr : ndarray of float
        `data` as a float array (no copy if it already is one).

    Raises
    ------
    TypeError
        If `data` cannot be converted to floats (strings, ragged
        nested lists, ...).

    Complexity
    ----------
    Time: O(n) for the conversion. Space: O(1) if no copy is needed,
    otherwise O(n).
    """
    try:
        return np.asarray(data, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{owner}() needs numeric input that NumPy can turn into a "
            f"float array: {exc}"
        ) from exc


def _normalise_axis(axis, ndim, owner):
    """Check `axis` and turn a negative axis into its non-negative form.

    Parameters
    ----------
    axis : int
        Axis to validate. Negative values count from the last axis.
    ndim : int
        Number of dimensions of the array the axis refers to.
    owner : str
        Name of the calling function, used in the error message.

    Returns
    -------
    axis : int
        The same axis as a value in ``[0, ndim - 1]``.

    Raises
    ------
    TypeError
        If `axis` is not an integer (bool is rejected too).
    ValueError
        If `axis` is outside ``[-ndim, ndim - 1]``.

    Complexity
    ----------
    Time: O(1). Space: O(1).
    """
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise TypeError(f"{owner}(): axis must be an integer, got {axis!r}")
    if not -ndim <= axis < ndim:
        raise ValueError(
            f"{owner}(): axis {axis} is out of bounds for an array "
            f"with {ndim} dimension(s)"
        )
    return int(axis) % ndim


def _rank_1d(x, method):
    """Rank a single 1D array, handling ties according to `method`.

    The caller (`rank`) has already checked that `method` is valid, so
    this helper does not re-check it.

    NaN policy
    ----------
    NaN values are always ranked last (highest ranks), regardless of
    `method`, and are treated as mutually tied with each other (so with
    method='average' multiple NaNs share the average of the top ranks).
    This matches pandas' ``na_option='bottom'`` (missing values get the
    highest ranks) rather than silently propagating NaN into every rank.
    Note that pandas' default, ``na_option='keep'``, would leave the NaN
    positions as NaN instead; we do not do that.

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

    # method == "average": mean of the ordinal ranks within each tied group.
    # cumsum(counts) gives the LAST ordinal rank of each group;
    # subtracting (count-1)/2 gives that group's average ordinal rank.
    group_last_rank = np.cumsum(counts)
    group_avg_rank = group_last_rank - (counts - 1) / 2.0
    avg_sorted = group_avg_rank[inverse]
    avg_ranks = np.empty(n, dtype=float)
    avg_ranks[order] = avg_sorted
    return avg_ranks


def rank(data, method="average", axis=0):
    """Assign ranks to data, handling ties according to `method`.

    Parameters
    ----------
    data : array_like, shape (n,) or (n, m)
        Input data, 1D or 2D. NaNs are ranked last (see `_rank_1d`).
    method : {'average', 'dense', 'ordinal'}, default 'average'
        - 'average': tied values get the mean of the ranks they would
          occupy (matches scipy.stats.rankdata(method='average')).
        - 'dense': like 'ordinal' but ties share a rank and the next rank
          is not skipped (like SQL DENSE_RANK).
        - 'ordinal': every value gets a distinct rank based on stable sort
          order, ties broken by original position.
    axis : int, default 0
        Axis to rank along. For 2D input, ``axis=0`` ranks each column
        independently (the default, so older calls give the same result)
        and ``axis=1`` (or ``-1``) ranks each row independently. For 1D
        input only ``0`` and ``-1`` are valid.

    Returns
    -------
    ranks : ndarray of float, same shape as `data`
        1-based ranks.

    Raises
    ------
    ValueError
        If `method` is invalid, `data` is not 1D or 2D, or `axis` is out
        of bounds. `method` is checked first, so a bad method is reported
        even for empty input.
    TypeError
        If `data` is not numeric (e.g. strings), or `axis` is not an int.

    Complexity
    ----------
    1D: O(n log n) time, O(n) space.
    2D with k lines to rank (k = m for axis=0, n for axis=1): O(N log L)
    time overall for N elements and line length L, looping over lines
    (a Python loop over lines is acceptable per the spec).
    """
    if method not in _RANK_METHODS:
        raise ValueError(
            f"Unknown method {method!r}; expected 'average', 'dense', or 'ordinal'"
        )

    data = _as_float_array(data, "rank")

    if data.ndim not in (1, 2):
        raise ValueError(f"rank() supports 1D or 2D input, got {data.ndim}D")

    axis = _normalise_axis(axis, data.ndim, "rank")

    if data.ndim == 1:
        return _rank_1d(data, method)

    # Move the axis to rank along to the front so every line to rank is a
    # column, rank each column, then move the axis back.
    moved = np.moveaxis(data, axis, 0)
    out = np.empty_like(moved, dtype=float)
    for line in range(moved.shape[1]):
        out[:, line] = _rank_1d(moved[:, line], method)
    return np.moveaxis(out, 0, axis)


def percentile(data, q, interpolation="linear", axis=None):
    """Compute the q-th percentile(s) of data, ignoring NaNs.

    Thin wrapper around numpy.nanpercentile. This is the single place
    that owns percentile semantics for the package -- stats.quantile()
    calls this instead of duplicating the logic.

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
    axis : int or None, default None
        Axis to compute along. ``None`` (the default) flattens `data`
        first, so 2D input gives one number per percentile. ``axis=0``
        gives one result per column and ``axis=1`` one per row. With an
        array-valued `q` the percentile index comes first in the result,
        exactly like ``np.nanpercentile``.

    Returns
    -------
    result : float or ndarray
        The requested percentile(s).

    Raises
    ------
    ValueError
        If `data` is empty, `q` is outside [0, 100] (or NaN),
        `interpolation` is not one of the supported names, or `axis` is
        out of bounds.
    TypeError
        If `data` is not numeric, or `axis` is not an int or None.

    Complexity
    ----------
    Time: O(n log n) (NumPy sorts internally). Space: O(n).
    """
    data = _as_float_array(data, "percentile")
    q_arr = _as_float_array(q, "percentile")

    if data.size == 0:
        raise ValueError("percentile(): data is empty")

    if not np.all((q_arr >= 0) & (q_arr <= 100)):
        raise ValueError("q must be between 0 and 100")

    if interpolation not in _PERCENTILE_METHODS:
        raise ValueError(
            f"Invalid interpolation {interpolation!r}; expected one of "
            f"{', '.join(_PERCENTILE_METHODS)}"
        )

    if axis is not None:
        axis = _normalise_axis(axis, data.ndim, "percentile")

    return np.nanpercentile(data, q, axis=axis, method=interpolation)
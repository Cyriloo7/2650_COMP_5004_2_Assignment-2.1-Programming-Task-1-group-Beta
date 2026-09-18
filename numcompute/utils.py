"""
utils.py -- Shared numeric helpers for NumCompute.

Small, general-purpose functions that more than one other module needs,
kept here once so the logic isn't copy-pasted (and potentially made
inconsistent) across files.

Provides:
    euclidean_distance(A, B)   -> pairwise Euclidean distance matrix
    manhattan_distance(A, B)   -> pairwise Manhattan (L1) distance matrix
    cosine_distance(A, B)      -> pairwise cosine distance matrix
    sigmoid(x)                 -> logistic sigmoid, overflow-safe
    relu(x)                    -> rectified linear unit
    softmax(x, axis=-1)        -> numerically stable softmax
    logsumexp(x, axis=None)    -> numerically stable log(sum(exp(x)))
    safe_divide(num, denom, fill=0.0) -> division with no crash / no NaN
    batch_iterator(X, batch_size)     -> yields successive slices of X
"""

import numpy as np


def _check_same_feature_count(A, B, fn_name):
    """Shared shape check used by every pairwise-distance function."""
    if A.shape[1] != B.shape[1]:
        raise ValueError(
            f"{fn_name}(): A has {A.shape[1]} columns but B has "
            f"{B.shape[1]} -- both must describe points in the same "
            f"number of dimensions"
        )


def euclidean_distance(A, B):
    """
    Pairwise Euclidean distance between every point in A and every point in B.

    Uses the expanded algebraic form
        ||a - b||^2 = ||a||^2 + ||b||^2 - 2 (a . b)
    so the whole (n, m) result is built from a handful of array
    operations and one matrix multiply -- never a full (n, m, d)
    intermediate array, which is what a naive broadcasted subtraction
    would create.

    Parameters
    ----------
    A : array_like, shape (n, d)
    B : array_like, shape (m, d)

    Returns
    -------
    ndarray, shape (n, m)
        distances[i, j] = Euclidean distance between A[i] and B[j].

    Raises
    ------
    ValueError
        If A and B don't have the same number of columns (dimensions).

    Complexity
    ----------
    O(n*m*d) time (dominated by the A @ B.T matrix multiply),
    O(n*m) space for the output.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    _check_same_feature_count(A, B, "euclidean_distance")

    a_sq = np.sum(A ** 2, axis=1, keepdims=True)       # shape (n, 1)
    b_sq = np.sum(B ** 2, axis=1, keepdims=True).T     # shape (1, m)
    cross = A @ B.T                                    # shape (n, m)

    sq_dists = a_sq + b_sq - 2 * cross
    # Floating-point rounding can push a true-zero distance slightly
    # negative (e.g. -1e-16). sqrt() of that is NaN, so clip it first.
    sq_dists = np.maximum(sq_dists, 0.0)
    return np.sqrt(sq_dists)


def manhattan_distance(A, B):
    """
    Pairwise Manhattan (L1) distance between every point in A and every
    point in B.

    Unlike Euclidean distance, L1 distance has no equivalent algebraic
    expansion, so this does use a (n, m, d) intermediate array. Fine for
    the data sizes in this toolkit; for very large A/B, chunk over rows
    of A and call this on each chunk.

    Parameters
    ----------
    A : array_like, shape (n, d)
    B : array_like, shape (m, d)

    Returns
    -------
    ndarray, shape (n, m)
        distances[i, j] = sum(|A[i, k] - B[j, k]| for k in range(d)).

    Raises
    ------
    ValueError
        If A and B don't have the same number of columns (dimensions).

    Complexity
    ----------
    O(n*m*d) time and space.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    _check_same_feature_count(A, B, "manhattan_distance")

    diffs = np.abs(A[:, None, :] - B[None, :, :])  # shape (n, m, d)
    return diffs.sum(axis=2)


def cosine_distance(A, B):
    """
    Pairwise cosine distance (1 - cosine similarity) between every point
    in A and every point in B.

    A zero vector has no defined direction, so pairs involving one are
    given a similarity of 0 (distance of 1) rather than raising or
    producing NaN -- see `safe_divide`.

    Parameters
    ----------
    A : array_like, shape (n, d)
    B : array_like, shape (m, d)

    Returns
    -------
    ndarray, shape (n, m)
        distances[i, j] = 1 - cosine_similarity(A[i], B[j]).

    Raises
    ------
    ValueError
        If A and B don't have the same number of columns (dimensions).

    Complexity
    ----------
    O(n*m*d) time (dominated by the A @ B.T matrix multiply),
    O(n*m) space for the output.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    _check_same_feature_count(A, B, "cosine_distance")

    a_norm = np.linalg.norm(A, axis=1, keepdims=True)  # shape (n, 1)
    b_norm = np.linalg.norm(B, axis=1, keepdims=True)  # shape (m, 1)

    similarity = safe_divide(A @ B.T, a_norm @ b_norm.T, fill=0.0)
    return 1.0 - similarity


def sigmoid(x):
    """
    Logistic sigmoid, 1 / (1 + exp(-x)), computed without overflow.

    A direct translation of the formula overflows `exp()` for very
    negative x (e.g. x = -1000), producing a RuntimeWarning even though
    the final answer would still come out correct. This avoids the
    overflow entirely by using an equivalent formula for negative x.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    ndarray
        Same shape as `x`, every value in (0, 1).

    Complexity
    ----------
    O(n) time and space, where n is the number of elements in `x`.
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)

    non_negative = x >= 0
    out[non_negative] = 1.0 / (1.0 + np.exp(-x[non_negative]))

    exp_x = np.exp(x[~non_negative])
    out[~non_negative] = exp_x / (1.0 + exp_x)

    return out


def relu(x):
    """
    Rectified linear unit: max(x, 0), elementwise.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    ndarray
        Same shape as `x`.

    Complexity
    ----------
    O(n) time and space.
    """
    x = np.asarray(x, dtype=float)
    return np.maximum(x, 0.0)


def softmax(x, axis=-1):
    """
    Numerically stable softmax: turns a vector of numbers into
    probabilities that sum to 1.

    Subtracts the max along `axis` before exponentiating. This doesn't
    change the mathematical answer, but keeps every exponentiated value
    from overflowing -- computing this directly on large inputs (e.g.
    [1000, 1001, 1002]) returns [nan, nan, nan]; this returns the
    correct [0.090, 0.245, 0.665].

    Parameters
    ----------
    x : array_like
    axis : int, default=-1
        Axis the probabilities should sum to 1 along.

    Returns
    -------
    ndarray
        Same shape as `x`.

    Complexity
    ----------
    O(n) time and space.
    """
    x = np.asarray(x, dtype=float)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=axis, keepdims=True)


def logsumexp(x, axis=None):
    """
    Numerically stable log(sum(exp(x))).

    Uses the same max-shift trick as `softmax`, so it never overflows
    even when `x` contains large values.

    Parameters
    ----------
    x : array_like
    axis : int or None, default=None
        Axis to reduce over. None reduces over the whole array.

    Returns
    -------
    float or ndarray
        A plain float if axis=None, otherwise an ndarray with `axis`
        removed from x.shape (matching the shape convention of
        `np.sum`).

    Complexity
    ----------
    O(n) time and space.
    """
    x = np.asarray(x, dtype=float)
    m = np.max(x, axis=axis, keepdims=True)
    summed = np.sum(np.exp(x - m), axis=axis, keepdims=True)
    result = m + np.log(summed)
    return np.squeeze(result, axis=axis) if axis is not None else result.item()


def safe_divide(num, denom, fill=0.0):
    """
    Elementwise division that substitutes `fill` wherever the
    denominator is zero, instead of raising, warning, or producing
    inf/NaN.

    Parameters
    ----------
    num : array_like
    denom : array_like
        Same shape as `num` (or broadcastable to it).
    fill : float, default=0.0
        Value to use wherever `denom` is exactly 0.

    Returns
    -------
    ndarray
        Elementwise num / denom, with `fill` substituted for divisions
        by zero.

    Complexity
    ----------
    O(n) time and space.
    """
    num = np.asarray(num, dtype=float)
    denom = np.asarray(denom, dtype=float)

    is_zero = denom == 0
    # Substitute a harmless placeholder before dividing, so the division
    # itself never actually happens by zero -- this is what keeps this
    # silent, with no RuntimeWarning, unlike np.where(denom==0, fill, num/denom).
    safe_denom = np.where(is_zero, 1.0, denom)
    return np.where(is_zero, fill, num / safe_denom)


def batch_iterator(X, batch_size):
    """
    Yield successive `batch_size`-sized slices of `X`.

    The final batch may be smaller than `batch_size` if `len(X)` isn't
    an exact multiple of it.

    Parameters
    ----------
    X : array_like
    batch_size : int
        Must be a positive integer.

    Yields
    ------
    ndarray
        Successive slices of `X`, in order, covering every element
        exactly once.

    Raises
    ------
    ValueError
        If `batch_size` is not a positive integer.

    Complexity
    ----------
    O(1) time and space per batch yielded (no copying beyond the slice
    itself); O(n) total across the full iteration.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_iterator(): batch_size must be positive, got {batch_size}")

    X = np.asarray(X)
    n = len(X)
    for start in range(0, n, batch_size):
        yield X[start:start + batch_size]
"""Shared numeric helpers.

Contents:
    euclidean_distance, manhattan_distance, cosine_distance: pairwise distances.
    sigmoid, relu: activation functions.
    softmax, log_softmax, logsumexp: numerically stable exponential sums.
    safe_divide: division with a fallback where the denominator is zero.
    topk_indices: indices of the k largest or smallest entries along an axis.
    batch_iterator: successive slices of an array.
"""

import numpy as np


def _is_integer(value):
    """Return True for Python and NumPy integers (bool excluded)."""
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _validate_points(A, B, fn_name):
    """Check that A and B are 2-D point sets with the same number of columns."""
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(
            f"{fn_name}(): A and B must be 2-D (n_points, n_features) arrays, "
            f"got shapes {A.shape} and {B.shape}"
        )
    if A.shape[1] != B.shape[1]:
        raise ValueError(
            f"{fn_name}(): A has {A.shape[1]} columns but B has "
            f"{B.shape[1]} -- both must describe points in the same "
            f"number of dimensions"
        )


def _finite_mean(A, B):
    """Column means over the finite entries of A and B (0 for a column with none).

    Ignoring non-finite entries stops a NaN or inf in one point from
    affecting the centre, and so the distances, of the others.
    """
    total = np.zeros(A.shape[1])
    count = np.zeros(A.shape[1])
    for points in (A, B):
        finite = np.isfinite(points)
        total += np.where(finite, points, 0.0).sum(axis=0)
        count += finite.sum(axis=0)
    return total / np.maximum(count, 1)


def _exp_shift(x, axis):
    """Return the slice maxima `m` and a shifted exponent array for exp-sums.

    `shifted` is <= 0 everywhere, so np.exp(shifted) cannot overflow. In a
    slice whose maximum is +inf, the +inf entries map to 0 and all others to
    -inf, so the +inf entries share the weight equally. A slice that is
    entirely -inf stays -inf.
    """
    m = np.max(x, axis=axis, keepdims=True)
    centre = np.where(np.isfinite(m), m, 0.0)
    shifted = np.where(
        np.isposinf(m), np.where(np.isposinf(x), 0.0, -np.inf), x - centre
    )
    return m, shifted


def euclidean_distance(A, B):
    """Pairwise Euclidean distances between the rows of A and the rows of B.

    Uses ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b, which avoids an (n, m, d)
    intermediate. That form loses precision for nearby points far from the
    origin, so both inputs are first shifted by their shared mean (distances
    are translation-invariant), at the cost of one extra copy of each.
    Distances that are tiny relative to the spread of the data can still be
    inexact.

    Parameters
    ----------
    A : array_like, shape (n, d)
    B : array_like, shape (m, d)

    Returns
    -------
    ndarray, shape (n, m)
        distances[i, j] is the distance between A[i] and B[j].

    Raises
    ------
    ValueError
        If A or B is not 2-D, or they have different numbers of columns.

    Complexity
    ----------
    O(n m d) time (matrix multiply); O((n + m) d + n m) space.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    _validate_points(A, B, "euclidean_distance")
    if A.shape[0] == 0 or B.shape[0] == 0:
        return np.empty((A.shape[0], B.shape[0]))

    # Distances are translation-invariant; centring limits cancellation error.
    centre = _finite_mean(A, B)
    A = A - centre
    B = B - centre

    a_sq = np.sum(A ** 2, axis=1, keepdims=True)       # shape (n, 1)
    b_sq = np.sum(B ** 2, axis=1, keepdims=True).T     # shape (1, m)
    cross = A @ B.T                                    # shape (n, m)

    sq_dists = a_sq + b_sq - 2 * cross
    # Rounding can make a true-zero squared distance slightly negative.
    sq_dists = np.maximum(sq_dists, 0.0)
    return np.sqrt(sq_dists)


def manhattan_distance(A, B):
    """Pairwise Manhattan (L1) distances between the rows of A and the rows of B.

    L1 has no algebraic expansion like Euclidean distance, so this builds an
    (n, m, d) intermediate; for very large inputs, call it on chunks of A.

    Parameters
    ----------
    A : array_like, shape (n, d)
    B : array_like, shape (m, d)

    Returns
    -------
    ndarray, shape (n, m)
        distances[i, j] is the sum of |A[i, k] - B[j, k]| over k.

    Raises
    ------
    ValueError
        If A or B is not 2-D, or they have different numbers of columns.

    Complexity
    ----------
    O(n m d) time and space.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    _validate_points(A, B, "manhattan_distance")

    diffs = np.abs(A[:, None, :] - B[None, :, :])  # shape (n, m, d)
    return diffs.sum(axis=2)


def cosine_distance(A, B):
    """Pairwise cosine distances (1 - cosine similarity) between rows of A and B.

    A zero vector has no direction, so pairs involving one get similarity 0
    (distance 1) rather than NaN; see `safe_divide`.

    Parameters
    ----------
    A : array_like, shape (n, d)
    B : array_like, shape (m, d)

    Returns
    -------
    ndarray, shape (n, m)
        distances[i, j] is 1 - cosine_similarity(A[i], B[j]).

    Raises
    ------
    ValueError
        If A or B is not 2-D, or they have different numbers of columns.

    Complexity
    ----------
    O(n m d) time (matrix multiply); O(n m) space.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    _validate_points(A, B, "cosine_distance")

    a_norm = np.linalg.norm(A, axis=1, keepdims=True)  # shape (n, 1)
    b_norm = np.linalg.norm(B, axis=1, keepdims=True)  # shape (m, 1)

    similarity = safe_divide(A @ B.T, a_norm @ b_norm.T, fill=0.0)
    return 1.0 - similarity


def sigmoid(x):
    """Logistic sigmoid, 1 / (1 + exp(-x)), computed without overflow.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    ndarray
        Same shape as `x`, with values in [0, 1].

    Raises
    ------
    ValueError
        If `x` cannot be converted to a float array.

    Complexity
    ----------
    O(N) time and space, where N = x.size.
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)

    non_negative = x >= 0
    out[non_negative] = 1.0 / (1.0 + np.exp(-x[non_negative]))

    exp_x = np.exp(x[~non_negative])
    out[~non_negative] = exp_x / (1.0 + exp_x)

    return out


def relu(x):
    """Rectified linear unit, max(x, 0), elementwise.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    ndarray
        Same shape as `x`.

    Raises
    ------
    ValueError
        If `x` cannot be converted to a float array.

    Complexity
    ----------
    O(N) time and space, where N = x.size.
    """
    x = np.asarray(x, dtype=float)
    return np.maximum(x, 0.0)


def softmax(x, axis=-1):
    """Numerically stable softmax along an axis.

    -inf entries get probability 0. If a slice contains +inf, those entries
    share all the probability equally. A slice that is entirely -inf has no
    valid distribution and gives NaN.

    Parameters
    ----------
    x : array_like
    axis : int, default=-1
        Axis the probabilities sum to 1 along.

    Returns
    -------
    ndarray
        Same shape as `x`.

    Raises
    ------
    ValueError
        If `x` is empty.

    Complexity
    ----------
    O(N) time and space, where N = x.size.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("softmax(): x is empty")

    _, shifted = _exp_shift(x, axis)
    exp_x = np.exp(shifted)
    with np.errstate(invalid="ignore"):  # all -inf slice: 0 / 0 gives NaN
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    """Numerically stable log of the softmax along an axis.

    Computed in the log domain, so it stays accurate where `np.log(softmax(x))`
    would underflow to -inf. Infinities follow the same rules as `softmax`.

    Parameters
    ----------
    x : array_like
    axis : int, default=-1
        Axis the underlying probabilities sum to 1 along.

    Returns
    -------
    ndarray
        Same shape as `x`, with values <= 0 (NaN for an entirely -inf slice).

    Raises
    ------
    ValueError
        If `x` is empty.

    Complexity
    ----------
    O(N) time and space, where N = x.size.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("log_softmax(): x is empty")

    _, shifted = _exp_shift(x, axis)
    with np.errstate(divide="ignore", invalid="ignore"):  # all -inf slice gives NaN
        return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


def logsumexp(x, axis=None):
    """Numerically stable log(sum(exp(x))).

    The result is -inf for an entirely -inf slice and +inf for a slice that
    contains +inf.

    Parameters
    ----------
    x : array_like
    axis : int or None, default=None
        Axis to reduce over; None reduces over the whole array.

    Returns
    -------
    float or ndarray
        A float if `axis` is None, otherwise an array with `axis` removed.

    Raises
    ------
    ValueError
        If `x` is empty.

    Complexity
    ----------
    O(N) time and space, where N = x.size.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("logsumexp(): x is empty")

    m, shifted = _exp_shift(x, axis)
    with np.errstate(divide="ignore"):  # log(0) = -inf is the right answer here
        result = m + np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis) if axis is not None else result.item()


def safe_divide(num, denom, fill=0.0):
    """Elementwise num / denom, using `fill` where denom is zero.

    Parameters
    ----------
    num : array_like
    denom : array_like
        Broadcastable against `num`.
    fill : float, default=0.0
        Value used where `denom` is exactly 0.

    Returns
    -------
    ndarray

    Raises
    ------
    ValueError
        If `num` and `denom` cannot be broadcast together.

    Complexity
    ----------
    O(N) time and space, where N is the size of the broadcast result.
    """
    num = np.asarray(num, dtype=float)
    denom = np.asarray(denom, dtype=float)

    is_zero = denom == 0
    # Divide by 1 where denom is 0, so no division by zero (or warning) occurs.
    safe_denom = np.where(is_zero, 1.0, denom)
    return np.where(is_zero, fill, num / safe_denom)


def topk_indices(X, k, axis=-1, largest=True):
    """Indices of the `k` largest (or smallest) entries along an axis.

    Selects with `np.partition` instead of a full sort. Ties are broken by
    lower index first and NaN ranks last in both modes, so the result equals
    the first `k` entries of a stable `np.argsort` (of `-X` when
    `largest=True`).

    Parameters
    ----------
    X : array_like
        Values to rank; converted to float.
    k : int
        Number of indices to return, with 1 <= k <= X.shape[axis].
    axis : int, default=-1
        Axis to select along.
    largest : bool, default=True
        Select the largest entries if True, the smallest if False (use False
        for nearest neighbours on a distance matrix).

    Returns
    -------
    ndarray of int
        Shape of `X` with `axis` resized to `k`, ordered best first.

    Raises
    ------
    ValueError
        If `k` is not an integer in [1, X.shape[axis]], or `axis` is out of
        range for `X`.

    Complexity
    ----------
    O(N) average time for the selection plus O(k log k) per slice for the
    ordering; O(N) extra space, where N = X.size.
    """
    X = np.asarray(X, dtype=float)
    moved = np.moveaxis(X, axis, -1)
    n = moved.shape[-1]
    if not _is_integer(k) or not 1 <= k <= n:
        raise ValueError(f"topk_indices(): k must be an integer in [1, {n}], got {k!r}")

    keys = (-moved if largest else moved).reshape(-1, n)  # smaller key is better
    kth = np.partition(keys, k - 1, axis=-1)[:, k - 1:k]  # k-th best key per row

    # Entries ranked strictly ahead of the k-th, and entries tied with it.
    kth_is_nan = np.isnan(kth)
    ahead = np.where(kth_is_nan, ~np.isnan(keys), keys < kth)
    tied = np.where(kth_is_nan, np.isnan(keys), keys == kth)
    # Fill the remaining slots with the earliest tied entries.
    spare = k - ahead.sum(axis=-1, keepdims=True)
    chosen = ahead | (tied & (np.cumsum(tied, axis=-1) <= spare))

    cols = np.nonzero(chosen)[1].reshape(-1, k)  # k indices per row, ascending
    order = np.argsort(np.take_along_axis(keys, cols, axis=-1), axis=-1, kind="stable")
    best_first = np.take_along_axis(cols, order, axis=-1)
    return np.moveaxis(best_first.reshape(moved.shape[:-1] + (k,)), -1, axis)


def batch_iterator(X, batch_size):
    """Yield successive `batch_size`-sized slices of `X`.

    The last batch is smaller if `len(X)` is not a multiple of `batch_size`.

    Parameters
    ----------
    X : array_like
    batch_size : int
        Positive integer.

    Yields
    ------
    ndarray
        Consecutive slices of `X`, covering every element exactly once.

    Raises
    ------
    ValueError
        If `batch_size` is not a positive integer (raised when iteration
        starts, since this is a generator).

    Complexity
    ----------
    O(batch_size) time and space per batch; O(len(X)) over the full iteration.
    """
    if not _is_integer(batch_size) or batch_size <= 0:
        raise ValueError(
            f"batch_iterator(): batch_size must be a positive integer, got {batch_size!r}"
        )

    X = np.asarray(X)
    n = len(X)
    for start in range(0, n, batch_size):
        yield X[start:start + batch_size]
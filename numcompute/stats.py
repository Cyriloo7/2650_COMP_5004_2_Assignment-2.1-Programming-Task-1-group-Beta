"""
stats.py -- Descriptive statistics for NumCompute.

Provides:
    describe(X, axis=0)     -> dict of mean/median/std/min/max, NaN-aware
    StreamingStats       -> running (streaming) mean and variance
    histogram(data, ...)     -> thin wrapper around np.histogram
    quantile(data, q, ...)   -> NaN-aware quantile, delegates to rank.percentile
"""

import warnings

import numpy as np

from .rank import percentile


def describe(X, axis=0):
    """
    Compute mean, median, standard deviation, min, and max along an axis.

    NaN values are ignored (treated as missing), so a single missing value
    in a column does not turn that whole column's statistics into NaN.

    Parameters
    ----------
    X : array_like
        Input data. May contain NaN.
    axis : int, default=0
        Axis to reduce over. axis=0 -> one result per column (the usual
        case for an (n_samples, n_features) table). axis=1 -> one result
        per row.

    Returns
    -------
    dict[str, ndarray or float]
        Keys: 'mean', 'median', 'std', 'min', 'max'.
        Each value has X's shape with `axis` removed.
        E.g. X.shape == (n, m) and axis=0 -> each value has shape (m,).

    Raises
    ------
    ValueError
        If X has zero length along `axis`.

    Complexity
    ----------
    Time O(n), space O(1) extra (beyond the output), where n is the size
    of X along `axis`.
    """
    X = np.asarray(X, dtype=float)
    if X.shape[axis] == 0:
        raise ValueError(f"describe(): axis {axis} is empty (X.shape={X.shape})")

    # A column/row that is ALL NaN has no valid data to summarise. NumPy's
    # nan* functions handle this correctly (they return NaN for it) but
    # print a warning every time -- suppress just that expected warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return {
            "mean": np.nanmean(X, axis=axis),
            "median": np.nanmedian(X, axis=axis),
            "std": np.nanstd(X, axis=axis),
            "min": np.nanmin(X, axis=axis),
            "max": np.nanmax(X, axis=axis),
        }


class StreamingStats:
    """
    Running (streaming) mean and variance, updated one sample at a time.

    Uses Welford's online algorithm, so the full dataset never needs to be
    held in memory -- each call to `update` only needs the running totals
    and the new value.

    Attributes
    ----------
    count : int
        Number of samples seen so far.
    mean : float
        Running mean of all samples seen so far.

    Complexity
    ----------
    update(): O(1) time, O(1) space, per call.
    """

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0  # running sum of squared differences from the mean

    def update(self, x):
        """Incorporate one new sample `x` (a plain number)."""
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.M2 += delta * (x - self.mean)
        return self

    @property
    def variance(self):
        """
        Population variance (divides by `count`, not `count - 1`) of every
        sample seen so far.

        Raises
        ------
        ValueError
            If no samples have been added yet.
        """
        if self.count == 0:
            raise ValueError("variance is undefined with zero samples")
        return self.M2 / self.count

    @property
    def std(self):
        """Population standard deviation -- the square root of `variance`."""
        return self.variance ** 0.5

    def merge(self, other):
        """
        Fold another StreamingStats's samples into this one, as if
        every sample `other` ever saw had been passed to `update` on this
        accumulator instead -- without re-processing any of them.

        Useful for combining per-chunk accumulators (e.g. one per chunk
        from `io.read_csv_chunks`) into a single overall result.

        Parameters
        ----------
        other : StreamingStats

        Returns
        -------
        StreamingStats
            This accumulator (updated in place), for chaining.
        """
        if other.count == 0:
            return self
        if self.count == 0:
            self.count, self.mean, self.M2 = other.count, other.mean, other.M2
            return self

        total = self.count + other.count
        delta = other.mean - self.mean
        self.mean += delta * other.count / total
        self.M2 += other.M2 + delta ** 2 * self.count * other.count / total
        self.count = total
        return self


def histogram(data, bins=10, range=None):
    """
    Bin `data` into a histogram.

    Thin wrapper around `np.histogram` -- there is no benefit to
    re-implementing bucket counting by hand.

    Parameters
    ----------
    data : array_like
        1-D input data.
    bins : int or sequence, default=10
        Number of equal-width bins, or explicit bin edges.
    range : tuple(float, float), optional
        Lower and upper range of the bins. Defaults to (data.min(), data.max()).

    Returns
    -------
    counts : ndarray, shape (len(bin_edges) - 1,)
        Number of data points in each bin.
    bin_edges : ndarray, shape (len(counts) + 1,)
        The edges of the bins.

    Complexity
    ----------
    O(n) typical case.
    """
    return np.histogram(data, bins=bins, range=range)


def quantile(data, q, interpolation="linear"):
    """
    NaN-aware quantile of `data` at fraction(s) `q` in [0, 1].

    Delegates to `rank.percentile` (the same computation, on a 0-1 scale
    instead of 0-100) so the interpolation logic lives in exactly one
    place in the toolkit.

    Parameters
    ----------
    data : array_like
        Input data. May contain NaN (ignored).
    q : float or array_like of float
        Quantile(s) to compute, each in [0, 1].
    interpolation : {'linear', 'lower', 'higher', 'midpoint'}, default='linear'
        How to interpolate when the quantile falls between two data points.

    Returns
    -------
    float or ndarray
        The value(s) at the given quantile(s).

    Raises
    ------
    ValueError
        If `data` is empty, or any value in `q` is outside [0, 1].

    Complexity
    ----------
    O(n log n) (dominated by sorting, done inside NumPy).
    """
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        raise ValueError("quantile(): data is empty")

    q = np.asarray(q, dtype=float)
    if np.any((q < 0) | (q > 1)):
        raise ValueError(f"quantile(): q must be in [0, 1], got {q}")

    return percentile(data, q * 100, interpolation=interpolation)
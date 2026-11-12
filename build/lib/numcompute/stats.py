"""Descriptive statistics.

Contents:
    describe: NaN-aware mean, median, std, min and max along an axis.
    StreamingStats: running mean and variance, one sample at a time.
    histogram: histogram of the non-NaN values.
    quantile: NaN-aware quantiles along an axis, via rank.percentile.
"""

import math
import warnings

import numpy as np

from .rank import percentile


def describe(X, axis=0):
    """NaN-aware mean, median, std, min and max along an axis.

    Parameters
    ----------
    X : array_like
        Input data; NaN values are ignored.
    axis : int, default=0
        Axis to reduce over; negative values count from the end. For an
        (n_samples, n_features) table, 0 gives one result per column and 1
        one result per row.

    Returns
    -------
    dict
        Keys 'mean', 'median', 'std', 'min', 'max'. Each value has the shape
        of `X` with `axis` removed, e.g. (n_features,) for axis=0. A slice
        that is entirely NaN gives NaN.

    Raises
    ------
    ValueError
        If `axis` is not a valid axis of `X`, or `X` is empty along `axis`.

    Complexity
    ----------
    O(N) time and O(N) extra space, where N = X.size (`nanmedian` works on
    a copy of the data).
    """
    X = np.asarray(X, dtype=float)
    if not isinstance(axis, (int, np.integer)) or not -X.ndim <= axis < X.ndim:
        raise ValueError(
            f"describe(): invalid axis {axis!r} for an array with {X.ndim} dimension(s)"
        )
    if X.shape[axis] == 0:
        raise ValueError(f"describe(): axis {axis} is empty (X.shape={X.shape})")

    # All-NaN slices give NaN as intended; silence NumPy's warning for them.
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
    """Running mean and variance, updated one sample at a time.

    Implements Welford's online algorithm, so the data never has to be held
    in memory. NaN samples are skipped. `mean`, `variance` and `std` raise
    ValueError until at least one non-NaN sample has been added.

    Attributes
    ----------
    count : int
        Number of non-NaN samples added.
    M2 : float
        Sum of squared deviations from the running mean.
    """

    def __init__(self):
        self.count = 0
        self._mean = 0.0
        self.M2 = 0.0

    def update(self, x):
        """Add one sample.

        Parameters
        ----------
        x : float
            Sample to add. NaN is skipped and leaves the statistics unchanged.

        Returns
        -------
        StreamingStats
            This accumulator, for chaining.

        Raises
        ------
        TypeError
            If `x` is not a real number.

        Complexity
        ----------
        O(1) time and space.
        """
        if math.isnan(x):
            return self
        self.count += 1
        delta = x - self._mean
        self._mean += delta / self.count
        self.M2 += delta * (x - self._mean)
        return self

    @property
    def mean(self):
        """Mean of the samples added.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If no samples have been added.

        Complexity
        ----------
        O(1).
        """
        if self.count == 0:
            raise ValueError("mean is undefined with zero samples")
        return self._mean

    @property
    def variance(self):
        """Population variance of the samples added (divisor `count`).

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If no samples have been added.

        Complexity
        ----------
        O(1).
        """
        if self.count == 0:
            raise ValueError("variance is undefined with zero samples")
        return self.M2 / self.count

    @property
    def std(self):
        """Population standard deviation: the square root of `variance`.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If no samples have been added.

        Complexity
        ----------
        O(1).
        """
        return self.variance ** 0.5

    def merge(self, other):
        """Fold another accumulator's samples into this one.

        The result is the same as passing all of `other`'s samples to
        `update`, without revisiting them. Use it to combine per-chunk
        accumulators.

        Parameters
        ----------
        other : StreamingStats
            Accumulator to merge in; it is not modified.

        Returns
        -------
        StreamingStats
            This accumulator, updated in place, for chaining.

        Complexity
        ----------
        O(1) time and space.
        """
        if other.count == 0:
            return self
        if self.count == 0:
            self.count, self._mean, self.M2 = other.count, other._mean, other.M2
            return self

        total = self.count + other.count
        delta = other._mean - self._mean
        self._mean += delta * other.count / total
        self.M2 += other.M2 + delta ** 2 * self.count * other.count / total
        self.count = total
        return self


# Name used in the project spec.
WelfordAccumulator = StreamingStats


def histogram(data, bins=10, range=None):
    """Histogram of the non-NaN values in `data`.

    Thin wrapper around `np.histogram`; NaN values are dropped before binning.

    Parameters
    ----------
    data : array_like
        Input values; multi-dimensional input is flattened.
    bins : int or sequence of float, default=10
        Number of equal-width bins, or explicit bin edges.
    range : tuple of float, optional
        (min, max) of the bins. Defaults to the min and max of the non-NaN
        values.

    Returns
    -------
    counts : ndarray, shape (n_bins,)
        Number of values in each bin; all zero if no non-NaN values remain.
    bin_edges : ndarray, shape (n_bins + 1,)
        Bin edges.

    Raises
    ------
    ValueError
        If `bins` or `range` is invalid, or `data` contains infinite values
        and `range` is not given.

    Complexity
    ----------
    O(N) time and space, where N = data.size.
    """
    data = np.asarray(data, dtype=float)
    return np.histogram(data[~np.isnan(data)], bins=bins, range=range)


def quantile(data, q, interpolation="linear", axis=None):
    """NaN-aware quantile(s) of `data`.

    Delegates to `rank.percentile`, rescaling `q` from [0, 1] to [0, 100].

    Parameters
    ----------
    data : array_like
        Input data; NaN values are ignored.
    q : float or array_like of float
        Quantile(s) to compute, each in [0, 1].
    interpolation : {'linear', 'lower', 'higher', 'midpoint'}, default='linear'
        Interpolation used when a quantile falls between two data points.
    axis : int or None, default=None
        Axis to compute along. None flattens `data` and uses all values.

    Returns
    -------
    float or ndarray
        A float if `q` is a scalar and `axis` is None. Otherwise `data`'s
        shape with `axis` removed, with a leading dimension for `q` if it is
        array-like.

    Raises
    ------
    ValueError
        If `data` is empty, any `q` is outside [0, 1], or `axis` is invalid.

    Complexity
    ----------
    O(N log N) time, where N = data.size.
    """
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        raise ValueError("quantile(): data is empty")

    q = np.asarray(q, dtype=float)
    if np.any((q < 0) | (q > 1)):
        raise ValueError(f"quantile(): q must be in [0, 1], got {q}")

    return percentile(data, q * 100, interpolation=interpolation, axis=axis)
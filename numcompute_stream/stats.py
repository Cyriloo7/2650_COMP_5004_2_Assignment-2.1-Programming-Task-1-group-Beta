"""Streaming descriptive statistics, updated one chunk at a time.

``StreamingStats.update_stats(X_chunk)`` keeps, for every feature (column):

* count, mean and variance - Welford's method in its chunk form (exact);
* min and max - exact;
* quantiles - from a random sample of at most ``sample_size`` rows
  (reservoir sampling), so they are exact until that many rows have been seen;
* a histogram - fixed bins, optionally over the last ``window`` chunks only.

NaN values are ignored and counted as missing; text and infinity are rejected
with the 2.1 input check. This is the vectorised, chunk-wise version of the
Assignment 2.1 ``numcompute.stats.StreamingStats`` (which adds one value at a
time); the 2.1 ``histogram`` and ``quantile`` functions and ``safe_divide``
are reused, not changed.
"""

import warnings
from collections import deque

import numpy as np

from numcompute.preprocessing import NotFittedError, _check_numeric  # Assignment 2.1 (reused)
from numcompute.stats import histogram as batch_histogram           # Assignment 2.1 (reused)
from numcompute.stats import quantile as batch_quantile             # Assignment 2.1 (reused)
from numcompute.utils import safe_divide                            # Assignment 2.1 (reused)

from ._checks import check_int_at_least


def update_moments(count, mean, m2, X):
    """Merge a chunk X into running per-column (count, mean, m2).

    ``m2`` is the sum of squared differences from the mean, so the variance
    is ``m2 / count``. This is the chunk version of Welford's algorithm
    (Chan et al., 1979) - the same merge as ``StreamingStats.merge`` in
    Assignment 2.1, done for every column at once. The result does not depend
    on how the data was split into chunks, and it avoids the round-off errors
    of E[x^2] - E[x]^2. NaN values are skipped.

    Parameters
    ----------
    count, mean, m2 : ndarray of shape (n_features,)
        The running state (start from zeros).
    X : ndarray of shape (n_samples, n_features)
        The chunk; NaN marks a missing value.

    Returns
    -------
    (count, mean, m2) : tuple of ndarray of shape (n_features,)
        The new state (the inputs are not modified).

    Notes
    -----
    Time O(n * f) for a chunk of n rows and f columns; memory O(f).
    """
    valid = ~np.isnan(X)
    chunk_count = valid.sum(axis=0)
    chunk_mean = safe_divide(np.where(valid, X, 0.0).sum(axis=0), chunk_count)
    chunk_m2 = (np.where(valid, X - chunk_mean, 0.0) ** 2).sum(axis=0)

    new_count = count + chunk_count
    delta = chunk_mean - mean
    new_mean = mean + delta * safe_divide(chunk_count, new_count)
    new_m2 = m2 + chunk_m2 + delta ** 2 * safe_divide(count * chunk_count, new_count)
    return new_count, new_mean, new_m2


def update_min_max(minimum, maximum, X):
    """Merge a non-empty chunk X into running per-column (minimum, maximum).

    Start both from arrays of NaN: ``np.fmin`` / ``np.fmax`` skip NaN, so a
    column with no values yet stays NaN. Used by StreamingStats, StandardScaler
    (to spot constant columns exactly) and MinMaxScaler.

    Parameters
    ----------
    minimum, maximum : ndarray of shape (n_features,)
        The running extremes (NaN where no value has been seen).
    X : ndarray of shape (n_samples, n_features)
        The chunk (n_samples >= 1); NaN values are skipped.

    Returns
    -------
    (minimum, maximum) : tuple of ndarray of shape (n_features,)
    """
    return np.fmin(minimum, np.fmin.reduce(X, axis=0)), np.fmax(maximum, np.fmax.reduce(X, axis=0))


class StreamingStats:
    """Running statistics of a stream of chunks (rows = samples, columns = features).

    Parameters
    ----------
    bins : int, default 10
        Number of histogram bins per feature.
    hist_range : (float, float) or None, default None
        Histogram range. None uses the min and max of the first chunk; later
        values outside that range are counted in ``out_of_range_``.
    window : int or None, default None
        If set, ``histogram()`` only counts the last ``window`` chunks.
    sample_size : int, default 1000
        Size of the random sample used for quantiles.
    random_state : int or None
        Seed for the random sample.

    Notes
    -----
    Each update costs O(n * f) for the moments and min/max, plus one histogram
    per feature (a loop over the few features, never over samples). Memory is
    O(f * (sample_size + bins * window)) however long the stream is.

    Example
    -------
    >>> stats = StreamingStats()
    >>> for chunk in chunks:
    ...     stats.update_stats(chunk)
    >>> stats.mean, stats.std, stats.quantile(0.5)
    """

    def __init__(self, bins=10, hist_range=None, window=None, sample_size=1000, random_state=None):
        check_int_at_least("bins", bins)
        check_int_at_least("sample_size", sample_size)
        check_int_at_least("window", window, allow_none=True)
        self.bins = bins
        self.hist_range = hist_range
        self.window = window
        self.sample_size = sample_size
        self.random_state = random_state
        self.n_features_in_ = None

    def update_stats(self, X_chunk):
        """Add one chunk to every running statistic.

        Parameters
        ----------
        X_chunk : array-like of shape (n_samples, n_features) or (n_samples,)
            Numbers; NaN marks a missing value. A 1-D chunk is one feature.
            The first non-empty chunk fixes ``n_features``; an empty chunk
            changes nothing.

        Returns
        -------
        self

        Raises
        ------
        TypeError
            If the chunk holds text (encode text columns first).
        ValueError
            If the chunk is not 1-D or 2-D, holds infinity, or has the wrong
            number of features. Nothing changes when a chunk is rejected.
        """
        X = np.asarray(X_chunk)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(f"X_chunk must be 1D or 2D, got shape {X.shape}.")
        X = _check_numeric(X, "StreamingStats")  # 2.1 check: numbers only, no infinity (NaN = missing)
        if self.n_features_in_ is not None and X.shape[1] != self.n_features_in_:
            raise ValueError(f"X_chunk has {X.shape[1]} features, expected {self.n_features_in_}.")
        if len(X) == 0:
            return self
        if self.n_features_in_ is None:
            self._start(X)  # the first non-empty chunk fixes the number of features

        self.count_, self.mean_, self._m2 = update_moments(self.count_, self.mean_, self._m2, X)
        self.min_, self.max_ = update_min_max(self.min_, self.max_, X)
        self.n_missing_ += np.isnan(X).sum(axis=0)
        self._update_sample(X)
        self._update_histogram(X)
        self.n_samples_seen_ += len(X)
        return self

    def _start(self, X):
        """Set up the running state from the first chunk."""
        n_features = X.shape[1]
        self.n_features_in_ = n_features
        self.n_samples_seen_ = 0
        self.count_ = np.zeros(n_features)
        self.mean_ = np.zeros(n_features)
        self._m2 = np.zeros(n_features)
        self.min_ = np.full(n_features, np.nan)
        self.max_ = np.full(n_features, np.nan)
        self.n_missing_ = np.zeros(n_features, dtype=int)
        self._rng = np.random.default_rng(self.random_state)
        self._sample = np.full((self.sample_size, n_features), np.nan)

        if self.hist_range is not None:
            low, high = np.full(n_features, self.hist_range[0]), np.full(n_features, self.hist_range[1])
        else:
            low = np.nan_to_num(np.fmin.reduce(X, axis=0), nan=0.0)   # all-missing column -> [0, 1]
            high = np.nan_to_num(np.fmax.reduce(X, axis=0), nan=1.0)
            constant = high <= low                                     # give constant columns a width
            low, high = low - 0.5 * constant, high + 0.5 * constant
        self.edges_ = np.linspace(low, high, self.bins + 1, axis=1)    # (n_features, bins + 1)
        self._total_counts = np.zeros((n_features, self.bins))
        self._recent_counts = deque(maxlen=self.window)
        self.out_of_range_ = np.zeros(n_features, dtype=int)

    def _update_sample(self, X):
        """Reservoir sampling: keep a uniform random sample of all rows seen so far."""
        position = self.n_samples_seen_ + np.arange(len(X))  # position of each row in the stream
        slot = np.where(position < self.sample_size, position,
                        self._rng.integers(0, position + 1))  # each row replaces a random slot ...
        keep = slot < self.sample_size                         # ... with probability sample_size / (position + 1)
        self._sample[slot[keep]] = X[keep]

    def _update_histogram(self, X):
        counts = np.zeros((self.n_features_in_, self.bins))
        for j in range(self.n_features_in_):  # one histogram per feature
            edges = self.edges_[j]
            self.out_of_range_[j] += np.sum((X[:, j] < edges[0]) | (X[:, j] > edges[-1]))  # NaN compares False
            counts[j] = batch_histogram(X[:, j], bins=edges)[0]                            # 2.1: drops NaN
        self._total_counts += counts
        self._recent_counts.append(counts)

    def _check_started(self):
        """Raise NotFittedError if no chunk has been added yet."""
        if self.n_features_in_ is None:
            raise NotFittedError("StreamingStats has no data yet: call update_stats first.")

    @property
    def mean(self):
        """Mean of each feature, ndarray of shape (n_features,) (NaN for a feature with no value yet)."""
        self._check_started()
        return np.where(self.count_ > 0, self.mean_, np.nan)

    @property
    def var(self):
        """Population variance (divides by n) of each feature, like the 2.1 StreamingStats.

        Returns an ndarray of shape (n_features,). A constant column has a
        variance of exactly 0 (round-off in the running mean could otherwise
        leave a tiny positive number).
        """
        self._check_started()
        variance = np.where(self.min_ == self.max_, 0.0, safe_divide(self._m2, self.count_))
        return np.where(self.count_ > 0, variance, np.nan)

    @property
    def std(self):
        """Population standard deviation of each feature, ndarray of shape (n_features,)."""
        return np.sqrt(self.var)

    @property
    def min(self):
        """Exact minimum of each feature, ndarray of shape (n_features,)."""
        self._check_started()
        return self.min_.copy()

    @property
    def max(self):
        """Exact maximum of each feature, ndarray of shape (n_features,)."""
        self._check_started()
        return self.max_.copy()

    def quantile(self, q):
        """Quantile(s) of each feature (exact until ``sample_size`` rows have been seen).

        Parameters
        ----------
        q : float or array-like of float in [0, 1]

        Returns
        -------
        ndarray of shape (n_features,), or (len(q), n_features) for several q
        """
        self._check_started()
        sample = self._sample[:min(self.n_samples_seen_, self.sample_size)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-missing columns give NaN
            return batch_quantile(sample, q, axis=0)

    def histogram(self):
        """Histogram of each feature (only the last ``window`` chunks when ``window`` is set).

        Returns
        -------
        counts : ndarray of shape (n_features, bins)
        edges : ndarray of shape (n_features, bins + 1)
            Fixed by the first chunk (or by ``hist_range``).
        """
        self._check_started()
        counts = sum(self._recent_counts) if self.window else self._total_counts
        return counts.copy(), self.edges_.copy()

    def summary(self):
        """Main statistics: dict with keys count, missing, mean, std, min, median and max (one value per feature)."""
        return {"count": self.count_.astype(int), "missing": self.n_missing_.copy(), "mean": self.mean,
                "std": self.std, "min": self.min, "median": self.quantile(0.5), "max": self.max}

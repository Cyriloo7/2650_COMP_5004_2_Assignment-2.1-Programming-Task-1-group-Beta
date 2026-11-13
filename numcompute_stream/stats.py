"""Streaming descriptive statistics, updated one chunk at a time.

``StreamingStats.update_stats(X_chunk)`` keeps, for every feature (column):

* count, mean and variance - Welford's method in its chunk form (exact);
* min and max - exact;
* quantiles - from a random sample of at most ``sample_size`` rows
  (reservoir sampling), so they are exact until that many rows have been seen;
* a histogram - fixed bins, optionally over the last ``window`` chunks only.

NaN values are ignored and counted as missing. This is the vectorised,
chunk-wise version of the Assignment 2.1 ``numcompute.stats.StreamingStats``
(which adds one value at a time); the 2.1 ``histogram`` and ``quantile``
functions and ``safe_divide`` are reused, not changed.
"""

import warnings
from collections import deque

import numpy as np

from numcompute.preprocessing import NotFittedError         # Assignment 2.1 (reused)
from numcompute.stats import histogram as batch_histogram  # Assignment 2.1 (reused)
from numcompute.stats import quantile as batch_quantile    # Assignment 2.1 (reused)
from numcompute.utils import safe_divide                   # Assignment 2.1 (reused)

from ._checks import check_int_at_least


def update_moments(count, mean, m2, X):
    """Merge a chunk X into running per-column (count, mean, m2).

    ``m2`` is the sum of squared differences from the mean, so the variance
    is ``m2 / count``. This is the chunk version of Welford's algorithm
    (Chan et al., 1979) - the same merge as ``StreamingStats.merge`` in
    Assignment 2.1, done for every column at once. The result does not depend
    on how the data was split into chunks, and it avoids the round-off errors
    of E[x^2] - E[x]^2. NaN values are skipped.

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
        """Add one chunk of shape (n_samples, n_features). A 1D chunk is one feature."""
        X = np.asarray(X_chunk, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(f"X_chunk must be 1D or 2D, got shape {X.shape}.")
        if len(X) == 0:
            return self
        if self.n_features_in_ is None:
            self._start(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X_chunk has {X.shape[1]} features, expected {self.n_features_in_}.")

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
        if self.n_features_in_ is None:
            raise NotFittedError("StreamingStats has no data yet: call update_stats first.")

    @property
    def mean(self):
        self._check_started()
        return np.where(self.count_ > 0, self.mean_, np.nan)

    @property
    def var(self):
        """Population variance (divides by n), like the 2.1 StreamingStats.

        A constant column has a variance of exactly 0 (round-off in the running
        mean could otherwise leave a tiny positive number).
        """
        self._check_started()
        variance = np.where(self.min_ == self.max_, 0.0, safe_divide(self._m2, self.count_))
        return np.where(self.count_ > 0, variance, np.nan)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def min(self):
        self._check_started()
        return self.min_.copy()

    @property
    def max(self):
        self._check_started()
        return self.max_.copy()

    def quantile(self, q):
        """Quantile(s) q in [0, 1] per feature (exact until ``sample_size`` rows have been seen)."""
        self._check_started()
        sample = self._sample[:min(self.n_samples_seen_, self.sample_size)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-missing columns give NaN
            return batch_quantile(sample, q, axis=0)

    def histogram(self):
        """Return (counts, edges), each with one row per feature.

        With ``window`` set, only the last ``window`` chunks are counted.
        """
        self._check_started()
        counts = sum(self._recent_counts) if self.window else self._total_counts
        return counts.copy(), self.edges_.copy()

    def summary(self):
        """Dictionary of the main statistics (one value per feature)."""
        return {"count": self.count_.astype(int), "missing": self.n_missing_.copy(), "mean": self.mean,
                "std": self.std, "min": self.min, "median": self.quantile(0.5), "max": self.max}

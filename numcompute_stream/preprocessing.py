"""Preprocessing transformers that learn incrementally.

The scalers and the imputer extend the Assignment 2.1 classes in
``numcompute.preprocessing`` (reused, not changed). They keep the 2.1
``transform``, input checks and error messages, and add ``partial_fit``,
which updates what was learned with one chunk. ``fit`` forgets everything and
learns from X, giving the same statistics as the 2.1 batch ``fit``.
``OneHotEncoder`` extends the 2.1 encoder: it adds categories as they appear,
and the 2.1 ``transform`` does the encoding.

Every transformer has:

    partial_fit(X)     update what has been learned with one chunk
    transform(X)       apply it
    fit(X)             start again and learn from X
    fit_transform(X)   fit, then transform

NaN values are ignored while learning. The scalers keep NaN in their output;
Imputer replaces it.
"""

import numpy as np

from numcompute import preprocessing as base               # Assignment 2.1 (reused)
from numcompute.utils import safe_divide                   # Assignment 2.1 (reused)

from ._checks import check_int_at_least
from .stats import update_min_max, update_moments


def _check_chunk(transformer, X):
    """Run the 2.1 input checks (2-D, numeric, no infinity) and check the number of features.

    The first non-empty chunk sets ``n_features_in_``; later chunks must match it.
    """
    X = base._check_numeric(X, type(transformer).__name__)
    if hasattr(transformer, "n_features_in_"):
        base._check_n_features(transformer, X, reset=False)
    elif len(X) > 0:
        base._check_n_features(transformer, X, reset=True)
    return X


class _StreamingMixin:
    """``fit`` = forget the learned state, then learn from X with ``partial_fit``."""

    _state = ()  # names of the attributes learned from data

    def _reset(self):
        for name in self._state:
            self.__dict__.pop(name, None)

    def fit(self, X, y=None):
        """Forget what was learned and learn from X."""
        self._reset()
        return self.partial_fit(X, y)


class StandardScaler(_StreamingMixin, base.StandardScaler):
    """The 2.1 StandardScaler plus ``partial_fit``: a running mean and variance.

    The statistics are updated chunk by chunk with Welford's method (see
    ``stats.update_moments``), so the result is the same as the 2.1 batch
    ``fit`` on all the data. As in 2.1, a constant column gets its exact value
    as the mean and ``scale_ = 0`` (it is not divided by zero), and a column
    with no values gets NaN statistics.
    """

    _state = ("n_features_in_", "n_samples_seen_", "_mean", "_m2", "_min", "_max", "mean_", "var_", "scale_")

    def partial_fit(self, X, y=None):
        X = _check_chunk(self, X)
        if len(X) == 0:
            return self
        if not hasattr(self, "n_samples_seen_"):
            n = X.shape[1]
            self.n_samples_seen_, self._mean, self._m2 = np.zeros(n), np.zeros(n), np.zeros(n)
            self._min, self._max = np.full(n, np.nan), np.full(n, np.nan)
        self.n_samples_seen_, self._mean, self._m2 = update_moments(self.n_samples_seen_, self._mean, self._m2, X)
        self._min, self._max = update_min_max(self._min, self._max, X)

        constant = self._min == self._max  # exact test, as in 2.1 (NaN == NaN is False)
        seen = self.n_samples_seen_ > 0
        mean = np.where(seen, np.where(constant, self._min, self._mean), np.nan)
        self.var_ = np.where(seen, np.where(constant, 0.0, safe_divide(self._m2, self.n_samples_seen_)), np.nan)
        self.mean_ = mean if self.with_mean else np.zeros(X.shape[1])
        self.scale_ = np.sqrt(self.var_) if self.with_std else np.ones(X.shape[1])
        return self


class MinMaxScaler(_StreamingMixin, base.MinMaxScaler):
    """The 2.1 MinMaxScaler plus ``partial_fit``: a running minimum and maximum."""

    _state = ("n_features_in_", "min_", "max_", "data_range_")

    def partial_fit(self, X, y=None):
        self._range()  # 2.1 check of feature_range
        X = _check_chunk(self, X)
        if len(X) == 0:
            return self
        if not hasattr(self, "min_"):
            self.min_, self.max_ = np.full(X.shape[1], np.nan), np.full(X.shape[1], np.nan)
        self.min_, self.max_ = update_min_max(self.min_, self.max_, X)
        with np.errstate(over="ignore"):
            self.data_range_ = self.max_ - self.min_
        return self


class Imputer(_StreamingMixin, base.SimpleImputer):
    """The 2.1 SimpleImputer plus ``partial_fit``: fill values updated on the fly.

    ``partial_fit`` supports ``strategy="mean"`` (running mean) and
    ``"constant"``. The 2.1 strategies "median" and "most_frequent" need all
    the data at once, so they work with ``fit`` only. A feature with no
    observed value yet is filled with ``fill_value``.
    """

    _state = ("n_features_in_", "_count", "_mean", "_m2", "statistics_")

    def partial_fit(self, X, y=None):
        if self.strategy not in ("mean", "constant"):
            raise ValueError(f"partial_fit supports strategy 'mean' or 'constant', got {self.strategy!r} "
                             "(use fit for the 2.1 batch strategies).")
        try:
            fill = float(self.fill_value)
        except (TypeError, ValueError):
            raise ValueError(f"fill_value must be a number, got {self.fill_value!r}") from None
        if not np.isfinite(fill):
            raise ValueError(f"fill_value must be finite, got {self.fill_value!r}")
        X = _check_chunk(self, X)
        if len(X) == 0:
            return self
        if not hasattr(self, "_count"):
            self._count, self._mean, self._m2 = np.zeros(X.shape[1]), np.zeros(X.shape[1]), np.zeros(X.shape[1])
        self._count, self._mean, self._m2 = update_moments(self._count, self._mean, self._m2, X)  # running mean
        if self.strategy == "mean":
            self.statistics_ = np.where(self._count > 0, self._mean, fill)
        else:
            self.statistics_ = np.full(X.shape[1], fill)
        return self

    def fit(self, X, y=None):
        """Learn from X: streaming strategies restart ``partial_fit``; the others use the 2.1 batch fit."""
        if self.strategy in ("mean", "constant"):
            return super().fit(X, y)
        self._reset()
        return base.SimpleImputer.fit(self, X, y)


def _is_missing(values):
    """True for None, NaN and empty text: these are never learned as categories."""
    if values.dtype.kind == "f":
        return np.isnan(values)
    if values.dtype.kind in "US":
        return np.char.strip(values) == ""
    return np.array([v is None or v != v or (isinstance(v, str) and not v.strip()) for v in values.tolist()],
                    dtype=bool)


class OneHotEncoder(_StreamingMixin, base.OneHotEncoder):
    """The 2.1 OneHotEncoder plus ``partial_fit``: categories are added as they appear.

    The 2.1 ``transform`` does the encoding (one 0/1 column per known
    category). For streaming:

    * new categories are added at the end, so older categories keep their columns;
    * ``handle_unknown`` defaults to "ignore": a category that arrives before
      ``partial_fit`` has learned it is encoded as all zeros (2.1 behaviour);
    * missing values (None, NaN or "") are never learned as categories, so
      they are encoded as all zeros too.

    Parameters
    ----------
    columns : list of int or None, default None
        Columns to encode; None encodes every column. The other columns pass
        through unchanged (as floats) and come first in the output.
    max_categories : int or None, default None
        None: one output column per category seen so far, so the output grows
        when a new category appears. An int k reserves exactly k columns per
        encoded column, so the width never changes - use this when a model
        follows the encoder in a streaming pipeline. Categories beyond the
        first k are treated as unknown.
    handle_unknown, dtype
        As in the 2.1 OneHotEncoder.
    """

    _state = ("n_features_in_", "categories_", "_kind", "_encoded", "_n_columns")

    def __init__(self, columns=None, max_categories=None, handle_unknown="ignore", dtype=np.float64):
        check_int_at_least("max_categories", max_categories, allow_none=True)
        super().__init__(handle_unknown=handle_unknown, dtype=dtype)
        self.columns = columns
        self.max_categories = max_categories

    def partial_fit(self, X, y=None):
        """Learn the new categories of one chunk."""
        if self.handle_unknown not in base._HANDLE_UNKNOWN:
            raise ValueError(f"handle_unknown must be one of {base._HANDLE_UNKNOWN}, got {self.handle_unknown!r}")
        X = base._check_2d(X, "OneHotEncoder")
        if not hasattr(self, "categories_"):
            if len(X) == 0:
                return self
            self._n_columns = X.shape[1]
            self._encoded = list(range(X.shape[1])) if self.columns is None else list(self.columns)
            encoded = X[:, self._encoded]
            self._kind = base._kind_group(encoded.dtype)
            base._check_n_features(self, encoded, reset=True)  # n_features_in_ = number of encoded columns
            self.categories_ = [encoded[:0, j] for j in range(encoded.shape[1])]  # empty, with the right dtype
        self._check_width(X)
        for j, column in enumerate(self._encoded):
            values = X[:, column]
            known = self.categories_[j]
            new = [v for v in dict.fromkeys(values[~_is_missing(values)].tolist()) if not np.any(known == v)]
            if self.max_categories is not None:
                new = new[:max(self.max_categories - len(known), 0)]
            if new:  # added at the end, in order of appearance
                new = np.array(new, dtype=object if known.dtype == object else None)
                self.categories_[j] = np.concatenate([known, new])
        return self

    def transform(self, X):
        """Encode the chosen columns with the 2.1 transform; other columns come first, as floats."""
        base._check_fitted(self, "categories_")
        X = base._check_2d(X, "OneHotEncoder")
        self._check_width(X)
        encoded = super().transform(X[:, self._encoded])
        if self.max_categories is not None:  # pad each column's block to max_categories columns
            blocks = np.split(encoded, np.cumsum([len(c) for c in self.categories_])[:-1], axis=1)
            encoded = np.hstack([np.pad(b, ((0, 0), (0, self.max_categories - b.shape[1]))) for b in blocks])
        passthrough = [j for j in range(X.shape[1]) if j not in self._encoded]
        try:
            numbers = X[:, passthrough].astype(float)
        except (TypeError, ValueError):
            raise TypeError("OneHotEncoder: a column that is not encoded contains text; add it to `columns`.") from None
        return np.hstack([numbers, encoded])

    def get_feature_names_out(self, input_features=None):
        """Names of the output columns, in order (2.1 naming: "<column>_<category>")."""
        base._check_fitted(self, "categories_")
        names = [f"x{j}" for j in range(self._n_columns)] if input_features is None else list(input_features)
        if len(names) != self._n_columns:
            raise ValueError(f"input_features has {len(names)} names, expected {self._n_columns}")
        named = super().get_feature_names_out([names[j] for j in self._encoded]).tolist()
        out, start = [names[j] for j in range(self._n_columns) if j not in self._encoded], 0
        for column, categories in zip(self._encoded, self.categories_):
            out += named[start:start + len(categories)]
            start += len(categories)
            if self.max_categories is not None:
                out += [f"{names[column]}_<unused>"] * (self.max_categories - len(categories))
        return np.asarray(out, dtype=str)

    def _check_width(self, X):
        if X.shape[1] != self._n_columns:
            raise ValueError(f"X has {X.shape[1]} columns, but OneHotEncoder was fitted with {self._n_columns} columns")

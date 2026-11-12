"""
preprocessing.py -- Scaling, imputation and one-hot encoding for NumCompute.

Provides:
    TransformerMixin
        -> base class that gives every transformer fit_transform(X, y=None)
    NotFittedError
        -> raised when transform() is called before fit()
    StandardScaler(with_mean=True, with_std=True)
        -> z-scores: fit stores mean_, var_ and scale_, and transform applies
           (X - mean_) / scale_safe, mapping zero-variance columns to 0
    MinMaxScaler(feature_range=(0, 1), clip=False)
        -> rescales each column into feature_range: fit stores min_ and max_
    SimpleImputer(strategy='mean'|'median'|'most_frequent'|'constant',
                  fill_value=0)
        -> replaces NaN with a per-column value stored in statistics_
    OneHotEncoder(handle_unknown='error'|'ignore', dtype=np.float64)
        -> one 0/1 column per category: fit stores categories_, and
           get_feature_names_out(input_features=None) names the outputs

Requires:
    Python >= 3.9 and NumPy >= 1.23; no other numcompute modules (works on
    the arrays returned by io.read_csv). Input must be 2-D (n_samples,
    n_features): numeric with NaN as missing for the scalers and imputer,
    and strings, numbers or objects for OneHotEncoder.

Transformers follow a scikit-learn style protocol:

* ``fit(X, y=None) -> self`` learns per-column statistics,
* ``transform(X) -> X_out`` applies them,
* ``fit_transform(X, y=None) -> X_out`` (from :class:`TransformerMixin`).

Conventions shared by every transformer
---------------------------------------
* ``X`` must be 2-D with shape ``(n_samples, n_features)``. 1-D input is
  rejected with a hint to use ``X.reshape(-1, 1)`` so a single feature is
  never confused with a single sample.
* ``transform`` raises ``ValueError`` when ``X`` has a different number of
  columns than the data seen in ``fit``, and :class:`NotFittedError` when
  called before ``fit``.
* Inputs are never modified in place; a new array is returned.
* Learned attributes end in ``_`` (``mean_``, ``categories_``, ...).
* Numeric transformers return ``float64`` and reject ``±inf`` (an infinite
  value makes every statistic of its column meaningless).
* Statistics are computed column-wise with vectorised NumPy reductions;
  the only Python loops run over *columns* where a per-column result has
  a different length (``OneHotEncoder.categories_``), never over rows.
"""

from __future__ import annotations

import warnings

import numpy as np


# ----------------------------------------------------------------------------
# Shared base classes and validation helpers
# ----------------------------------------------------------------------------
class NotFittedError(ValueError, AttributeError):
    """Raised when ``transform`` is called before ``fit``.

    Subclasses both ``ValueError`` and ``AttributeError`` so callers can
    catch it either way.
    """


class TransformerMixin:
    """Mixin giving every transformer ``fit_transform`` and a readable repr.

    Subclasses implement ``fit(X, y=None) -> self`` and ``transform(X)``.
    """

    def fit_transform(self, X, y=None) -> np.ndarray:
        """Fit to ``X`` and return the transformed ``X``.

        Equivalent to ``self.fit(X, y).transform(X)``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : ignored
            Present for pipeline compatibility.

        Returns
        -------
        X_out : numpy.ndarray, shape (n_samples, n_features_out)
            Transformed data.

        Raises
        ------
        TypeError, ValueError
            Whatever the subclass's ``fit`` or ``transform`` raises for bad
            input, passed through unchanged (for example non-numeric data, a
            non-2-D array, an empty array or an invalid parameter). See each
            transformer's ``fit`` and ``transform`` for the exact cases.

        Complexity
        ----------
        Cost of ``fit`` plus cost of ``transform``.
        """
        return self.fit(X, y).transform(X)

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self._init_params().items())
        return f"{type(self).__name__}({params})"

    def _init_params(self) -> dict:
        """Constructor arguments (public attributes without a trailing '_')."""
        return {k: v for k, v in vars(self).items() if not k.startswith("_") and not k.endswith("_")}


def _check_2d(X, owner: str) -> np.ndarray:
    """Return ``X`` as an ndarray, raising if it is not 2-D."""
    X = np.asarray(X)
    if X.ndim != 2:
        hint = " Use X.reshape(-1, 1) for a single feature." if X.ndim == 1 else ""
        raise ValueError(
            f"{owner} expects a 2-D array of shape (n_samples, n_features), "
            f"got {X.ndim}-D array with shape {X.shape}.{hint}"
        )
    return X


def _check_numeric(X, owner: str) -> np.ndarray:
    """Validate ``X`` as a finite-or-NaN 2-D float64 array (a new or viewed array).

    Raises
    ------
    TypeError
        If ``X`` is not numeric (e.g. strings).
    ValueError
        If ``X`` is not 2-D or contains ``±inf``.

    Complexity: O(n·m) time; O(n·m) memory when a dtype conversion is needed.
    """
    X = _check_2d(X, owner)
    if X.dtype.kind in "biuf":
        X = X.astype(np.float64, copy=False)
    elif X.dtype.kind == "O":
        try:
            X = X.astype(np.float64)
        except (TypeError, ValueError):
            raise TypeError(f"{owner} requires numeric input; X contains non-numeric objects") from None
    else:
        raise TypeError(
            f"{owner} requires numeric input, got dtype {X.dtype}. "
            "Encode categorical columns with OneHotEncoder instead."
        )
    if np.isinf(X).any():
        r, c = np.argwhere(np.isinf(X))[0]
        raise ValueError(f"{owner}: X contains infinity at row {r}, column {c}")
    return X


def _check_fitted(est, attr: str) -> None:
    if not hasattr(est, attr):
        raise NotFittedError(
            f"This {type(est).__name__} instance is not fitted yet; call fit(X) before transform(X)."
        )


def _check_n_features(est, X: np.ndarray, *, reset: bool) -> None:
    """Record (``reset=True``) or verify the number of columns of ``X``."""
    owner = type(est).__name__
    n_samples, n_features = X.shape
    if reset:
        if n_samples == 0:
            raise ValueError(f"{owner}.fit requires at least one sample, got X with shape {X.shape}")
        if n_features == 0:
            raise ValueError(f"{owner}.fit requires at least one feature, got X with shape {X.shape}")
        est.n_features_in_ = n_features
    elif n_features != est.n_features_in_:
        raise ValueError(
            f"X has {n_features} features, but {owner} was fitted with "
            f"{est.n_features_in_} features"
        )


def _nan_stat(func, X: np.ndarray) -> np.ndarray:
    """Apply a ``np.nan*`` reduction column-wise, silencing all-NaN warnings.

    All-NaN columns yield NaN; callers decide how to handle them.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return func(X, axis=0)


def _safe_divisor(d: np.ndarray) -> np.ndarray:
    """Replace 0 and NaN divisors with 1.0 (zero-variance / empty-column guard)."""
    return np.where((d == 0) | np.isnan(d), 1.0, d)


# ----------------------------------------------------------------------------
# StandardScaler
# ----------------------------------------------------------------------------
class StandardScaler(TransformerMixin):
    """Standardise features to zero mean and unit variance (z-scores).

    ``transform`` computes ``(X - mean_) / scale_safe`` where
    ``scale_safe = np.where(scale_ == 0, 1.0, scale_)``.

    Parameters
    ----------
    with_mean : bool, default True
        Subtract the column mean. If False, ``mean_`` is all zeros.
    with_std : bool, default True
        Divide by the column standard deviation. If False, ``scale_`` is
        all ones.

    Attributes
    ----------
    mean_ : ndarray of shape (n_features,)
        Per-column mean of the non-NaN training values.
    var_ : ndarray of shape (n_features,)
        Per-column population variance (``ddof=0``) of the non-NaN values.
    scale_ : ndarray of shape (n_features,)
        ``sqrt(var_)``, the true standard deviation (0 for constant
        columns); the zero guard is applied only at transform time.
    n_features_in_ : int
        Number of columns seen in ``fit``.

    Notes
    -----
    * **Zero variance:** a constant column (``nanmax == nanmin``) gets
      ``scale_ = 0`` and ``mean_`` exactly equal to its value, so it
      transforms to exactly 0 instead of tiny rounding noise.
    * **NaN:** ignored in ``fit`` (``np.nanmean``/``np.nanstd``) and
      propagated by ``transform``. An all-NaN column has ``mean_`` and
      ``scale_`` of NaN and transforms to NaN; impute first.
    * **Stability:** each column is divided by its max-abs value before the
      mean and standard deviation are computed and then rescaled, so values
      near ``1e300`` do not overflow when squared.
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X, y=None) -> "StandardScaler":
        """Compute per-column mean and standard deviation.

        Parameters
        ----------
        X : array-like of numbers, shape (n_samples, n_features)
            Training data; NaN marks missing values.
        y : ignored

        Returns
        -------
        self : StandardScaler

        Raises
        ------
        TypeError
            If ``X`` is not numeric.
        ValueError
            If ``X`` is not 2-D, is empty, or contains ``±inf``.

        Complexity
        ----------
        O(n·m) time, O(n·m) temporary memory, O(m) stored state.
        """
        X = _check_numeric(X, "StandardScaler")
        _check_n_features(self, X, reset=True)

        # Divide by max |x| per column so mean/std cannot overflow, then rescale.
        amax = _safe_divisor(_nan_stat(np.nanmax, np.abs(X)))
        Xs = X / amax
        mean = _nan_stat(np.nanmean, Xs) * amax
        scale = _nan_stat(np.nanstd, Xs) * amax

        constant = _nan_stat(np.nanmax, X) == _nan_stat(np.nanmin, X)
        mean = np.where(constant, _nan_stat(np.nanmin, X), mean)
        scale = np.where(constant, 0.0, scale)

        n = X.shape[1]
        with np.errstate(over="ignore"):  # var_ of |x| > 1e154 is honestly inf
            self.var_ = scale ** 2
        self.mean_ = mean if self.with_mean else np.zeros(n)
        self.scale_ = scale if self.with_std else np.ones(n)
        return self

    def transform(self, X) -> np.ndarray:
        """Standardise ``X`` with the statistics learned in ``fit``.

        Parameters
        ----------
        X : array-like of numbers, shape (n_samples, n_features_in_)

        Returns
        -------
        X_out : ndarray of float64, shape (n_samples, n_features_in_)
            ``(X - mean_) / scale_safe``; NaN stays NaN.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        TypeError
            If ``X`` is not numeric.
        ValueError
            If ``X`` is not 2-D, contains ``±inf``, or its column count
            differs from ``n_features_in_``.

        Complexity
        ----------
        O(n·m) time and memory (one broadcast subtract and divide).
        """
        _check_fitted(self, "mean_")
        X = _check_numeric(X, "StandardScaler")
        _check_n_features(self, X, reset=False)
        return (X - self.mean_) / _safe_divisor(self.scale_)


# ----------------------------------------------------------------------------
# MinMaxScaler
# ----------------------------------------------------------------------------
class MinMaxScaler(TransformerMixin):
    """Scale each feature linearly into ``feature_range``.

    ``transform`` computes
    ``lo + (X - min_) / range_safe * (hi - lo)`` with
    ``range_safe = np.where(max_ - min_ == 0, 1.0, max_ - min_)``.

    Parameters
    ----------
    feature_range : tuple (lo, hi), default (0, 1)
        Target range; requires finite ``lo < hi``.
    clip : bool, default False
        If True, clip transformed values to ``feature_range``. Otherwise
        values outside the training range map outside ``feature_range``.

    Attributes
    ----------
    min_ : ndarray of shape (n_features,)
        Per-column minimum of the non-NaN training values.
    max_ : ndarray of shape (n_features,)
        Per-column maximum of the non-NaN training values.
    data_range_ : ndarray of shape (n_features,)
        ``max_ - min_`` (may be ``inf`` if the span exceeds float range;
        the transform itself stays finite, see Notes).
    n_features_in_ : int
        Number of columns seen in ``fit``.

    Notes
    -----
    * **Zero range:** a constant column maps to ``lo``.
    * **NaN:** ignored in ``fit`` and propagated by ``transform``; an
      all-NaN column has NaN ``min_``/``max_`` and transforms to NaN.
    * **Stability:** the scaling is computed on halves,
      ``(X/2 - min_/2) / (max_/2 - min_/2)``, which cannot overflow even
      when ``max_ - min_`` exceeds the float64 range.
    """

    def __init__(self, feature_range: tuple[float, float] = (0, 1), clip: bool = False):
        self.feature_range = feature_range
        self.clip = clip

    def _range(self) -> tuple[float, float]:
        try:
            lo, hi = (float(v) for v in self.feature_range)
        except (TypeError, ValueError):
            raise ValueError(
                f"feature_range must be a pair of numbers (lo, hi), got {self.feature_range!r}"
            ) from None
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise ValueError(f"feature_range must satisfy finite lo < hi, got {self.feature_range!r}")
        return lo, hi

    def fit(self, X, y=None) -> "MinMaxScaler":
        """Compute per-column minimum and maximum.

        Parameters
        ----------
        X : array-like of numbers, shape (n_samples, n_features)
            Training data; NaN marks missing values.
        y : ignored

        Returns
        -------
        self : MinMaxScaler

        Raises
        ------
        TypeError
            If ``X`` is not numeric.
        ValueError
            If ``feature_range`` is invalid, or ``X`` is not 2-D, is empty,
            or contains ``±inf``.

        Complexity
        ----------
        O(n·m) time, O(m) stored state.
        """
        self._range()  # fail fast on a bad feature_range
        X = _check_numeric(X, "MinMaxScaler")
        _check_n_features(self, X, reset=True)
        self.min_ = _nan_stat(np.nanmin, X)
        self.max_ = _nan_stat(np.nanmax, X)
        with np.errstate(over="ignore"):
            self.data_range_ = self.max_ - self.min_
        return self

    def transform(self, X) -> np.ndarray:
        """Scale ``X`` into ``feature_range`` using the fitted min and max.

        Parameters
        ----------
        X : array-like of numbers, shape (n_samples, n_features_in_)

        Returns
        -------
        X_out : ndarray of float64, shape (n_samples, n_features_in_)

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        TypeError
            If ``X`` is not numeric.
        ValueError
            If ``X`` is not 2-D, contains ``±inf``, or its column count
            differs from ``n_features_in_``.

        Complexity
        ----------
        O(n·m) time and memory.
        """
        _check_fitted(self, "min_")
        lo, hi = self._range()
        X = _check_numeric(X, "MinMaxScaler")
        _check_n_features(self, X, reset=False)

        half_range = _safe_divisor(self.max_ / 2 - self.min_ / 2)
        X_out = (X / 2 - self.min_ / 2) / half_range * (hi - lo) + lo
        if self.clip:
            X_out = np.clip(X_out, lo, hi)  # NaN passes through np.clip
        return X_out


# ----------------------------------------------------------------------------
# SimpleImputer
# ----------------------------------------------------------------------------
_IMPUTE_STRATEGIES = ("mean", "median", "most_frequent", "constant")


def _nan_mode(X: np.ndarray) -> np.ndarray:
    """Column-wise most frequent non-NaN value, fully vectorised.

    Ties resolve to the smallest value. All-NaN columns return NaN.

    Method: sort each column (NaNs go last), compute for every position the
    length of the run of equal values ending there, then ``argmax`` over the
    run lengths. Because columns are sorted ascending and ``argmax`` returns
    the first maximum, ties pick the smallest value.

    Complexity: O(n log n · m) time, O(n·m) memory.
    """
    n, m = X.shape
    S = np.sort(X, axis=0)
    rows = np.arange(n)[:, None]
    new_run = np.ones_like(S, dtype=bool)
    new_run[1:] = S[1:] != S[:-1]
    run_start = np.maximum.accumulate(np.where(new_run, rows, 0), axis=0)
    run_len = np.where(np.isnan(S), -1, rows - run_start + 1)
    return S[np.argmax(run_len, axis=0), np.arange(m)]


class SimpleImputer(TransformerMixin):
    """Replace NaN with a per-column statistic.

    ``transform`` computes ``np.where(np.isnan(X), statistics_, X)``.

    Parameters
    ----------
    strategy : {'mean', 'median', 'most_frequent', 'constant'}, default 'mean'
        * ``'mean'`` / ``'median'``: ``np.nanmean`` / ``np.nanmedian``.
        * ``'most_frequent'``: column mode; ties go to the smallest value.
        * ``'constant'``: always ``fill_value``.
    fill_value : float, default 0
        Value used by ``'constant'``, and the fallback for columns that are
        entirely NaN during ``fit`` (see Notes).

    Attributes
    ----------
    statistics_ : ndarray of float64, shape (n_features,)
        Fill value for each column.
    n_features_in_ : int
        Number of columns seen in ``fit``.

    Notes
    -----
    * **All-NaN column:** mean, median and mode are undefined, so the
      column's statistic falls back to ``fill_value`` and a ``UserWarning``
      names the affected columns. The column is kept (not dropped) so the
      output shape always equals the input shape.
    * **Numeric only:** NaN is the missing marker. Categorical columns read
      with ``read_csv(dtype=str)`` mark missing as ``''``, which
      :class:`OneHotEncoder` treats as its own category.
    """

    def __init__(self, strategy: str = "mean", fill_value: float = 0):
        self.strategy = strategy
        self.fill_value = fill_value

    def fit(self, X, y=None) -> "SimpleImputer":
        """Learn the fill value of every column.

        Parameters
        ----------
        X : array-like of numbers, shape (n_samples, n_features)
            Training data; NaN marks missing values.
        y : ignored

        Returns
        -------
        self : SimpleImputer

        Raises
        ------
        TypeError
            If ``X`` is not numeric.
        ValueError
            If ``strategy`` is unknown, ``fill_value`` is not a finite
            number, or ``X`` is not 2-D, is empty, or contains ``±inf``.

        Warns
        -----
        UserWarning
            If a column is entirely NaN and the strategy is not 'constant'.

        Complexity
        ----------
        O(n·m) for 'mean'/'constant', O(n·m) average for 'median'
        (partition-based), O(n log n · m) for 'most_frequent'.
        """
        if self.strategy not in _IMPUTE_STRATEGIES:
            raise ValueError(f"strategy must be one of {_IMPUTE_STRATEGIES}, got {self.strategy!r}")
        try:
            fill = float(self.fill_value)
        except (TypeError, ValueError):
            raise ValueError(f"fill_value must be a number, got {self.fill_value!r}") from None
        if not np.isfinite(fill):
            raise ValueError(f"fill_value must be finite, got {self.fill_value!r}")

        X = _check_numeric(X, "SimpleImputer")
        _check_n_features(self, X, reset=True)

        if self.strategy == "constant":
            stats = np.full(X.shape[1], fill)
        else:
            if self.strategy == "mean":
                amax = _safe_divisor(_nan_stat(np.nanmax, np.abs(X)))  # overflow guard
                stats = _nan_stat(np.nanmean, X / amax) * amax
            elif self.strategy == "median":
                stats = _nan_stat(np.nanmedian, X)
            else:
                stats = _nan_mode(X)

            empty = np.isnan(stats)
            if empty.any():
                warnings.warn(
                    f"SimpleImputer: columns {np.flatnonzero(empty).tolist()} are entirely NaN; "
                    f"filling them with fill_value={fill}",
                    UserWarning,
                    stacklevel=2,
                )
                stats = np.where(empty, fill, stats)

        self.statistics_ = stats
        return self

    def transform(self, X) -> np.ndarray:
        """Replace every NaN in ``X`` with its column's ``statistics_`` value.

        Parameters
        ----------
        X : array-like of numbers, shape (n_samples, n_features_in_)

        Returns
        -------
        X_out : ndarray of float64, shape (n_samples, n_features_in_)
            Copy of ``X`` with no NaN values.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        TypeError
            If ``X`` is not numeric.
        ValueError
            If ``X`` is not 2-D, contains ``±inf``, or its column count
            differs from ``n_features_in_``.

        Complexity
        ----------
        O(n·m) time and memory.
        """
        _check_fitted(self, "statistics_")
        X = _check_numeric(X, "SimpleImputer")
        _check_n_features(self, X, reset=False)
        return np.where(np.isnan(X), self.statistics_, X)


# ----------------------------------------------------------------------------
# OneHotEncoder
# ----------------------------------------------------------------------------
_HANDLE_UNKNOWN = ("error", "ignore")
_TEXT_KINDS, _NUMBER_KINDS = "US", "biuf"


def _kind_group(dtype: np.dtype) -> str:
    """Classify a dtype as 'text', 'number' or 'object' for compatibility checks."""
    if dtype.kind in _TEXT_KINDS:
        return "text"
    if dtype.kind in _NUMBER_KINDS:
        return "number"
    if dtype.kind == "O":
        return "object"
    raise TypeError(f"OneHotEncoder cannot encode dtype {dtype}")


class OneHotEncoder(TransformerMixin):
    """Encode each categorical column as a block of 0/1 indicator columns.

    ``fit`` stores ``categories_[j] = np.unique(X[:, j])``; ``transform``
    builds each block by broadcasting ``X[:, j][:, None] == categories_[j]``
    and joins the blocks with ``np.hstack``.

    Parameters
    ----------
    handle_unknown : {'error', 'ignore'}, default 'error'
        What to do with a category at transform time that was not seen in
        ``fit``:

        * ``'error'`` (default): raise ``ValueError`` naming the column and
          the unseen values. Safest: a new category usually means the
          training data was not representative, or the data is wrong.
        * ``'ignore'``: encode the unseen value as an all-zero block for
          that column. Output shape is unchanged, so downstream steps keep
          working; the row simply matches no known category.
    dtype : numeric dtype, default numpy.float64
        dtype of the output array.

    Attributes
    ----------
    categories_ : list of ndarray, length n_features
        Sorted unique values of each column (``np.unique``). Column ``j``
        contributes ``len(categories_[j])`` output columns, in this order.
    n_features_in_ : int
        Number of columns seen in ``fit``.

    Notes
    -----
    * Accepts string columns (e.g. ``read_csv(..., dtype=str)``), numeric
      columns, or object arrays.
    * **Missing values are a category:** ``''`` in string data and ``NaN``
      in numeric data each get their own indicator column (all NaNs count
      as one category).
    * Strings and numbers never match each other: transforming numeric
      data with an encoder fitted on strings (or vice versa) raises
      ``TypeError`` rather than silently reporting every value as unknown.
    * The loop runs over columns, not rows, because each column has a
      different number of categories; each block is fully vectorised.
    """

    def __init__(self, handle_unknown: str = "error", dtype=np.float64):
        self.handle_unknown = handle_unknown
        self.dtype = dtype

    def fit(self, X, y=None) -> "OneHotEncoder":
        """Learn the sorted unique categories of every column.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Categorical data (strings, numbers or objects).
        y : ignored

        Returns
        -------
        self : OneHotEncoder

        Raises
        ------
        TypeError
            If the dtype cannot be encoded or an object column mixes
            incomparable types (e.g. ``str`` and ``int``).
        ValueError
            If ``handle_unknown`` is invalid, or ``X`` is not 2-D or empty.

        Complexity
        ----------
        O(n log n · m) time (one sort per column); O(Σ k_j) stored state
        for ``k_j`` categories in column ``j``.
        """
        if self.handle_unknown not in _HANDLE_UNKNOWN:
            raise ValueError(
                f"handle_unknown must be one of {_HANDLE_UNKNOWN}, got {self.handle_unknown!r}"
            )
        X = _check_2d(X, "OneHotEncoder")
        self._kind = _kind_group(X.dtype)
        _check_n_features(self, X, reset=True)
        try:
            self.categories_ = [np.unique(X[:, j]) for j in range(X.shape[1])]
        except TypeError as exc:
            raise TypeError(f"OneHotEncoder: a column mixes incomparable types ({exc})") from None
        return self

    def transform(self, X) -> np.ndarray:
        """Return the one-hot encoding of ``X``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features_in_)

        Returns
        -------
        X_out : ndarray of ``dtype``, shape (n_samples, sum_j len(categories_[j]))
            For each input column, one indicator column per category, in
            the order of ``categories_``.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        TypeError
            If ``X`` holds strings but the encoder was fitted on numbers,
            or vice versa.
        ValueError
            If ``X`` is not 2-D, its column count differs from
            ``n_features_in_``, or (with ``handle_unknown='error'``) it
            contains a category not seen in ``fit``.

        Complexity
        ----------
        O(n · Σ k_j) time and memory: each block is an ``n × k_j``
        broadcast comparison.
        """
        _check_fitted(self, "categories_")
        X = _check_2d(X, "OneHotEncoder")
        kind = _kind_group(X.dtype)
        if "object" not in (kind, self._kind) and kind != self._kind:
            raise TypeError(
                f"OneHotEncoder was fitted on {self._kind} data but transform got "
                f"{kind} data (dtype {X.dtype})"
            )
        _check_n_features(self, X, reset=False)

        blocks = []
        for j, cats in enumerate(self.categories_):
            col = X[:, j]
            block = col[:, None] == cats[None, :]
            if col.dtype.kind == "f" and cats.dtype.kind == "f":
                block |= np.isnan(col)[:, None] & np.isnan(cats)[None, :]

            unknown = ~block.any(axis=1)
            if unknown.any() and self.handle_unknown == "error":
                seen = np.unique(col[unknown])
                shown = ", ".join(repr(v.item() if hasattr(v, "item") else v) for v in seen[:5])
                more = f" and {seen.size - 5} more" if seen.size > 5 else ""
                raise ValueError(
                    f"OneHotEncoder: found unknown categories [{shown}]{more} in column {j} "
                    f"during transform (first at row {int(np.argmax(unknown))}). "
                    "Use handle_unknown='ignore' to encode them as all zeros."
                )
            blocks.append(block)

        return np.hstack(blocks).astype(self.dtype, copy=False)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Names of the output columns, e.g. ``['city_Adelaide', 'city_Perth']``.

        Parameters
        ----------
        input_features : sequence of str, optional
            Input column names (e.g. the ``colnames`` returned by
            ``read_csv``). Default: ``'x0', 'x1', ...``.

        Returns
        -------
        names : ndarray of str, shape (n_features_out,)
            ``f"{input_feature}_{category}"`` for every output column.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        ValueError
            If ``input_features`` has the wrong length.

        Complexity
        ----------
        O(Σ k_j) time and memory.
        """
        _check_fitted(self, "categories_")
        if input_features is None:
            input_features = [f"x{j}" for j in range(self.n_features_in_)]
        elif len(input_features) != self.n_features_in_:
            raise ValueError(
                f"input_features has {len(input_features)} names, expected {self.n_features_in_}"
            )
        names = [f"{name}_{cat}" for name, cats in zip(input_features, self.categories_)
                            for cat in cats.tolist()]
        return np.asarray(names, dtype=str)

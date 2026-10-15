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


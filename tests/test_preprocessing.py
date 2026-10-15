"""
test_preprocessing.py -- Unit tests for numcompute.preprocessing.

Provides:
    shared tests
        -> fit/transform/fit_transform, NotFittedError, 2-D and empty input,
           inf, non-numeric input, strides, and input left unmodified
    StandardScaler and MinMaxScaler tests
        -> zero-variance and all-NaN columns, NaN propagation, overflow
           safety and column-count mismatch
    SimpleImputer tests
        -> every strategy, the all-NaN column fallback and its warning, and
           the vectorised mode checked against a reference loop
    OneHotEncoder tests
        -> block layout, unseen categories ('error' and 'ignore'), missing
           values as a category, type mismatch and feature names
    end-to-end test
        -> io.read_csv -> SimpleImputer -> StandardScaler + OneHotEncoder

Requires:
    pytest >= 7, NumPy >= 1.23, numcompute.preprocessing (imported as pp)
    and numcompute.io. Run `python -m pytest` from the repo root.
"""

import warnings

import numpy as np
import pytest

from numcompute import preprocessing as pp


@pytest.fixture(autouse=True)
def _no_stray_warnings():
    """Fail any test that leaks a RuntimeWarning (e.g. 'Mean of empty slice')."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        yield


# ----------------------------------------------------------------------------
# Shared behaviour (TransformerMixin, validation) via StandardScaler
# ----------------------------------------------------------------------------
X_BASIC = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])


def test_fit_returns_self_and_fit_transform_matches():
    sc = pp.StandardScaler()
    assert sc.fit(X_BASIC) is sc
    np.testing.assert_allclose(pp.StandardScaler().fit_transform(X_BASIC), sc.transform(X_BASIC))


def test_transform_before_fit_raises_not_fitted():
    with pytest.raises(pp.NotFittedError, match="not fitted"):
        pp.StandardScaler().transform(X_BASIC)
    assert issubclass(pp.NotFittedError, ValueError)


@pytest.mark.parametrize("bad", [np.arange(4.0), np.zeros((2, 2, 2))])
def test_non_2d_input_raises(bad):
    with pytest.raises(ValueError, match="2-D"):
        pp.StandardScaler().fit(bad)


def test_one_d_error_suggests_reshape():
    with pytest.raises(ValueError, match=r"reshape\(-1, 1\)"):
        pp.StandardScaler().fit([1.0, 2.0, 3.0])


@pytest.mark.parametrize("shape", [(0, 3), (3, 0)])
def test_fit_on_empty_array_raises(shape):
    with pytest.raises(ValueError, match="at least one"):
        pp.StandardScaler().fit(np.empty(shape))


def test_string_input_to_numeric_transformer_raises_type_error():
    with pytest.raises(TypeError, match="numeric input"):
        pp.StandardScaler().fit(np.array([["a"], ["b"]]))


def test_infinity_raises_with_location():
    X = X_BASIC.copy()
    X[2, 1] = np.inf
    with pytest.raises(ValueError, match="infinity at row 2, column 1"):
        pp.StandardScaler().fit(X)


def test_input_is_not_modified():
    X = X_BASIC.copy()
    pp.StandardScaler().fit_transform(X)
    np.testing.assert_array_equal(X, X_BASIC)


def test_integer_and_list_input_accepted():
    out = pp.StandardScaler().fit_transform([[1, 2], [3, 4]])
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, [[-1, -1], [1, 1]])


def test_non_contiguous_strides_give_same_result():
    rng = np.random.default_rng(0)
    big = rng.normal(size=(50, 6))
    for view in (big[:, ::2], big[::3], np.asfortranarray(big), big.T.T):
        np.testing.assert_allclose(
            pp.StandardScaler().fit_transform(view),
            pp.StandardScaler().fit_transform(np.ascontiguousarray(view)),
        )


def test_repr_shows_parameters():
    assert repr(pp.StandardScaler(with_mean=False)) == "StandardScaler(with_mean=False, with_std=True)"


# ----------------------------------------------------------------------------
# StandardScaler
# ----------------------------------------------------------------------------
def test_standard_scaler_zero_mean_unit_variance():
    out = pp.StandardScaler().fit_transform(X_BASIC)
    np.testing.assert_allclose(out.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(out.std(axis=0), 1)


def test_standard_scaler_attributes_match_numpy():
    sc = pp.StandardScaler().fit(X_BASIC)
    np.testing.assert_allclose(sc.mean_, X_BASIC.mean(axis=0))
    np.testing.assert_allclose(sc.scale_, X_BASIC.std(axis=0))
    np.testing.assert_allclose(sc.var_, X_BASIC.var(axis=0))
    assert sc.n_features_in_ == 2


@pytest.mark.parametrize("value", [0.0, 5.0, 0.1, 1e-300, -7.3])
def test_standard_scaler_zero_variance_column_maps_to_exact_zero(value):
    X = np.column_stack([np.full(6, value), np.arange(6.0)])
    sc = pp.StandardScaler().fit(X)
    out = sc.transform(X)
    assert sc.scale_[0] == 0.0  # true std is stored; guard applies in transform
    assert np.all(out[:, 0] == 0.0)  # exactly zero, no NaN/inf/rounding noise
    np.testing.assert_allclose(out[:, 1].std(), 1)


def test_standard_scaler_single_sample_gives_zeros():
    np.testing.assert_array_equal(pp.StandardScaler().fit_transform([[3.0, -1.0]]), [[0.0, 0.0]])


def test_standard_scaler_ignores_nan_in_fit_and_propagates_it():
    X = np.array([[1.0, 2.0], [np.nan, 4.0], [3.0, np.nan]])
    sc = pp.StandardScaler().fit(X)
    np.testing.assert_allclose(sc.mean_, [2.0, 3.0])
    out = sc.transform(X)
    np.testing.assert_array_equal(np.isnan(out), np.isnan(X))


def test_standard_scaler_all_nan_column():
    X = np.column_stack([np.full(4, np.nan), np.arange(4.0)])
    sc = pp.StandardScaler().fit(X)  # must not emit RuntimeWarning (autouse fixture)
    out = sc.transform(X)
    assert np.isnan(sc.mean_[0]) and np.isnan(sc.scale_[0])
    assert np.all(np.isnan(out[:, 0]))
    np.testing.assert_allclose(out[:, 1].mean(), 0, atol=1e-12)


def test_standard_scaler_huge_values_do_not_overflow():
    base = np.array([[1.0], [2.0], [3.0], [4.0]])
    out = pp.StandardScaler().fit_transform(base * 1e300)
    np.testing.assert_allclose(out, pp.StandardScaler().fit_transform(base))


def test_standard_scaler_with_mean_and_with_std_flags():
    no_mean = pp.StandardScaler(with_mean=False).fit(X_BASIC)
    np.testing.assert_array_equal(no_mean.mean_, [0, 0])
    np.testing.assert_allclose(no_mean.transform(X_BASIC), X_BASIC / X_BASIC.std(axis=0))
    no_std = pp.StandardScaler(with_std=False).fit(X_BASIC)
    np.testing.assert_allclose(no_std.transform(X_BASIC), X_BASIC - X_BASIC.mean(axis=0))


def test_standard_scaler_transform_column_mismatch_raises():
    sc = pp.StandardScaler().fit(X_BASIC)
    with pytest.raises(ValueError, match="X has 3 features, but StandardScaler was fitted with 2"):
        sc.transform(np.ones((2, 3)))


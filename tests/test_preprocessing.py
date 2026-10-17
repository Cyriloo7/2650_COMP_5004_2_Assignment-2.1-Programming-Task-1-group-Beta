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

# ----------------------------------------------------------------------------
# MinMaxScaler
# ----------------------------------------------------------------------------
def test_minmax_default_range_is_zero_to_one():
    mm = pp.MinMaxScaler().fit(X_BASIC)
    np.testing.assert_array_equal(mm.min_, [1, 10])
    np.testing.assert_array_equal(mm.max_, [4, 40])
    np.testing.assert_allclose(mm.transform(X_BASIC)[:, 0], [0, 1 / 3, 2 / 3, 1])


def test_minmax_custom_feature_range():
    out = pp.MinMaxScaler(feature_range=(-1, 1)).fit_transform(X_BASIC)
    np.testing.assert_allclose(out.min(axis=0), -1)
    np.testing.assert_allclose(out.max(axis=0), 1)


def test_minmax_zero_range_column_maps_to_lower_bound():
    X = np.column_stack([np.full(3, 7.0), [0.0, 5.0, 10.0]])
    out = pp.MinMaxScaler(feature_range=(2, 5)).fit_transform(X)
    np.testing.assert_array_equal(out[:, 0], [2, 2, 2])
    np.testing.assert_allclose(out[:, 1], [2, 3.5, 5])


def test_minmax_all_nan_column_and_nan_propagation():
    X = np.array([[np.nan, 1.0], [np.nan, np.nan], [np.nan, 3.0]])
    out = pp.MinMaxScaler().fit_transform(X)
    assert np.all(np.isnan(out[:, 0]))
    np.testing.assert_array_equal(out[[0, 2], 1], [0, 1])
    assert np.isnan(out[1, 1])


def test_minmax_unseen_range_and_clip():
    mm = pp.MinMaxScaler().fit([[0.0], [10.0]])
    np.testing.assert_allclose(mm.transform([[-5.0], [20.0]]).ravel(), [-0.5, 2.0])
    mm.clip = True
    np.testing.assert_allclose(mm.transform([[-5.0], [20.0], [np.nan]])[:2].ravel(), [0, 1])
    assert np.isnan(mm.transform([[np.nan]])[0, 0])


def test_minmax_extreme_span_does_not_overflow():
    X = np.array([[-1e308], [0.0], [1e308]])
    mm = pp.MinMaxScaler().fit(X)
    assert np.isinf(mm.data_range_[0])  # span itself exceeds float64
    np.testing.assert_allclose(mm.transform(X).ravel(), [0, 0.5, 1])


@pytest.mark.parametrize("rng", [(1, 0), (0, 0), (0, np.inf), ("a", "b"), (1,)])
def test_minmax_invalid_feature_range_raises(rng):
    with pytest.raises(ValueError, match="feature_range"):
        pp.MinMaxScaler(feature_range=rng).fit(X_BASIC)


def test_minmax_transform_column_mismatch_raises():
    mm = pp.MinMaxScaler().fit(X_BASIC)
    with pytest.raises(ValueError, match="X has 1 features, but MinMaxScaler was fitted with 2"):
        mm.transform(np.ones((2, 1)))


# ----------------------------------------------------------------------------
# SimpleImputer
# ----------------------------------------------------------------------------
X_MISSING = np.array(
    [
        [1.0, 7.0, 3.0],
        [np.nan, 7.0, 3.0],
        [4.0, 1.0, np.nan],
        [10.0, np.nan, 5.0],
        [np.nan, 1.0, 5.0],
    ]
)


@pytest.mark.parametrize(
    "strategy, expected",
    [
        ("mean", [5.0, 4.0, 4.0]),
        ("median", [4.0, 4.0, 4.0]),
        ("most_frequent", [1.0, 1.0, 3.0]),  # ties -> smallest value
        ("constant", [0.0, 0.0, 0.0]),
    ],
)
def test_imputer_strategies(strategy, expected):
    imp = pp.SimpleImputer(strategy=strategy).fit(X_MISSING)
    np.testing.assert_allclose(imp.statistics_, expected)
    out = imp.transform(X_MISSING)
    assert out.shape == X_MISSING.shape and not np.isnan(out).any()
    observed = ~np.isnan(X_MISSING)
    np.testing.assert_array_equal(out[observed], X_MISSING[observed])  # untouched
    rows, cols = np.nonzero(~observed)
    np.testing.assert_allclose(out[rows, cols], np.asarray(expected)[cols])


def test_imputer_constant_fill_value():
    out = pp.SimpleImputer(strategy="constant", fill_value=-1).fit_transform(X_MISSING)
    assert np.sum(out == -1) == np.isnan(X_MISSING).sum()


@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent"])
def test_imputer_all_nan_column_falls_back_to_fill_value_with_warning(strategy):
    X = np.column_stack([np.arange(3.0), np.full(3, np.nan)])
    imp = pp.SimpleImputer(strategy=strategy, fill_value=9)
    with pytest.warns(UserWarning, match=r"columns \[1\] are entirely NaN"):
        out = imp.fit_transform(X)
    assert out.shape == X.shape  # column kept, not dropped
    np.testing.assert_array_equal(out[:, 1], [9, 9, 9])


def test_imputer_constant_all_nan_column_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = pp.SimpleImputer(strategy="constant").fit_transform(np.full((2, 1), np.nan))
    np.testing.assert_array_equal(out, [[0], [0]])


def test_imputer_transform_uses_training_statistics():
    imp = pp.SimpleImputer().fit([[0.0], [10.0]])
    np.testing.assert_array_equal(imp.transform([[np.nan], [3.0]]), [[5.0], [3.0]])


def test_vectorised_mode_matches_reference_loop():
    rng = np.random.default_rng(1)
    X = rng.integers(0, 5, size=(200, 7)).astype(float)
    X[rng.random(X.shape) < 0.2] = np.nan

    def ref(col):
        vals, counts = np.unique(col[~np.isnan(col)], return_counts=True)
        return vals[np.argmax(counts)]  # np.unique is sorted -> smallest on ties

    expected = [ref(X[:, j]) for j in range(X.shape[1])]
    np.testing.assert_array_equal(pp.SimpleImputer(strategy="most_frequent").fit(X).statistics_, expected)


def test_imputer_mean_of_huge_values_does_not_overflow():
    imp = pp.SimpleImputer().fit([[1e308], [1e308], [np.nan]])
    np.testing.assert_allclose(imp.statistics_, [1e308])


@pytest.mark.parametrize(
    "kwargs", [{"strategy": "mode"}, {"fill_value": np.nan}, {"fill_value": "zero"}]
)
def test_imputer_invalid_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        pp.SimpleImputer(**kwargs).fit(X_MISSING)


def test_imputer_transform_column_mismatch_raises():
    imp = pp.SimpleImputer().fit(X_MISSING)
    with pytest.raises(ValueError, match="X has 2 features, but SimpleImputer was fitted with 3"):
        imp.transform(np.ones((1, 2)))


# ----------------------------------------------------------------------------
# OneHotEncoder
# ----------------------------------------------------------------------------
CITIES = np.array([["Perth", "M"], ["Adelaide", "F"], ["Perth", "F"], ["Darwin", "M"]])


def test_onehot_categories_sorted_and_blocks_in_column_order():
    enc = pp.OneHotEncoder().fit(CITIES)
    assert [c.tolist() for c in enc.categories_] == [["Adelaide", "Darwin", "Perth"], ["F", "M"]]
    out = enc.transform(CITIES)
    assert out.dtype == np.float64
    np.testing.assert_array_equal(
        out,
        [
            [0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 1, 0, 0, 1],
        ],
    )
    np.testing.assert_array_equal(out.sum(axis=1), 2)  # one hot per input column


def test_onehot_unseen_category_raises_by_default():
    enc = pp.OneHotEncoder().fit(CITIES)
    unseen = np.array([["Perth", "M"], ["Hobart", "F"]])
    with pytest.raises(ValueError, match=r"unknown categories \['Hobart'\] in column 0.*row 1"):
        enc.transform(unseen)


def test_onehot_unseen_category_ignore_gives_all_zero_block():
    enc = pp.OneHotEncoder(handle_unknown="ignore").fit(CITIES)
    out = enc.transform(np.array([["Hobart", "F"], ["Perth", "X"]]))
    assert out.shape == (2, 5)
    np.testing.assert_array_equal(out, [[0, 0, 0, 1, 0], [0, 0, 1, 0, 0]])


def test_onehot_unseen_message_truncates_long_lists():
    enc = pp.OneHotEncoder().fit([[0], [1]])
    with pytest.raises(ValueError, match=r"\[2, 3, 4, 5, 6\] and 4 more"):
        enc.transform(np.arange(2, 11).reshape(-1, 1))


def test_onehot_missing_values_are_their_own_category():
    enc = pp.OneHotEncoder().fit(np.array([["a"], [""], ["a"]]))
    assert enc.categories_[0].tolist() == ["", "a"]
    num = pp.OneHotEncoder().fit(np.array([[1.0], [np.nan], [np.nan], [2.0]]))
    assert num.categories_[0].size == 3  # all NaNs collapse to one category
    np.testing.assert_array_equal(num.transform([[np.nan], [2.0]]), [[0, 0, 1], [0, 1, 0]])


def test_onehot_numeric_categories_and_custom_dtype():
    enc = pp.OneHotEncoder(dtype=np.int8).fit([[3], [1], [3]])
    out = enc.transform([[1], [3]])
    assert out.dtype == np.int8
    np.testing.assert_array_equal(out, [[1, 0], [0, 1]])
    np.testing.assert_array_equal(enc.transform([[1.0]]), [[1, 0]])  # int/float compatible


def test_onehot_string_vs_number_mismatch_raises_type_error():
    enc = pp.OneHotEncoder().fit(CITIES)
    with pytest.raises(TypeError, match="fitted on text data but transform got number"):
        enc.transform(np.ones((2, 2)))


def test_onehot_mixed_object_column_raises_type_error():
    with pytest.raises(TypeError, match="incomparable types"):
        pp.OneHotEncoder().fit(np.array([["a"], [1]], dtype=object))


def test_onehot_invalid_handle_unknown_raises():
    with pytest.raises(ValueError, match="handle_unknown"):
        pp.OneHotEncoder(handle_unknown="skip").fit(CITIES)


def test_onehot_transform_column_mismatch_raises():
    enc = pp.OneHotEncoder().fit(CITIES)
    with pytest.raises(ValueError, match="X has 1 features, but OneHotEncoder was fitted with 2"):
        enc.transform(CITIES[:, :1])


def test_onehot_feature_names():
    enc = pp.OneHotEncoder().fit(CITIES)
    assert enc.get_feature_names_out().tolist()[:2] == ["x0_Adelaide", "x0_Darwin"]
    names = enc.get_feature_names_out(["city", "sex"]).tolist()
    assert names == ["city_Adelaide", "city_Darwin", "city_Perth", "sex_F", "sex_M"]
    with pytest.raises(ValueError, match="expected 2"):
        enc.get_feature_names_out(["city"])


def test_end_to_end_with_read_csv(tmp_path):
    from numcompute.io import read_csv

    path = tmp_path / "people.csv"
    path.write_text("age,city,income\n25,Adelaide,50000\n,NA,61000\n40,Perth,\n31,Perth,58000\n")
    X_num, _ = read_csv(path, usecols=["age", "income"])
    X_cat, cat_cols = read_csv(path, usecols=["city"], dtype=str)

    X_num = pp.StandardScaler().fit_transform(pp.SimpleImputer(strategy="median").fit_transform(X_num))
    enc = pp.OneHotEncoder().fit(X_cat)
    features = np.hstack([X_num, enc.transform(X_cat)])

    assert features.shape == (4, 5) and not np.isnan(features).any()
    assert enc.get_feature_names_out(cat_cols).tolist() == ["city_", "city_Adelaide", "city_Perth"]

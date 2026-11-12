"""Tests for preprocessing.py: the 2.1 transformers extended with partial_fit."""

import numpy as np
import pytest

from numcompute import preprocessing as base21
from numcompute.preprocessing import NotFittedError
from numcompute_stream.preprocessing import Imputer, MinMaxScaler, OneHotEncoder, StandardScaler


def data_with_gaps(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(5, 3, size=(300, 3))
    X[rng.random(X.shape) < 0.05] = np.nan
    return X


def test_scaler_partial_fit_matches_the_2_1_batch_fit():
    X = data_with_gaps()
    scaler = StandardScaler()
    for chunk in np.array_split(X, 7):
        scaler.partial_fit(chunk)
    batch = base21.StandardScaler().fit(X)
    np.testing.assert_allclose(scaler.mean_, batch.mean_)
    np.testing.assert_allclose(scaler.scale_, batch.scale_)
    np.testing.assert_allclose(scaler.transform(X), batch.transform(X))  # NaN stays NaN in both
    assert isinstance(scaler, base21.StandardScaler)


def test_constant_and_all_missing_columns_match_the_2_1_scaler():
    # 0.1 cannot be stored exactly, so a running mean drifts by round-off; the scaler must still give 0
    X = np.array([[0.1, 1.0, np.nan], [0.1, 2.0, np.nan], [0.1, 4.0, np.nan]])
    ours, theirs = StandardScaler().fit(X), base21.StandardScaler().fit(X)
    np.testing.assert_array_equal(ours.transform(X)[:, 0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(ours.mean_, theirs.mean_)  # [0.1, 2.33, NaN]
    np.testing.assert_allclose(ours.scale_, theirs.scale_)
    np.testing.assert_allclose(ours.transform(X), theirs.transform(X))


def test_scaler_options_and_constant_features():
    X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    scaler = StandardScaler().fit(X)
    assert scaler.scale_[0] == 0.0 and np.all(scaler.transform(X)[:, 0] == 0)  # 2.1 rule: no division by zero
    np.testing.assert_allclose(StandardScaler(with_mean=False).fit(X).transform(X)[:, 1], X[:, 1] / np.std(X[:, 1]))


def test_scaler_errors():
    with pytest.raises(NotFittedError):
        StandardScaler().transform([[1.0]])
    scaler = StandardScaler().fit([[1.0, 2.0]])
    with pytest.raises(ValueError):
        scaler.partial_fit([[1.0, 2.0, 3.0]])  # wrong number of features
    with pytest.raises(ValueError, match="reshape"):
        scaler.partial_fit([1.0, 2.0])  # 1-D input
    with pytest.raises(TypeError):
        scaler.partial_fit([["a", "b"]])  # text
    with pytest.raises(ValueError):
        scaler.partial_fit([[np.inf, 1.0]])
    assert scaler.partial_fit(np.empty((0, 2))).n_samples_seen_.tolist() == [1, 1]  # empty chunk: no change


def test_minmax_partial_fit_matches_the_2_1_batch_fit():
    X = data_with_gaps(1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    for chunk in np.array_split(X, 5):
        scaler.partial_fit(chunk)
    batch = base21.MinMaxScaler(feature_range=(-1, 1)).fit(X)
    np.testing.assert_allclose(scaler.min_, batch.min_)
    np.testing.assert_allclose(scaler.transform(X), batch.transform(X))
    with pytest.raises(ValueError):
        MinMaxScaler(feature_range=(1, 0)).partial_fit(X)


def test_imputer_updates_its_means_on_the_fly():
    imputer = Imputer().partial_fit([[1.0, np.nan], [3.0, np.nan]])
    np.testing.assert_allclose(imputer.statistics_, [2.0, 0.0])  # no value yet -> fill_value
    imputer.partial_fit([[5.0, 10.0]])
    np.testing.assert_allclose(imputer.statistics_, [3.0, 10.0])
    np.testing.assert_allclose(imputer.transform([[np.nan, np.nan]]), [[3.0, 10.0]])
    X = data_with_gaps(2)
    np.testing.assert_allclose(Imputer().fit(X).statistics_, base21.SimpleImputer().fit(X).statistics_)


def test_imputer_strategies():
    constant = Imputer(strategy="constant", fill_value=-1).partial_fit([[1.0], [np.nan]])
    np.testing.assert_allclose(constant.transform([[np.nan]]), [[-1.0]])
    median = Imputer(strategy="median")
    with pytest.raises(ValueError, match="partial_fit"):
        median.partial_fit([[1.0]])  # needs all the data at once
    np.testing.assert_allclose(median.fit([[1.0], [2.0], [10.0]]).statistics_, [2.0])  # the 2.1 batch fit
    with pytest.raises(ValueError):
        Imputer(strategy="mean", fill_value="x").partial_fit([[1.0]])


def test_one_hot_encoder_adds_new_categories_incrementally():
    encoder = OneHotEncoder().partial_fit([["red"], ["blue"]])
    assert encoder.transform([["red"]]).shape == (1, 2)
    encoder.partial_fit([["green"], [""]])  # "" is missing, not a category
    assert [c.tolist() for c in encoder.categories_] == [["red", "blue", "green"]]  # in order of arrival
    np.testing.assert_array_equal(encoder.transform([["green"], ["red"], [None]]), [[0, 0, 1], [1, 0, 0], [0, 0, 0]])
    assert isinstance(encoder, base21.OneHotEncoder)  # the 2.1 transform does the encoding


def test_one_hot_encoder_fixed_width_and_passthrough_columns():
    X = np.array([[1.5, "a"], [2.5, "b"]], dtype=object)
    encoder = OneHotEncoder(columns=[1], max_categories=3).fit(X)
    np.testing.assert_array_equal(encoder.transform(X), [[1.5, 1, 0, 0], [2.5, 0, 1, 0]])
    encoder.partial_fit(np.array([[0.0, "c"], [0.0, "d"]], dtype=object))  # "d" does not fit
    assert [c.tolist() for c in encoder.categories_] == [["a", "b", "c"]]
    assert encoder.transform(np.array([[0.0, "d"]], dtype=object)).shape == (1, 4)  # width unchanged
    assert encoder.get_feature_names_out(["price", "size"]).tolist() == ["price", "size_a", "size_b", "size_c"]


def test_one_hot_encoder_errors():
    with pytest.raises(ValueError):
        OneHotEncoder(max_categories=0)
    with pytest.raises(NotFittedError):
        OneHotEncoder().transform([["a"]])
    with pytest.raises(TypeError):
        OneHotEncoder(columns=[0]).fit([["x", "text"]]).transform([["x", "text"]])  # text left unencoded
    strict = OneHotEncoder(handle_unknown="error").fit([["a"]])
    with pytest.raises(ValueError, match="unknown categories"):  # the 2.1 error message
        strict.transform([["b"]])
    with pytest.raises(ValueError):
        OneHotEncoder(handle_unknown="maybe").partial_fit([["a"]])


def test_empty_chunks_and_shape_errors():
    empty = np.empty((0, 2))
    assert not hasattr(MinMaxScaler().partial_fit(empty), "min_")  # an empty chunk changes nothing
    with pytest.raises(NotFittedError):
        Imputer().partial_fit(empty).transform([[1.0, 2.0]])
    assert not hasattr(OneHotEncoder().partial_fit(np.empty((0, 1), dtype=object)), "categories_")
    with pytest.raises(ValueError):
        Imputer(fill_value=np.inf).partial_fit([[1.0]])
    encoder = OneHotEncoder(columns=[1]).fit(np.array([[1.0, "a"]], dtype=object))
    with pytest.raises(ValueError):
        encoder.transform(np.array([["a"]], dtype=object))  # wrong number of columns
    with pytest.raises(ValueError):
        encoder.get_feature_names_out(["only one name"])


def test_one_hot_encoder_on_numeric_categories_with_missing_values():
    encoder = OneHotEncoder().partial_fit(np.array([[1.0], [np.nan], [2.0]]))  # NaN is missing, not a category
    assert [c.tolist() for c in encoder.categories_] == [[1.0, 2.0]]
    np.testing.assert_array_equal(encoder.transform(np.array([[2.0], [np.nan]])), [[0, 1], [0, 0]])
